#!/usr/bin/env python3
"""Compare Online SPARTAN against the batch reference and write the paper's
tables.

Reads what each side already wrote to disk, lines them up, prints the six
metrics we kept, and writes one CSV per figure into --out. The CSVs are meant
to be read straight by PGFPlots or Lilaq, so the plot in the PDF and the
number on this terminal can never drift apart.

The distances are computed by THEIR code, loaded from their checkout, so the
headline numbers are not something this script invented.

Run the C first (with out_dir set: it writes settings.cfg there, which is
where every setting of the run is read from), then tools/ref_spartan.py.

    python3 tools/compare_stages.py tests/synthetic/cfg/stream_ref.cfg \
            --spartan-repo ~/dev/SPARTAN-main --out out/stream_ref

WHAT IS NOT COMPARED, AND WHY
Letters are not compared symbol by symbol. When two eigenvalues are equal,
the coordinates inside the plane they span are not defined: both sides are
free to pick any pair of perpendicular arrows in it, and both are right.
Measured here on a stationary series the two bases agree on the SPANS to
within a few degrees while the individual arrows sit tens of degrees apart,
so a letter-by-letter comparison reports a disagreement that does not exist.
What is compared instead is the DISTANCES the letters induce, which is what a
search actually uses and which none of that affects.

Covers EVERY epoch. Two windows written in the same epoch are bounded through
that epoch's dictionary by the authors' own spartan_pca_mindist. Two windows
written in different epochs are bounded by the cross-epoch bound, which is a
line-for-line port of word_box, box_through and cross_epoch_mindist from
src/spartan.c, so the number here is the number the C would produce.

WHAT IS IN THE CSVs

    conv_subspace.csv   pos, ev_online, ev_batch, bits_online, bits_batch
    epochs.csv          epoch, first_window, windows, angle_prev, angle_hind
    tlb_cdf.csv         tlb, frac_online, frac_batch, frac_hindsight,
                        frac_stored, frac_search
    tlb_summary.csv     method, mean, median, p10, zeros, violations
                        (method: online, batch, hindsight, stored, search)
    tlb_pairs.csv       kind, share, online, stored, hindsight, search
    pruning.csv         k, prune_online, prune_batch, prune_hindsight,
                        prune_stored, prune_search
    agreement.csv       k, overlap, overlap_hindsight
    scaling.csv         n, bytes_online, bytes_batch

WHAT THE COLUMNS MEAN

Our side, three ways of bounding a pair written in DIFFERENT epochs (inside
one epoch all three are the same MINDIST):
    search   the query window is in hand, as in a k-NN search. It is
             projected again onto the OLDER window's basis and written as a
             word of that epoch, then compared by the ordinary MINDIST. One
             extra projection per older epoch, no slack.
    box      the query's word is turned into a box, carried into the older
             epoch by the epoch map, and the slack sin(theta) * r is taken
             off, with r measured on the query window. CSV name: online.
    join     the same, but both windows survive only as words, so r is
             replaced by the largest it could be. A join over the archive.
             CSV name: stored.
The batch side, the authors' own code:
    batch warm-up   fitted on the warm-up windows only. CSV name: batch.
    batch all       fitted on every scored window, i.e. it has seen the whole
                    stream in advance. No streaming method can have that, so
                    it is the ceiling. CSV name: hindsight.
    scalars.tex         one \\newcommand per single number

Every table is kept under a few hundred rows on purpose. PGFPlots is TeX: a
table of several thousand rows is slow to typeset and can exhaust TeX's
memory. So the TLB distribution is written as a CDF on a fixed grid rather
than as one row per pair, and nothing here ever writes a row per window.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import sys

import numpy as np

try:
    from scipy.stats import rankdata
except ImportError:
    sys.exit(
        "needs scipy. Activate the virtualenv you installed their requirements into."
    )

# What follows are settings of the EVALUATION, not of the algorithm: how many
# windows to sample, which neighbourhood sizes to score. The algorithm's own
# settings and constants are read from the C's settings.cfg.
TLB_SAMPLE = 1000  # windows drawn for the distance matrices
TLB_SEED = 0
TOP_K = 10  # neighbourhood size for the headline overlap
CDF_POINTS = 201  # rows in tlb_cdf.csv, grid over [0, 1]
KNN_K = (1, 5, 10, 20, 50, 100)
ANGLE_BINS = ((0, 10), (10, 20), (20, 45), (45, 90))  # degrees, (lo, hi]
MAX_SKEW_S = 1800  # how far apart the inputs of one run may be
BYTES_PER_DOUBLE = 8
EUCLID_ROWS = 100  # rows per block of the true distances, see euclid_blocks


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_their_distances(repo):
    """Their distance_vectorized.py, loaded as a standalone file.

    Importing it through the package would run their __init__.py, which pulls
    in every method in the repo and each one's dependencies. This file needs
    only numpy and numba.

    benchmark/util is the copy their own eval_tlb.py imports, so it is the one
    their published numbers came from. TSB_Symbolic/util holds a second copy;
    fall back to it only if the first is missing.
    """
    root = os.path.expanduser(repo)
    src = os.path.join(root, "benchmark", "util", "distance_vectorized.py")
    if not os.path.exists(src):
        alt = os.path.join(root, "TSB_Symbolic", "util", "distance_vectorized.py")
        src = alt if os.path.exists(alt) else src
    if not os.path.exists(src):
        sys.exit("cannot find distance_vectorized.py under %s" % root)
    spec = importlib.util.spec_from_file_location("ref_dist", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from read_settings import geometry


def read_windows(path, d, m_time, stride):
    """Mode-1 unfolded windows, matching stream.c:191-203."""
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals.extend(float(x) for x in s.replace(",", " ").split())
    flat = np.asarray(vals, dtype=float)
    if len(flat) % d != 0:
        sys.exit(
            "%s: %d numbers is not a multiple of channels = %d" % (path, len(flat), d)
        )
    frames = flat.reshape(-1, d)
    if len(frames) < m_time:
        sys.exit(
            "%s: %d frames is shorter than one window of %d"
            % (path, len(frames), m_time)
        )
    n_win = 1 + (len(frames) - m_time) // stride
    out = np.empty((n_win, d * m_time))
    for w in range(n_win):
        out[w] = frames[w * stride : w * stride + m_time].T.reshape(-1)
    return out


def znorm_rows(X, d, eps):
    """z-norm PER CHANNEL, the way main.c:366-368 does it.

    The C loops the channels and calls znorm_row on each m_time slice, so
    every channel gets its own mean and its own spread. Normalising the whole
    concatenated row of d*m_time numbers in one go is a DIFFERENT operation:
    it would let a loud channel set the scale for a quiet one and would leave
    each channel with a non-zero mean of its own.

    Identical to the whole-row version when d = 1, which is why this went
    unnoticed on every univariate dataset so far.

    Layout is Mode-1 unfolding (stream.c:191-203): channel c occupies the
    m_time contiguous doubles at [c*m_time : (c+1)*m_time], so the reshape to
    (n, d, m_time) splits exactly along the channel boundaries.

    Divides by m_time, not m_time-1, and adds SPARTAN_EPS, matching
    spartan.c:34 znorm_row.
    """
    B = X.reshape(len(X), d, -1)
    mu = B.mean(axis=2, keepdims=True)
    sd = np.sqrt(((B - mu) ** 2).mean(axis=2, keepdims=True)) + eps
    return ((B - mu) / sd).reshape(len(X), -1)


def read_our_dict(path, k):
    """dict_0.csv -> (ev, bits, alphabet, breakpoints).

    Columns are looked up BY NAME. A change in the C's column order then shows
    up as a clear error instead of as silently shifted numbers.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != k:
        sys.exit("%s has %d rows, expected %d" % (path, len(rows), k))
    names = list(rows[0].keys())
    for want in ("ev", "bits", "alphabet"):
        if want not in names:
            sys.exit(
                "%s has no column '%s'. Columns are: %s"
                % (path, want, ", ".join(names))
            )
    n_bkpt = sum(1 for c in names if c.startswith("bkpt"))

    ev, bits, alpha, bkpt = [], [], [], []
    for r in rows:
        ev.append(float(r["ev"]))
        bits.append(int(r["bits"]))
        a = int(r["alphabet"])
        alpha.append(a)
        cuts = []
        for b in range(min(a, n_bkpt)):
            cell = (r["bkpt%d" % b] or "").strip()
            cuts.append(float("inf") if cell == "inf" else float(cell))
        bkpt.append(cuts)
    return np.array(ev), bits, alpha, bkpt


def read_our_basis(path):
    """basis_0.csv -> k rows of m numbers. First column is the row index."""
    out = []
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            out.append([float(v) for v in row[1:]])
    return np.array(out)


def read_words(path):
    idx, syms = [], []
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        head = next(rdr)
        # Only the PCA letters. With level_letters on, the C appends
        # level<c>,spread<c> columns after them; those are not positions of
        # the SPARTAN word and their dictionary has no breakpoints here, so
        # passing them through would index past the end of it.
        keep = [i for i, h in enumerate(head) if h.startswith("sym")]
        for row in rdr:
            idx.append(int(row[0]))
            syms.append([int(row[i]) for i in keep])
    return np.array(idx), np.array(syms)


def refuse_stale(paths, skew):
    """Stop if the inputs were not all written by the same run.

    This exists because of a failure that has already happened three times
    and never once announced itself. A step of the pipeline dies, the next
    step finds YESTERDAY's file sitting where it expected today's, reads it
    without complaint, and prints a full page of plausible numbers that
    describe two different runs stitched together. Nothing is out of range,
    nothing looks wrong, and the result is simply false.

    One pipeline run writes all of these within seconds of each other, so
    any real gap means a step was skipped. The check is on modification
    times rather than on content because content gives no way to tell a
    stale file from a fresh one that happens to agree.

    paths is a list of (label, path). Missing files are reported as missing,
    which is the other half of the same problem.
    """
    missing = [(lab, p) for lab, p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(
            "missing input, so an earlier step of the pipeline did not "
            "finish:\n" + "\n".join("  %-26s %s" % (lab, p) for lab, p in missing)
        )
    seen = [(lab, p, os.path.getmtime(p)) for lab, p in paths]
    newest = max(t for _, _, t in seen)
    stale = [(lab, p, newest - t) for lab, p, t in seen if newest - t > skew]
    if stale:
        sys.exit(
            "these inputs are older than the newest one by more than %d s, "
            "so they are\nleft over from an earlier run and comparing them "
            "would mix two runs:\n"
            % skew
            + "\n".join(
                "  %-26s %7.0f s older   %s" % (lab, age, p) for lab, p, age in stale
            )
            + "\n\nRe-run the whole pipeline for this cfg. If a step failed, "
            "read the log\nfrom the top: the error is there, not at the "
            "end."
        )


def read_epochs(out_dir):
    """epochs.csv -> one dict per epoch, in order.

    first_window is the index of the first window WRITTEN by that epoch: the
    C logs it at the promotion, before the window in hand is projected, so
    that window already goes through the new basis (main.c:815, :924).
    """
    p = os.path.join(out_dir, "epochs.csv")
    if not os.path.exists(p):
        sys.exit("missing %s: the C was not run with out_dir set" % p)
    out = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            out.append(
                dict(
                    epoch=int(r["epoch"]),
                    first=int(r["first_window"]),
                    theta_cut=float(r["theta_cut_rad"]),
                    staleness=float(r["staleness_rad"]),
                )
            )
    if not out:
        sys.exit("%s has no epochs: the run never left the warm-up" % p)
    if [e["epoch"] for e in out] != list(range(len(out))):
        sys.exit("%s: epochs are not numbered 0, 1, 2, ..." % p)
    return out


# --------------------------------------------------------------------------
# the cross-epoch bound, ported from src/spartan.c
#
# Each function below is the same arithmetic as the C function of the same
# name, done on whole arrays at once. Nothing is improved or tightened here:
# the point is to report what the C's bound is worth, so a difference
# between the two would make every number in this file about something else.
# --------------------------------------------------------------------------


def digitize(Y, bkpt, alphabet):
    """Letters for projections Y (n, k), as spartan.c digitize writes them.

    The C walks up while y is STRICTLY greater than a cut point, so the
    letter is the number of cut points strictly below y, which is
    searchsorted with side='left'.
    """
    out = np.empty(Y.shape, dtype=int)
    for j, (cuts, a) in enumerate(zip(bkpt, alphabet)):
        out[:, j] = np.searchsorted(np.asarray(cuts[: a - 1]), Y[:, j], side="left")
    return out


def word_box(words, bkpt, alphabet, x_max):
    """spartan.c word_box: the k intervals a word stands for, clipped to
    [-x_max, x_max]. words is (n, k); returns lo, hi of the same shape."""
    n, k = words.shape
    lo = np.empty((n, k))
    hi = np.empty((n, k))
    for j, (cuts, a) in enumerate(zip(bkpt, alphabet)):
        # edges[s] is the low end of cell s, edges[s+1] its high end
        edges = np.concatenate(
            (
                [-x_max],
                np.clip(np.asarray(cuts[: a - 1], dtype=float), -x_max, x_max),
                [x_max],
            )
        )
        lo[:, j] = edges[words[:, j]]
        hi[:, j] = edges[words[:, j] + 1]
    return lo, hi


def epoch_map(B_to, B_from, eps):
    """spartan.c epoch_map: the k x k table that carries coordinates of the
    FROM epoch into the TO epoch, and the sine of the largest angle between
    the two subspaces. Returns (None, None) where the C returns -3, a pair
    of epochs at 90 degrees that cannot be translated."""
    M = B_to @ B_from.T
    s_min = float(np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0).min())
    if s_min < eps:
        return None, None
    return M, math.sqrt(1.0 - s_min * s_min)


def box_through(M, lo, hi):
    """spartan.c box_through: the smallest upright box around the image of
    each box under M. Interval arithmetic: a positive coefficient takes the
    low end for the low bound, a negative one takes the high end."""
    pos, neg = np.maximum(M, 0.0), np.minimum(M, 0.0)
    return lo @ pos.T + hi @ neg.T, hi @ pos.T + lo @ neg.T


def box_gaps(q_lo, q_hi, c_lo, c_hi):
    """The first half of spartan.c cross_epoch_mindist: for every query box
    against every candidate box, sqrt of the summed squared gaps. Returns
    (n_query, n_candidate). The slack is taken off by the caller, once."""
    g1 = c_lo[None, :, :] - q_hi[:, None, :]
    g2 = q_lo[:, None, :] - c_hi[None, :, :]
    gap = np.maximum(0.0, np.maximum(g1, g2))
    return np.sqrt((gap * gap).sum(axis=2))


def residual_exact(x, Y):
    """spartan.c residual_len with the window in hand: the length of the part
    of the window its own basis did not capture."""
    d = (x * x).sum(axis=1) - (Y * Y).sum(axis=1)
    return np.sqrt(np.maximum(d, 0.0))


def residual_worst(lo, hi, x_max):
    """spartan.c residual_len when only the word survives.

    The window is gone, so its length is replaced by the largest it could
    have been, x_max, as the C's comment prescribes. The kept part is at
    least as long as the point of the word's box nearest the origin, so that
    much is known to have been captured and is taken off. Still an
    over-estimate of r, which is the direction that keeps the bound valid.
    """
    near = np.where((lo <= 0.0) & (hi >= 0.0), 0.0, np.minimum(np.abs(lo), np.abs(hi)))
    d = x_max * x_max - (near * near).sum(axis=1)
    return np.sqrt(np.maximum(d, 0.0))


# --------------------------------------------------------------------------
# writing tables a TeX plotting package can read
# --------------------------------------------------------------------------


def write_table(path, header, rows):
    """One CSV, comma separated, with a header of plain column names.

    Guarantees the three things that make PGFPlots fail with an unhelpful
    message: no empty cell, no nan, no inf. A row carrying any of them is a
    measurement that does not exist, so it is dropped rather than written as
    a hole for TeX to trip over.

    Returns the number of DATA rows written, not counting the header.
    """
    clean = []
    for r in rows:
        vals = []
        ok = True
        for v in r:
            if isinstance(v, str):
                vals.append(v)
                continue
            fv = float(v)
            if not math.isfinite(fv):
                ok = False
                break
            # %.6g keeps six significant digits and never emits an exponent
            # TeX cannot parse, e.g. 1e-07 stays readable as 1e-07 which
            # pgfplots does accept.
            vals.append("%.6g" % fv)
        if ok:
            clean.append(vals)
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for vals in clean:
            f.write(",".join(vals) + "\n")
    return len(clean)


def write_scalars(path, pairs, prefix=""):
    """One \\newcommand per single number, so the prose can cite them.

    TeX command names may contain LETTERS ONLY: no digits, no underscore. The
    names here are therefore written out in words (tlbptenonline, not
    tlb_p10_online). A name that breaks the rule is rejected here rather than
    at typeset time, where the error points at the wrong line.

    prefix exists because \\newcommand REFUSES to redefine a name that already
    exists. Without it, \\input of a second dataset's scalars aborts the
    build. Give each dataset its own prefix and \\thetamax becomes
    \\streamthetamax, \\stationarythetamax and so on.
    """
    if prefix and not prefix.isalpha():
        sys.exit("--tex-prefix %r must be letters only; TeX will reject it" % prefix)
    out = [(prefix + name, val) for name, val in pairs]
    for name, _ in out:
        if not name.isalpha():
            sys.exit("scalar name %r is not letters only; TeX will reject it" % name)
    with open(path, "w") as f:
        f.write("%% written by tools/compare_stages.py, do not edit\n")
        for name, val in out:
            if isinstance(val, str):
                f.write("\\newcommand{\\%s}{%s}\n" % (name, val))
            else:
                f.write("\\newcommand{\\%s}{%.4g}\n" % (name, float(val)))
    return len(out)


# --------------------------------------------------------------------------
# the pieces of the comparison
# --------------------------------------------------------------------------


def to_mindist_breakpoints(bkpt, alphabet):
    """Our cut points in THEIR layout.

    They carry a leading -MAX and a trailing +MAX, so their array has
    alphabet+1 entries and theirs[j] equals ours[j-1]. Their mindist indexes
    it directly, so it has to match exactly or the distances are silently
    wrong.
    """
    out = []
    for j, (cuts, a) in enumerate(zip(bkpt, alphabet)):
        inner = list(cuts[: a - 1])
        # An alphabet of a needs exactly a-1 internal cuts. Fewer means
        # dict_0.csv has fewer bkpt columns than this alphabet requires, and
        # the short row that would result is read by THEIR mindist as if it
        # were full: it indexes bkpt[sym] directly, so the distances come out
        # wrong rather than the program coming out broken.
        if len(inner) != a - 1:
            sys.exit(
                "dict_0.csv position %d: alphabet %d needs %d cut "
                "points but only %d were written. Does the C export "
                "enough bkpt columns for MAX_ALPHABET?" % (j, a, a - 1, len(inner))
            )
        if not all(np.isfinite(v) for v in inner):
            sys.exit(
                "dict_0.csv position %d: a cut point is nan or inf. "
                "Only the sentinel past the last cell may be infinite." % j
            )
        row = [-sys.float_info.max] + inner + [sys.float_info.max]
        out.append(np.asarray(row, dtype=float))
    return out


def principal_angles_deg(A, B):
    """Angles between the two row SPACES, in degrees.

    Blind to the order of the rows and to their signs, which is correct: both
    describe the same space. Whatever it does not see is reported separately
    by alignment() below.
    """
    sv = np.linalg.svd(A @ B.T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))


def alignment(ours, theirs):
    """For each of our directions, the reference direction it matches best.

    Returns a list of (their index, signed dot, angle in degrees). A negative
    dot means the arrow points the opposite way along the same axis; an index
    out of order means the two sides numbered a pair differently; an angle
    well above zero while the subspace angle is small means the arrow was
    rotated INSIDE the plane of a pair. None of the three makes either side
    wrong, and all three make letters incomparable.

    Kept for the terminal only. It is a diagnostic, not a score, so it is
    deliberately NOT written to a CSV: the angles inside a degenerate pair are
    arbitrary and a plot of them would invite a reader to compare noise.
    """
    M = ours @ theirs.T
    out = []
    for i in range(len(ours)):
        j = int(np.argmax(np.abs(M[i])))
        dot = float(M[i, j])
        out.append((j, dot, float(np.degrees(np.arccos(min(1.0, abs(dot)))))))
    return out


def tlb_ratios(md, euclid):
    """Their TLB convention exactly, as one ratio per ordered pair.

    eval_tlb.py:194-200 takes the mean of mindist/euclidean over ALL ordered
    pairs including the diagonal, with a pair that is zero on both sides
    counted as 1. The mean is taken by the caller so the same array can also
    give the median, the tail and the share of exact zeros.
    """
    both_zero = (md == 0) & (euclid == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(both_zero, 1.0, md / euclid)
    return np.where(np.isfinite(ratio), ratio, 1.0)


def tlb_summary(ratio, md, euclid):
    """Mean, median, worst tenth, share of exact zeros, violations.

    The mean alone hides the left tail, and the left tail is the part that
    costs: a pair whose bound reports zero can never be pruned, whatever the
    average says. p10 is the TLB at the worst tenth of pairs.

    A violation is a pair where the 'lower' bound came out LARGER than the
    true distance. That would be a wrong search answer, not a loose one. Their
    own script never checks it for SPARTAN. It must be zero.
    """
    return dict(
        mean=float(ratio.mean()),
        median=float(np.median(ratio)),
        p10=float(np.percentile(ratio, 10)),
        zeros=float(np.mean(md == 0)),
        violations=int(np.sum(md > euclid)),
    )


def tlb_cdf(ratio, grid):
    """Fraction of pairs whose TLB is at or below each point of the grid.

    Written instead of the raw ratios because there are a million of them and
    TeX would choke. The grid is fixed and shared by both sides, so the two
    curves are sampled at exactly the same places and can be subtracted.
    """
    r = np.sort(ratio.ravel())
    return np.searchsorted(r, grid, side="right") / float(len(r))


def rank_agreement(A, B):
    """Mean Spearman correlation between the two distance matrices, computed
    ROW BY ROW and then averaged.

    Row by row because that is the question a search asks: for THIS query, do
    the two sides put the candidates in the same order? A single correlation
    over the flattened matrix would mix queries together and hide a side that
    orders every neighbourhood wrongly but gets the overall scale right.

    rankdata, not argsort, because mindist produces many exact ties (any pair
    whose cells are adjacent in every position scores zero) and argsort would
    break them arbitrarily and differently on the two sides.
    """
    ra, rb = rankdata(A, axis=1), rankdata(B, axis=1)
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, 1.0)
    return float(np.mean(r))


def neighbour_overlap(A, B, topk):
    """Of the topk nearest each side finds for a query, how many are shared.

    The practical twin of the correlation: a search pulls a candidate set and
    verifies it exactly, so what matters is whether the two sets hold the same
    items, not the order inside them. The diagonal is removed first, since a
    window is always its own nearest neighbour on both sides and would inflate
    the number for free.
    """
    A = A.copy()
    B = B.copy()
    np.fill_diagonal(A, np.inf)
    np.fill_diagonal(B, np.inf)
    ta = np.argpartition(A, topk, axis=1)[:, :topk]
    tb = np.argpartition(B, topk, axis=1)[:, :topk]
    shared = [len(set(ta[i]).intersection(tb[i])) for i in range(len(A))]
    return float(np.mean(shared)) / topk


def pruning_power(md, euclid, kk):
    """Fraction of candidates a k-NN query can skip without reading them.

    For each query row: the radius r is the k-th smallest TRUE distance to
    anything other than itself, so it is the distance to the last answer the
    query will return. Every candidate whose mindist exceeds r is provably not
    in the answer and is dropped without touching the raw window. Everything
    at or below r survives as a candidate and has to be read and verified.

    This is what we wrongly expected TLB to report. TLB says how much of the
    true distance the bound recovers; this says how much of the data you get
    to skip, which is the number that turns into saved I/O.

    Self is excluded on both sides. A query with fewer than k other candidates
    is not counted.

    Returns a dict k -> mean fraction pruned.
    """
    n = len(md)
    e = euclid.copy()
    np.fill_diagonal(e, np.inf)
    m = md.copy()
    # -inf, not +inf: a window is not a pruning decision about itself, and
    # +inf here would score every query as having pruned one extra candidate.
    np.fill_diagonal(m, -np.inf)
    out = {}
    for k in kk:
        if k >= n:
            continue
        r = np.partition(e, k - 1, axis=1)[:, k - 1]  # k-th nearest, true
        pruned = (m > r[:, None]).sum(axis=1)
        out[k] = float(np.mean(pruned)) / (n - 1)
    return out


def state_bytes_online(k, m, ring_w, max_alphabet):
    """Bytes the streaming side holds while it runs, whatever the stream
    length. Counted from what main.c allocates:

        4 bases of k x m          live, averaged, frozen, the one being built
        1 ring of ring_w x m      the last windows themselves; a re-basing
                                  is PCA on them
        2 rings of ring_w x k     projections for the type 1 test, and the
                                  re-basing's projections, each with 2k sums
        3 vectors of m            running mean, the window, its centred
                                  copy (Oja's input only)
        the cut points            k x MAX_ALPHABET
        two scratch blocks        2k^2 + k for the angles,
                                  2k^2 + k + 2m for subspace_advance

    The warm-up buffer is not here: it is freed when epoch 0 opens.

    Computed from the geometry rather than measured: a measurement would
    include the input buffer and the allocator's slack, which are not the
    algorithm.
    """
    doubles = (
        4 * k * m
        + ring_w * m
        + 2 * (ring_w * k + 2 * k)
        + 3 * m
        + k * max_alphabet
        + (2 * k * k + k)
        + (2 * k * k + k + 2 * m)
    )
    return BYTES_PER_DOUBLE * doubles


def epoch_bytes(k, m, max_alphabet):
    """Bytes ONE closed epoch leaves behind: its basis and its cut points.
    This is the only thing that grows, and it grows with the number of
    epochs, not with the number of windows."""
    return BYTES_PER_DOUBLE * (k * m + k * max_alphabet)


def state_bytes_batch(n, m):
    """Bytes the batch side must hold to do what it does.

    It fits PCA on the whole corpus and takes empirical quantiles over the
    whole corpus, so every window has to be resident: n x m doubles. The
    covariance and the components are O(m^2) and O(k*m) on top and are not
    counted, which makes this an UNDERSTATEMENT of the batch footprint, in the
    batch side's favour.
    """
    return BYTES_PER_DOUBLE * n * m


def human_bytes(b):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return "%.1f %s" % (b, unit)
        b /= 1024.0


# --------------------------------------------------------------------------


def fmt(vals, f="%8.4f"):
    return " ".join(f % v for v in vals)


def online_distances(dist, take_idx, wo, ep_of, eps, Z, x, m, tiny):
    """Our lower bound for every ordered pair of the sampled windows.

    take_idx   window index of each sampled window, increasing
    wo         their words, (n, k)
    ep_of      the epoch that wrote each, (n,)
    eps        per-epoch dicts with basis, bkpt, alpha, mdb
    x          the windows as the C projected them, z-normed and nothing
               subtracted, (n, m)
    tiny       SPARTAN_EPS, below which epoch_map calls a pair untranslatable

    Returns (md_search, md_live, md_stored, apart, skipped).

    Same epoch: the authors' spartan_pca_mindist through that epoch's cut
    points, identical in both matrices.

    Different epochs: the LATER window is the query and the earlier one the
    candidate, which is the only order a stream produces: the past is stored
    and the present asks. The query's box is carried into the candidate's
    epoch and the slack sin(theta) * r is taken off, where r is the part of
    the query its own basis did not capture.
        md_live    r computed from the query window itself
        md_stored  r replaced by the largest value it could have
    Both are filled symmetrically so the matrix can be read by rows.

    apart[i, j] is the largest angle, in degrees, between the bases that
    wrote the two windows: 0 inside an epoch, -1 for a pair epoch_map
    refused. It is the quantity the slack is built from, which is why the
    report splits on it and NOT on how many re-basings lie between the two
    windows: measured on stationary data, two neighbouring epochs sat 8.5
    degrees apart and the next two 56, so "one re-basing apart" averaged two
    cases that have nothing in common.

    skipped is the number of refused pairs; their bound is set to 0, which
    is always valid and never prunes.
    """
    n = len(take_idx)
    # a z-normed window has length at most sqrt(m) and nothing is
    # subtracted before projecting (spartan.c word_box)
    x_max = math.sqrt(m)
    search = np.zeros((n, n))
    live = np.zeros((n, n))
    stored = np.zeros((n, n))
    apart = np.zeros((n, n))
    skipped = 0

    rows = {e: np.flatnonzero(ep_of == e) for e in np.unique(ep_of)}
    box = {}
    r_live = {}
    r_worst = {}
    for e, ix in rows.items():
        E = eps[e]
        lo, hi = word_box(wo[ix], E["bkpt"], E["alpha"], x_max)
        box[e] = (lo, hi)
        r_live[e] = residual_exact(x[ix], x[ix] @ E["basis"].T)
        r_worst[e] = residual_worst(lo, hi, x_max)

        same = dist.spartan_pca_mindist(wo[ix], wo[ix], E["mdb"])
        search[np.ix_(ix, ix)] = same
        live[np.ix_(ix, ix)] = same
        stored[np.ix_(ix, ix)] = same

    for eq, iq in rows.items():  # the query's epoch, the later
        for ec, ic in rows.items():  # the candidate's, the earlier
            if ec >= eq:
                continue
            M, sin_t = epoch_map(eps[ec]["basis"], eps[eq]["basis"], tiny)
            if M is None:
                skipped += 2 * len(iq) * len(ic)
                apart[np.ix_(iq, ic)] = -1.0
                apart[np.ix_(ic, iq)] = -1.0
                continue
            deg = math.degrees(math.asin(min(1.0, sin_t)))
            apart[np.ix_(iq, ic)] = deg
            apart[np.ix_(ic, iq)] = deg
            q_lo, q_hi = box_through(M, *box[eq])
            gaps = box_gaps(q_lo, q_hi, *box[ec])
            a = np.maximum(0.0, gaps - sin_t * r_live[eq][:, None])
            b = np.maximum(0.0, gaps - sin_t * r_worst[eq][:, None])
            live[np.ix_(iq, ic)] = a
            live[np.ix_(ic, iq)] = a.T
            stored[np.ix_(iq, ic)] = b
            stored[np.ix_(ic, iq)] = b.T

            # search: the query window is projected onto the candidate's
            # basis and cut with the candidate's dictionary, so both words
            # are in one epoch and the ordinary MINDIST applies
            Ec = eps[ec]
            wq = digitize(x[iq] @ Ec["basis"].T, Ec["bkpt"], Ec["alpha"])
            c = dist.spartan_pca_mindist(wq, wo[ic], Ec["mdb"])
            search[np.ix_(iq, ic)] = c
            search[np.ix_(ic, iq)] = c.T
    return search, live, stored, apart, skipped


def euclid_blocks(dist, X, rows=EUCLID_ROWS):
    """Their euclidean_vectorized, called on blocks of rows.

    Theirs forms an n x n x m array in one go: 3.8 GB at n = 1000, m = 512,
    which killed the run. Each block is computed by their own function, so
    every distance is the same arithmetic on the same numbers.
    """
    return np.vstack(
        [dist.euclidean_vectorized(X[i : i + rows], X) for i in range(0, len(X), rows)]
    )


def load_reference(ref_dir, name, suffix):
    """One batch reference: its json and its words. suffix is 'ref' for the
    warm-up fit and 'hindsight' for the fit on the scored windows."""
    with open(os.path.join(ref_dir, "%s.%s.json" % (name, suffix))) as f:
        j = json.load(f)
    stem = name if suffix == "ref" else name + "." + suffix
    widx, words = read_words(os.path.join(ref_dir, stem + ".words.csv"))
    return dict(
        ev=np.asarray(j["ev"]),
        bits=list(j["bits"]),
        basis=np.asarray(j["directions"]),
        mdb=[np.asarray(b) for b in j["mindist_breakpoints"]],
        widx=widx,
        words=words,
        fit=j.get("fit_windows"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cfg")
    ap.add_argument("--spartan-repo", required=True)
    ap.add_argument(
        "--out", required=True, help="directory for the CSVs the paper reads"
    )
    ap.add_argument(
        "--ref-dir",
        default=None,
        help="where ref_spartan.py wrote its words and json. "
        "Must match the --out-dir it was given. Defaults to "
        "a 'ref' folder beside the data.",
    )
    ap.add_argument("--sample", type=int, default=TLB_SAMPLE)
    ap.add_argument(
        "--seed",
        type=int,
        default=TLB_SEED,
        help="which windows the distance matrices are drawn "
        "from. Every run uses the same seed by default so "
        "two configurations are compared on identical "
        "windows. Change it to find out how much of a "
        "difference between runs is just the draw.",
    )
    ap.add_argument(
        "--max-skew",
        type=int,
        default=MAX_SKEW_S,
        help="seconds an input may lag behind the newest one "
        "before it is treated as left over from an earlier "
        "run (default %d)" % MAX_SKEW_S,
    )
    ap.add_argument(
        "--tex-prefix",
        default="",
        help="letters-only prefix for the macro names in "
        "scalars.tex, so several datasets can be \\input "
        "into the same document without clashing",
    )
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        help="also print the arrow-by-arrow alignment and the "
        "breakpoints of epoch 0 against the warm-up "
        "reference, which are debugging aids and not in "
        "the paper",
    )
    args = ap.parse_args()

    g = geometry(args.cfg)
    k, m = g["k"], g["m"]
    dist = load_their_distances(args.spartan_repo)
    os.makedirs(args.out, exist_ok=True)

    name = os.path.splitext(os.path.basename(g["input"]))[0]
    ref_dir = args.ref_dir or os.path.join(
        os.path.dirname(os.path.dirname(g["input"])), "ref"
    )

    epochs = read_epochs(g["out_dir"])
    ne = len(epochs)

    inputs = [
        ("our words.csv", os.path.join(g["out_dir"], "words.csv")),
        ("our epochs.csv", os.path.join(g["out_dir"], "epochs.csv")),
    ]
    for e in range(ne):
        inputs.append(
            ("our dict_%d.csv" % e, os.path.join(g["out_dir"], "dict_%d.csv" % e))
        )
        inputs.append(
            ("our basis_%d.csv" % e, os.path.join(g["out_dir"], "basis_%d.csv" % e))
        )
    for suffix, stem in (("ref", name), ("hindsight", name + ".hindsight")):
        inputs.append(
            (
                "reference %s.json" % suffix,
                os.path.join(ref_dir, "%s.%s.json" % (name, suffix)),
            )
        )
        inputs.append(
            ("reference %s words" % suffix, os.path.join(ref_dir, stem + ".words.csv"))
        )
    refuse_stale(inputs, args.max_skew)

    # ------------------------------------------------------------ loading
    eps = []
    for e in range(ne):
        dpath = os.path.join(g["out_dir"], "dict_%d.csv" % e)
        ev, bits, alpha, bkpt = read_our_dict(dpath, k)
        # The dictionary must be internally consistent before anything is
        # read through it: the cut-point count comes from the alphabet, so an
        # alphabet that does not match the bits would hand their mindist the
        # wrong number of cut points and produce plausible wrong distances.
        for j, (b, a) in enumerate(zip(bits, alpha)):
            if a != (1 << b):
                sys.exit(
                    "%s position %d: alphabet %d is not 2^%d. The two "
                    "must agree or every distance through this "
                    "dictionary is wrong." % (dpath, j, a, b)
                )
        eps.append(
            dict(
                ev=ev,
                bits=bits,
                alpha=alpha,
                bkpt=bkpt,
                basis=read_our_basis(os.path.join(g["out_dir"], "basis_%d.csv" % e)),
                mdb=to_mindist_breakpoints(bkpt, alpha),
            )
        )

    widx_o, words_o = read_words(os.path.join(g["out_dir"], "words.csv"))
    warm = load_reference(ref_dir, name, "ref")
    hind = load_reference(ref_dir, name, "hindsight")

    # Four orderings meet here: the rows of our words.csv, the rows of each
    # reference's, and the window indices that index Z. They line up only if
    # every file is in increasing window order. Checked rather than assumed,
    # because if it ever stops holding every word pairs with the wrong window
    # and the only symptom is a worse TLB with no error anywhere.
    for who, idx in (
        ("ours", widx_o),
        ("the warm-up reference", warm["widx"]),
        ("the hindsight reference", hind["widx"]),
    ):
        if not np.all(np.diff(idx) > 0):
            sys.exit(
                "the words file from %s is not in strictly increasing "
                "window order, so words cannot be matched to windows" % who
            )

    common = np.intersect1d(np.intersect1d(widx_o, warm["widx"]), hind["widx"])
    if len(common) == 0:
        sys.exit("the three sides wrote words for no window in common")
    wo = words_o[np.isin(widx_o, common)]
    ww = warm["words"][np.isin(warm["widx"], common)]
    wh = hind["words"][np.isin(hind["widx"], common)]

    Z = znorm_rows(
        read_windows(g["input"], g["d"], g["m_time"], g["stride"]), g["d"], g["eps"]
    )
    n_total = len(Z)

    firsts = np.array([e["first"] for e in epochs])
    ep_common = np.searchsorted(firsts, common, side="right") - 1
    if ep_common.min() < 0:
        sys.exit("a word was written before epoch 0 opened at window %d" % firsts[0])

    # Re-derive every word: project the z-normed window, nothing subtracted
    # (main.c, project(symbol_basis, win, y)), on its epoch's basis,
    # cut with its epoch's dictionary, compare with what the C wrote. One
    # mismatch means this script and the C disagree about which epoch wrote
    # a window, or about the centring, or about a cut point, and then no
    # distance below would be about the run it claims to describe.
    mismatched = 0
    for e in range(ne):
        ix = np.flatnonzero(ep_common == e)
        if len(ix) == 0:
            continue
        w = common[ix]
        again = digitize(Z[w] @ eps[e]["basis"].T, eps[e]["bkpt"], eps[e]["alpha"])
        mismatched += int((again != wo[ix]).any(axis=1).sum())
    if mismatched:
        sys.exit(
            "%d of %d words could not be re-derived from the raw "
            "windows with the bases and dictionaries the C exported. "
            "The files do not describe one run, or this script reads "
            "an epoch boundary differently from the C." % (mismatched, len(common))
        )

    # --------------------------------------------- the distance matrices
    rng = np.random.default_rng(args.seed)
    take = rng.choice(len(common), size=min(args.sample, len(common)), replace=False)
    take.sort()
    n = len(take)
    win = common[take]
    ep_take = ep_common[take]

    # Their mindist subtracts -MAX from +MAX in cells it then masks out, and
    # numpy warns about the overflow. The masked value is never used.
    with np.errstate(over="ignore", invalid="ignore"):
        euclid = euclid_blocks(dist, Z[win])
        md_q, md_o, md_s, apart, skipped = online_distances(
            dist, win, wo[take], ep_take, eps, Z, Z[win], m, g["eps"]
        )
        md_w = dist.spartan_pca_mindist(ww[take], ww[take], warm["mdb"])
        md_h = dist.spartan_pca_mindist(wh[take], wh[take], hind["mdb"])

    ep_of_row = ep_take[:, None] == ep_take[None, :]
    same = ep_of_row
    cross = ~same
    share_same = float(same.mean())

    # ------------------------------------------------------------ report
    # Numbers only. What each one means is in the module docstring and in the
    # notes we keep beside the paper; the terminal is for reading results.
    ang0 = np.sort(principal_angles_deg(eps[0]["basis"], warm["basis"]))
    ev_err = np.abs(eps[0]["ev"] - warm["ev"])
    worst_pos = int(np.argmax(ev_err))

    ratio_q = tlb_ratios(md_q, euclid)
    ratio_o = tlb_ratios(md_o, euclid)
    ratio_s = tlb_ratios(md_s, euclid)
    ratio_w = tlb_ratios(md_w, euclid)
    ratio_h = tlb_ratios(md_h, euclid)
    s_q = tlb_summary(ratio_q, md_q, euclid)
    s_o = tlb_summary(ratio_o, md_o, euclid)
    s_s = tlb_summary(ratio_s, md_s, euclid)
    s_w = tlb_summary(ratio_w, md_w, euclid)
    s_h = tlb_summary(ratio_h, md_h, euclid)
    broken = any(s["violations"] for s in (s_q, s_o, s_s, s_w, s_h))

    pp_q = pruning_power(md_q, euclid, KNN_K)
    pp_o = pruning_power(md_o, euclid, KNN_K)
    pp_s = pruning_power(md_s, euclid, KNN_K)
    pp_w = pruning_power(md_w, euclid, KNN_K)
    pp_h = pruning_power(md_h, euclid, KNN_K)

    rho_w = rank_agreement(md_o, md_w)
    rho_h = rank_agreement(md_o, md_h)
    ov_w = {kk: neighbour_overlap(md_o, md_w, kk) for kk in KNN_K if kk < n}
    ov_h = {kk: neighbour_overlap(md_o, md_h, kk) for kk in KNN_K if kk < n}
    if not ov_w:
        sys.exit(
            "--sample %d is too small for any neighbourhood size in %s"
            % (n, list(KNN_K))
        )
    head_k = TOP_K if TOP_K in ov_w else max(ov_w)

    b_on = state_bytes_online(k, m, g["ring_w"], g["max_alphabet"])
    b_ep = epoch_bytes(k, m, g["max_alphabet"])
    b_ba = state_bytes_batch(n_total, m)

    W = 11  # width of one number column

    def row(label, *vals):
        print(
            "  %-26s" % label
            + "".join(
                v.rjust(W) if isinstance(v, str) else ("%*.4f" % (W, v)) for v in vals
            )
        )

    def pct(v):
        return "%.1f %%" % (100.0 * v)

    print(
        "%s   %d windows x %d   warm-up %d   scored %d   sample %d   "
        "seed %d" % (name, n_total, m, g["warmup"], len(common), n, args.seed)
    )
    print(
        "words re-derived %d/%d   pairs same epoch %s   cross %s"
        % (len(common), len(common), pct(share_same), pct(1.0 - share_same))
    )
    print()

    print("1 SUBSPACE")
    print(
        "  %-9s %8s %8s %11s %13s   %s"
        % ("", "first", "windows", "to prev", "to batch all", "bits")
    )
    ep_rows = []
    for e in range(ne):
        last = firsts[e + 1] if e + 1 < ne else n_total
        a_prev = (
            float(principal_angles_deg(eps[e]["basis"], eps[e - 1]["basis"]).max())
            if e
            else 0.0
        )
        a_hind = float(principal_angles_deg(eps[e]["basis"], hind["basis"]).max())
        ep_rows.append([e, firsts[e], last - firsts[e], a_prev, a_hind])
        print(
            "  %-9s %8d %8d %11s %9.2f deg   %s"
            % (
                "epoch %d" % e,
                firsts[e],
                last - firsts[e],
                ("%.2f deg" % a_prev) if e else "-",
                a_hind,
                " ".join(str(b) for b in eps[e]["bits"]),
            )
        )
    a_wh = float(principal_angles_deg(warm["basis"], hind["basis"]).max())
    print(
        "  %-9s %8d %8s %11s %9.2f deg   %s"
        % (
            "b.warm-up",
            0,
            g["warmup"],
            "-",
            a_wh,
            " ".join(str(b) for b in warm["bits"]),
        )
    )
    print(
        "  %-9s %8d %8d %11s %13s   %s"
        % (
            "b.all",
            g["warmup"],
            n_total - g["warmup"],
            "-",
            "-",
            " ".join(str(b) for b in hind["bits"]),
        )
    )
    print(
        "  epoch 0 vs batch warm-up   angle %.2f deg   ev worst diff %.4f "
        "at pos %d" % (ang0.max(), ev_err[worst_pos], worst_pos)
    )
    print()

    def head(title, *cols):
        top = [c.split("/")[0] for c in cols]
        bot = [c.split("/")[1] if "/" in c else "" for c in cols]
        print(("%-28s" % title) + "".join(t.rjust(W) for t in top))
        if any(bot):
            print(" " * 28 + "".join(b.rjust(W) for b in bot))

    head("2 TLB", "search", "box", "join", "batch/warm-up", "batch/all")
    for key in ("mean", "median", "p10"):
        row(key, s_q[key], s_o[key], s_s[key], s_w[key], s_h[key])
    row(
        "zero",
        pct(s_q["zeros"]),
        pct(s_o["zeros"]),
        pct(s_s["zeros"]),
        pct(s_w["zeros"]),
        pct(s_h["zeros"]),
    )
    row(
        "violations",
        str(s_q["violations"]),
        str(s_o["violations"]),
        str(s_s["violations"]),
        str(s_w["violations"]),
        str(s_h["violations"]),
    )
    if broken:
        print("  *** BOUND BROKEN ***")
    print()

    pair_rows = []
    head("2b TLB by pair", "share", "search", "box", "join", "batch/all")

    def part(label, mask, kind):
        if not mask.any():
            return
        r_ = [
            kind,
            float(mask.mean()),
            float(ratio_o[mask].mean()),
            float(ratio_s[mask].mean()),
            float(ratio_h[mask].mean()),
            float(ratio_q[mask].mean()),
        ]
        pair_rows.append(r_)
        row(label, pct(r_[1]), r_[5], r_[2], r_[3], r_[4])

    part("same epoch", same, "same")
    part("cross epoch", cross, "cross")
    for lo_deg, hi_deg in ANGLE_BINS:
        part(
            "  %d-%d deg" % (lo_deg, hi_deg),
            cross & (apart > lo_deg) & (apart <= hi_deg),
            "deg%dto%d" % (lo_deg, hi_deg),
        )
    part("all", np.ones_like(same), "all")
    if skipped:
        row("refused (bound 0)", str(skipped))
    print()

    head("3 PRUNING", "search", "box", "join", "batch/warm-up", "batch/all")
    for kk in sorted(pp_o):
        row("k = %d" % kk, pp_q[kk], pp_o[kk], pp_s[kk], pp_w[kk], pp_h[kk])
    print()

    head("4 AGREEMENT, box against", "batch/warm-up", "batch/all")
    row("spearman", rho_w, rho_h)
    row("top-%d overlap" % head_k, ov_w[head_k], ov_h[head_k])
    print()

    print("5 MEMORY")
    row("online state", human_bytes(b_on))
    row("online epochs (%d)" % ne, human_bytes(ne * b_ep))
    row("batch (%d windows)" % n_total, human_bytes(b_ba))
    print()

    # Queries use the cross-epoch bound the run was configured with; joins
    # have no window and are always box with the worst r
    qname = g["cross_epoch"]
    s_qry, pp_qry = (s_q, pp_q) if qname == "search" else (s_o, pp_o)
    print("6 OURS / BATCH ALL")
    row("TLB mean, query (%s)" % qname, s_qry["mean"] / s_h["mean"])
    row("TLB mean, join", s_s["mean"] / s_h["mean"])
    if head_k in pp_o and pp_h[head_k] > 0:
        row("prune k=%d, query (%s)" % (head_k, qname), pp_qry[head_k] / pp_h[head_k])
        row("prune k=%d, join" % head_k, pp_s[head_k] / pp_h[head_k])
    row("memory", (b_on + ne * b_ep) / float(b_ba))
    print()

    # ---------------------------------------------------- the diagnostics
    if args.diagnostics:
        Zc = Z[common[ep_common == 0]]
        bo, basis_o, basis_r = eps[0]["mdb"], eps[0]["basis"], warm["basis"]
        print("   ALIGNMENT, epoch 0 against batch warm-up (diagnostic)")
        flips = swaps = 0
        for i, (j, dot, deg) in enumerate(alignment(basis_o, basis_r)):
            flips += dot < 0
            swaps += j != i
            print(
                "     our %d -> their %d   dot %+.4f   arrow %5.1f deg "
                "apart%s%s"
                % (
                    i,
                    j,
                    dot,
                    deg,
                    "  FLIPPED" if dot < 0 else "",
                    "  SWAPPED" if j != i else "",
                )
            )
        print(
            "     %d of %d arrows point the opposite way, %d are numbered "
            "differently" % (flips, k, swaps)
        )
        print(
            "     an arrow tens of degrees away while the subspace angle "
            "is small means it was"
        )
        print("     rotated inside the plane of an equal pair. Neither side is wrong.")
        print()

        sd_o = (Zc @ basis_o.T).std(axis=0)
        sd_r = (Zc @ basis_r.T).std(axis=0)
        print("   BREAKPOINTS in standard deviations of their own axis (diagnostic)")
        for j, (a, b) in enumerate(zip(bo, warm["mdb"])):
            ia, ib = a[1:-1] / sd_o[j], b[1 : len(a) - 1] / sd_r[j]
            if len(ia) == 0 or len(ia) != len(ib):
                continue
            print(
                "     pos %d   spread ours %6.3f  theirs %6.3f   worst cut "
                "differs by %.3f sd"
                % (j, sd_o[j], sd_r[j], float(np.abs(ia - ib).max()))
            )
        print("     they round projections to 4 decimals before binning and we do not,")
        print("     so a difference at that scale is expected by construction.")
        print()

    # --------------------------------------------------------- the tables
    written = []

    p = os.path.join(args.out, "conv_subspace.csv")
    written.append(
        (
            p,
            write_table(
                p,
                ["pos", "ev_online", "ev_batch", "bits_online", "bits_batch"],
                [
                    [
                        j,
                        eps[0]["ev"][j],
                        warm["ev"][j],
                        eps[0]["bits"][j],
                        warm["bits"][j],
                    ]
                    for j in range(k)
                ],
            ),
        )
    )

    p = os.path.join(args.out, "epochs.csv")
    written.append(
        (
            p,
            write_table(
                p,
                ["epoch", "first_window", "windows", "angle_prev", "angle_hind"],
                ep_rows,
            ),
        )
    )

    grid = np.linspace(0.0, 1.0, CDF_POINTS)
    cdfs = [tlb_cdf(r, grid) for r in (ratio_o, ratio_w, ratio_h, ratio_s, ratio_q)]
    p = os.path.join(args.out, "tlb_cdf.csv")
    written.append(
        (
            p,
            write_table(
                p,
                [
                    "tlb",
                    "frac_online",
                    "frac_batch",
                    "frac_hindsight",
                    "frac_stored",
                    "frac_search",
                ],
                [[grid[i]] + [c[i] for c in cdfs] for i in range(CDF_POINTS)],
            ),
        )
    )

    p = os.path.join(args.out, "tlb_summary.csv")
    written.append(
        (
            p,
            write_table(
                p,
                ["method", "mean", "median", "p10", "zeros", "violations"],
                [
                    [lab, s["mean"], s["median"], s["p10"], s["zeros"], s["violations"]]
                    for lab, s in (
                        ("online", s_o),
                        ("batch", s_w),
                        ("hindsight", s_h),
                        ("stored", s_s),
                        ("search", s_q),
                    )
                ],
            ),
        )
    )

    p = os.path.join(args.out, "tlb_pairs.csv")
    written.append(
        (
            p,
            write_table(
                p,
                ["kind", "share", "online", "stored", "hindsight", "search"],
                pair_rows,
            ),
        )
    )

    p = os.path.join(args.out, "pruning.csv")
    written.append(
        (
            p,
            write_table(
                p,
                [
                    "k",
                    "prune_online",
                    "prune_batch",
                    "prune_hindsight",
                    "prune_stored",
                    "prune_search",
                ],
                [
                    [kk, pp_o[kk], pp_w[kk], pp_h[kk], pp_s[kk], pp_q[kk]]
                    for kk in sorted(pp_o)
                ],
            ),
        )
    )

    p = os.path.join(args.out, "agreement.csv")
    written.append(
        (
            p,
            write_table(
                p,
                ["k", "overlap", "overlap_hindsight"],
                [[kk, ov_w[kk], ov_h[kk]] for kk in sorted(ov_w)],
            ),
        )
    )

    # One row per decade-ish of stream length, so the figure shows the shape
    # of the two curves rather than one point each. The online row does not
    # depend on n, which IS the result: a horizontal line against a diagonal.
    ns = [
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        100000,
        200000,
        500000,
        1000000,
        2000000,
        5000000,
    ]
    p = os.path.join(args.out, "scaling.csv")
    written.append(
        (
            p,
            write_table(
                p,
                ["n", "bytes_online", "bytes_batch"],
                [[nn, b_on, state_bytes_batch(nn, m)] for nn in ns],
            ),
        )
    )

    cross_mean = float(ratio_o[cross].mean()) if cross.any() else 0.0
    p = os.path.join(args.out, "scalars.tex")
    n_mac = write_scalars(
        p,
        [
            ("thetamax", ang0.max()),
            ("everrmax", ev_err[worst_pos]),
            ("everrpos", str(worst_pos)),
            ("spearman", rho_w),
            ("spearmanhind", rho_h),
            ("overlapten", ov_w[head_k]),
            ("overlaptenhind", ov_h[head_k]),
            ("tlbsearch", s_q["mean"]),
            ("tlbonline", s_o["mean"]),
            ("tlbstored", s_s["mean"]),
            ("tlbbatch", s_w["mean"]),
            ("tlbhind", s_h["mean"]),
            ("tlbratio", s_o["mean"] / s_w["mean"]),
            ("tlbratiohind", s_o["mean"] / s_h["mean"]),
            ("tlbsame", float(ratio_o[same].mean())),
            ("tlbcross", cross_mean),
            ("tlbptenonline", s_o["p10"]),
            ("tlbptenbatch", s_w["p10"]),
            ("tlbptenhind", s_h["p10"]),
            ("nepochs", str(ne)),
            ("nwindows", str(n)),
        ],
        prefix=args.tex_prefix,
    )
    written.append((p, n_mac))

    for path, nrows in written:
        tag = "macros" if path.endswith(".tex") else "rows"
        print("wrote %-44s %4d %s" % (path, nrows, tag))
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
