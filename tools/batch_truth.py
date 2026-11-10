#!/usr/bin/env python3
"""The offline answer for one dataset: what batch PCA says about the windows
that actually get words.

This is the reference an online estimate is graded against. It is NOT the
online algorithm and it reads none of its output; it only needs the settings
file, so it can be run before the C has ever been run on the data.

It works on ANY frames file, synthetic or real UCR, which is why it is a tool
of its own rather than part of the generator.

What it writes, next to the data in ../truth/<name>.truth.json:

    shares       the top-k variance shares, normalised among themselves,
                 which is the quantity the C's own eigenvalue estimate tracks
    directions   the top-k eigenvectors, k rows of m numbers. Without these
                 you can compare percentages but you cannot ask the only
                 question that matters: did the online basis find the right
                 directions? That answer is one angle, in degrees.
    best_bits    the optimal allocation for those shares, by exhaustive search
    ring_window  the ring size the collected-quantile rule demands here

    python3 tools/batch_truth.py tests/synthetic/cfg/stationary_theta89.cfg
"""

import json
import os
import sys

import numpy as np

# Must match src/spartan.h:20-21. There the cap is DERIVED:
#     #define MAX_ALPHABET 16
#     #define MAX_BITS (__builtin_ctz(MAX_ALPHABET))
# so it is derived here the same way instead of being written out as a second
# number that can drift. For a power of two, bit_length()-1 is ctz.
# Change MAX_ALPHABET in the header and change the one number below to match.
MAX_ALPHABET = 16
MAX_BITS = MAX_ALPHABET.bit_length() - 1

# Must match SPARTAN_EPS in src/spartan.h. znorm divides by (std + EPS).
SPARTAN_EPS = 1e-12


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def read_cfg(path):
    """Parse the settings file the C reads, with the same rules: one
    `key = value` per line, '#' starts a comment, blanks ignored.

    Returns a plain dict of strings. Unlike the C this does NOT reject unknown
    keys: the C owns that check, and duplicating it here would mean two places
    to update every time a knob is added.
    """
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                sys.exit("%s: line is not 'key = value': %r" % (path, line))
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def geometry(cfg):
    """The handful of settings this tool needs, with the C's own defaults
    filled in where the file is silent.

    stride defaults to window and total_bits to 2*components, exactly as
    config.c:283-284 resolves them, so a file that leaves them out is read the
    same way here as there.
    """
    g = dict(
        input=cfg["input"],
        d=int(cfg.get("channels", 1)),
        m_time=int(cfg["window"]),
        k=int(cfg["components"]),
        warmup=int(cfg["warmup"]),
        lam=float(cfg.get("lambda", 0.5)),
        ring_w=int(cfg.get("ring_window", 0)),
    )
    g["stride"] = int(cfg.get("stride", g["m_time"]))
    g["total_bits"] = int(cfg.get("total_bits", 2 * g["k"]))
    g["m"] = g["d"] * g["m_time"]
    return g


# --------------------------------------------------------------------------
# reading the stream the way the C reads it
# --------------------------------------------------------------------------


def read_windows(path, d, m_time, stride):
    """Every window the stream hands out, as rows of an (n_windows, d*m_time)
    array.

    The layout matches stream.c:191-203, Mode-1 unfolding: channel c occupies
    m_time contiguous doubles, so element [c*m_time + i] is channel c at time
    i within the window. For d = 1 that is just the contiguous slice.

    Blank lines and lines starting with '#' are skipped, as fetch_frame does.
    """
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals.extend(float(x) for x in s.replace(",", " ").split())

    flat = np.asarray(vals, dtype=float)
    if len(flat) % d != 0:
        sys.exit("%s: %d numbers is not a multiple of channels = %d"
                 % (path, len(flat), d))
    frames = flat.reshape(-1, d)                   # (n_frames, d)

    n_frames = len(frames)
    n_win = 0 if n_frames < m_time else 1 + (n_frames - m_time) // stride

    out = np.empty((n_win, d * m_time))
    for w in range(n_win):
        block = frames[w * stride: w * stride + m_time]      # (m_time, d)
        out[w] = block.T.reshape(-1)                         # channels first
    return out


def znorm_rows(X, d):
    """z-norm PER CHANNEL, the way main.c:366-368 does it.

    The C loops the channels and calls znorm_row on each m_time slice, so
    every channel gets its own mean and its own spread. Normalising the whole
    concatenated row of d*m_time numbers in one go is a DIFFERENT operation
    and would compute the offline answer on data the C never saw.

    Identical to the whole-row version when d = 1.

    Divides by m_time, not m_time-1, and adds SPARTAN_EPS so a flat channel
    does not divide by zero, matching spartan.c:34 znorm_row.

    Applied here even though the synthetic generator already z-normed: the C
    normalises everything it reads, so the truth must be computed on the same
    numbers. On already-normalised univariate data this is near-identity, off
    only by the epsilon.
    """
    B = X.reshape(len(X), d, -1)
    mu = B.mean(axis=2, keepdims=True)
    sd = np.sqrt(((B - mu) ** 2).mean(axis=2, keepdims=True)) + SPARTAN_EPS
    return ((B - mu) / sd).reshape(len(X), -1)


# --------------------------------------------------------------------------
# the offline answer
# --------------------------------------------------------------------------


def batch_pca(Z, k):
    """Top-k eigenvalues and eigenvectors of the covariance of these windows.

    Centring is by the COLUMN MEAN of the whole set, which is what PCA does
    and what sklearn does inside the reference. The C instead subtracts a
    RUNNING mean that it updates window by window. The two agree in the limit
    and differ early in the stream, and that difference is one of the things
    the comparison is meant to size.

    Returns (shares, directions). shares are the top-k eigenvalues normalised
    among themselves, which is what the C's estimate tracks and what
    spartan.py reports after its own re-normalisation. directions is k x m,
    one eigenvector per row, in descending eigenvalue order.
    """
    Zc = Z - Z.mean(axis=0)
    C = Zc.T @ Zc / len(Z)
    w, V = np.linalg.eigh(C)              # ascending, orthonormal columns
    order = np.argsort(w)[::-1][:k]       # descending, keep k
    vals = w[order]
    dirs = V[:, order].T                  # rows are directions
    return vals / vals.sum(), dirs


def best_split(shares, total_bits, lam, max_bits=MAX_BITS):
    """The optimal bit allocation, by exhaustive search over every
    non-increasing allocation that spends the budget exactly.

    Per position i with importance e_i and bits b_i:

        gain_i = b_i * e_i  -  lam * (b_i - ref)^2 * e_i
        ref    = total_bits // k

    Slow and obviously correct, which is what a reference is for. The C solves
    the same problem with a dynamic program; if they disagree, the DP is
    wrong, not this.

    b_i may be ZERO. The batch reference floors it at 1; we allow 0, and that
    difference is one of the things the comparison exists to quantify.

    Returns (allocation, objective).
    """
    best, best_score = None, -1e18
    k = len(shares)
    ref = total_bits // k

    def rec(i, left, prev, acc):
        nonlocal best, best_score
        if i == k:
            if left != 0:
                return
            sc = sum(b * e - lam * (b - ref) ** 2 * e
                     for b, e in zip(acc, shares))
            if sc > best_score:
                best_score, best = sc, list(acc)
            return
        for b in range(min(prev, max_bits, left), -1, -1):
            acc.append(b)
            rec(i + 1, left - b, b, acc)
            acc.pop()

    rec(0, total_bits, max_bits, [])
    return best, best_score


def principal_angle_deg(A, B):
    """The largest principal angle between the row spaces of A and B, in
    degrees. Both must have orthonormal rows.

    Here arccos IS the right tool: the angle is genuinely non-zero, because a
    finite sample never recovers the planted directions exactly. The sqrt(eps)
    floor that makes arccos useless for testing "did the span move at all"
    sits at 6e-7 degrees, far below anything meaningful here.
    """
    sv = np.linalg.svd(A @ B.T, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)).max()))


# --------------------------------------------------------------------------


def fmt_row(label, vals, fmt):
    return "  %-9s %s" % (label, " ".join(fmt % v for v in vals))


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    cfg_path = sys.argv[1]
    g = geometry(read_cfg(cfg_path))

    print("stream   %s" % g["input"])
    X = read_windows(g["input"], g["d"], g["m_time"], g["stride"])
    Z = znorm_rows(X, g["d"])

    n = len(Z)
    if g["warmup"] >= n:
        sys.exit("warmup = %d but the stream only has %d windows: nothing "
                 "would get a word" % (g["warmup"], n))
    post = Z[g["warmup"]:]
    print("windows  %d total, %d after the %d warm-up"
          % (n, len(post), g["warmup"]))

    shares, dirs = batch_pca(post, g["k"])
    bits, score = best_split(shares.tolist(), g["total_bits"], g["lam"])
    alphabet = [1 << b for b in bits]

    print()
    print("batch PCA over the windows that get words")
    print(fmt_row("share", shares, "%6.4f"))
    print(fmt_row("bits", bits, "%6d"))
    print(fmt_row("alphabet", alphabet, "%6d"))
    print("  %-9s %.6f" % ("objective", score))

    # The collected-quantile rule: a breakpoint is an empirical quantile of
    # the projections in the ring, so each cell needs enough points to mean
    # something. Ten per cell is the rule we settled on.
    need_ring = 10 * max(alphabet)
    note = ""
    if g["ring_w"] and g["ring_w"] < need_ring:
        note = "   <-- cfg has %d, TOO SMALL" % g["ring_w"]
    print("  %-9s %d%s" % ("ring need", need_ring, note))

    # If the data was planted by make_datasets.py, grade the batch answer
    # against what was planted. On real data there is nothing to compare to.
    meta_path = os.path.splitext(g["input"])[0] + ".meta.json"
    planted_deg = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        planted = np.asarray(meta["shapes"]["directions"])
        planted_deg = principal_angle_deg(planted, dirs)
        print()
        print("angle to the planted shapes  %.2f deg" % planted_deg)

    name = os.path.splitext(os.path.basename(g["input"]))[0]
    out_dir = os.path.join(os.path.dirname(os.path.dirname(g["input"])),
                           "truth")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, name + ".truth.json")

    with open(out_path, "w") as f:
        json.dump(dict(
            name=name,
            source=g["input"],
            cfg=cfg_path,
            geometry={kk: g[kk] for kk in
                      ("d", "m_time", "m", "stride", "k", "total_bits",
                       "lam", "warmup")},
            over_windows=[g["warmup"], n],
            shares=shares.tolist(),
            directions=dirs.tolist(),
            best_bits=bits,
            alphabet=alphabet,
            objective=score,
            ring_window_needed=need_ring,
            angle_to_planted_deg=planted_deg,
        ), f, indent=1)

    print("wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
