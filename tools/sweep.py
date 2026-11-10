#!/usr/bin/env python3
"""Run many settings over one dataset and print one line per setting.

Takes a baseline settings file, varies it, and for each variation runs the C,
runs the reference, runs the comparison, and records a single row. The full
output of every run is kept in a log so a row that looks interesting can be
opened without running it again.

    python3 tools/sweep.py tests/synthetic/cfg/stream_ref.cfg \
            --bin ./build/spartan_toy --spartan-repo ~/dev/SPARTAN-main \
            --out out/sweep

WHAT IS SWEPT TOGETHER AND WHY

Two groups of settings are varied combinatorially, and they are the two
groups we have measured a connection in. Everything else moves one knob at a
time around the baseline, which costs levels*knobs runs instead of
levels^knobs and is the only way the grid stays finishable.

Block A, step x product x stride. All three enter the same quantity:

    lr      = step * stride / (m * m_time)
    product = warmup * lr

product is the dimensionless amount of learning done before the first word is
written. Below about 1 the basis has not converged and every later number is
measuring that instead of the algorithm. Sweeping step and warmup separately
would waste most of the grid on combinations that differ only in product, so
warmup is DERIVED from the product rather than chosen.

Block B, lambda x total_bits. The allocation objective penalises distance
from an equal split, (b - total_bits/k)^2, so the budget sets the point the
penalty pulls towards and the two cannot be read apart. Measured separately:
the 4-bit cap in the C starts to bind below lambda 0.1, so that corner is
where the two sides can stop being comparable.

Block C, everything else, one at a time.

WHAT IS NOT SWEPT

batch_truth.py is not run. It is the oracle and the comparison does not read
it; running it per config would be minutes of work nobody looks at. Run it
once per dataset by hand.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time


# Must match src/spartan.h:20-21.
MAX_ALPHABET = 16
MAX_BITS = MAX_ALPHABET.bit_length() - 1

# Levels. Edit here, not in the code below.
STEPS = (0.06, 0.12, 0.24, 0.48)
PRODUCTS = (0.5, 1.0, 1.9, 4.0)
STRIDES = (256, 192, 128, 64)
LAMBDAS = (0.1, 0.15, 0.3, 0.5, 0.8, 1.0)
TOTAL_BITS = (8, 16, 32)

ONE_AT_A_TIME = (
    ("basis_average", (0.01, 0.1, 0.3, 1.0)),
    ("theta_threshold_deg", (5, 10, 20, 45, 89)),
    ("ring_window", (160, 256, 512, 1024)),
    # How far past the data the incoming basis is pushed when it is frozen,
    # as a fraction of the threshold. 0 freezes where the data is. The C
    # accepts [0, 2] (config.c validate).
    ("freeze_ahead", (0.0, 0.25, 0.5, 1.0)),
    # These two are NOT independent: the C requires angle_period to be a
    # multiple of gs_period, because the angle has to be read on a basis
    # that has just been straightened or the number is not an angle. So the
    # levels are chosen to stay legal against the OTHER one's baseline:
    # divisors of angle_period 10, and multiples of gs_period 5.
    ("gs_period", (1, 2, 5, 10)),
    ("angle_period", (5, 10, 20, 50)),
    ("window", (128, 256, 512)),
    ("components", (4, 8, 16)),
)

# What the C uses when the settings file is silent (config.c
# config_default_mv), for the knobs swept here that a cfg may leave out.
C_DEFAULTS = {"freeze_ahead": "0", "basis_average": "0.1"}

# A config must leave at least this many windows unscored by the warm-up,
# or the comparison has too little to measure and the row is noise.
MIN_SCORED = 500


# --------------------------------------------------------------------------
# settings files
# --------------------------------------------------------------------------


def read_cfg(path):
    """Settings as an ordered dict of strings, comments and blanks dropped."""
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def write_cfg(path, cfg):
    with open(path, "w") as f:
        f.write("# written by tools/sweep.py, do not edit\n")
        for k, v in cfg.items():
            f.write("%-20s= %s\n" % (k, v))


def count_frames(path):
    """Frames in the data file. Needed to know how many windows a stride
    yields, which decides whether a warm-up leaves anything to score."""
    n = 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
    return n


def n_windows(n_frames, m_time, stride):
    if n_frames < m_time:
        return 0
    return 1 + (n_frames - m_time) // stride


def derived(cfg):
    """The numbers the C derives, computed the same way here.

    m is window * channels. lr is step / m, exactly as config.c:306-308
    computes it. There is NO stride term: an earlier version of this function
    had one, step * stride / (m * m_time), which the C never adopted. The two
    agree only when stride equals window, so every overlapping row of Block A
    was given a warm-up sized for a learning rate the C was not using, and
    its "product" label was off by window / stride (a factor of 4 at stride
    64 with window 256).
    """
    d = int(cfg.get("channels", 1))
    m_time = int(cfg["window"])
    m = d * m_time
    stride = int(cfg.get("stride", m_time))
    step = float(cfg["step"])
    return m, m_time, stride, step / float(m)


def warmup_for(cfg, product):
    """The warm-up that reaches this product with this cfg's step and stride.

    warmup = product / lr. Rounded to a whole window, which moves the product
    by less than one window's worth and is reported as asked for rather than
    as achieved.
    """
    _, _, _, lr = derived(cfg)
    return int(round(product / lr))


# --------------------------------------------------------------------------
# building the list of runs
# --------------------------------------------------------------------------


def feasible(cfg, n_frames):
    """Why this config cannot run, or None if it can.

    Checked before anything is executed so an impossible corner of the grid
    costs nothing and is reported as skipped with its reason, rather than
    failing halfway with a message about a missing file.
    """
    m, m_time, stride, _ = derived(cfg)
    k = int(cfg["components"])
    tb = int(cfg["total_bits"])
    warm = int(cfg["warmup"])

    if tb % k:
        return ("total_bits %d is not divisible by components %d, so the "
                "reference cannot express it" % (tb, k))
    if tb > MAX_BITS * k:
        return ("total_bits %d exceeds MAX_BITS %d times components %d"
                % (tb, MAX_BITS, k))
    nw = n_windows(n_frames, m_time, stride)
    if nw == 0:
        return "window %d is longer than the data" % m_time
    if nw - warm < MIN_SCORED:
        return ("warmup %d leaves %d of %d windows, under the %d needed"
                % (warm, nw - warm, nw, MIN_SCORED))
    gs = int(cfg.get("gs_period", 1))
    ang = int(cfg.get("angle_period", 1))
    if gs and ang % gs:
        return ("angle_period %d is not a multiple of gs_period %d, so the "
                "angle would be read on a basis that is not orthonormal"
                % (ang, gs))
    ring = int(cfg.get("ring_window", 0))
    need = 10 * (1 << min(MAX_BITS, tb // k + 1))
    if ring and ring < need:
        return "ring_window %d is under the %d this alphabet needs" % (ring,
                                                                       need)
    return None


def plan(base, n_frames, blocks):
    """Every run to do, as (block, label, cfg, axes).

    axes records WHICH knobs this row varies and to what, as real numbers
    rather than as text inside the label. The plotting step pivots on it, so
    nothing downstream has to pick a label apart with a regular expression
    and silently mis-parse a knob whose name later changes.
    """
    runs = [("--", "baseline", dict(base), {})]

    if "A" in blocks:
        for st in STEPS:
            for pr in PRODUCTS:
                for sd in STRIDES:
                    c = dict(base)
                    c["step"] = repr(st)
                    c["stride"] = str(sd)
                    c["warmup"] = str(warmup_for(c, pr))
                    runs.append(("A", "step=%g prod=%g stride=%d"
                                 % (st, pr, sd), c,
                                 dict(step=st, product=pr, stride=sd)))

    if "B" in blocks:
        for lam in LAMBDAS:
            for tb in TOTAL_BITS:
                c = dict(base)
                c["lambda"] = repr(lam)
                c["total_bits"] = str(tb)
                runs.append(("B", "lambda=%g total_bits=%d" % (lam, tb), c,
                             dict(lam=lam, total_bits=tb)))

    if "C" in blocks:
        prod = float(base["warmup"]) * derived(base)[3]
        for knob, levels in ONE_AT_A_TIME:
            for v in levels:
                # That level IS the baseline row. Compared as numbers, so a
                # cfg that says 0 matches a level written 0.0, and a knob the
                # cfg leaves out is taken at the C's default.
                at_base = base.get(knob, C_DEFAULTS.get(knob))
                if at_base is not None and float(at_base) == float(v):
                    continue
                c = dict(base)
                c[knob] = repr(v) if isinstance(v, float) else str(v)
                # window changes m, which changes lr, which changes how much
                # learning a given warm-up buys. Hold the product instead of
                # the window count, or the row would measure the warm-up.
                if knob == "window":
                    c["warmup"] = str(warmup_for(c, prod))
                    # A baseline with no overlap must stay with no overlap.
                    # Left alone, a baseline that spells out stride = 256
                    # keeps it when the window drops to 128, and the C
                    # rejects a stride longer than the window
                    # (config.c:106), so the row died before it ran.
                    if base.get("stride", base["window"]) == base["window"]:
                        c["stride"] = str(v)
                runs.append(("C", "%s=%s" % (knob, v), c,
                             dict(knob=knob, value=float(v))))

    return runs


# --------------------------------------------------------------------------
# one table per ROW of the sensitivity matrix
# --------------------------------------------------------------------------


# Every row file carries every metric, so the figure picks a column rather
# than a file. Left name is the CSV column, right name is what read_metrics
# produced.
ROW_COLUMNS = (
    ("theta",      "thetamax"),
    ("everr",      "everrmax"),
    ("tlb",        "tlb_online"),
    ("tlbbatch",   "tlb_batch"),
    ("p10",        "p10_online"),
    ("p10batch",   "p10_batch"),
    ("prune",      "prune10_online"),
    ("prunebatch", "prune10_batch"),
    ("spearman",   "spearman"),
    # not "overlap": the overlap ROW uses that name for its x axis, and two
    # columns with one name is a figure that silently plots the wrong one
    ("topten",     "overlapten"),
    ("epochs",     "epochs"),
)

# The settings each row of the matrix varies, with the name the figure uses.
# The order here is the order of the rows in the PDF.
ROW_TITLES = (
    ("budget",        "Learning budget"),
    ("step",          "Oja step"),
    ("overlap",       "Window overlap"),
    ("lambda",        "Allocation penalty"),
    ("bits",          "Bit budget"),
    ("basis_average", "Basis averaging"),
    ("theta_threshold_deg", "Epoch gate threshold"),
    ("ring_window",   "Candidate ring size"),
    ("freeze_ahead",  "Freeze ahead"),
    ("gs_period",     "Orthonormalisation period"),
    ("angle_period",  "Angle check period"),
    ("window",        "Window length"),
    ("components",    "Components"),
)


def write_row_csv(path, xname, points):
    """One row of the matrix: x in the first column, every metric after it.

    points is a list of (x, metrics dict). A metric the run never produced
    is written as nan, which pgfplots skips, so a panel with no data draws
    an empty frame instead of a wrong line.
    """
    with open(path, "w") as f:
        f.write(",".join([xname] + [c for c, _ in ROW_COLUMNS]) + "\n")
        for x, m in sorted(points, key=lambda p: p[0]):
            cells = ["%.6g" % float(x)]
            for _, key in ROW_COLUMNS:
                v = m.get(key, None)
                try:
                    cells.append("%.6g" % float(v))
                except (TypeError, ValueError):
                    cells.append("nan")
            f.write(",".join(cells) + "\n")
    return len(points)


def nearest(values, want):
    return min(values, key=lambda v: abs(v - want)) if values else None


def write_plots(rows, out, base):
    """One CSV per row of the sensitivity matrix.

    A row holds ONE setting varying with everything else at the baseline,
    which is what makes a panel readable: two lines at most, ours and the
    reference. The three-way block is cut into three such rows rather than
    drawn as a cube.

    Rows that failed are still used for their epoch count, because a run
    the comparison refused BECAUSE it opened epochs is exactly the data the
    epoch panel is about.
    """
    have = [r for r in rows if r.get("axes")]
    d = os.path.join(out, "plots")
    os.makedirs(d, exist_ok=True)
    made = []

    base_step = float(base["step"])
    base_stride = float(base.get("stride", base["window"]))
    base_prod = int(base["warmup"]) * derived(base)[3]
    base_lam = float(base.get("lambda", 0.5))
    base_bits = float(base["total_bits"])

    a = [r for r in have if r["block"] == "A"]
    if a:
        steps = {r["axes"]["step"] for r in a}
        prods = {r["axes"]["product"] for r in a}
        strides = {r["axes"]["stride"] for r in a}
        st, pr, sd = (nearest(steps, base_step), nearest(prods, base_prod),
                      nearest(strides, base_stride))

        sel = [(r["axes"]["product"], r) for r in a
               if r["axes"]["step"] == st and r["axes"]["stride"] == sd]
        if sel:
            p = os.path.join(d, "row_budget.csv")
            made.append((p, write_row_csv(p, "budget", sel),
                         "step %g, stride %g" % (st, sd)))

        sel = [(r["axes"]["step"], r) for r in a
               if r["axes"]["product"] == pr and r["axes"]["stride"] == sd]
        if sel:
            p = os.path.join(d, "row_step.csv")
            made.append((p, write_row_csv(p, "step", sel),
                         "budget %g, stride %g" % (pr, sd)))

        # Overlap is plotted as a PERCENTAGE, not as a stride: a stride of
        # 64 in a window of 256 means nothing on its own, and the reader
        # should not have to divide.
        mt = float(base["window"])
        sel = [(100.0 * (1.0 - r["axes"]["stride"] / mt), r) for r in a
               if r["axes"]["step"] == st and r["axes"]["product"] == pr]
        if sel:
            p = os.path.join(d, "row_overlap.csv")
            made.append((p, write_row_csv(p, "overlap", sel),
                         "step %g, budget %g" % (st, pr)))

    b = [r for r in have if r["block"] == "B"]
    if b:
        bits = {r["axes"]["total_bits"] for r in b}
        lams = {r["axes"]["lam"] for r in b}
        tb, lm = nearest(bits, base_bits), nearest(lams, base_lam)
        sel = [(r["axes"]["lam"], r) for r in b
               if r["axes"]["total_bits"] == tb]
        if sel:
            p = os.path.join(d, "row_lambda.csv")
            made.append((p, write_row_csv(p, "lambda", sel),
                         "total_bits %g" % tb))
        sel = [(r["axes"]["total_bits"], r) for r in b
               if r["axes"]["lam"] == lm]
        if sel:
            p = os.path.join(d, "row_bits.csv")
            made.append((p, write_row_csv(p, "bits", sel),
                         "lambda %g" % lm))

    # The planner skips a level equal to the baseline, since rerunning the
    # baseline would be a duplicate row. The FIGURE still needs that point.
    base_row = next((r for r in rows
                     if r["block"] == "--" and not r.get("error")), None)
    c = [r for r in have if r["block"] == "C"]
    for knob in sorted({r["axes"]["knob"] for r in c}):
        sel = [(r["axes"]["value"], r) for r in c
               if r["axes"]["knob"] == knob]
        if base_row is not None and knob in base:
            try:
                bv = float(base[knob])
            except ValueError:
                bv = None
            if bv is not None and all(x != bv for x, _ in sel):
                sel = sel + [(bv, base_row)]
        p = os.path.join(d, "row_%s.csv" % knob)
        made.append((p, write_row_csv(p, knob, sel), "one setting"))
    return made



# --------------------------------------------------------------------------
# the tables the document typesets
# --------------------------------------------------------------------------


def tex_safe(text):
    """Escape the characters LaTeX treats as commands.

    These four tables exist only to be typeset, so the escaping belongs
    here. The alternative, replacing underscores with spaces, would print
    setting names that do not match the ones in the code, and a name you
    cannot grep for is worse than a backslash in a file nobody reads.
    """
    out = str(text)
    for ch in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        out = out.replace(ch, "\\" + ch)
    return out.replace(",", " ")


def classify(r):
    """One short phrase saying what happened in this run.

    Derived from the numbers rather than written by hand, so it cannot drift
    away from the data it describes. The order matters: a run that never
    started has no metrics to judge, and a collapsed bound says nothing
    about the subspace.
    """
    e = r.get("error", "")
    if e:
        g = re.search(r"opened (\d+) epochs", e)
        if g:
            return "gate fired %s times" % g.group(1)
        if "exited" in e:
            return "settings rejected by the program"
        return "did not finish"
    on, ba = r.get("tlb_online"), r.get("tlb_batch")
    th = r.get("thetamax")
    if on is not None and ba is not None:
        if on < 0.01 and ba < 0.01:
            return "both collapsed to one bit everywhere"
        if ba < 0.01:
            return "batch collapsed to one bit everywhere"
        if on < 0.01:
            return "ours collapsed to one bit everywhere"
    if th is not None:
        if th > 20:
            return "subspace did not converge"
        if th > 10:
            return "subspace only partly converged"
    return "ok"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def write_tables(rows, out, base):
    """The three tables the document reads, plus the full listing.

    Kept separate from the figures because a reader looks these up rather
    than scans them, and because the bit budget turned out to matter more
    than anything a curve could show.
    """
    d = os.path.join(out, "plots")
    os.makedirs(d, exist_ok=True)
    made = []
    for r in rows:
        r["status"] = classify(r)
    ok = [r for r in rows if r["status"] == "ok"]

    # ---- the whole lambda x budget grid, which is the crucial one --------
    b = [r for r in rows if r["block"] == "B" and r.get("axes")]
    if b:
        budgets = sorted({int(r["axes"]["total_bits"]) for r in b})
        head = ["lambda"]
        for tb in budgets:
            head += ["tlb%d" % tb, "tlbb%d" % tb,
                     "prune%d" % tb, "pruneb%d" % tb]
        lines = []
        for lam in sorted({r["axes"]["lam"] for r in b}):
            line = ["%g" % lam]
            for tb in budgets:
                hit = next((r for r in b if r["axes"]["lam"] == lam
                            and int(r["axes"]["total_bits"]) == tb), None)
                for key in ("tlb_online", "tlb_batch",
                            "prune10_online", "prune10_batch"):
                    v = _num(hit.get(key)) if hit else None
                    line.append("%.4f" % v if v is not None else "nan")
            lines.append(line)
        p = os.path.join(d, "table_bits.csv")
        with open(p, "w") as f:
            f.write(",".join(head) + "\n")
            for line in lines:
                f.write(",".join(line) + "\n")
        made.append((p, len(lines), "every lambda against every budget"))

    # ---- five winners, each against a stated criterion -------------------
    # There is no single score, so "best" has to name what it is best AT.
    # Four of the five are judged at the baseline budget, because a run that
    # spends twice the bits is not a better setting, it is a dearer one.
    tb0 = base["total_bits"]
    at_budget = [r for r in ok if str(r.get("total_bits", tb0)) == str(tb0)]

    def pick(label, pool, key, biggest=True):
        cand = [r for r in pool if _num(r.get(key)) is not None]
        if not cand:
            return None
        r = (max if biggest else min)(cand, key=lambda r: _num(r[key]))
        return [tex_safe(label), tex_safe(r["label"]),
                "%.2f" % _num(r.get("thetamax", float("nan"))),
                "%.4f" % _num(r.get("everrmax", float("nan"))),
                "%.4f" % _num(r.get("tlb_online", float("nan"))),
                "%.4f" % _num(r.get("tlb_batch", float("nan"))),
                "%.4f" % _num(r.get("p10_online", float("nan"))),
                "%.4f" % _num(r.get("p10_batch", float("nan"))),
                "%.4f" % _num(r.get("prune10_online", float("nan"))),
                "%.4f" % _num(r.get("prune10_batch", float("nan")))]

    ratio = []
    for r in at_budget:
        on, ba = _num(r.get("tlb_online")), _num(r.get("tlb_batch"))
        if on is not None and ba and ba > 0.01:
            rr = dict(r)
            rr["_ratio"] = on / ba
            ratio.append(rr)

    best = [x for x in (
        pick("tightest bound at the standard budget", at_budget,
             "tlb_online"),
        pick("most candidates pruned at the standard budget", at_budget,
             "prune10_online"),
        pick("closest to the reference", ratio, "_ratio"),
        pick("best subspace", ok, "thetamax", biggest=False),
        pick("best agreement on the variance shares", ok, "everrmax",
             biggest=False),
    ) if x]
    p = os.path.join(d, "table_best.csv")
    with open(p, "w") as f:
        f.write("criterion,setting,theta,everr,tlb,tlbbatch,"
                "p10,p10batch,prune,prunebatch\n")
        for line in best:
            f.write(",".join(line) + "\n")
    made.append((p, len(best), "one winner per stated criterion"))

    # ---- what broke, one line per distinct cause -------------------------
    # Grouped by CAUSE, not by status text: "gate fired 6 times" and
    # "gate fired 16 times" are one failure mode seen twice, and listing
    # them apart would hide that.
    bad = {}
    for r in rows:
        st = r["status"]
        if st == "ok":
            continue
        # A settings file the program refused is not a failure of the
        # method, it is a pair of knobs that have to agree. That belongs in
        # the limitations, not in a table of things that broke.
        if st == "settings rejected by the program":
            continue
        key = re.sub(r"\d+", "N", st)
        bad.setdefault(key, []).append((r["label"], st))
    p = os.path.join(d, "table_broken.csv")
    with open(p, "w") as f:
        f.write("what happened,how many,example setting\n")
        for key in sorted(bad, key=lambda k: -len(bad[k])):
            lab, st = bad[key][0]
            f.write("%s,%d,%s\n"
                    % (tex_safe(key), len(bad[key]), tex_safe(lab)))
    made.append((p, len(bad), "distinct failure modes"))

    # ---- everything, for the appendix ------------------------------------
    p = os.path.join(d, "table_all.csv")
    with open(p, "w") as f:
        f.write("n,block,setting,theta,everr,tlb,tlbbatch,p10,p10batch,"
                "prune,prunebatch,epochs,status\n")
        for r in sorted(rows, key=lambda r: r.get("n", 0)):
            def c(k, fmt="%.4f"):
                v = _num(r.get(k))
                return fmt % v if v is not None else ""
            f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                r.get("n", ""), r.get("block", ""),
                tex_safe(r.get("label", "")),
                c("thetamax", "%.2f"), c("everrmax"), c("tlb_online"),
                c("tlb_batch"), c("p10_online"), c("p10batch")
                or c("p10_batch"), c("prune10_online"), c("prune10_batch"),
                c("epochs", "%.0f"), tex_safe(r["status"])))
        made.append((p, len(rows), "every run, for the appendix"))
    return made


# --------------------------------------------------------------------------
# running one config
# --------------------------------------------------------------------------


def ref_key(cfg):
    """What the reference's answer depends on.

    It fits batch PCA on the warm-up windows and allocates bits from them, so
    it sees the geometry and the budget and nothing else. None of step,
    basis_average, theta_threshold_deg, ring_window, gs_period or
    angle_period reaches it. Two configs with the same key get the same
    reference, so it is computed once and reused, which is most of the time
    saved in a sweep over our own knobs.
    """
    return (cfg["input"], cfg.get("channels", "1"), cfg["window"],
            cfg.get("stride", cfg["window"]), cfg["components"],
            cfg["total_bits"], cfg.get("lambda", "0.5"), cfg["warmup"])


def run(cmd, log):
    log.write("\n$ %s\n" % " ".join(cmd))
    log.flush()
    p = subprocess.run(cmd, capture_output=True, text=True)
    log.write(p.stdout)
    if p.stderr:
        log.write("--- stderr ---\n" + p.stderr)
    log.flush()
    return p


def read_metrics(res_dir, out_dir, c_stdout):
    """Pull the row's numbers out of what the tools wrote.

    Read from the CSVs rather than scraped from the terminal, so a change in
    the printed layout cannot silently shift a column. The two exceptions are
    the C's own health numbers, which only exist on its stdout.
    """
    m = {}
    tex = os.path.join(res_dir, "scalars.tex")
    if os.path.exists(tex):
        for line in open(tex):
            g = re.match(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", line.strip())
            if g:
                try:
                    m[g.group(1)] = float(g.group(2))
                except ValueError:
                    pass

    p = os.path.join(res_dir, "tlb_summary.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            m["tlb_" + r["method"]] = float(r["mean"])
            m["p10_" + r["method"]] = float(r["p10"])
            m["viol_" + r["method"]] = float(r["violations"])

    p = os.path.join(res_dir, "pruning.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            m["prune%s_online" % r["k"]] = float(r["prune_online"])
            m["prune%s_batch" % r["k"]] = float(r["prune_batch"])

    p = os.path.join(out_dir, "epochs.csv")
    if os.path.exists(p):
        m["epochs"] = max(0, sum(1 for _ in open(p)) - 2)

    for name, pat in (("noise", r"noise\s*=\s*([\d.]+) deg"),
                      ("rate", r"rate\s*=\s*([\d.]+) deg"),
                      ("algo_s", r"time\s*=.*?,\s*([\d.]+) s algorithm")):
        g = re.search(pat, c_stdout)
        if g:
            m[name] = float(g.group(1))
    return m


# --------------------------------------------------------------------------


HEAD = ("  #  blk  %-34s %6s %7s %7s %7s %7s %6s %7s %6s %6s"
        % ("knob", "theta", "ev_err", "TLB_on", "TLB_bat", "p10",
           "prune10", "ep", "noise", "sec"))


def row_line(i, blk, label, m, secs):
    def g(key, fmt, width):
        v = m.get(key)
        return (fmt % v) if v is not None else "-".rjust(width)
    return ("%3d  %-3s  %-34s %s %s %s %s %s %s %s %s %6.1f"
            % (i, blk, label[:34],
               g("thetamax", "%6.2f", 6), g("everrmax", "%7.4f", 7),
               g("tlb_online", "%7.4f", 7), g("tlb_batch", "%7.4f", 7),
               g("p10_online", "%7.4f", 7), g("prune10_online", "%6.4f", 6),
               g("epochs", "%7d", 7), g("noise", "%6.2f", 6),
               secs))


def flush(rows, out):
    """Write sweep.csv and the plot tables with what is finished so far.

    Called after EVERY row, not once at the end. A sweep is tens of minutes
    and the obvious reason to stop one early is that the first rows already
    answered the question; losing all of them for that would be absurd. The
    files are small, so rewriting them each time costs nothing worth saving.

    Columns are collected across all rows because a failed row carries an
    error and no metrics, and a later row may report a metric an earlier one
    did not.
    """
    # The axes become ax_* columns here, in one place rather than at each
    # of the four spots a row is appended. They are what --plots-only reads
    # back to redraw without rerunning anything.
    flat = []
    for r in rows:
        f = {k: v for k, v in r.items() if k != "axes"}
        for a, v in r.get("axes", {}).items():
            f["ax_" + a] = v
        flat.append(f)
    cols = []
    for r in flat:
        for c in r:
            if c not in cols:
                cols.append(c)
    path = os.path.join(out, "sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        for r in flat:
            w.writerow(r)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cfg", help="the baseline settings file")
    ap.add_argument("--bin", required=True, help="the compiled C")
    ap.add_argument("--spartan-repo", required=True)
    ap.add_argument("--out", required=True, help="where everything goes")
    ap.add_argument("--blocks", default="ABC",
                    help="which blocks to run, e.g. AB (default ABC)")
    ap.add_argument("--sample", type=int, default=1000,
                    help="windows the comparison draws for its distances")
    ap.add_argument("--plots-only", action="store_true",
                    help="rebuild out/plots from an existing sweep.csv and "
                         "exit. Runs nothing, so a change to the figures "
                         "costs seconds instead of the whole grid.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and what would be skipped, run "
                         "nothing. Always do this first on a new grid.")
    args = ap.parse_args()

    base = read_cfg(args.cfg)

    if args.plots_only:
        path = os.path.join(args.out, "sweep.csv")
        if not os.path.exists(path):
            sys.exit("no %s to redraw from" % path)
        rows = []
        for r in csv.DictReader(open(path)):
            # the axes were written as ax_* columns for exactly this
            axes = {k[3:]: float(v) if k != "ax_knob" else v
                    for k, v in r.items() if k.startswith("ax_") and v != ""}
            row = {k: v for k, v in r.items() if not k.startswith("ax_")}
            for k in list(row):
                if k not in ("block", "label", "error"):
                    try:
                        row[k] = float(row[k])
                    except (ValueError, TypeError):
                        pass
            row["axes"] = axes
            rows.append(row)
        for p, n, note in (write_plots(rows, args.out, base)
                           + write_tables(rows, args.out, base)):
            print("wrote %-44s %4d rows   %s" % (p, n, note))
        return 0

    n_frames = count_frames(base["input"])
    runs = plan(base, n_frames, args.blocks.upper())

    live = [t for t in runs if feasible(t[2], n_frames) is None]
    dead = [(t[0], t[1], feasible(t[2], n_frames)) for t in runs
            if feasible(t[2], n_frames) is not None]

    print("sweep on %s" % os.path.basename(base["input"]))
    print("baseline %s" % args.cfg)
    for b in "ABC":
        if b in args.blocks.upper():
            nb = sum(1 for x in live if x[0] == b)
            nd = sum(1 for x in dead if x[0] == b)
            print("block %s  %3d to run, %2d skipped" % (b, nb, nd))
    keys = {ref_key(t[2]) for t in live}
    print("         %3d runs in all, %d distinct references to compute"
          % (len(live), len(keys)))
    print()

    if dead:
        print("skipped:")
        for b, l, why in dead:
            print("  %-3s %-34s %s" % (b, l[:34], why))
        print()
    if args.dry_run:
        print("dry run, nothing executed")
        return 0

    for sub in ("cfg", "run", "res", "ref"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    log_path = os.path.join(args.out, "sweep.log")
    rows = []
    ref_done = {}

    print(HEAD)
    with open(log_path, "w") as log:
        for i, (blk, label, cfg, axes) in enumerate(live, 1):
            t0 = time.time()
            tag = "%03d" % i
            out_dir = os.path.join(args.out, "run", tag)
            res_dir = os.path.join(args.out, "res", tag)
            os.makedirs(out_dir, exist_ok=True)

            cfg = dict(cfg)
            cfg["out_dir"] = out_dir
            cfg["words"] = os.path.join(out_dir, "words.csv")
            cfg_path = os.path.join(args.out, "cfg", tag + ".cfg")
            write_cfg(cfg_path, cfg)

            log.write("\n%s\n=== %s  %s  %s\n" % ("=" * 70, tag, blk, label))

            p = run([args.bin, cfg_path], log)
            if p.returncode != 0:
                print("%3d  %-3s  %-34s  C FAILED, see the log"
                      % (i, blk, label[:34]))
                rows.append(dict(n=i, block=blk, label=label, axes=axes,
                                 error="C exited %d" % p.returncode))
                flush(rows, args.out)
                continue

            key = ref_key(cfg)
            if key not in ref_done:
                rd = os.path.join(args.out, "ref", "k%02d" % len(ref_done))
                os.makedirs(rd, exist_ok=True)
                q = run(["python3", "tools/ref_spartan.py", cfg_path,
                         "--spartan-repo", args.spartan_repo,
                         "--out-dir", rd], log)
                if q.returncode != 0:
                    print("%3d  %-3s  %-34s  REFERENCE FAILED, see the log"
                          % (i, blk, label[:34]))
                    rows.append(dict(n=i, block=blk, label=label,
                                     axes=axes,
                                     error="ref_spartan exited %d"
                                           % q.returncode))
                    flush(rows, args.out)
                    continue
                ref_done[key] = rd
            ref_dir = ref_done[key]

            r = run(["python3", "tools/compare_stages.py", cfg_path,
                     "--spartan-repo", args.spartan_repo,
                     "--ref-dir", ref_dir, "--out", res_dir,
                     "--sample", str(args.sample)], log)
            secs = time.time() - t0
            if r.returncode != 0:
                both = r.stdout + r.stderr
                last = both.strip().split("\n")[-1][:46]
                print("%3d  %-3s  %-34s  %s" % (i, blk, label[:34], last))
                fail = dict(n=i, block=blk, label=label, axes=axes,
                            error=last, sec=round(secs, 1))
                # A run the comparison refused BECAUSE it opened epochs is
                # exactly the data the epoch panel is about, so keep the
                # count even though every other metric is missing.
                g = re.search(r"opened (\d+) epochs", both)
                if g:
                    fail["epochs"] = int(g.group(1))
                rows.append(fail)
                flush(rows, args.out)
                continue

            # the C's health lines go to stderr, the rest to stdout, and
            # which is which is not worth remembering: read both
            m = read_metrics(res_dir, out_dir, p.stdout + "\n" + p.stderr)
            print(row_line(i, blk, label, m, secs))

            row = dict(n=i, block=blk, label=label, sec=round(secs, 1),
                       error="", axes=axes)
            for knob in ("window", "stride", "components", "total_bits",
                         "lambda", "step", "basis_average",
                         "theta_threshold_deg", "gs_period", "angle_period",
                         "warmup", "ring_window", "freeze_ahead"):
                row[knob] = cfg.get(knob, "")
            row["lr"] = "%.9g" % derived(cfg)[3]
            row["product"] = "%.4g" % (int(cfg["warmup"]) * derived(cfg)[3])
            row.update(m)
            rows.append(row)
            flush(rows, args.out)
            write_plots(rows, args.out, base)
            write_tables(rows, args.out, base)

    csv_path = os.path.join(args.out, "sweep.csv")

    print()
    print("wrote %-40s %4d rows" % (csv_path, len(rows)))
    print("wrote %-40s full output of every run" % log_path)
    for p, n, note in (write_plots(rows, args.out, base)
                       + write_tables(rows, args.out, base)):
        print("wrote %-40s %4d rows   %s" % (p, n, note))

    ok = [r for r in rows if not r.get("error") and "tlb_online" in r]
    if ok:
        print()
        print("best 8 by TLB, with the baseline marked")
        for r in sorted(ok, key=lambda r: -r["tlb_online"])[:8]:
            print("   %.4f  %-3s %-40s%s"
                  % (r["tlb_online"], r["block"], r["label"][:40],
                     "   <- baseline" if r["block"] == "--" else ""))
        base_row = next((r for r in ok if r["block"] == "--"), None)
        if base_row and base_row["tlb_online"] >= max(
                r["tlb_online"] for r in ok):
            print("   the baseline is already the best of these")
    bad = [r for r in rows if r.get("error")]
    if bad:
        print()
        print("%d runs did not finish:" % len(bad))
        for r in bad:
            print("   %3d  %-34s %s" % (r["n"], r["label"][:34], r["error"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
