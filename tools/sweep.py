#!/usr/bin/env python3
"""Run many settings over one or more datasets, one line per setting.

For every dataset: the C runs once on the baseline cfg, and the settings.cfg
it writes is the baseline, every default already resolved by the C. Then each
variation is written as the baseline cfg plus the knobs it changes, and for
each one the C runs, the batch reference runs (once per distinct reference),
the comparison runs, and one row is recorded. Everything a row reports is
read from what those programs wrote; nothing here knows an algorithm default.

    python3 tools/sweep.py                         the DATASETS below
    python3 tools/sweep.py tests/tsb/cfg/tsb_clean.cfg
    python3 tools/sweep.py --dry-run               the plan, runs only the C
                                                   on each baseline

Everything you would change is in the CONTROL PANEL below the imports.

WHAT IS SWEPT TOGETHER AND WHY

Block A, step x stride, every combination. Both set how much Oja moves per
frame of data: step is the size of one update, stride how many frames lie
between two updates. Since epoch 0 starts from exact PCA, no warm-up length
is needed for Oja to converge, so the old third axis (the learning budget)
has nothing left to measure.

Block B, lambda x total_bits, every combination. The allocation penalty
pulls towards the even split total_bits / k, so the budget sets where lambda
pulls and the two cannot be read apart.

Block C, everything else, one knob at a time around the baseline.

WHAT IS NOT DONE HERE

No feasibility check. The C validates its own settings and stops before
reading any data, so a combination it refuses costs nothing and the row
carries its message. A copy of its rules here would drift from the C.

batch_truth.py is not run: the comparison does not read it. Run it once per
dataset by hand.
"""

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time

from read_settings import load

# ==========================================================================
# CONTROL PANEL. Datasets, programs and every level swept live here and
# nowhere else. These are choices of the EXPERIMENT; the algorithm's own
# defaults live in the C and reach this script through settings.cfg.
# ==========================================================================

# Run when no cfg is given on the command line
DATASETS = (
    "tests/tsb/cfg/tsb_noisy.cfg",
    "tests/tsb/cfg/tsb_drift.cfg",
)

BIN = "./build/spartan_toy"  # the compiled C
SPARTAN_REPO = "~/dev/SPARTAN-main"  # the authors' checkout
OUT = "out/sweep"  # one folder per dataset goes in here
SAMPLE = 1000  # windows the comparison draws
PRUNE_K = 10  # neighbourhood the pruning column uses

# Block A: every combination
BLOCK_A = {
    "step": (0.06, 0.12, 0.24, 0.48),
    "stride": (256, 192, 128, 64),
}

# Block B: every combination
BLOCK_B = {
    "lambda": (0.1, 0.15, 0.3, 0.5, 0.8, 1.0),
    "total_bits": (8, 16, 32),
}

# Block C: one knob at a time, the rest at the baseline
BLOCK_C = {
    "basis_average": (0.01, 0.1, 0.3, 1.0),
    "theta_threshold_deg": (5, 10, 20, 45, 89),
    "ring_window": (160, 256, 512, 1024),
    "gs_period": (1, 2, 5, 10),
    "angle_period": (5, 10, 20, 50),
    "window": (128, 256, 512),
    "components": (4, 8, 16),
    "warmup": (256, 512, 1024, 2048),
    "cross_epoch": ("search", "box"),
}

# Settings held at the baseline's value while a Block C knob moves. The C
# derives warmup from ring_window when warmup is not given, so moving the
# ring alone would move the warm-up too and the row would measure both.
HOLD = {
    "ring_window": ("warmup",),
}

# ==========================================================================

# What the batch reference's answer depends on (ref_spartan.py reads these
# and nothing else), so configs that agree on them share one reference
REF_KEYS = (
    "input",
    "channels",
    "window",
    "stride",
    "components",
    "total_bits",
    "lambda",
    "warmup",
)

# compare_stages.py's CSV names -> the names used here
METHODS = (
    ("search", "search"),
    ("online", "box"),
    ("stored", "join"),
    ("batch", "batch_warmup"),
    ("hindsight", "batch_all"),
)

# Settings copied into every row, as the C resolved them
ROW_SETTINGS = (
    "window",
    "stride",
    "components",
    "total_bits",
    "lambda",
    "step",
    "basis_average",
    "theta_threshold_deg",
    "gs_period",
    "angle_period",
    "warmup",
    "ring_window",
    "cross_epoch",
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Below this mean TLB a bound prunes nothing: the diagonal alone gives
# 1 / SAMPLE. The batch reference lands here at one bit per position, where
# every pair of its letters is equal or adjacent and MINDIST is 0, and a
# ratio against it would be a division by almost nothing
COLLAPSED_TLB = 0.01


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def read_cfg(path):
    """A cfg as an ordered dict of strings, comments and blanks dropped."""
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def write_cfg(path, cfg):
    with open(path, "w") as f:
        f.write("# written by tools/sweep.py, do not edit\n")
        for k, v in cfg.items():
            f.write("%-20s= %s\n" % (k, v))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def same(a, b):
    """Equal as numbers when both are numbers (0 and 0.0 match), else as
    text (search and box)."""
    x, y = num(a), num(b)
    if x is not None and y is not None:
        return x == y
    return str(a) == str(b)


def text(v):
    """A level as it goes into a cfg and a label."""
    return "%g" % v if isinstance(v, float) else str(v)


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


def plan(base_cfg, base, blocks):
    """Every run, as (block, label, overrides, knob).

    base is the baseline's settings.cfg. A variation equal to the baseline
    on every knob it sets is the baseline itself and is not run twice.
    knob names the Block C knob a row moves, empty for the other blocks.
    """
    runs = [("--", "baseline", {}, "")]

    def is_base(over):
        return all(same(base[k], v) for k, v in over.items())

    for blk, grid in (("A", BLOCK_A), ("B", BLOCK_B)):
        if blk not in blocks:
            continue
        for combo in itertools.product(*grid.values()):
            over = dict(zip(grid, combo))
            if not is_base(over):
                runs.append(
                    (
                        blk,
                        " ".join("%s=%s" % (k, text(v)) for k, v in over.items()),
                        over,
                        "",
                    )
                )

    if "C" in blocks:
        for knob, levels in BLOCK_C.items():
            for v in levels:
                over = {knob: v}
                if is_base(over):
                    continue
                for held in HOLD.get(knob, ()):
                    over[held] = base[held]
                # A baseline cfg that spells out stride = window means no
                # overlap; keep it that way, or the C rejects a stride longer
                # than a shorter window
                if (
                    knob == "window"
                    and "stride" in base_cfg
                    and same(base_cfg["stride"], base["window"])
                ):
                    over["stride"] = v
                runs.append(("C", "%s=%s" % (knob, text(v)), over, knob))
    return runs


def ref_count(base, runs):
    """Distinct batch references the plan needs, counted from the planned
    settings. Exact as long as no knob moves a reference key through a C
    default it does not hold (HOLD exists for that)."""
    keys = set()
    for _, _, over, _ in runs:
        keys.add(tuple(str(over.get(k, base[k])) for k in REF_KEYS))
    return len(keys)


# --------------------------------------------------------------------------
# running one config
# --------------------------------------------------------------------------


def run(cmd, log):
    log.write("\n$ %s\n" % " ".join(cmd))
    log.flush()
    p = subprocess.run(cmd, capture_output=True, text=True)
    log.write(p.stdout)
    if p.stderr:
        log.write("--- stderr ---\n" + p.stderr)
    log.flush()
    return p


def last_words(p):
    """The last line a tool printed, or how it died if it printed nothing:
    a process killed for memory leaves no message, only its exit code."""
    lines = (p.stdout + p.stderr).strip().split("\n")
    if lines and lines[-1].strip():
        return lines[-1]
    return "exit %d with no message%s" % (
        p.returncode,
        ", killed (out of memory?)" if p.returncode in (-9, 137) else "",
    )


def c_refusal(stderr):
    """The C's own reason for stopping: its config message if it refused
    the settings, else its last line."""
    lines = [l for l in stderr.strip().split("\n") if l.strip()]
    for l in lines:
        if l.startswith("config: "):
            return "rejected by the C: " + l[len("config: ") :]
    return "C failed: " + (lines[-1] if lines else "no message")


def read_metrics(res_dir, c_log, query):
    """The row's numbers, from the CSVs the comparison wrote and the summary
    lines the C printed. query is search or box, the run's cross_epoch."""
    m = {}
    names = dict(METHODS)
    p = os.path.join(res_dir, "tlb_summary.csv")
    for r in csv.DictReader(open(p)):
        n = names[r["method"]]
        m["tlb_" + n] = float(r["mean"])
        m["p10_" + n] = float(r["p10"])
        m["zero_" + n] = float(r["zeros"])
        m["viol_" + n] = int(float(r["violations"]))
    p = os.path.join(res_dir, "pruning.csv")
    for r in csv.DictReader(open(p)):
        if int(float(r["k"])) == PRUNE_K:
            for left, right in METHODS:
                m["prune_" + right] = float(r["prune_" + left])

    g = re.search(r"epochs\s*=.*?(\d+) type 1.*?(\d+) type 2", c_log)
    if g:
        m["t1"], m["t2"] = int(g.group(1)), int(g.group(2))
    g = re.search(r"staleness\s*=\s*([\d.]+) deg", c_log)
    if g:
        m["staleness_deg"] = float(g.group(1))
    g = re.search(r"late\s*=\s*(\d+) windows", c_log)
    if g:
        m["late"] = int(g.group(1))

    m["tlb_query"] = m.get("tlb_" + query)
    m["prune_query"] = m.get("prune_" + query)
    m["viol"] = sum(m.get("viol_" + n, 0) for n in ("search", "box", "join"))
    collapsed = (m.get("tlb_batch_all") or 0.0) < COLLAPSED_TLB
    for what in ("tlb", "prune"):
        top = None if collapsed else m.get(what + "_batch_all")
        for side in ("query", "join"):
            v = m.get("%s_%s" % (what, side))
            m["ratio_%s_%s" % (what, side)] = v / top if v is not None and top else None
    return m


# --------------------------------------------------------------------------
# what the terminal shows
# --------------------------------------------------------------------------

ROW = "%3s  %-3s  %-26s %3s %3s %9s %9s %9s %9s %5s %5s"
# group titles over the columns of ROW: 36 for "#  blk  setting", then
# " t1 t2" is 8 wide and each " query join" pair 20
HEAD1 = (
    " " * 36
    + "epochs".rjust(8)
    + "TLB / batch all".rjust(20)
    + ("prune k=%d / b.all" % PRUNE_K).rjust(20)
)
HEAD2 = ROW % (
    "#",
    "blk",
    "setting",
    "t1",
    "t2",
    "query",
    "join",
    "query",
    "join",
    "viol",
    "sec",
)


def row_line(r):
    def f(key, fmt):
        v = r.get(key)
        return fmt % v if v is not None and v != "" else "-"

    return ROW % (
        r["n"],
        r["block"],
        r["setting"][:26],
        f("t1", "%d"),
        f("t2", "%d"),
        f("ratio_tlb_query", "%.4f"),
        f("ratio_tlb_join", "%.4f"),
        f("ratio_prune_query", "%.4f"),
        f("ratio_prune_join", "%.4f"),
        f("viol", "%d"),
        f("sec", "%.1f"),
    )


def fail_line(r):
    return "%3d  %-3s  %-26s %s" % (
        r["n"],
        r["block"],
        r["setting"][:26],
        r["error"][:46],
    )


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def flush(rows, out):
    """sweep.csv with every finished row. Rewritten after each row, so a
    sweep stopped early keeps everything it finished."""
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    path = os.path.join(out, "sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def read_back(path):
    """sweep.csv as rows, numbers as numbers, for --plots-only."""
    rows = []
    for r in csv.DictReader(open(path)):
        row = {}
        for k, v in r.items():
            x = num(v)
            row[k] = (
                x
                if x is not None and k not in ("setting", "knob", "error", "block")
                else v
            )
        rows.append(row)
    return rows


# Every row file carries every column, so a figure picks a column, not a
# file. Left: the CSV column. Right: the row's key.
ROW_COLUMNS = (
    ("tlbquery", "tlb_query"),
    ("tlbjoin", "tlb_join"),
    ("tlbbatchwarm", "tlb_batch_warmup"),
    ("tlbbatchall", "tlb_batch_all"),
    ("prunequery", "prune_query"),
    ("prunejoin", "prune_join"),
    ("prunebatchall", "prune_batch_all"),
    ("t1", "t1"),
    ("t2", "t2"),
)


def cell(v, fmt="%.6g"):
    x = num(v)
    return fmt % x if x is not None else ("nan" if v in (None, "") else str(v))


def write_row_csv(path, xname, points):
    """One row of the sensitivity matrix: x first, every column after it.
    A missing number is nan, which pgfplots skips."""

    def key(p):
        x = num(p[0])
        return (0, x, "") if x is not None else (1, 0.0, str(p[0]))

    with open(path, "w") as f:
        f.write(",".join([xname] + [c for c, _ in ROW_COLUMNS]) + "\n")
        for x, r in sorted(points, key=key):
            f.write(
                ",".join([cell(x)] + [cell(r.get(k)) for _, k in ROW_COLUMNS]) + "\n"
            )
    return len(points)


def write_plots(rows, out):
    """One CSV per row of the sensitivity matrix: one setting moving, the
    rest at the baseline, the baseline point included. Selected on the
    settings each run actually used, so a level list that leaves out the
    baseline value still gets the baseline point."""
    d = os.path.join(out, "plots")
    os.makedirs(d, exist_ok=True)
    ok = [r for r in rows if not r.get("error")]
    base = next((r for r in ok if r["block"] == "--"), None)
    if base is None:
        return []
    made = []

    def emit(name, xname, sel, xf, note):
        if sel:
            p = os.path.join(d, "row_%s.csv" % name)
            made.append(
                (p, write_row_csv(p, xname, [(xf(r), r) for r in sel + [base]]), note)
            )

    a = [r for r in ok if r["block"] == "A"]
    emit(
        "step",
        "step",
        [r for r in a if same(r["stride"], base["stride"])],
        lambda r: r["step"],
        "stride %s" % text(base["stride"]),
    )
    # overlap as a percentage of the window: a stride means nothing alone
    emit(
        "overlap",
        "overlap",
        [r for r in a if same(r["step"], base["step"])],
        lambda r: 100.0 * (1.0 - num(r["stride"]) / num(r["window"])),
        "step %s" % text(base["step"]),
    )

    b = [r for r in ok if r["block"] == "B"]
    emit(
        "lambda",
        "lambda",
        [r for r in b if same(r["total_bits"], base["total_bits"])],
        lambda r: r["lambda"],
        "total_bits %s" % text(base["total_bits"]),
    )
    emit(
        "bits",
        "bits",
        [r for r in b if same(r["lambda"], base["lambda"])],
        lambda r: r["total_bits"],
        "lambda %s" % text(base["lambda"]),
    )

    c = [r for r in ok if r["block"] == "C"]
    for knob in BLOCK_C:
        emit(
            knob,
            knob,
            [r for r in c if r.get("knob") == knob],
            lambda r, k=knob: r[k],
            "one setting",
        )
    return made


def tex_safe(s):
    """These tables are typeset, so LaTeX's command characters are escaped
    rather than renamed: a setting name you cannot grep for is worse."""
    out = str(s)
    for ch in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        out = out.replace(ch, "\\" + ch)
    return out.replace(",", " ")


def classify(r):
    """One short phrase for what happened, derived from the row."""
    e = r.get("error") or ""
    if e:
        return e.split(":", 1)[0]
    if num(r.get("viol")):
        return "bound broken"
    q, b = num(r.get("tlb_query")), num(r.get("tlb_batch_all"))
    if q is not None and q < COLLAPSED_TLB:
        return "both collapsed" if (b or 0) < COLLAPSED_TLB else "ours collapsed"
    if b is not None and b < COLLAPSED_TLB:
        return "batch collapsed"
    return "ok"


def write_tables(rows, out):
    d = os.path.join(out, "plots")
    os.makedirs(d, exist_ok=True)
    made = []
    for r in rows:
        r["status"] = classify(r)

    # the lambda x budget grid, baseline cell included
    b = [r for r in rows if r["block"] in ("B", "--") and not r.get("error")]
    if len(b) > 1:
        budgets = sorted({num(r["total_bits"]) for r in b})
        lams = sorted({num(r["lambda"]) for r in b})
        head = ["lambda"]
        for tb in budgets:
            head += ["tlbq%d" % tb, "tlbb%d" % tb, "pruneq%d" % tb, "pruneb%d" % tb]
        p = os.path.join(d, "table_bits.csv")
        with open(p, "w") as f:
            f.write(",".join(head) + "\n")
            for lam in lams:
                line = ["%g" % lam]
                for tb in budgets:
                    hit = next(
                        (
                            r
                            for r in b
                            if num(r["lambda"]) == lam and num(r["total_bits"]) == tb
                        ),
                        {},
                    )
                    line += [
                        cell(hit.get(k), "%.4f")
                        for k in (
                            "tlb_query",
                            "tlb_batch_all",
                            "prune_query",
                            "prune_batch_all",
                        )
                    ]
                f.write(",".join(line) + "\n")
        made.append((p, len(lams), "every lambda against every budget"))

    # what broke, one line per cause. A setting the C refused is a pair of
    # knobs that must agree, a limitation and not a failure of the method
    bad = {}
    for r in rows:
        if r["status"] not in ("ok", "rejected by the C"):
            bad.setdefault(r["status"], []).append(r)
    p = os.path.join(d, "table_broken.csv")
    with open(p, "w") as f:
        f.write("what happened,how many,example setting\n")
        for k in sorted(bad, key=lambda k: -len(bad[k])):
            f.write(
                "%s,%d,%s\n"
                % (tex_safe(k), len(bad[k]), tex_safe(bad[k][0]["setting"]))
            )
    made.append((p, len(bad), "distinct failure modes"))

    p = os.path.join(d, "table_all.csv")
    with open(p, "w") as f:
        f.write(
            "n,block,setting,t1,t2,tlbquery,tlbjoin,tlbbatchall,"
            "prunequery,prunejoin,prunebatchall,viol,status\n"
        )
        for r in rows:
            f.write(
                ",".join(
                    [cell(r["n"], "%d"), r["block"], tex_safe(r["setting"])]
                    + [cell(r.get(k), "%d") for k in ("t1", "t2")]
                    + [
                        cell(r.get(k), "%.4f")
                        for k in (
                            "tlb_query",
                            "tlb_join",
                            "tlb_batch_all",
                            "prune_query",
                            "prune_join",
                            "prune_batch_all",
                        )
                    ]
                    + [cell(r.get("viol"), "%d"), tex_safe(r["status"])]
                )
                + "\n"
            )
    made.append((p, len(rows), "every run, for the appendix"))
    return made


def write_all(rows, out):
    flush(rows, out)
    return write_plots(rows, out) + write_tables(rows, out)


# --------------------------------------------------------------------------
# one dataset
# --------------------------------------------------------------------------


def sweep_one(cfg_path, args):
    name = os.path.splitext(os.path.basename(cfg_path))[0]
    out = os.path.join(args.out, name)
    for sub in ("cfg", "run", "res", "ref"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    log = open(os.path.join(out, "sweep.log"), "w")
    base_cfg = read_cfg(cfg_path)

    def prepare(i, over):
        """Write run i's cfg (baseline cfg + overrides) and run the C."""
        tag = "%03d" % i
        out_dir = os.path.join(out, "run", tag)
        os.makedirs(out_dir, exist_ok=True)
        cfg = dict(base_cfg)
        cfg.update({k: text(v) for k, v in over.items()})
        cfg["out_dir"] = out_dir
        cfg["words"] = os.path.join(out_dir, "words.csv")
        path = os.path.join(out, "cfg", tag + ".cfg")
        write_cfg(path, cfg)
        log.write("\n%s\n=== %s\n" % ("=" * 70, tag))
        return path, os.path.join(out, "res", tag), run([args.bin, path], log)

    # the baseline first: its settings.cfg is what every level is read
    # against, so nothing here needs to know a default
    base_path, base_res, base_p = prepare(1, {})
    if base_p.returncode != 0:
        sys.exit(
            "%s: the C refused the baseline: %s" % (cfg_path, c_refusal(base_p.stderr))
        )
    base = load(base_path)

    runs = plan(base_cfg, base, args.blocks)
    print("sweep   %s" % cfg_path)
    print(
        "base    k %d  window %d  stride %d  bits %d  lambda %s  warmup %d"
        "  ring %d  cross_epoch %s"
        % (
            base["components"],
            base["window"],
            base["stride"],
            base["total_bits"],
            text(base["lambda"]),
            base["warmup"],
            base["ring_window"],
            base["cross_epoch"],
        )
    )
    print()
    for blk, what in (
        ("A", " x ".join(BLOCK_A)),
        ("B", " x ".join(BLOCK_B)),
        ("C", "one knob at a time"),
    ):
        if blk in args.blocks:
            print(
                "block %s   %-24s %3d runs"
                % (blk, what, sum(1 for r in runs if r[0] == blk))
            )
    print(
        "          %d runs with the baseline, %d batch references"
        % (len(runs), ref_count(base, runs))
    )
    print()
    if args.dry_run:
        for i, (blk, label, _, _) in enumerate(runs, 1):
            print("%3d  %-3s  %s" % (i, blk, label))
        print()
        print("dry run, only the baseline C was run")
        log.close()
        return

    print(HEAD1)
    print(HEAD2)
    rows = []
    refs = {}
    for i, (blk, label, over, knob) in enumerate(runs, 1):
        t0 = time.time()
        if i == 1:
            path, res, p = base_path, base_res, base_p
        else:
            path, res, p = prepare(i, over)
        row = dict(n=i, block=blk, setting=label, knob=knob, error="")

        def done(error=None):
            if error:
                row["error"] = error
            row["sec"] = round(time.time() - t0, 1)
            rows.append(row)
            print(fail_line(row) if error else row_line(row))
            write_all(rows, out)

        if p.returncode != 0:
            done(c_refusal(p.stderr))
            continue
        s = load(path)
        for k in ROW_SETTINGS:
            row[k] = s[k]

        key = tuple(str(s[k]) for k in REF_KEYS)
        if key not in refs:
            rd = os.path.join(out, "ref", "r%02d" % len(refs))
            os.makedirs(rd, exist_ok=True)
            q = run(
                [
                    sys.executable,
                    os.path.join(HERE, "ref_spartan.py"),
                    path,
                    "--spartan-repo",
                    args.spartan_repo,
                    "--out-dir",
                    rd,
                ],
                log,
            )
            if q.returncode != 0:
                done("reference failed: " + last_words(q))
                continue
            refs[key] = rd
        # A reference is reused across runs that share its key, so it can be
        # older than compare_stages' staleness guard allows. It is the right
        # reference for this run by construction (same key), so its files
        # are re-stamped rather than the guard turned off
        for fn in os.listdir(refs[key]):
            os.utime(os.path.join(refs[key], fn))

        r = run(
            [
                sys.executable,
                os.path.join(HERE, "compare_stages.py"),
                path,
                "--spartan-repo",
                args.spartan_repo,
                "--ref-dir",
                refs[key],
                "--out",
                res,
                "--sample",
                str(args.sample),
            ],
            log,
        )
        # compare_stages exits 1 both when it stops and when a bound is
        # broken; only the second writes its tables, so a fresh table tells
        # the two apart, and a broken bound is a row, not a failure
        summary = os.path.join(res, "tlb_summary.csv")
        if r.returncode != 0 and not (
            os.path.exists(summary) and os.path.getmtime(summary) >= t0
        ):
            done("comparison failed: " + last_words(r))
            continue
        row.update(read_metrics(res, p.stderr, s["cross_epoch"]))
        done()
    log.close()

    print()
    print("wrote %-40s %4d rows" % (os.path.join(out, "sweep.csv"), len(rows)))
    print("wrote %-40s full output of every run" % os.path.join(out, "sweep.log"))
    for p, n, note in write_all(rows, out):
        print("wrote %-40s %4d rows   %s" % (p, n, note))

    ok = [r for r in rows if not r["error"] and r.get("ratio_tlb_query") is not None]
    if ok:
        print()
        print("best 5 by TLB query / batch all")
        best = sorted(ok, key=lambda r: -r["ratio_tlb_query"])
        show = best[:5]
        base_row = next((r for r in ok if r["block"] == "--"), None)
        if base_row is not None and base_row not in show:
            print_best = show + [None, base_row]
        else:
            print_best = show
        for r in print_best:
            if r is None:
                print("   ...")
                continue
            print(
                (
                    "   %.4f  %-3s %-26s%s"
                    % (
                        r["ratio_tlb_query"],
                        r["block"],
                        r["setting"][:26],
                        "  <- baseline" if r["block"] == "--" else "",
                    )
                ).rstrip()
            )
    flat = [
        r for r in rows if not r["error"] and r.get("status", "").endswith("collapsed")
    ]
    if flat:
        print()
        print(
            "%d runs where a bound collapsed (mean TLB under %g, absolute):"
            % (len(flat), COLLAPSED_TLB)
        )
        for r in flat:
            print(
                "   %3d  %-3s  %-26s query %.4f  batch all %.4f  %s"
                % (
                    r["n"],
                    r["block"],
                    r["setting"][:26],
                    r["tlb_query"],
                    r["tlb_batch_all"],
                    r["status"],
                )
            )
    bad = [r for r in rows if r["error"]]
    if bad:
        print()
        print("%d runs did not finish:" % len(bad))
        for r in bad:
            print("   " + fail_line(r))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "cfg", nargs="*", help="baseline cfgs; the DATASETS list when none given"
    )
    ap.add_argument("--bin", default=BIN)
    ap.add_argument("--spartan-repo", default=SPARTAN_REPO)
    ap.add_argument("--out", default=OUT, help="one folder per dataset goes in here")
    ap.add_argument(
        "--blocks", default="ABC", help="which blocks to run, e.g. AB (default ABC)"
    )
    ap.add_argument("--sample", type=int, default=SAMPLE)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="run only the C on each baseline and print the plan",
    )
    ap.add_argument(
        "--plots-only",
        action="store_true",
        help="rebuild plots/ from each existing sweep.csv, run nothing",
    )
    args = ap.parse_args()
    args.blocks = args.blocks.upper()
    args.spartan_repo = os.path.expanduser(args.spartan_repo)

    for cfg_path in args.cfg or DATASETS:
        if args.plots_only:
            out = os.path.join(
                args.out, os.path.splitext(os.path.basename(cfg_path))[0]
            )
            path = os.path.join(out, "sweep.csv")
            if not os.path.exists(path):
                sys.exit("no %s to redraw from" % path)
            for p, n, note in write_plots(read_back(path), out) + write_tables(
                read_back(path), out
            ):
                print("wrote %-40s %4d rows   %s" % (p, n, note))
            continue
        sweep_one(cfg_path, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
