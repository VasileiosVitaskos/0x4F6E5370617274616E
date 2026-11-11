#!/usr/bin/env python3
"""Run the batch reference SPARTAN on the same windows our C sees.

Reads the settings the C used for this run (settings.cfg in its out_dir, so
the C runs first), builds the windows exactly as src/stream.c does,
then hands them to the authors' own SPARTAN class. Nothing is reimplemented:
the PCA, the bit allocation and the breakpoints all come from their code, so
any difference we later measure is a real difference between the two
algorithms and not between two readings of the same paper.

Two references are written, from the same code, on the same scored windows:

    warm-up    fits on the warm-up windows, the data our algorithm had before
               it wrote its first word.
    hindsight  fits on the scored windows themselves. No streaming method can
               have that; it is the ceiling, and the gap to it is the price
               of being streaming.

    python3 tools/ref_spartan.py tests/synthetic/cfg/stream_s006.cfg \
            --spartan-repo ~/dev/SPARTAN-main

Needs their virtualenv active: numpy, scipy, scikit-learn, numba, pandas.
"""

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np
from read_settings import geometry

# --------------------------------------------------------------------------
# our side: windows, read the same way batch_truth.py reads them
# --------------------------------------------------------------------------


def read_windows(path, d, m_time, stride):
    """Every window the stream hands out, as rows of (n_windows, d*m_time).

    Mode-1 unfolding, matching stream.c:191-203: channel c occupies m_time
    contiguous doubles. For d = 1 that is just the contiguous slice.
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
        sys.exit(
            "%s: %d numbers is not a multiple of channels = %d" % (path, len(flat), d)
        )
    frames = flat.reshape(-1, d)
    n_frames = len(frames)
    n_win = 0 if n_frames < m_time else 1 + (n_frames - m_time) // stride
    out = np.empty((n_win, d * m_time))
    for w in range(n_win):
        out[w] = frames[w * stride : w * stride + m_time].T.reshape(-1)
    return out


def znorm_rows(X, d, eps):
    """z-norm PER CHANNEL, the way main.c:366-368 does it.

    The C loops the channels and calls znorm_row on each m_time slice, so
    every channel gets its own mean and its own spread. Normalising the whole
    concatenated row in one go is a different operation and would hand the
    reference different numbers from the ones our C saw, which would make
    every later comparison meaningless.

    Identical to the whole-row version when d = 1.

    Divides by m_time, not m_time-1, and adds SPARTAN_EPS, matching
    spartan.c:34 znorm_row.
    """
    B = X.reshape(len(X), d, -1)
    mu = B.mean(axis=2, keepdims=True)
    sd = np.sqrt(((B - mu) ** 2).mean(axis=2, keepdims=True)) + eps
    return ((B - mu) / sd).reshape(len(X), -1)


# --------------------------------------------------------------------------


def alphabet_for(total_bits, k):
    """The alphabet_size to pass so the reference ends up with OUR bit budget.

    Their constructor silently overrides whatever bit_budget you give it with
    log2(alphabet_size) * word_length (spartan.py:125-126), so the budget is
    really set by the alphabet. Invert that: ask for 2 ** (total_bits / k).

    Fails loudly if the budget is not a whole number of bits per position,
    because then no alphabet_size can express it and the comparison would be
    quietly running a different experiment.
    """
    if total_bits % k != 0:
        sys.exit(
            "total_bits = %d is not divisible by components = %d. The "
            "reference sets its budget through alphabet_size, so it can "
            "only express budgets that are a whole number of bits per "
            "position." % (total_bits, k)
        )
    return 2 ** (total_bits // k)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cfg")
    ap.add_argument(
        "--spartan-repo",
        default=None,
        help="path to the SPARTAN checkout, added to sys.path",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="where to write the reference's words and json. "
        "Defaults to a 'ref' folder beside the data, which "
        "is keyed on the DATA FILE NAME only: a sweep that "
        "runs many settings over one dataset would have "
        "every config overwrite the last one's reference.",
    )
    args = ap.parse_args()

    # Load spartan.py as a STANDALONE FILE, not through the package.
    #
    # `from TSB_Symbolic.symbolic.spartan import SPARTAN` runs their
    # __init__.py, which imports every method in the repo, so it fails on the
    # first dependency that is not installed (ruptures, pulled in by esax).
    # spartan.py itself needs only numpy and scikit-learn.
    repo = os.path.expanduser(args.spartan_repo or ".")
    sys.path.insert(0, repo)  # for any absolute import inside the file
    src = os.path.join(repo, "TSB_Symbolic", "symbolic", "spartan", "spartan.py")
    if not os.path.exists(src):
        sys.exit("cannot find %s" % src)
    spec = importlib.util.spec_from_file_location("ref_spartan_mod", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    SPARTAN = mod.SPARTAN

    g = geometry(args.cfg)
    alpha = alphabet_for(g["total_bits"], g["k"])

    print("stream    %s" % g["input"])
    Z = znorm_rows(
        read_windows(g["input"], g["d"], g["m_time"], g["stride"]), g["d"], g["eps"]
    )
    n = len(Z)
    if g["warmup"] >= n:
        sys.exit("warmup = %d but only %d windows exist" % (g["warmup"], n))
    scored = Z[g["warmup"] :]
    print("windows   %d total, %d warm-up, %d scored" % (n, g["warmup"], len(scored)))
    print(
        "settings  word_length %d   alphabet %d   bit_budget %d   "
        "lambda %g   solver full" % (g["k"], alpha, g["total_bits"], g["lam"])
    )

    name = os.path.splitext(os.path.basename(g["input"]))[0]
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(g["input"])), "ref"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Two references from the same code, differing only in what they fit on.
    #
    #   warm-up    fit on [0, warmup). Exactly the data our C had before it
    #              wrote its first word, so the two start level.
    #   hindsight  fit on [warmup, n), the very windows that are scored. No
    #              streaming method can have this, and it is how batch
    #              SPARTAN is meant to be used: fit on the data, then encode
    #              it. The distance from us to THIS one is the price of
    #              being streaming.
    #
    # Both are needed. On a stream that drifts, the warm-up reference goes
    # stale and beating it proves little; on a stream that does not, the two
    # coincide, which is itself a check that the pipeline is sound.
    for tag, suffix, train, lo, hi in (
        ("warm-up", "ref", Z[: g["warmup"]], 0, g["warmup"]),
        ("hindsight", "hindsight", scored, g["warmup"], n),
    ):
        sp = SPARTAN(
            alphabet_size=alpha,
            word_length=g["k"],
            window_size=0,  # 0 means "the whole series is one window"
            binning_method="equi-depth",
            assignment_policy="DAA",
            bit_budget=g["total_bits"],
            lamda=g["lam"],
            pca_solver="full",
        )
        sp.fit(train)
        words = sp.transform(scored).astype(int)

        ev = np.asarray(sp.evcr[: g["k"]], dtype=float)
        ev = ev / ev.sum()  # they re-normalise among the kept k
        bits = [int(round(math.log2(a))) for a in sp.alphabet_size]

        print()
        print("reference SPARTAN, %s: fit on windows [%d, %d)" % (tag, lo, hi))
        print("  ev        %s" % " ".join("%.4f" % v for v in ev))
        print("  bits      %s" % " ".join("%6d" % b for b in bits))
        print("  alphabet  %s" % " ".join("%6d" % a for a in sp.alphabet_size))

        # Words in OUR csv shape, so compare_stages reads files of one format
        # and never needs to know which side produced which. The warm-up
        # reference keeps the file names it always had.
        wstem = name if suffix == "ref" else name + "." + suffix
        wpath = os.path.join(out_dir, wstem + ".words.csv")
        with open(wpath, "w") as f:
            f.write("window," + ",".join("sym%d" % j for j in range(g["k"])) + "\n")
            for i, row in enumerate(words):
                f.write(
                    "%d,%s\n" % (g["warmup"] + i, ",".join(str(int(v)) for v in row))
                )

        jpath = os.path.join(out_dir, name + "." + suffix + ".json")
        with open(jpath, "w") as f:
            json.dump(
                dict(
                    source=g["input"],
                    cfg=args.cfg,
                    fit=tag,
                    fit_windows=[lo, hi],
                    test_windows=[g["warmup"], n],
                    alphabet_size_in=alpha,
                    ev=ev.tolist(),
                    bits=bits,
                    alphabet=[int(a) for a in sp.alphabet_size],
                    directions=sp.pca.components_[: g["k"]].tolist(),
                    # theirs[j] = ours[j-1]: they carry a leading -MAX sentinel
                    # and a trailing +MAX, ours has only the trailing one
                    breakpoints=[list(map(float, b)) for b in sp.breakpoints],
                    mindist_breakpoints=[
                        list(map(float, b)) for b in sp.mindist_breakpoints
                    ],
                ),
                f,
                indent=1,
            )

        print("  wrote %s   %d words" % (wpath, len(words)))
        print("  wrote %s" % jpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
