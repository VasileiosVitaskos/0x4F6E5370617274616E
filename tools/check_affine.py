#!/usr/bin/env python3
"""Attack test: the C's words must not change under x -> a*x + b.

With window_znorm = false the windows reach the words raw. The claim that
makes this fair to a stream: a shift b adds the same constant to every
projection of a position and a scale a > 0 multiplies them all, and the
equi-depth cut points are taken from those same projections, so they move
and stretch with them; the basis is PCA on centred windows, which neither
sees. Oja gets a causal scale (main.c). So the words, the splits and the
epochs must be IDENTICAL for x and a*x + b, bit for bit.

Runs the C on a random-walk series with planted level shifts, once as is and
once per (a, b), and compares words.csv, every dict_n.csv bits column and
epochs.csv. Exit 0 if all identical.

  python3 tools/check_affine.py --bin ./build/spartan_toy      (make affine)
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np

PAIRS = [(1000.0, 5.0), (1e-3, -7.0), (3.7, 1e4)]

# A test setting, not an algorithm default: the shape of Task III (window 100,
# 16 positions, 64 bits), raw windows, short warm-up so several epochs open
TEST_CFG = """channels = 1
window = 100
stride = 10
components = 16
total_bits = 64
lambda = 0.5
window_znorm = false
ring_window = 256
warmup = 256
"""


def run(bin_path, series, work):
    os.makedirs(work, exist_ok=True)
    inp = os.path.join(work, "in.txt")
    np.savetxt(inp, series, fmt="%.17g")
    cfg = os.path.join(work, "run.cfg")
    with open(cfg, "w") as f:
        f.write(TEST_CFG + f"input = {inp}\nout_dir = {work}\nwords = {work}/words.csv\n")
    p = subprocess.run([bin_path, cfg], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("C failed:\n" + "\n".join(p.stderr.splitlines()[-5:]))
    words = open(os.path.join(work, "words.csv")).read()
    epochs = [r for r in csv.DictReader(open(os.path.join(work, "epochs.csv")))]
    bits = []
    for r in epochs:
        n = int(r["epoch"])
        rows = list(csv.DictReader(open(os.path.join(work, f"dict_{n}.csv"))))
        bits.append([x["bits"] for x in rows])
    ep = [(r["first_window"], r["type"]) for r in epochs]
    return words, ep, bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./build/spartan_toy")
    ap.add_argument("--n", type=int, default=30000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=args.n)) * 0.1 + np.sin(np.arange(args.n) / 7.0)
    x[args.n // 3:] += 4.0            # level shift
    x[2 * args.n // 3:] *= 2.5        # amplitude jump
    with tempfile.TemporaryDirectory() as tmp:
        ref = run(args.bin, x, os.path.join(tmp, "ref"))
        print(f"reference   {ref[0].count(chr(10)) - 1} words, {len(ref[1])} epochs")
        bad = 0
        for a, b in PAIRS:
            got = run(args.bin, a * x + b, os.path.join(tmp, f"a{a}b{b}"))
            same = [got[0] == ref[0], got[1] == ref[1], got[2] == ref[2]]
            bad += not all(same)
            print(f"a={a:<8g} b={b:<8g} words {'same' if same[0] else 'DIFFER'}  "
                  f"epochs {'same' if same[1] else 'DIFFER'}  "
                  f"splits {'same' if same[2] else 'DIFFER'}")
    print("PASS" if not bad else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
