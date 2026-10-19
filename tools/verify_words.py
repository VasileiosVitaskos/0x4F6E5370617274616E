#!/usr/bin/env python3
"""Independent check of the letters the C wrote.

Rebuilds every window from the built-in demo array, replays the running mean,
projects onto that window's epoch basis, digitises against that epoch's cut
points, and compares letter by letter with what the C produced.

Nothing here reads the C source. If the two agree on all 190 symbolised
windows, then the freeze, the promotion, the row permutation and the cut
points are all consistent with one another AND with the files on disk.

Usage:
    python3 verify_words.py <out_dir> <words.csv>

e.g.
    python3 verify_words.py debug_tests/out/epochs debug_tests/out/demo_words.csv

Pure standard library: no numpy, nothing to install.
"""

import csv
import math
import os
import struct
import sys


def _ordinal(x):
    """Map a double onto an integer so that ordering is preserved. Adjacent
    doubles get adjacent integers, which is what makes a ULP count possible."""
    n = struct.unpack("<q", struct.pack("<d", x))[0]
    if n < 0:
        n = -0x8000000000000000 - n
    return n


def ulp_distance(a, b):
    """How many representable doubles sit between a and b. 0 means the two are
    the same number; 1 means they are neighbours and no double lies between."""
    if math.isinf(a) or math.isinf(b) or math.isnan(a) or math.isnan(b):
        return None
    return abs(_ordinal(a) - _ordinal(b))


# Must match SPARTAN_EPS in spartan.h. znorm divides by (std + EPS), so a wrong
# value here shifts every projection by a hair and every borderline letter flips.
# If this script reports mismatches that are all near a cut point, check it:
#     grep SPARTAN_EPS spartan.h
SPARTAN_EPS = 1e-12

# The built-in demo stream, copied from main.c:33-53. 20 patterns of 8 samples,
# the block repeated DEMO_REPEATS times.
DEMO_ROWS = [
    [1.0, 2.1, 2.9, 4.2, 5.0, 6.1, 6.9, 8.0],
    [0.8, 1.9, 3.1, 3.9, 5.2, 5.9, 7.1, 7.9],
    [1.2, 2.0, 3.2, 4.1, 4.8, 6.2, 7.0, 8.1],
    [0.9, 2.2, 2.8, 4.0, 5.1, 5.8, 7.2, 8.2],
    [1.1, 1.8, 3.0, 4.3, 4.9, 6.0, 7.1, 7.8],
    [1.0, 2.0, 3.1, 3.8, 5.0, 6.2, 6.8, 8.0],
    [0.7, 2.1, 3.0, 4.1, 5.2, 6.1, 7.0, 8.3],
    [8.0, 6.9, 6.1, 5.0, 4.2, 2.9, 2.1, 1.0],
    [7.9, 7.1, 5.9, 5.2, 3.9, 3.1, 1.9, 0.8],
    [8.1, 7.0, 6.2, 4.8, 4.1, 3.2, 2.0, 1.2],
    [8.2, 7.2, 5.8, 5.1, 4.0, 2.8, 2.2, 0.9],
    [7.8, 7.1, 6.0, 4.9, 4.3, 3.0, 1.8, 1.1],
    [8.0, 6.8, 6.2, 5.0, 3.8, 3.1, 2.0, 1.0],
    [8.3, 7.0, 6.1, 5.2, 4.1, 3.0, 2.1, 0.7],
    [8.0, 6.0, 4.1, 2.0, 2.1, 3.9, 6.1, 7.9],
    [7.8, 5.9, 3.8, 1.9, 2.2, 4.1, 5.8, 8.1],
    [8.2, 6.1, 4.0, 2.2, 1.8, 4.0, 6.2, 7.8],
    [7.9, 6.2, 3.9, 1.8, 2.0, 4.2, 6.0, 8.0],
    [8.1, 5.8, 4.2, 2.1, 1.9, 3.8, 5.9, 8.2],
    [8.0, 6.0, 4.0, 2.0, 2.0, 4.0, 6.0, 8.0],
]
DEMO_REPEATS = 10

# Stream geometry for the demo: one channel, 8 frames per window, stride 8,
# so the windows tile the array without overlap. 1600 / 8 = 200 windows.
D = 1
M_TIME = 8
STRIDE = 8


def demo_stream():
    """The flat sample array the C builds at main.c:431-435."""
    flat = []
    for _ in range(DEMO_REPEATS):
        for row in DEMO_ROWS:
            flat.extend(row)
    return flat


def windows(flat):
    """Every window the stream hands out, in order."""
    m = D * M_TIME
    out = []
    start = 0
    while start + m <= len(flat):
        out.append(flat[start : start + m])
        start += STRIDE * D
    return out


def znorm(x):
    """spartan.c:34 znorm_row, per channel. Divides by m, not m-1, and adds
    SPARTAN_EPS to the deviation so a flat window does not divide by zero."""
    m = len(x)
    mean = sum(x) / m
    var = sum((v - mean) ** 2 for v in x) / m
    std = math.sqrt(var) + SPARTAN_EPS
    return [(v - mean) / std for v in x]


def read_epochs(path):
    """epochs.csv -> list of (epoch, first_window), in file order."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["epoch"]), int(r["first_window"])))
    return rows


def read_basis(path):
    """basis_n.csv -> list of k rows, each m floats."""
    out = []
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        next(rdr)  # header: dir,v0,...
        for row in rdr:
            out.append([float(v) for v in row[1:]])
    return out


def read_dict(path):
    """dict_n.csv -> (alphabet per position, cut points per position).

    A slot past the alphabet size is an empty field, and the last live slot is
    the literal 'inf' that digitize always stops at."""
    alphabet, bkpt = [], []
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        first_bkpt = header.index("bkpt0")
        for row in rdr:
            a = int(row[1])
            alphabet.append(a)
            cuts = []
            for b in range(a):
                cell = row[first_bkpt + b].strip()
                cuts.append(float("inf") if cell == "inf" else float(cell))
            bkpt.append(cuts)
    return alphabet, bkpt


def read_words(path):
    """demo_words.csv -> {window index: [symbols]}. Only the numeric form;
    a run with pretty = 1 writes letters and this check does not apply."""
    out = {}
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        next(rdr)  # header: window,sym0,...
        for row in rdr:
            out[int(row[0])] = [int(v) for v in row[1:]]
    return out


def digitize(y, alphabet, bkpt):
    """spartan.c:508. Strictly greater, so a value sitting exactly on a cut
    point stays in the lower cell."""
    word = []
    for j, yj in enumerate(y):
        s = 0
        while s < alphabet[j] - 1 and yj > bkpt[j][s]:
            s += 1
        word.append(s)
    return word


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    out_dir, words_path = sys.argv[1], sys.argv[2]

    epochs = read_epochs(os.path.join(out_dir, "epochs.csv"))
    tables = {}
    for n, _ in epochs:
        basis = read_basis(os.path.join(out_dir, "basis_%d.csv" % n))
        alphabet, bkpt = read_dict(os.path.join(out_dir, "dict_%d.csv" % n))
        tables[n] = (basis, alphabet, bkpt)

    expected = read_words(words_path)
    wins = windows(demo_stream())
    m = D * M_TIME

    # The running mean the C keeps, replayed over every window including the
    # warm-up ones. spartan.c:76. It is updated BEFORE the window is centred,
    # so the current window is part of its own mean.
    rmean = [0.0] * m
    n_seen = 0

    checked = 0
    mismatches = []

    for w_idx, raw in enumerate(wins):
        # Channel-wise z-norm. One channel here, so one call.
        win = []
        for c in range(D):
            win.extend(znorm(raw[c * M_TIME : (c + 1) * M_TIME]))

        n_seen += 1
        for i in range(m):
            rmean[i] += (win[i] - rmean[i]) / n_seen
        xc = [win[i] - rmean[i] for i in range(m)]

        if w_idx not in expected:
            continue  # warm-up window, no letters written

        # Which epoch owns this window: the last one that started at or before
        # it. No end column in epochs.csv, by design.
        ep = None
        for n, first in epochs:
            if first <= w_idx:
                ep = n
            else:
                break
        if ep is None:
            mismatches.append((w_idx, "no epoch covers this window", None, None, None))
            continue

        basis, alphabet, bkpt = tables[ep]
        y = [sum(xc[i] * row[i] for i in range(m)) for row in basis]
        got = digitize(y, alphabet, bkpt)

        checked += 1
        if got != expected[w_idx]:
            mismatches.append((w_idx, expected[w_idx], got, y, ep))

    print("windows in stream      : %d" % len(wins))
    print("windows with letters   : %d" % len(expected))
    print("windows checked        : %d" % checked)
    print(
        "epochs                 : %s"
        % ", ".join("%d from w%d" % (n, f) for n, f in epochs)
    )
    print("mismatches             : %d" % len(mismatches))

    if mismatches:
        print("\nfirst 10:")
        for w_idx, exp, got, y, ep in mismatches[:10]:
            print("  window %-4d  C wrote %-14s  recomputed %s" % (w_idx, exp, got))
            if y is None:
                continue
            _, alphabet, bkpt = tables[ep]
            for j in range(len(got)):
                if got[j] == exp[j]:
                    continue
                # digitize walks up while y > cut, so the cut that decided it
                # is the one at the index of the LOWER of the two letters.
                lo = min(exp[j], got[j])
                cut = bkpt[j][lo]
                d = ulp_distance(y[j], cut)
                rel = (
                    abs(y[j] - cut) / abs(cut)
                    if cut not in (0.0, float("inf"))
                    else float("nan")
                )
                print("      pos %d  y = %.17g" % (j, y[j]))
                print("             cut[%d] = %.17g" % (lo, cut))
                print(
                    "             gap = %s ULP, relative %.3e"
                    % ("n/a" if d is None else d, rel)
                )
        print("\nA gap of a few ULP means the two sides agree on the number and")
        print("disagree only on which side of the cut point it falls. That is")
        print("the summation order: spartan.c:15 adds in four partial sums,")
        print("this script adds left to right. Not a logic error.")
        print("A large gap means something genuinely differs. Say which it is.")
        return 1

    print("\nPASS: every letter the C wrote is reproducible from the files alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
