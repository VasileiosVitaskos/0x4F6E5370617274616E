#!/usr/bin/env python3
"""Convert community time-series archives into the format the C reader expects.

Output format (what stream_init_file reads):
    one FRAME per line, `channels` numbers per line, space separated,
    written with %.17g so a double survives the round trip exactly.

Every converter also writes a sidecar with whatever ground truth the source
carried, because that information must not be lost and must never be fed to
the algorithm as if it were data.

    convert.py tsb          in.csv  -o out.txt   ->  out.txt, out.truth
    convert.py ucr-anomaly  in.txt  -o out.txt   ->  out.txt, out.truth
    convert.py ucr-tsv      in.tsv  -o out.txt   ->  out.txt, out.labels, out.bounds
    convert.py plain        in.txt  -o out.txt   ->  out.txt

Standard library only, so it runs anywhere without a virtualenv.
"""

import argparse
import csv
import os
import re
import sys

FMT = "%.17g"


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def write_frames(path, frames):
    """frames is a list of lists, one inner list per time step."""
    with open(path, "w") as f:
        for fr in frames:
            f.write(" ".join(FMT % v for v in fr))
            f.write("\n")


def write_sidecar(path, values, header):
    with open(path, "w") as f:
        f.write("# " + header + "\n")
        for v in values:
            f.write(str(v) + "\n")


def sidecar(out_path, suffix):
    root, _ = os.path.splitext(out_path)
    return root + suffix


# --------------------------------------------------------------------------
# sanity report
# --------------------------------------------------------------------------


def report(frames, out_path):
    """Print what was produced and shout about the one mistake that would
    otherwise run silently to the end: a label column fed in as a channel."""
    if not frames:
        print("  nothing written: no frames found", file=sys.stderr)
        return

    d = len(frames[0])
    print("  frames   = %d" % len(frames), file=sys.stderr)
    print(
        "  channels = %d   -> set 'channels = %d' in the settings file" % (d, d),
        file=sys.stderr,
    )

    for c in range(d):
        col = [fr[c] for fr in frames]
        lo, hi = min(col), max(col)
        distinct = set(col)
        note = ""
        if len(distinct) == 1:
            note = "   <-- CONSTANT, is this really data?"
        elif distinct <= {0.0, 1.0}:
            note = "   <-- only {0,1}, is this a LABEL column?"
        print(
            "  channel %d: min %-12.6g max %-12.6g%s" % (c, lo, hi, note),
            file=sys.stderr,
        )

    print("  wrote %s" % out_path, file=sys.stderr)


# --------------------------------------------------------------------------
# TSB-AD / TSB-UAD:  CSV with a header row and a trailing 'Label' column
# --------------------------------------------------------------------------


def conv_tsb(in_path, out_path):
    with open(in_path, newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        sys.exit("error: %s is empty" % in_path)

    header = [h.strip() for h in rows[0]]
    body = rows[1:]

    # The label is the last column in this family; accept a few spellings
    label_idx = None
    for i, name in enumerate(header):
        if name.lower() in ("label", "is_anomaly", "anomaly"):
            label_idx = i
    if label_idx is None:
        label_idx = len(header) - 1
        print(
            "  note: no column named 'Label', assuming the last one (%r)"
            % header[label_idx],
            file=sys.stderr,
        )

    frames, labels = [], []
    for n, row in enumerate(body, start=2):
        if not row or all(c.strip() == "" for c in row):
            continue
        try:
            vals = [float(row[i]) for i in range(len(row)) if i != label_idx]
            labels.append(int(float(row[label_idx])))
        except (ValueError, IndexError):
            sys.exit("error: %s line %d is not numeric: %r" % (in_path, n, row))
        frames.append(vals)

    write_frames(out_path, frames)
    write_sidecar(
        sidecar(out_path, ".truth"),
        labels,
        "per-frame anomaly label, aligned with %s" % os.path.basename(out_path),
    )
    report(frames, out_path)


# --------------------------------------------------------------------------
# UCR Anomaly Archive:  one value per line, the FILENAME carries the answer
#   ..._<train_end>_<anom_start>_<anom_end>.txt   (1-indexed in the original)
# --------------------------------------------------------------------------


def conv_ucr_anomaly(in_path, out_path):
    vals = []
    with open(in_path) as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                vals.append(float(line))
            except ValueError:
                sys.exit("error: %s line %d is not a number: %r" % (in_path, n, line))

    frames = [[v] for v in vals]
    write_frames(out_path, frames)

    # Pull the three trailing integers out of the file name
    base = os.path.basename(in_path)
    nums = re.findall(r"_(\d+)", base)
    truth_path = sidecar(out_path, ".truth")
    if len(nums) >= 3:
        train_end, a0, a1 = (int(x) for x in nums[-3:])
        labels = [0] * len(vals)
        # the archive counts from 1; our frames count from 0
        for i in range(max(0, a0 - 1), min(len(vals), a1)):
            labels[i] = 1
        write_sidecar(
            truth_path,
            labels,
            "per-frame anomaly label from the file name "
            "(train_end=%d, anomaly=%d..%d)" % (train_end, a0, a1),
        )
        print(
            "  anomaly  = frames %d..%d, train split at %d"
            % (a0 - 1, a1 - 1, train_end),
            file=sys.stderr,
        )
    else:
        print(
            "  warning: file name carries no _train_start_end numbers, "
            "no truth written",
            file=sys.stderr,
        )

    report(frames, out_path)


# --------------------------------------------------------------------------
# UCR classification TSV:  one whole SERIES per line, class label first
#
# These are pre-segmented short series, not a stream. Concatenating them makes
# every window that straddles two series meaningless, so the boundaries are
# written out and the analysis must drop those windows.
# --------------------------------------------------------------------------


def conv_ucr_tsv(in_path, out_path):
    series, labels = [], []
    with open(in_path) as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\s,]+", line)
            try:
                labels.append(parts[0])
                vals = [float(x) for x in parts[1:] if x != ""]
            except ValueError:
                sys.exit(
                    "error: %s line %d is not numeric: %r" % (in_path, n, line[:60])
                )
            # Some datasets pad variable-length series with NaN
            vals = [v for v in vals if v == v]
            if vals:
                series.append(vals)

    frames, bounds, pos = [], [], 0
    for s in series:
        frames.extend([v] for v in s)
        pos += len(s)
        bounds.append(pos)

    write_frames(out_path, frames)
    write_sidecar(
        sidecar(out_path, ".labels"), labels, "class label, one per series, in order"
    )
    write_sidecar(
        sidecar(out_path, ".bounds"),
        bounds,
        "end frame index of each series, exclusive; "
        "windows crossing these are meaningless and must be dropped",
    )

    print("  series   = %d, concatenated" % len(series), file=sys.stderr)
    print(
        "  WARNING: these are separate series, not a stream. Any window that "
        "crosses a boundary in the .bounds file mixes two series.",
        file=sys.stderr,
    )
    report(frames, out_path)


# --------------------------------------------------------------------------
# plain: already one frame per line, just normalise the number formatting
# --------------------------------------------------------------------------


def conv_plain(in_path, out_path):
    frames = []
    with open(in_path) as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\s,;]+", line)
            try:
                frames.append([float(x) for x in parts if x != ""])
            except ValueError:
                sys.exit(
                    "error: %s line %d is not numeric: %r" % (in_path, n, line[:60])
                )

    widths = {len(fr) for fr in frames}
    if len(widths) > 1:
        sys.exit(
            "error: %s has lines of differing width: %s" % (in_path, sorted(widths))
        )

    write_frames(out_path, frames)
    report(frames, out_path)


# --------------------------------------------------------------------------

CONVERTERS = {
    "tsb": conv_tsb,
    "ucr-anomaly": conv_ucr_anomaly,
    "ucr-tsv": conv_ucr_tsv,
    "plain": conv_plain,
}


def main():
    ap = argparse.ArgumentParser(
        description="Convert a time-series archive file into the format the "
        "Online SPARTAN reader expects."
    )
    ap.add_argument("format", choices=sorted(CONVERTERS))
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    print("converting %s (%s)" % (args.input, args.format), file=sys.stderr)
    CONVERTERS[args.format](args.input, args.output)


if __name__ == "__main__":
    main()
