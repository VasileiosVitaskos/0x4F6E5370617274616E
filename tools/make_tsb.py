#!/usr/bin/env python3
"""Synthetic streams shaped like TSB-AD sensor data. DATA ONLY.

This script decides what the data is and nothing about how it is read: no
window, no stride, no k, no warm-up. Those are settings of the algorithm and
live in the C alone. The cfg written next to each stream names the data
(input, channels) and where the run goes (out_dir, words), nothing else, so
the C runs on its own defaults.

Every stream is ONE continuous series, so any stride is meaningful. The
signal is four periodic parts whose periods do not divide a 256-frame window,
so each window catches them at a different phase, the way a sensor or an ECG
lead does. A periodic part fills two directions (its cosine and its sine).

Every period is under 256 frames. A period longer than the window looks,
inside one window, like a trend plus a little curvature: one strong
direction and one very weak one that ties with the noise. Measured at window
256: with periods up to 611 the gap between the 8th and 9th variance was 1.2
at every noise level; with all periods under 256 it is 3.3 at 20 % noise and
2.1 at 50 %.

    tsb_clean   the signal alone, no drift, no noise
    tsb_noisy   the same signal, no drift, a lot of red noise
    tsb_drift   the same signal, one period sliding from 97 to 126 frames
                over the stream, moderate red noise

    tsb_drift_strong   DEBUGGING set, not for the paper. The shortest period
                slides from 61 to 72 frames, enough motion to cross the C's
                default threshold, so real type 2 epochs open. It exists to
                check the re-basing machinery itself: does the staging start
                in time (overshoot, late) and how far behind the data the
                incoming basis is when it takes over (staleness)

Red noise is AR(1): each sample keeps rho of the previous one plus a fresh
shock. With rho = 0.95 its power sits at low frequencies, in a few
directions, as real sensor noise does.

The three share one seed for the signal, so they differ ONLY in what the
name says. Everything needed to rebuild the noise-free signal is in the
.meta.json, and batch_truth.py does that to grade the run against what was
planted, with whatever window the run used.

    python3 tools/make_tsb.py            write everything under tests/tsb
    python3 tools/make_tsb.py --list     print the plan, write nothing
"""

import argparse
import json
import os
import sys

import numpy as np

N_FRAMES = 1_024_128  # length of every stream [frames]
PERIODS = [61.0, 97.0, 139.0, 197.0]  # [frames]
AMPS = [1.00, 0.70, 0.50, 0.35]
RHO = 0.95
SEED_SIGNAL = 3001
SEED_NOISE = 3002

PLAN = [
    # name, drift (which period slides, and to what) or None, noise share
    ("tsb_clean", None, 0.00),
    ("tsb_noisy", None, 0.50),
    ("tsb_drift", dict(part=1, to=126.0), 0.20),
    ("tsb_drift_strong", dict(part=0, to=72.0), 0.20),
]

OUT = os.path.join("tests", "tsb")


# --------------------------------------------------------------------------
# the series
# --------------------------------------------------------------------------


def signal(n, periods, amps, drift, seed):
    """Sum of the periodic parts, noise-free.

    drift is None, or dict(part=i, to=P): period i slides linearly from its
    value to P over the n frames. Built from the PHASE, the running sum of
    the frequency, rather than as cos(2 pi t / P(t)): with a period that
    changes, the second form jumps and folds, the first is a smooth chirp.

    Pure: everything comes in as arguments, so batch_truth.py can rebuild
    exactly this from a .meta.json.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    out = np.zeros(n)
    for i, (p, a) in enumerate(zip(periods, amps)):
        if drift is not None and i == drift["part"]:
            period = p + (drift["to"] - p) * t / (n - 1)
        else:
            period = np.full(n, p)
        phase = 2.0 * np.pi * np.cumsum(1.0 / period) + rng.uniform(0, 2 * np.pi)
        out += a * np.cos(phase)
    return out


def red_noise(n, rho, seed):
    """AR(1) with coefficient rho, unit variance. A plain loop: a million
    steps is under a second and needs nothing beyond numpy."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n) * np.sqrt(1.0 - rho * rho)
    e = np.empty(n)
    e[0] = rng.standard_normal()
    for i in range(1, n):
        e[i] = rho * e[i - 1] + w[i]
    return e


def planted_directions(periods, m_time):
    """The cosine and sine of each period on a window of m_time frames, mean
    removed (z-norm removes it), orthonormalised. Takes the window as an
    argument because the window is the C's setting, not this script's."""
    t = np.arange(m_time, dtype=float)
    rows = []
    for p in periods:
        rows += [np.cos(2 * np.pi * t / p), np.sin(2 * np.pi * t / p)]
    R = np.array(rows)
    R -= R.mean(axis=1, keepdims=True)
    Q, _ = np.linalg.qr(R.T)
    return Q.T


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

CFG = """# {name}
# generated by tools/make_tsb.py. Names the data and where the run goes,
# nothing else: every algorithm setting is the C's default unless you add it
input    = {frames}
channels = 1
out_dir  = {out_dir}
words    = {out_dir}/words.csv
"""


def write_one(name, drift, share):
    for sub in ("data", "cfg", "out/" + name):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    x = signal(N_FRAMES, PERIODS, AMPS, drift, SEED_SIGNAL)
    if share > 0.0:
        e = red_noise(N_FRAMES, RHO, SEED_NOISE)
        # var(noise) / (var(signal) + var(noise)) = share
        x = x + e * np.sqrt(share / (1.0 - share) * x.var() / e.var())

    frames = os.path.join(OUT, "data", name + ".frames")
    with open(frames, "w") as f:
        f.write("\n".join("%.17g" % v for v in x))
        f.write("\n")

    out_dir = os.path.join(OUT, "out", name)
    with open(os.path.join(OUT, "cfg", name + ".cfg"), "w") as f:
        f.write(CFG.format(name=name, frames=frames, out_dir=out_dir))

    meta = dict(
        name=name,
        generator="tools/make_tsb.py",
        n_frames=N_FRAMES,
        channels=1,
        signal=dict(periods=PERIODS, amplitudes=AMPS, drift=drift, seed=SEED_SIGNAL),
        noise=dict(kind="AR(1)", rho=RHO, share=share, seed=SEED_NOISE),
    )
    with open(os.path.join(OUT, "data", name + ".meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    periods = " / ".join(
        ("%g->%g" % (p, drift["to"])) if (drift and i == drift["part"]) else "%g" % p
        for i, p in enumerate(PERIODS)
    )
    print("%-12s %d frames, 1 channel" % (name, N_FRAMES))
    print(
        "  signal     periods %s frames, amplitudes %s"
        % (periods, " / ".join("%g" % a for a in AMPS))
    )
    print(
        "  noise      %s"
        % (
            "none"
            if share <= 0
            else "red, rho %.2f, %d %% of the energy" % (RHO, round(100 * share))
        )
    )
    print("  wrote      %s  .meta.json  cfg/%s.cfg" % (frames, name))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--list", action="store_true", help="print the plan and write nothing"
    )
    args = ap.parse_args()
    if args.list:
        for name, drift, share in PLAN:
            print("%-12s drift %-26s noise %d %%" % (name, drift, round(100 * share)))
        return 0
    for name, drift, share in PLAN:
        write_one(name, drift, share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
