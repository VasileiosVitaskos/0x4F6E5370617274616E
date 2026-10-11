#!/usr/bin/env python3
"""Synthetic datasets for Online SPARTAN, each with its answer.

The signal is built backwards, from the answer. We choose r orthonormal
directions w_i in R^m and an amplitude lambda_i for each, and a window is

    x = sum_i sqrt(lambda_i) * g_i * w_i  +  sigma * n,     g_i, n ~ N(0, 1)

so the two things the algorithm estimates, the directions and the variance
shares, are both things we chose. That buys the separation the whole test plan
rests on:

    a type 1 event changes the lambdas and leaves the w alone
        -> the subspace does not move, the shares do
        -> the angle must stay at zero, the gap must rise

    a type 2 event rotates the w and leaves the lambdas alone
        -> the subspace moves, the shares do not
        -> the angle must rise, the gap must stay at zero

Each detector is then independently falsifiable: in a pure type 1 set, any
angle above the floor is a false positive, and in a pure type 2 set, any gap is.

The truth file does NOT record the lambdas we chose. The C z-normalises every
window before it sees it, which changes the covariance, so the lambdas are not
what the algorithm is trying to estimate. What it records instead is what BATCH
PCA says about the windows actually written, after the same normalisation. That
is the offline answer on the same data, which is the only reference worth
comparing an online estimate against, and it doubles as the batch SPARTAN
baseline.

The w are chosen orthogonal to the all-ones vector, so that removing a window's
mean does not disturb them.

    python3 make_datasets.py            writes everything into ./datasets
    python3 make_datasets.py --list     prints the plan and writes nothing
"""

import json
import os
import sys

import numpy as np

M_TIME = 32          # numbers per window; also the stride, so no overlap
K = 3                # kept directions, and letters per word
TOTAL_BITS = 6
LAMBDA_REG = 0.5
WARMUP = 200         # 10 * the largest alphabet, which is the code's own rule
RING_W = 200
OUT = "datasets"


# ----------------------------------------------------------------- geometry

def orthonormal_basis(rng, m, count, forbidden):
    """count orthonormal vectors in R^m, all orthogonal to every row of
    forbidden. Modified Gram-Schmidt, twice: one pass leaves an orthogonality
    error around 1e-13, which would quietly become part of the answer."""
    out = []
    while len(out) < count:
        v = rng.standard_normal(m)
        for _ in range(2):
            for f in forbidden:
                v -= np.dot(v, f) * f
            for u in out:
                v -= np.dot(v, u) * u
        nrm = np.linalg.norm(v)
        if nrm < 1e-8:
            continue
        out.append(v / nrm)
    return np.array(out)


def rotate_in_plane(W, i, axis, theta):
    """Rotate row i of W by theta towards axis, inside the plane they span.
    axis must be a unit vector orthogonal to row i. Returns a new W."""
    out = W.copy()
    out[i] = np.cos(theta) * W[i] + np.sin(theta) * axis
    return out


def principal_angles(A, B):
    """The principal angles between the row spaces of A and B, both with
    orthonormal rows. Exact here because we know A and B on paper."""
    sv = np.linalg.svd(A @ B.T, compute_uv=False)
    return np.arccos(np.clip(sv, -1.0, 1.0))


# ----------------------------------------------------------------- schedules

def ramp(t, t0, t1, a, b):
    """a before t0, b after t1, straight line in between."""
    if t <= t0:
        return a
    if t >= t1:
        return b
    f = (t - t0) / (t1 - t0)
    return a + f * (b - a)


def calibrate_lambda(target, sigma, W, rng_seed=99, iters=25, n=4000):
    """Find the amplitudes that make the BATCH shares come out equal to target.

    There is no usable formula for this. The z-norm divides each window by its
    own spread, which is dominated by whichever direction is largest in that
    window, so it compresses the gaps between directions by an amount that
    depends on the whole configuration. Asking for 0.62 of the variance on the
    first direction and setting lambda to 0.62 gives about 0.49 in the file.

    So it is solved numerically: measure, scale each amplitude by how far off
    its share is, repeat. Converges to machine precision in about twenty
    passes. The fixed seed keeps the measurement from wandering between passes,
    which is what makes it converge instead of chase noise.
    """
    target = np.asarray(target, dtype=float)
    lam = target * 12.0
    for _ in range(iters):
        got = measure_shares(lam, W, sigma, rng_seed, n)
        lam = lam * (target / np.maximum(got, 1e-12))
        lam = lam / lam.sum() * 12.0
    return lam


def measure_shares(lam, W, sigma, rng_seed, n):
    rng = np.random.default_rng(rng_seed)
    g = rng.standard_normal((n, len(lam)))
    X = (g * np.sqrt(lam)) @ W + sigma * rng.standard_normal((n, W.shape[1]))
    mu = X.mean(axis=1, keepdims=True)
    sd = np.sqrt(((X - mu) ** 2).mean(axis=1, keepdims=True)) + 1e-12
    return batch_shares((X - mu) / sd)


def shares_at(t, schedule):
    """schedule is a list of (t0, t1, shares_before, shares_after) ramps applied
    in order; the last one that has started wins."""
    cur = np.array(schedule[0][2], dtype=float)
    for t0, t1, a, b in schedule:
        if t >= t0:
            cur = np.array([ramp(t, t0, t1, a[i], b[i]) for i in range(len(a))])
    return cur / cur.sum()


# ----------------------------------------------------------------- scenarios

def scenario_specs():
    """Each entry says how the shares and the rotation move over time, and what
    the run is supposed to demonstrate. n is the number of windows."""
    return [
        dict(
            name="e1_type1_must_not_fire",
            n=3000, sigma=0.20, outlier_rate=0.0,
            # Drifts a long way and stops just short of the point where the
            # best bit split changes. Every window still wants 3 2 1.
            shares=[(0, 0, [0.62, 0.26, 0.12], [0.62, 0.26, 0.12]),
                    (400, 2600, [0.62, 0.26, 0.12], [0.560, 0.260, 0.180])],
            rotation=None,
            expect=dict(
                events=[],
                angle="must stay at the resolution floor, under 1e-5 rad",
                gap="rises and stays up, but never justifies a re-tuning",
                note="a threshold read off the gap alone produces false "
                     "positives here; the allocation never changes",
            ),
        ),
        dict(
            name="e2_type2_must_not_fire",
            n=3000, sigma=0.20, outlier_rate=0.0,
            # Two equal shares make the top plane degenerate: no rotation inside
            # it is more correct than another, so the live directions are free
            # to wander. The SUBSPACE never moves.
            shares=[(0, 0, [0.40, 0.40, 0.20], [0.40, 0.40, 0.20])],
            calibrate=False,   # see below
            rotation=dict(kind="inside", i=0, j=1, t0=600, t1=2400,
                          theta=1.2),
            expect=dict(
                events=[],
                angle="exactly zero throughout, because the span is unchanged",
                gap="zero, because the shares are unchanged",
                note="degenerate top eigenvalues, the classic hard case. A "
                     "detector that measures angles between VECTORS screams "
                     "here; one that measures principal angles says nothing.",
            ),
        ),
        dict(
            name="t1_easy",
            n=3000, sigma=0.15, outlier_rate=0.0,
            shares=[(0, 0, [0.75, 0.18, 0.07], [0.75, 0.18, 0.07]),
                    (1500, 1500, [0.75, 0.18, 0.07], [0.40, 0.35, 0.25])],
            rotation=None,
            expect=dict(
                events=[dict(window=1500, type="retune")],
                angle="at the floor throughout",
                gap="near zero before 1500, a step up after it",
                note="the allocation crosses from 3 2 1 to 2 2 2",
            ),
        ),
        dict(
            name="t1_hard",
            n=4000, sigma=0.45, outlier_rate=0.01,
            shares=[(0, 0, [0.75, 0.18, 0.07], [0.75, 0.18, 0.07]),
                    (1200, 2800, [0.75, 0.18, 0.07], [0.40, 0.35, 0.25])],
            rotation=None,
            expect=dict(
                events=[dict(window=None, type="retune")],
                angle="at the floor throughout",
                gap="climbs slowly through the ramp",
                note="the question is not whether it is detected but HOW LATE. "
                     "The crossing window is computed below and stored.",
            ),
        ),
        dict(
            name="t2_pure",
            n=4000, sigma=0.20, outlier_rate=0.0,
            shares=[(0, 0, [0.70, 0.20, 0.10], [0.70, 0.20, 0.10])],
            rotation=dict(kind="outside", i=2, t0=1200, t1=2800, theta=0.35),
            expect=dict(
                events=[dict(window=1200, type="rebase")],
                angle="follows the stored true angle, within the lag of the "
                      "live basis",
                gap="stays near zero, the shares never move",
                note="truth.angle_true holds the exact subspace angle per "
                     "window, so the measured curve can be laid over it",
            ),
        ),
        dict(
            name="m1_sequential",
            n=6000, sigma=0.25, outlier_rate=0.0,
            shares=[(0, 0, [0.72, 0.19, 0.09], [0.72, 0.19, 0.09]),
                    (1000, 1000, [0.72, 0.19, 0.09], [0.45, 0.33, 0.22]),
                    (4200, 4200, [0.45, 0.33, 0.22], [0.70, 0.21, 0.09])],
            rotation=dict(kind="outside", i=2, t0=2200, t1=3200, theta=0.30),
            expect=dict(
                events=[dict(window=1000, type="retune"),
                        dict(window=2200, type="rebase"),
                        dict(window=4200, type="retune")],
                angle="flat, then a ramp between 2200 and 3200, then flat",
                gap="steps at 1000 and at 4200, quiet during the rotation",
                note="the three events do not overlap, so each path is "
                     "exercised on its own and the registry order is checkable",
            ),
        ),
        dict(
            name="m2_overlapping",
            n=6000, sigma=0.50, outlier_rate=0.02,
            shares=[(0, 0, [0.72, 0.19, 0.09], [0.72, 0.19, 0.09]),
                    (2200, 2200, [0.72, 0.19, 0.09], [0.42, 0.34, 0.24])],
            rotation=dict(kind="outside", i=2, t0=1500, t1=3000, theta=0.40),
            expect=dict(
                events=[dict(window=1500, type="rebase"),
                        dict(window=2200, type="retune_suppressed")],
                angle="ramps from 1500 to 3000",
                gap="steps at 2200, in the middle of the rotation",
                note="THE case. A re-tuning becomes justified at 2200 while a "
                     "re-basing is already under way. The right answer is to "
                     "suppress it and let the re-basing do both jobs, so a run "
                     "that records a retune at 2200 has the policy wrong even "
                     "though every individual detector was right.",
            ),
        ),
    ]


# ----------------------------------------------------------------- generation

def build(spec, seed):
    """Returns the written windows, the true basis per window, and the realised
    share curve (what batch PCA will say, measured, per ramp fraction)."""
    rng = np.random.default_rng(seed)
    m, n = M_TIME, spec["n"]

    ones = np.ones(m) / np.sqrt(m)
    W0 = orthonormal_basis(rng, m, K, [ones])
    # one spare direction, outside the span, for a rotation to move into
    spare = orthonormal_basis(rng, m, 1, [ones] + list(W0))[0]

    # The schedule is written in terms of the shares we WANT to see in the
    # file. Solve for the amplitudes that produce them, at every distinct
    # target in the schedule, then interpolate between those solutions.
    if not spec.get("calibrate", True):
        # Two of the targets are deliberately EQUAL here, which makes the
        # calibration fixed point degenerate: the two directions cannot be
        # told apart, so correcting one against the other chases the sampling
        # asymmetry and runs away. This set is not about hitting exact shares,
        # it is about the span not moving, so the amplitudes are used as given
        # and the shares that come out are measured and recorded.
        lam_fixed = np.array(spec["shares"][0][2], dtype=float) * 12.0

        def lam_at(t):
            return lam_fixed
    else:
        lam_at = None

    knots = []
    for t0, t1, a, b in spec["shares"]:
        knots += [tuple(np.array(a) / sum(a)), tuple(np.array(b) / sum(b))]
    knots = list(dict.fromkeys(knots))
    lam_of = ({kk: calibrate_lambda(kk, spec["sigma"], W0) for kk in knots}
              if spec.get("calibrate", True) else {})

    def lam_calibrated(t):
        s = shares_at(t, spec["shares"])
        # nearest two knots by share distance, blended; with ramps between two
        # endpoints this is exactly a straight line in amplitude space
        d = sorted(knots, key=lambda kk: np.sum((np.array(kk) - s) ** 2))
        if np.allclose(d[0], s, atol=1e-12):
            return lam_of[d[0]]
        a, b = d[0], d[1]
        va, vb = np.array(a), np.array(b)
        denom = np.sum((vb - va) ** 2)
        f = 0.0 if denom < 1e-18 else np.clip(np.sum((s - va) * (vb - va)) / denom, 0.0, 1.0)
        return (1 - f) * lam_of[a] + f * lam_of[b]

    if lam_at is None:
        lam_at = lam_calibrated

    X = np.empty((n, m))
    W_of_t = []
    for t in range(n):
        lam = lam_at(t)

        W = W0
        if spec["rotation"] is not None:
            rot = spec["rotation"]
            theta = ramp(t, rot["t0"], rot["t1"], 0.0, rot["theta"])
            if rot["kind"] == "outside":
                W = rotate_in_plane(W0, rot["i"], spare, theta)
            else:
                # inside the span: mix two rows with each other, which changes
                # the directions and leaves the span untouched
                i, j = rot["i"], rot["j"]
                W = W0.copy()
                W[i] = np.cos(theta) * W0[i] + np.sin(theta) * W0[j]
                W[j] = -np.sin(theta) * W0[i] + np.cos(theta) * W0[j]
        W_of_t.append(W)

        g = rng.standard_normal(K)
        x = (g * np.sqrt(lam)) @ W + spec["sigma"] * rng.standard_normal(m)
        if spec["outlier_rate"] > 0 and rng.random() < spec["outlier_rate"]:
            x = x + rng.standard_normal(m) * 6.0 * spec["sigma"]
        X[t] = x

    # the z-norm the C applies, applied here so the file already holds what the
    # algorithm will see and the truth is computed on exactly those numbers
    mu = X.mean(axis=1, keepdims=True)
    sd = np.sqrt(((X - mu) ** 2).mean(axis=1, keepdims=True)) + 1e-12
    Z = (X - mu) / sd

    # the realised share curve, measured rather than assumed, one point every
    # 1 percent of the stream, plus the split each point implies
    curve = []
    for frac in np.linspace(0.0, 1.0, 101):
        t = int(frac * (n - 1))
        sh = measure_shares(lam_at(t), W0, spec["sigma"], 99, 4000)
        curve.append(dict(window=t, shares=sh.tolist(),
                          best_bits=best_split(sh)[0]))
    return Z, np.array(W_of_t), W0, curve


# ----------------------------------------------------------------- the truth

def batch_shares(Z):
    """What offline PCA says about these windows: the shares of the top K
    eigenvalues among themselves, which is what estimate_eigenvalues computes
    and what spartan.py reports after its own re-normalisation."""
    Zc = Z - Z.mean(axis=0)
    C = Zc.T @ Zc / len(Z)
    w = np.sort(np.linalg.eigvalsh(C))[::-1][:K]
    return w / w.sum()


def best_split(shares, total_bits=TOTAL_BITS, lam=LAMBDA_REG, max_bits=4):
    """The optimal bit split for these shares, by exhaustive search over every
    non-increasing allocation. Slow and obviously correct, which is what a
    reference is for; the C does the same thing with dynamic programming."""
    best, best_score = None, -1e18
    def rec(i, left, prev, acc):
        nonlocal best, best_score
        if i == len(shares):
            if left != 0:
                return
            ref = total_bits // len(shares)
            sc = sum(b * e - lam * (b - ref) ** 2 * e for b, e in zip(acc, shares))
            if sc > best_score:
                best_score, best = sc, list(acc)
            return
        for b in range(min(prev, max_bits, left), -1, -1):
            acc.append(b); rec(i + 1, left - b, b, acc); acc.pop()
    rec(0, total_bits, max_bits, [])
    return best, best_score


def make_truth(spec, Z, W_of_t, curve, seed):
    n = len(Z)
    rot = spec["rotation"]

    # stable segments: cut at every event the spec declares
    cross = None
    first_bits = curve[0]["best_bits"]
    for pt in curve:
        if pt["best_bits"] != first_bits:
            cross = pt["window"]
            break
    cuts = sorted({0, n} | {e["window"] for e in spec["expect"]["events"]
                            if isinstance(e["window"], int)})
    if rot is not None:
        cuts = sorted(set(cuts) | {rot["t0"], rot["t1"]})
    if cross is not None:
        cuts = sorted(set(cuts) | {cross})
    segments = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        lo, hi = a + (WARMUP if a == 0 else 0), b
        if hi - lo < 200:
            continue
        sh = batch_shares(Z[lo:hi])
        bits, score = best_split(sh)
        segments.append(dict(start=lo, end=hi, batch_shares=sh.tolist(),
                             best_bits=bits, best_score=score))

    # the exact subspace angle against the basis as it was at window 0
    angle_true = [float(principal_angles(W_of_t[0], W_of_t[t]).max())
                  for t in range(n)]

    # the window where the best split first changes, read off the measured
    # share curve rather than estimated from a sliding batch window
    crossing = None
    first = curve[0]["best_bits"]
    for pt in curve:
        if pt["best_bits"] != first:
            crossing = pt["window"]
            break

    # a spec may leave an event window unknown because it depends on where the
    # ramp crosses; fill it in now that the crossing is measured
    events = []
    for e in spec["expect"]["events"]:
        e = dict(e)
        if not isinstance(e["window"], int):
            e["window"] = crossing
        events.append(e)

    return dict(
        name=spec["name"], seed=seed,
        geometry=dict(m_time=M_TIME, channels=1, stride=M_TIME, k=K,
                      total_bits=TOTAL_BITS, lamda=LAMBDA_REG,
                      warmup=WARMUP, ring_window=RING_W, n_windows=n),
        noise=dict(sigma=spec["sigma"], outlier_rate=spec["outlier_rate"]),
        events=events,
        expect={kk: vv for kk, vv in spec["expect"].items() if kk != "events"},
        segments=segments,
        share_curve=curve,
        allocation_crossing_window=crossing,
        angle_true=angle_true,
    )


# ----------------------------------------------------------------- output

CFG = """# {name}
input         = {name}.frames
channels      = 1
window        = {m}
stride        = {m}
components    = {k}
total_bits    = {tb}
lambda        = {lam}
learning_rate = 0.01
gs_period     = 5
warmup        = {warm}
ring_window   = {ring}
words         = {name}.words.csv
dictionary    = {name}.dict.csv
pretty        = 0
"""


def write_all():
    os.makedirs(OUT, exist_ok=True)
    index = []
    for i, spec in enumerate(scenario_specs()):
        seed = 1000 + i
        Z, W_of_t, _, curve = build(spec, seed)
        name = spec["name"]

        with open(os.path.join(OUT, name + ".frames"), "w") as f:
            f.write("\n".join("%.17g" % v for v in Z.ravel()))
            f.write("\n")

        truth = make_truth(spec, Z, W_of_t, curve, seed)
        with open(os.path.join(OUT, name + ".truth.json"), "w") as f:
            json.dump(truth, f, indent=1)

        with open(os.path.join(OUT, name + ".cfg"), "w") as f:
            f.write(CFG.format(name=name, m=M_TIME, k=K, tb=TOTAL_BITS,
                               lam=LAMBDA_REG, warm=WARMUP, ring=RING_W))

        seg = truth["segments"]
        print("%-26s %5d windows  segments %d  first split %s  crossing %s"
              % (name, spec["n"], len(seg),
                 seg[0]["best_bits"] if seg else "-",
                 truth["allocation_crossing_window"]))
        for s in seg:
            print("      %5d..%-5d shares %s  best %s"
                  % (s["start"], s["end"],
                     [round(x, 4) for x in s["batch_shares"]], s["best_bits"]))
        index.append(dict(name=name, windows=spec["n"],
                          events=truth["events"]))

    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    if "--list" in sys.argv:
        for s in scenario_specs():
            print("%-26s n=%-5d sigma=%.2f  %s"
                  % (s["name"], s["n"], s["sigma"], s["expect"]["note"]))
    else:
        write_all()
