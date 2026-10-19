#!/usr/bin/env python3
"""Synthetic datasets for the Online SPARTAN / batch SPARTAN comparison.

Three runs, two data files. The point of each one is to remove a suspect:

    D1  stationary, theta = 89 deg   one epoch, the gate never opens.
        Any disagreement with spartan.py here is NOT the epoch machinery.
        It is one of four things: Oja's subspace estimate against exact PCA,
        the bit allocation, the breakpoints, or the digitize convention.

    D2  stationary, theta = 45 deg   THE SAME BYTES as D1, one line changed.
        Now the gate opens on Oja's own wander, on data that does not move.
        D2 minus D1 is the cost of the epoch mechanism on stationary data,
        measured with the data held exactly fixed.

    D3  degenerate span               two equal top shares, so the top plane
        has no preferred rotation inside it. The directions wander, the
        SPAN does not move at all. Every epoch here is self-inflicted.

The signal is built backwards from the answer. We pick k orthonormal
directions w_i in R^m and an amplitude lambda_i for each, and a window is

    x = sum_i sqrt(lambda_i) * g_i * w_i  +  sigma * n,    g_i, n ~ N(0,1)

so the directions and the variance shares are both things we chose, not
things we hope for.

The truth file does NOT record the lambdas. The C z-normalises every window
before it sees it, which changes the covariance, so the lambdas are not what
the algorithm is trying to estimate. What the truth records instead is what
BATCH PCA says about the windows actually written, after the same
normalisation. That is the offline answer on the same numbers, which is the
only reference worth comparing an online estimate against, and it doubles as
the batch SPARTAN baseline.

The w are chosen orthogonal to the all-ones vector, so that removing a
window's mean does not disturb them.

    python3 tools/make_datasets.py           write everything under tests/synthetic
    python3 tools/make_datasets.py --list    print the plan, write nothing
"""

import argparse
import json
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# Geometry. UCR scale: a window of 256 is a typical series length in the UCR
# classification archive, which is the archive SPARTAN's own experiments use.
# The demo runs at m = 8, which hides every cost that grows with m: Gram-
# Schmidt is O(k^2 m), the Oja step is O(k m), and an accidental O(m^2) is
# invisible at 8 and obvious at 256.
# --------------------------------------------------------------------------

M_TIME = 256  # numbers per window
CHANNELS = 1
STRIDE = M_TIME  # no overlap: each window is an independent draw, and
# the UCR classification archive is separate series too
K = 8  # kept directions, and letters per word
TOTAL_BITS = 16  # even split would be 2 bits per position
LAMBDA_REG = 0.5  # how hard the objective pulls towards the even split
MAX_BITS = 8  # must equal MAX_BITS in src/spartan.h
N_WINDOWS = 4000  # 4000 * 256 = 1,024,000 samples, about 25 MB per file
SIGMA = 0.20  # isotropic noise added to every window
WARMUP = 512  # windows collected before the first dictionary

OUT = os.path.join("tests", "synthetic")

# Shares we want to SEE IN THE FILE after z-norm, not the lambdas we feed in.
# A moderate decay: steep enough that the allocation is lopsided and the
# sorting invariant matters, flat enough that the tail positions still get
# bits. Must sum to 1.
SHARES_STATIONARY = [0.30, 0.20, 0.14, 0.10, 0.08, 0.07, 0.06, 0.05]

# Same tail, but the top two are EQUAL. That is what makes the top plane
# degenerate: no rotation inside it is more correct than any other.
SHARES_DEGENERATE = [0.25, 0.25, 0.14, 0.10, 0.08, 0.07, 0.06, 0.05]

# D3 rotation: mix rows 0 and 1 with each other, theta ramping over this span.
ROT_T0, ROT_T1, ROT_THETA = 1000, 3000, 1.2  # radians


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------


def orthonormal_basis(rng, m, count, forbidden):
    """count orthonormal vectors in R^m, every one orthogonal to every row of
    forbidden.

    Modified Gram-Schmidt run TWICE. One pass leaves an orthogonality error
    around 1e-13, which would quietly become part of the answer: the whole
    design rests on the span being exactly what we chose, and a 1e-13 leak is
    larger than the 1e-12 tolerance the D3 assert uses.

    Returns an array of shape (count, m).
    """
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
            continue  # drew something in the killed span
        out.append(v / nrm)
    return np.array(out)


def principal_angles(A, B):
    """Principal angles, in radians, between the row spaces of A and B.

    Both must have orthonormal rows. The singular values of A @ B.T are the
    cosines of the principal angles, so arccos of them is the answer. This
    measures the SUBSPACE, not the individual vectors: rotating a basis
    inside its own span leaves every principal angle at zero, which is
    exactly the property D3 exists to test.
    """
    sv = np.linalg.svd(A @ B.T, compute_uv=False)
    return np.arccos(np.clip(sv, -1.0, 1.0))


# --------------------------------------------------------------------------
# the objective, computed the slow obvious way
# --------------------------------------------------------------------------


def best_split(shares, total_bits=TOTAL_BITS, lam=LAMBDA_REG, max_bits=MAX_BITS):
    """The optimal bit allocation for these importances, by exhaustive search
    over every non-increasing allocation that spends the budget exactly.

    The objective, per position i with importance e_i and bits b_i:

        gain_i = b_i * e_i  -  lam * (b_i - ref)^2 * e_i
        ref    = total_bits // k            (the even split, in bits)

    Slow and obviously correct, which is what a reference is for. The C
    computes the same thing with a dynamic program; if the two ever disagree,
    the DP is wrong, not this.

    Note b_i may be ZERO here. The batch reference uses a floor of 1 bit per
    position; we allow 0, and that difference is one of the things the
    comparison is meant to quantify.

    Returns (allocation as a list, objective value).
    """
    best, best_score = None, -1e18
    k = len(shares)
    ref = total_bits // k

    def rec(i, left, prev, acc):
        nonlocal best, best_score
        if i == k:
            if left != 0:
                return
            sc = sum(b * e - lam * (b - ref) ** 2 * e for b, e in zip(acc, shares))
            if sc > best_score:
                best_score, best = sc, list(acc)
            return
        # non-increasing: never more than the position before it
        for b in range(min(prev, max_bits, left), -1, -1):
            acc.append(b)
            rec(i + 1, left - b, b, acc)
            acc.pop()

    rec(0, total_bits, max_bits, [])
    return best, best_score


def ring_window_for(shares):
    """The ring size the collected-quantile rule demands for these shares.

    A breakpoint is an empirical quantile of the projections in the ring. With
    an alphabet of A you are asking for A-1 cut points, so the ring needs
    enough points that each cell holds a meaningful count; the rule we settled
    on is ten points per cell:

        ring_window >= 10 * (largest alphabet across positions)

    Computed rather than chosen, so it cannot silently fall out of step with
    the bit budget. Floor of 512 so the warm-up is never the binding limit.
    """
    bits, _ = best_split(shares)
    largest_alphabet = 1 << max(bits)
    return max(512, 10 * largest_alphabet)


# --------------------------------------------------------------------------
# calibration: solve for the amplitudes that produce the shares we asked for
# --------------------------------------------------------------------------


def batch_shares(Z):
    """What offline PCA says about these windows: the shares of the top K
    eigenvalues among themselves.

    This is the quantity the C's own eigenvalue estimate is trying to track,
    and the one spartan.py reports after its re-normalisation, so it is the
    common ground the two can be compared on.

    Z is (n_windows, m), already z-normalised per window.
    """
    Zc = Z - Z.mean(axis=0)
    C = Zc.T @ Zc / len(Z)
    w = np.sort(np.linalg.eigvalsh(C))[::-1][:K]
    return w / w.sum()


def znorm_rows(X):
    """The per-window z-norm the C applies at spartan.c:34, applied here so
    the file already holds exactly the numbers the algorithm will see.

    Divides by m, not m-1, and adds SPARTAN_EPS to the deviation so a flat
    window does not divide by zero. Keep the epsilon in step with
    SPARTAN_EPS in src/spartan.h.
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = np.sqrt(((X - mu) ** 2).mean(axis=1, keepdims=True)) + 1e-12
    return (X - mu) / sd


def measure_shares(lam, W, sigma, seed, n=4000):
    """Draw n windows with these amplitudes, z-norm them, and report the batch
    shares that come out. Fixed seed on purpose: see calibrate_lambda."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, len(lam)))
    X = (g * np.sqrt(lam)) @ W + sigma * rng.standard_normal((n, W.shape[1]))
    return batch_shares(znorm_rows(X))


def calibrate_lambda(target, sigma, W, seed=99, iters=25, n=4000):
    """Find the amplitudes that make the BATCH shares come out equal to target.

    There is no usable formula. The z-norm divides each window by its own
    spread, and that spread is dominated by whichever direction happens to be
    largest in that particular window, so the normalisation compresses the
    gaps between directions by an amount that depends on the whole
    configuration. Ask for 0.30 on the first direction and set lambda to 0.30
    and you get about 0.24 in the file.

    So it is solved numerically: measure what came out, scale each amplitude
    by how far short its share fell, repeat. Converges in roughly twenty
    passes.

    The FIXED SEED is what makes it converge rather than chase its own tail:
    with a fresh seed each pass, the correction would be fighting sampling
    noise of the same order as the error it is trying to remove.

    Returns the amplitude vector, normalised so it sums to 12 (an arbitrary
    overall scale; z-norm removes it anyway).
    """
    target = np.asarray(target, dtype=float)
    lam = target * 12.0
    for _ in range(iters):
        got = measure_shares(lam, W, sigma, seed, n)
        lam = lam * (target / np.maximum(got, 1e-12))
        lam = lam / lam.sum() * 12.0
    return lam


# --------------------------------------------------------------------------
# the two generators
# --------------------------------------------------------------------------


def build_stationary(seed):
    """D1 and D2 share this. Nothing moves: one fixed basis, one fixed share
    vector, independent draws.

    Returns (Z, angle_true, W0) where Z is (N_WINDOWS, M_TIME) after z-norm
    and angle_true is identically zero, recorded anyway so every truth file
    has the same shape.
    """
    rng = np.random.default_rng(seed)
    ones = np.ones(M_TIME) / np.sqrt(M_TIME)
    W0 = orthonormal_basis(rng, M_TIME, K, [ones])

    lam = calibrate_lambda(SHARES_STATIONARY, SIGMA, W0)

    g = rng.standard_normal((N_WINDOWS, K))
    X = (g * np.sqrt(lam)) @ W0 + SIGMA * rng.standard_normal((N_WINDOWS, M_TIME))
    return znorm_rows(X), [0.0] * N_WINDOWS, W0


def build_degenerate(seed):
    """D3. The top two shares are equal, and rows 0 and 1 are rotated into
    each other over time.

    Because the rotation mixes two rows WITH EACH OTHER, the plane they span
    is untouched: the pair (cos t, sin t), (-sin t, cos t) is an orthogonal
    change of basis inside that plane. So every principal angle against the
    starting basis stays at zero while the individual direction vectors swing
    through 1.2 radians.

    Calibration is DISABLED here. With two equal targets the fixed point is
    degenerate: the two directions cannot be told apart, so correcting one
    against the other chases the sampling asymmetry and runs away. This set is
    not about hitting exact shares, it is about the span not moving, so the
    amplitudes are used as given and whatever shares come out are measured and
    recorded.

    Returns (Z, angle_true, W0). Raises AssertionError if the span moved.
    """
    rng = np.random.default_rng(seed)
    ones = np.ones(M_TIME) / np.sqrt(M_TIME)
    W0 = orthonormal_basis(rng, M_TIME, K, [ones])

    lam = np.array(SHARES_DEGENERATE, dtype=float) * 12.0

    X = np.empty((N_WINDOWS, M_TIME))
    angle_true = []
    worst = 0.0
    for t in range(N_WINDOWS):
        # theta ramps 0 -> ROT_THETA across [ROT_T0, ROT_T1], flat outside
        if t <= ROT_T0:
            th = 0.0
        elif t >= ROT_T1:
            th = ROT_THETA
        else:
            th = ROT_THETA * (t - ROT_T0) / (ROT_T1 - ROT_T0)

        W = W0.copy()
        W[0] = np.cos(th) * W0[0] + np.sin(th) * W0[1]
        W[1] = -np.sin(th) * W0[0] + np.cos(th) * W0[1]

        a = float(principal_angles(W0, W).max())
        worst = max(worst, a)
        angle_true.append(a)

        g = rng.standard_normal(K)
        X[t] = (g * np.sqrt(lam)) @ W + SIGMA * rng.standard_normal(M_TIME)

    # The whole point of this set. If the span moved, it is not testing what
    # we think it is testing, and silently producing it would be worse than
    # producing nothing.
    assert worst < 1e-12, (
        "degenerate_span: the subspace MOVED, worst principal angle %.3e rad. "
        "The rotation is supposed to stay inside the span." % worst
    )
    return znorm_rows(X), angle_true, W0


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

CFG = """# {name}
# generated by tools/make_datasets.py -- edit the generator, not this file
input               = {frames}
channels            = {ch}
window              = {m}
stride              = {stride}
components          = {k}
total_bits          = {tb}
lambda              = {lam}
step                = 0.24
gs_period           = 5
angle_period        = 10
theta_threshold_deg = {theta}
ring_window         = {ring}
warmup              = {warm}
out_dir             = {out_dir}
"""


def write_frames(path, Z):
    """One FRAME per line, `channels` numbers per line, space separated.

    This matches tools/convert.py. The old generator wrote Z.ravel(), one
    number per line, which happens to be identical for channels = 1 and
    silently wrong the first time anyone generates a multi-channel set.

    %.17g is the shortest decimal form that round-trips an IEEE-754 double
    exactly, so nothing is lost on the way to the C.
    """
    with open(path, "w") as f:
        for row in Z:
            for i in range(0, len(row), CHANNELS):
                f.write(" ".join("%.17g" % v for v in row[i : i + CHANNELS]))
                f.write("\n")


def make_truth(name, Z, angle_true, ring, theta_deg, seed, note):
    """Everything a comparison script needs that the C must never be told."""
    post = Z[WARMUP:]  # the windows that actually get words
    sh = batch_shares(post)
    bits, score = best_split(sh.tolist())
    return dict(
        name=name,
        seed=seed,
        note=note,
        geometry=dict(
            m_time=M_TIME,
            channels=CHANNELS,
            stride=STRIDE,
            k=K,
            total_bits=TOTAL_BITS,
            lamda=LAMBDA_REG,
            max_bits=MAX_BITS,
            warmup=WARMUP,
            ring_window=ring,
            n_windows=N_WINDOWS,
            theta_threshold_deg=theta_deg,
        ),
        noise=dict(sigma=SIGMA),
        batch=dict(
            shares=sh.tolist(),
            best_bits=bits,
            best_score=score,
            alphabet=[1 << b for b in bits],
            over_windows=[WARMUP, N_WINDOWS],
        ),
        angle_true_rad=angle_true,
        angle_true_max_deg=float(np.degrees(max(angle_true))),
    )


def write_set(
    name, Z, angle_true, theta_deg, shares_target, seed, note, frames_name=None
):
    """Write one run: frames (unless shared with another run), cfg, truth."""
    os.makedirs(os.path.join(OUT, "cfg"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "truth"), exist_ok=True)

    frames_name = frames_name or name
    frames_rel = os.path.join(OUT, "data", frames_name + ".frames")
    if not os.path.exists(frames_rel):
        write_frames(frames_rel, Z)

    ring = ring_window_for(shares_target)

    with open(os.path.join(OUT, "cfg", name + ".cfg"), "w") as f:
        f.write(
            CFG.format(
                name=name,
                frames=frames_rel,
                ch=CHANNELS,
                m=M_TIME,
                stride=STRIDE,
                k=K,
                tb=TOTAL_BITS,
                lam=LAMBDA_REG,
                theta=theta_deg,
                ring=ring,
                warm=WARMUP,
                out_dir=os.path.join(OUT, "out", name),
            )
        )

    truth = make_truth(name, Z, angle_true, ring, theta_deg, seed, note)
    with open(os.path.join(OUT, "truth", name + ".truth.json"), "w") as f:
        json.dump(truth, f, indent=1)

    b = truth["batch"]
    print(
        "%-24s theta=%2d  ring=%4d  shares %s"
        % (name, theta_deg, ring, [round(x, 4) for x in b["shares"]])
    )
    print(
        "%-24s           bits %s  alphabet %s  score %.6f"
        % ("", b["best_bits"], b["alphabet"], b["best_score"])
    )
    print("%-24s           true angle max %.3e deg" % ("", truth["angle_true_max_deg"]))
    return truth


PLAN = [
    (
        "stationary_theta89",
        89,
        "one epoch, the gate never opens. Any disagreement with spartan.py is "
        "NOT the epoch machinery.",
    ),
    (
        "stationary_theta45",
        45,
        "the same bytes as theta89. The difference is the cost of the epoch "
        "mechanism on data that does not move.",
    ),
    (
        "degenerate_span",
        45,
        "equal top shares: the directions wander, the span does not. Every "
        "epoch here is a false positive.",
    ),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--list", action="store_true", help="print the plan and write nothing"
    )
    args = ap.parse_args()

    if args.list:
        print(
            "m=%d  k=%d  total_bits=%d  stride=%d  windows=%d  sigma=%.2f"
            % (M_TIME, K, TOTAL_BITS, STRIDE, N_WINDOWS, SIGMA)
        )
        print("ring_window (stationary) = %d" % ring_window_for(SHARES_STATIONARY))
        print("ring_window (degenerate) = %d" % ring_window_for(SHARES_DEGENERATE))
        print()
        for name, theta, note in PLAN:
            print("%-24s theta=%2d  %s" % (name, theta, note))
        return 0

    Zs, ang_s, _ = build_stationary(seed=1001)
    write_set(
        "stationary_theta89",
        Zs,
        ang_s,
        89,
        SHARES_STATIONARY,
        1001,
        PLAN[0][2],
        frames_name="stationary",
    )
    write_set(
        "stationary_theta45",
        Zs,
        ang_s,
        45,
        SHARES_STATIONARY,
        1001,
        PLAN[1][2],
        frames_name="stationary",
    )

    Zd, ang_d, _ = build_degenerate(seed=1002)
    write_set("degenerate_span", Zd, ang_d, 45, SHARES_DEGENERATE, 1002, PLAN[2][2])

    print("\nwrote %s/{data,cfg,truth}" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
