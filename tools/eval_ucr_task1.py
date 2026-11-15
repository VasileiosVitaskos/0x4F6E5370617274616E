#!/usr/bin/env python3
"""Task-I (1-NN Classification) & Task-IV (Test-split TLB) on the 128 UCR Archive.

Memory-safe version: computes Euclidean and Symbolic L1 distances in row blocks
of 100 (preventing 80+ GB 3D broadcasting OOM kills on Crop / StarLightCurves /
HandOutlines), wraps each dataset in try/except, and automatically resumes
completed configurations.

Place this file in:  tools/eval_ucr_task1.py
"""

import argparse
import csv
import gc
import importlib.util
import math
import os
import subprocess
import sys
import time
import numpy as np

try:
    from scipy.stats import wilcoxon, rankdata
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
except ImportError as e:
    sys.exit(f"Missing Python package ({e}). Activate your .venv first.")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_stages import (
    load_their_distances,
    znorm_rows,
    read_our_dict,
    read_our_basis,
    read_words,
    read_epochs,
    to_mindist_breakpoints,
    online_distances,
    euclid_blocks,
    tlb_ratios,
    word_box,
    epoch_map,
    box_through,
    box_gaps,
    residual_exact,
    residual_worst,
    digitize,
)

# ==========================================================================
# CONTROL PANEL
# ==========================================================================

DEFAULT_UCR_DIR = os.path.expanduser("~/devvit/Univariate2018_ts/Univariate_ts")
DEFAULT_SPARTAN_REPO = os.path.expanduser("~/devvit/SPARTAN")
DEFAULT_BIN = "./build/spartan_toy"
DEFAULT_OUT = "out/ucr_task1"

WARMUP_CAP = 512
SEED = 0
BLOCK_ROWS = 100  # Never allocate more than 100 queries at once -> < 20 MB RAM

BASE_CFG = dict(
    k=8,
    total_bits=16,
    lam=0.5,
    theta=45.0,
    step=0.24,
    beta=0.1,
    gs_period=5,
    freeze_ahead=0.0,
    warmup_cap=512,
)
SWEEP_WARMUP = (32, 64, 128, 256, 512)
SWEEP_THETA = (20, 25, 30, 35, 40, 45, 50)
SWEEP_LAMBDA = (0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0)
SWEEP_ALPHA_BITS = (8, 16, 24, 32)
SWEEP_WORDLEN = (2, 4, 8, 12, 16)
SWEEP_STEP = (0.06, 0.12, 0.24, 0.48)
SWEEP_AHEAD = (0.0, 0.25, 0.5, 1.0)


# ==========================================================================
# 1. MEMORY-SAFE BLOCKED DISTANCE HELPERS (OOM PREVENTION)
# ==========================================================================

def euclid_cross_blocks(dist_mod, Q, C, rows=BLOCK_ROWS):
    """Compute (len(Q), len(C)) Euclidean distance matrix in small row blocks."""
    return np.vstack(
        [dist_mod.euclidean_vectorized(Q[i : i + rows], C) for i in range(0, len(Q), rows)]
    )


def symbolic_l1_blocks(W_query, W_cand, rows=BLOCK_ROWS):
    """Compute (len(W_query), len(W_cand)) Symbolic L1 matrix in small row blocks."""
    out = np.empty((len(W_query), len(W_cand)), dtype=float)
    for i in range(0, len(W_query), rows):
        q_blk = W_query[i : i + rows]
        out[i : i + rows] = np.abs(q_blk[:, None, :] - W_cand[None, :, :]).sum(axis=2)
    return out


def mindist_blocks(dist_mod, W_query, W_cand, mdb, rows=BLOCK_ROWS):
    """Compute spartan_pca_mindist in row blocks to avoid large intermediate arrays."""
    return np.vstack(
        [dist_mod.spartan_pca_mindist(W_query[i : i + rows], W_cand, mdb) for i in range(0, len(W_query), rows)]
    )


def box_gaps_blocks(q_lo, q_hi, c_lo, c_hi, rows=BLOCK_ROWS):
    """Compute box_gaps in row blocks to avoid (n_q, n_c, k) memory spikes."""
    out = np.empty((len(q_lo), len(c_lo)), dtype=float)
    for i in range(0, len(q_lo), rows):
        out[i : i + rows] = box_gaps(q_lo[i : i + rows], q_hi[i : i + rows], c_lo, c_hi)
    return out


# ==========================================================================
# 2. LOADING & PREPROCESSING UCR .ts FILES
# ==========================================================================

def load_spartan_class(repo):
    root = os.path.expanduser(repo)
    sys.path.insert(0, root)
    src = os.path.join(root, "TSB_Symbolic", "symbolic", "spartan", "spartan.py")
    if not os.path.exists(src):
        sys.exit(f"Cannot find {src}")
    spec = importlib.util.spec_from_file_location("ref_spartan_mod", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SPARTAN


_THEIR_DATASET = {}


def their_create_numpy_dataset(repo):
    """Their benchmark/util/dataset.py, loaded as a file, so every dataset is
    read, padded, resampled and gap-filled exactly as their eval scripts do.

    dataset.py imports strtobool from distutils, which Python 3.12 removed.
    It never calls it on this path; a stand-in is registered only when the
    real module is missing, and their file itself is not touched.
    """
    root = os.path.expanduser(repo)
    if root in _THEIR_DATASET:
        return _THEIR_DATASET[root]
    try:
        import distutils.util  # noqa: F401
    except ModuleNotFoundError:
        import types
        du = types.ModuleType("distutils")
        duu = types.ModuleType("distutils.util")
        duu.strtobool = lambda v: {"y": 1, "yes": 1, "t": 1, "true": 1, "on": 1, "1": 1,
                                   "n": 0, "no": 0, "f": 0, "false": 0, "off": 0, "0": 0}[str(v).lower()]
        du.util = duu
        sys.modules["distutils"] = du
        sys.modules["distutils.util"] = duu
    src = os.path.join(root, "benchmark", "util", "dataset.py")
    if not os.path.exists(src):
        sys.exit(f"Cannot find {src}")
    spec = importlib.util.spec_from_file_location("their_dataset_mod", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _THEIR_DATASET[root] = mod.create_numpy_dataset
    return mod.create_numpy_dataset


def load_ucr_dataset(ucr_dir, name, repo):
    """Train and test exactly as their eval_tlb.py:61-73 prepares them: their
    loader (equal length, gaps filled), then each series z-normed by its own
    mean and spread, a zero spread replaced by 1 on both sides. Returns
    (X_tr, y_tr, X_te, y_te) or None when the files are missing.
    """
    p_tr = os.path.join(ucr_dir, name, f"{name}_TRAIN.ts")
    p_te = os.path.join(ucr_dir, name, f"{name}_TEST.ts")
    if not (os.path.exists(p_tr) and os.path.exists(p_te)):
        return None
    create = their_create_numpy_dataset(repo)
    X_tr, y_tr, X_te, y_te = create(name=name, path=ucr_dir)
    X_tr = np.asarray(X_tr, dtype=float)[:, 0, :]
    X_te = np.asarray(X_te, dtype=float)[:, 0, :]
    if np.isnan(X_tr).any() or np.isnan(X_te).any():
        return None
    s_tr = X_tr.std(axis=1, keepdims=True)
    s_te = X_te.std(axis=1, keepdims=True)
    s_tr[s_tr == 0] = 1
    s_te[s_te == 0] = 1
    X_tr = (X_tr - X_tr.mean(axis=1, keepdims=True)) / s_tr
    X_te = (X_te - X_te.mean(axis=1, keepdims=True)) / s_te
    return X_tr, np.asarray(y_tr), X_te, np.asarray(y_te)


def build_stratified_stream(X_tr, y_tr, X_te, warmup_cap=WARMUP_CAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n_tr = len(X_tr)
    perm_tr = rng.permutation(n_tr)
    X_tr_shuf = X_tr[perm_tr]
    y_tr_shuf = y_tr[perm_tr]

    w_size = min(warmup_cap, n_tr)
    if w_size == n_tr:
        X_warm = X_tr_shuf.copy()
    else:
        classes, counts = np.unique(y_tr_shuf, return_counts=True)
        chosen = []
        for c, cnt in zip(classes, counts):
            idx_c = np.flatnonzero(y_tr_shuf == c)
            take_c = max(1, int(round(w_size * (cnt / float(n_tr)))))
            chosen.extend(idx_c[: min(take_c, len(idx_c))])
        chosen = np.array(chosen, dtype=int)
        if len(chosen) > w_size:
            chosen = chosen[:w_size]
        elif len(chosen) < w_size:
            rest = np.setdiff1d(np.arange(n_tr), chosen, assume_unique=True)
            chosen = np.concatenate([chosen, rest[: (w_size - len(chosen))]])
        rng.shuffle(chosen)
        X_warm = X_tr_shuf[chosen]

    stream = np.vstack([X_warm, X_tr_shuf, X_te])
    return stream, X_tr_shuf, y_tr_shuf, X_te, w_size


# ==========================================================================
# 3. RUNNING ONE DATASET (WITH MEMORY CLEANUP & ERROR GUARD)
# ==========================================================================

def recut_check(Y, words, bkpt, alphabet, tol=1e-9):
    """Letters the C wrote that cutting Y again here does NOT reproduce.

    A cut point is a quantile, i.e. the projection of one of the rows it was
    cut from. That row sits exactly on it, and Python's matrix product can
    land it one rounding step to the other side of the C's dot product. Such
    a letter is not counted: the projection is within tol of the cut it
    crossed. Anything else is a real disagreement and should be 0.
    """
    again = digitize(Y, bkpt, alphabet)
    bad = 0
    for j in range(Y.shape[1]):
        rows = np.flatnonzero(again[:, j] != words[:, j])
        if len(rows) == 0:
            continue
        cuts = np.asarray(bkpt[j][:alphabet[j] - 1], dtype=float)
        if len(cuts) == 0:
            bad += len(rows)
            continue
        near = np.min(np.abs(Y[rows, j][:, None] - cuts[None, :]), axis=1)
        bad += int(np.sum(near > tol * (1.0 + np.abs(Y[rows, j]))))
    return bad


def knn1_accuracy(D, y_tr, y_te):
    nn = np.argmin(D, axis=1)
    return float(np.mean(y_tr[nn] == y_te))


def write_cfg(cfg_path, setting, frames_path, m, k, warmup, ring_w, run_dir, words_path):
    with open(cfg_path, "w") as f:
        f.write(f"input               = {frames_path}\n")
        f.write("channels            = 1\n")
        f.write(f"window              = {m}\n")
        f.write(f"stride              = {m}\n")
        f.write(f"components          = {k}\n")
        f.write(f"total_bits          = {setting['total_bits']}\n")
        f.write(f"lambda              = {setting['lam']}\n")
        f.write(f"step                = {setting['step']}\n")
        f.write(f"basis_average       = {setting['beta']}\n")
        f.write(f"gs_period           = {setting['gs_period']}\n")
        f.write(f"angle_period        = {setting['gs_period']}\n")
        f.write(f"theta_threshold_deg = {setting['theta']}\n")
        f.write(f"warmup              = {warmup}\n")
        f.write(f"ring_window         = {ring_w}\n")
        f.write(f"freeze_ahead        = {setting['freeze_ahead']}\n")
        f.write(f"out_dir             = {run_dir}\n")
        f.write(f"words               = {words_path}\n")


def clean_run_dir(run_dir):
    os.makedirs(run_dir, exist_ok=True)
    for fn in os.listdir(run_dir):
        if fn.startswith(("dict_", "basis_", "epochs", "words", "settings")):
            os.remove(os.path.join(run_dir, fn))


def write_frames(path, rows):
    """Writes the frames once. Into a private .tmp first, then renamed: a
    rename is atomic, so another process (task2 shares this cache) sees either
    no file or the whole file, never a half-written one."""
    if not os.path.exists(path):
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            for row in rows:
                f.write("\n".join(f"{v:.17g}" for v in row) + "\n")
        os.replace(tmp, path)


def run_offline(bin_path, setting, X_tr, m, k, run_dir, frames_dir, name):
    """Our method OFFLINE: the C with the warm-up = the whole train, streamed
    once. Epoch 0 is then exact PCA on all of train with OUR split and OUR
    cut points, nothing streamed. Batch minus offline is the cost of the bit
    rule (theirs has at least 1 bit and at most floor(maxEV*N) per position,
    ours neither); offline minus online is the cost of streaming.
    Returns (basis (k, m), dict of bits/alpha/bkpt/mdb) or None.
    """
    n_tr = len(X_tr)
    clean_run_dir(run_dir)
    frames = os.path.join(frames_dir, f"{name}_train_only.frames")
    write_frames(frames, X_tr)
    cfg = os.path.join(run_dir, "run.cfg")
    write_cfg(cfg, setting, frames, m, k, n_tr, max(16, n_tr), run_dir,
              os.path.join(run_dir, "words.csv"))
    p = subprocess.run([bin_path, cfg], capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(os.path.join(run_dir, "dict_0.csv")):
        return None
    ev, bits, alpha, bkpt = read_our_dict(os.path.join(run_dir, "dict_0.csv"), k)
    return (read_our_basis(os.path.join(run_dir, "basis_0.csv")),
            dict(bits=bits, alpha=alpha, bkpt=bkpt, mdb=to_mindist_breakpoints(bkpt, alpha)))


def evaluate_dataset_setting(
    ucr_dir, name, setting, bin_path, spartan_cls, dist_mod, work_dir, spartan_repo
):
    loaded = load_ucr_dataset(ucr_dir, name, spartan_repo)
    if loaded is None:
        return dict(error="missing .ts files or unfilled gaps")
    X_tr_raw, y_tr_raw, X_te_raw, y_te = loaded
    m = X_tr_raw.shape[1]
    k = setting["k"]
    if k > m:
        return dict(error=f"k={k} > m={m}")
    w_cap = setting.get("warmup_cap", WARMUP_CAP)
    stream, X_tr, y_tr, X_te, warmup = build_stratified_stream(
        X_tr_raw, y_tr_raw, X_te_raw, warmup_cap=w_cap
    )
    n_tr, n_te = len(X_tr), len(X_te)
    ring_w = max(16, warmup)

    frames_dir = os.path.join(work_dir, "_frames_cache")
    os.makedirs(frames_dir, exist_ok=True)
    frames_path = os.path.join(frames_dir, f"{name}_w{warmup}.frames")
    write_frames(frames_path, stream)

    run_dir = os.path.join(work_dir, "runs", name)
    clean_run_dir(run_dir)
    cfg_path = os.path.join(run_dir, "run.cfg")
    words_path = os.path.join(run_dir, "words.csv")
    write_cfg(cfg_path, setting, frames_path, m, k, warmup, ring_w, run_dir, words_path)

    t0 = time.perf_counter()
    p = subprocess.run([bin_path, cfg_path], capture_output=True, text=True)
    t_online = time.perf_counter() - t0
    if p.returncode != 0:
        return dict(error=f"C exit {p.returncode}: {p.stderr.splitlines()[-1] if p.stderr else ''}")

    epochs = read_epochs(run_dir)
    ne = len(epochs)
    eps = []
    for e in range(ne):
        ev, bits, alpha, bkpt = read_our_dict(os.path.join(run_dir, f"dict_{e}.csv"), k)
        eps.append(
            dict(
                ev=ev,
                bits=bits,
                alpha=alpha,
                bkpt=bkpt,
                basis=read_our_basis(os.path.join(run_dir, f"basis_{e}.csv")),
                mdb=to_mindist_breakpoints(bkpt, alpha),
            )
        )
    widx_o, words_o = read_words(words_path)
    if len(words_o) != n_tr + n_te:
        return dict(error=f"expected {n_tr + n_te} words, got {len(words_o)}")

    t1_cnt = sum(1 for r in csv.DictReader(open(os.path.join(run_dir, "epochs.csv"))) if int(r["type"]) == 1)
    t2_cnt = sum(1 for r in csv.DictReader(open(os.path.join(run_dir, "epochs.csv"))) if int(r["type"]) == 2)

    Z_all = znorm_rows(stream, 1, 1e-12)
    Z_scored = Z_all[warmup:]
    Z_tr = Z_scored[:n_tr]
    Z_te = Z_scored[n_tr:]

    firsts = np.array([e["first"] for e in epochs])
    ep_scored = np.searchsorted(firsts, widx_o, side="right") - 1

    # Every word the C wrote, cut again here from its window through its own
    # epoch's basis and dictionary. Must be 0: every search distance below
    # re-cuts windows the same way, so a mismatch means they are wrong too
    recut_mismatch = 0
    for e in np.unique(ep_scored):
        ix = np.flatnonzero(ep_scored == e)
        E = eps[e]
        recut_mismatch += recut_check(Z_scored[ix] @ E["basis"].T, words_o[ix],
                                      E["bkpt"], E["alpha"])

    alpha_in = 2 ** (setting["total_bits"] // k)
    t0 = time.perf_counter()
    sp_batch = spartan_cls(
        alphabet_size=alpha_in,
        word_length=k,
        window_size=0,
        binning_method="equi-depth",
        assignment_policy="DAA",
        bit_budget=setting["total_bits"],
        lamda=setting["lam"],
        pca_solver="full",
    )
    sp_batch.fit(Z_tr)
    W_tr_batch = sp_batch.transform(Z_tr).astype(int)
    W_te_batch = sp_batch.transform(Z_te).astype(int)
    t_batch = time.perf_counter() - t0
    mdb_batch = [np.asarray(b, dtype=float) for b in sp_batch.mindist_breakpoints]

    sp_warm = spartan_cls(
        alphabet_size=alpha_in,
        word_length=k,
        window_size=0,
        binning_method="equi-depth",
        assignment_policy="DAA",
        bit_budget=setting["total_bits"],
        lamda=setting["lam"],
        pca_solver="full",
    )
    sp_warm.fit(Z_all[:warmup])
    W_tr_warm = sp_warm.transform(Z_tr).astype(int)
    W_te_warm = sp_warm.transform(Z_te).astype(int)
    mdb_warm = [np.asarray(b, dtype=float) for b in sp_warm.mindist_breakpoints]

    # Our method offline: same C, warm-up = all of train, nothing streamed
    off = run_offline(bin_path, setting, X_tr, m, k,
                      os.path.join(work_dir, "runs_offline", name), frames_dir, name)
    if off is not None:
        B_off, E_off = off
        W_tr_off = digitize(Z_tr @ B_off.T, E_off["bkpt"], E_off["alpha"])
        W_te_off = digitize(Z_te @ B_off.T, E_off["bkpt"], E_off["alpha"])
    bits_batch = " ".join(str(int(round(math.log2(a)))) for a in sp_batch.alphabet_size)
    bits_online0 = " ".join(str(b) for b in eps[0]["bits"])
    bits_offline = " ".join(str(b) for b in E_off["bits"]) if off is not None else ""

    x_max = math.sqrt(m)
    ep_tr = ep_scored[:n_tr]
    ep_te = ep_scored[n_tr:]
    wo_tr = words_o[:n_tr]
    wo_te = words_o[n_tr:]

    D_search = np.zeros((n_te, n_tr))
    D_box = np.zeros((n_te, n_tr))
    D_join = np.zeros((n_te, n_tr))
    D_l1_search = np.zeros((n_te, n_tr))

    rows_tr = {e: np.flatnonzero(ep_tr == e) for e in np.unique(ep_tr)}
    rows_te = {e: np.flatnonzero(ep_te == e) for e in np.unique(ep_te)}

    box_tr = {e: word_box(wo_tr[ix], eps[e]["bkpt"], eps[e]["alpha"], x_max) for e, ix in rows_tr.items()}
    box_te = {}
    r_live_te = {}
    r_worst_te = {}
    for e, ix in rows_te.items():
        lo, hi = word_box(wo_te[ix], eps[e]["bkpt"], eps[e]["alpha"], x_max)
        box_te[e] = (lo, hi)
        r_live_te[e] = residual_exact(Z_te[ix], Z_te[ix] @ eps[e]["basis"].T)
        r_worst_te[e] = residual_worst(lo, hi, x_max)

    with np.errstate(over="ignore", invalid="ignore"):
        for eq, iq in rows_te.items():
            for ec, ic in rows_tr.items():
                if eq == ec:
                    same_d = mindist_blocks(dist_mod, wo_te[iq], wo_tr[ic], eps[eq]["mdb"])
                    D_search[np.ix_(iq, ic)] = same_d
                    D_box[np.ix_(iq, ic)] = same_d
                    D_join[np.ix_(iq, ic)] = same_d
                    D_l1_search[np.ix_(iq, ic)] = symbolic_l1_blocks(wo_te[iq], wo_tr[ic])
                else:
                    M, sin_t = epoch_map(eps[ec]["basis"], eps[eq]["basis"], 1e-12)
                    if M is not None:
                        q_lo, q_hi = box_through(M, *box_te[eq])
                        gaps = box_gaps_blocks(q_lo, q_hi, *box_tr[ec])
                        D_box[np.ix_(iq, ic)] = np.maximum(0.0, gaps - sin_t * r_live_te[eq][:, None])
                        D_join[np.ix_(iq, ic)] = np.maximum(0.0, gaps - sin_t * r_worst_te[eq][:, None])
                    Ec = eps[ec]
                    wq = digitize(Z_te[iq] @ Ec["basis"].T, Ec["bkpt"], Ec["alpha"])
                    D_search[np.ix_(iq, ic)] = mindist_blocks(dist_mod, wq, wo_tr[ic], Ec["mdb"])
                    D_l1_search[np.ix_(iq, ic)] = symbolic_l1_blocks(wq, wo_tr[ic])

        D_batch = mindist_blocks(dist_mod, W_te_batch, W_tr_batch, mdb_batch)
        D_warm = mindist_blocks(dist_mod, W_te_warm, W_tr_warm, mdb_warm)
        if off is not None:
            D_off = mindist_blocks(dist_mod, W_te_off, W_tr_off, E_off["mdb"])
            D_l1_off = symbolic_l1_blocks(W_te_off, W_tr_off)

    D_l1_batch = symbolic_l1_blocks(W_te_batch, W_tr_batch)
    D_l1_warm = symbolic_l1_blocks(W_te_warm, W_tr_warm)
    D_ed = euclid_cross_blocks(dist_mod, Z_te, Z_tr)

    acc_md_q = knn1_accuracy(D_search, y_tr, y_te)
    acc_md_b = knn1_accuracy(D_box, y_tr, y_te)
    acc_md_j = knn1_accuracy(D_join, y_tr, y_te)
    acc_md_ba = knn1_accuracy(D_batch, y_tr, y_te)
    acc_md_wa = knn1_accuracy(D_warm, y_tr, y_te)
    acc_l1_on = knn1_accuracy(D_l1_search, y_tr, y_te)
    acc_l1_ba = knn1_accuracy(D_l1_batch, y_tr, y_te)
    acc_l1_wa = knn1_accuracy(D_l1_warm, y_tr, y_te)
    acc_ed_val = knn1_accuracy(D_ed, y_tr, y_te)
    acc_md_off = knn1_accuracy(D_off, y_tr, y_te) if off is not None else float("nan")
    acc_l1_off = knn1_accuracy(D_l1_off, y_tr, y_te) if off is not None else float("nan")
    if off is not None:
        del D_off, D_l1_off

    # Free large (n_te, n_tr) matrices immediately before TLB step
    del D_search, D_box, D_join, D_batch, D_warm, D_l1_search, D_l1_batch, D_l1_warm, D_ed

    rng = np.random.default_rng(SEED)
    take_te = rng.choice(n_te, size=min(500, n_te), replace=False) if n_te > 500 else np.arange(n_te)
    take_te.sort()
    with np.errstate(over="ignore", invalid="ignore"):
        eu_te = euclid_blocks(dist_mod, Z_te[take_te])
        md_q_te, md_o_te, md_s_te, _, _ = online_distances(
            dist_mod, widx_o[n_tr + take_te], wo_te[take_te], ep_te[take_te], eps, Z_all, Z_te[take_te], m, 1e-12
        )
        md_b_te = mindist_blocks(dist_mod, W_te_batch[take_te], W_te_batch[take_te], mdb_batch)
        md_w_te = mindist_blocks(dist_mod, W_te_warm[take_te], W_te_warm[take_te], mdb_warm)
        if off is not None:
            md_off_te = mindist_blocks(dist_mod, W_te_off[take_te], W_te_off[take_te], E_off["mdb"])

    viol_search = int(np.sum(md_q_te > eu_te + 1e-9))
    viol_box = int(np.sum(md_o_te > eu_te + 1e-9))

    return dict(
        dataset=name,
        n_train=n_tr,
        n_test=n_te,
        m=m,
        warmup=warmup,
        t1=t1_cnt,
        t2=t2_cnt,
        acc_mindist_search=acc_md_q,
        acc_mindist_box=acc_md_b,
        acc_mindist_join=acc_md_j,
        acc_mindist_batch=acc_md_ba,
        acc_mindist_warm=acc_md_wa,
        acc_l1_online=acc_l1_on,
        acc_l1_batch=acc_l1_ba,
        acc_l1_warm=acc_l1_wa,
        acc_ed=acc_ed_val,
        acc_mindist_offline=acc_md_off,
        acc_l1_offline=acc_l1_off,
        tlb_offline=float(tlb_ratios(md_off_te, eu_te).mean()) if off is not None else float("nan"),
        bits_batch=bits_batch,
        bits_online0=bits_online0,
        bits_offline=bits_offline,
        recut_mismatch=recut_mismatch,
        tlb_search=float(tlb_ratios(md_q_te, eu_te).mean()),
        tlb_box=float(tlb_ratios(md_o_te, eu_te).mean()),
        tlb_join=float(tlb_ratios(md_s_te, eu_te).mean()),
        tlb_batch=float(tlb_ratios(md_b_te, eu_te).mean()),
        tlb_warm=float(tlb_ratios(md_w_te, eu_te).mean()),
        viol=viol_search + viol_box,
        sec_online=t_online,
        sec_batch=t_batch,
        error="",
    )


# ==========================================================================
# 4. PLOTTING IN EXACT SIGMOD'25 PAPER STYLE
# ==========================================================================

def plot_pairwise_scatter(x, y, xlabel, ylabel, title, out_path):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    wins = int(np.sum(y > x + 1e-12))
    ties = int(np.sum(np.abs(y - x) <= 1e-12))
    losses = int(np.sum(y < x - 1e-12))

    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=300)
    tri = Polygon([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], closed=True, facecolor="#c8e6c9", edgecolor="none", zorder=1)
    ax.add_patch(tri)
    ax.plot([0, 1], [0, 1], color="#e57373", linewidth=1.3, zorder=2)
    ax.scatter(x, y, color="#0d1b6e", s=14, alpha=0.85, zorder=3)

    table_data = [["Win", f"{wins}"], ["Tie", f"{ties}"], ["Loss", f"{losses}"]]
    tbl = ax.table(
        cellText=table_data,
        colLabels=[f"{ylabel} vs {xlabel}", ""],
        loc="lower right",
        cellLoc="center",
        bbox=[0.52, 0.05, 0.44, 0.26],
        zorder=4,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for _, cell in tbl.get_celld().items():
        cell.set_edgecolor("#888888")
        cell.set_linewidth(0.7)
        cell.set_facecolor("white")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png")
    plt.close(fig)


def plot_critical_diagram(scores_dict, title, out_path):
    names = list(scores_dict.keys())
    k = len(names)
    M = np.column_stack([scores_dict[nm] for nm in names])
    n = len(M)
    if n < 2 or k < 2:
        return

    ranks = np.empty_like(M, dtype=float)
    for i in range(n):
        ranks[i] = rankdata(-M[i], method="average")
    avg_ranks = ranks.mean(axis=0)

    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q_alpha = q_table.get(k, 2.728)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))

    order = np.argsort(avg_ranks)
    sorted_names = [names[i] for i in order]
    sorted_ranks = [avg_ranks[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.2, 2.3), dpi=300)
    ax.set_xlim(k + 0.5, 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    y_axis = 0.72
    ax.plot([1, k], [y_axis, y_axis], color="black", lw=1.2)
    for tick in range(1, k + 1):
        ax.plot([tick, tick], [y_axis, y_axis + 0.04], color="black", lw=1.0)
        ax.text(tick, y_axis + 0.07, str(tick), ha="center", va="bottom", fontsize=10)

    half = (k + 1) // 2
    for idx, (nm, rnk) in enumerate(zip(sorted_names, sorted_ranks)):
        if idx < half:
            y_lab = 0.48 - idx * (0.38 / max(1, half))
            ax.plot([rnk, rnk], [y_axis, y_lab], color="black", lw=1.2)
            ax.plot([rnk, 0.75], [y_lab, y_lab], color="black", lw=1.2)
            ax.text(0.70, y_lab, f"{nm} ({rnk:.2f})", ha="left", va="center", fontsize=9.5, fontweight="bold")
        else:
            idx_l = idx - half
            y_lab = 0.48 - idx_l * (0.38 / max(1, k - half))
            ax.plot([rnk, rnk], [y_axis, y_lab], color="black", lw=1.2)
            ax.plot([rnk, k + 0.25], [y_lab, y_lab], color="black", lw=1.2)
            ax.text(k + 0.30, y_lab, f"{nm} ({rnk:.2f})", ha="right", va="center", fontsize=9.5, fontweight="bold")

    cliques = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(sorted_ranks[j] - sorted_ranks[i]) <= cd:
                if not any(c_i <= i and j <= c_j for c_i, c_j in cliques):
                    cliques = [(c_i, c_j) for c_i, c_j in cliques if not (i <= c_i and c_j <= j)]
                    cliques.append((i, j))
    for c_idx, (i, j) in enumerate(cliques):
        y_bar = y_axis - 0.07 - c_idx * 0.05
        ax.plot([sorted_ranks[i], sorted_ranks[j]], [y_bar, y_bar], color="black", lw=3.2)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png")
    plt.close(fig)


def plot_sensitivity_curve(x_vals, series_dict, xlabel, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(5.0, 3.3), dpi=300)
    styles = [
        ("#1f77b4", "-o", 2.2),
        ("#2ca02c", "--s", 1.8),
        ("#ff7f0e", "-.^", 1.8),
        ("#8c564b", "-.", 1.6),
        ("#d62728", ":", 1.6),
    ]
    for (label, y_vals), (col, fmt, lw) in zip(series_dict.items(), styles):
        ax.plot(x_vals, y_vals, fmt, color=col, lw=lw, markersize=5, label=label)
    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png")
    plt.close(fig)


def plot_grouped_bars(x_labels, bar_dict, ed_val, xlabel, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=300)
    x = np.arange(len(x_labels))
    m_names = list(bar_dict.keys())
    width = 0.78 / len(m_names)
    colors = ["#2b4c7e", "#568ea6", "#f1c40f", "#e67e22"]
    hatches = ["||", "--", "//", ".."]

    for i, nm in enumerate(m_names):
        offset = (i - (len(m_names) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            bar_dict[nm],
            width=width * 0.92,
            label=nm,
            color=colors[i % len(colors)],
            hatch=hatches[i % len(hatches)],
            edgecolor="black",
            linewidth=0.6,
        )
    if ed_val is not None:
        ax.axhline(ed_val, color="#d62728", linestyle="--", linewidth=1.6, label="ED")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=8.5, ncol=min(5, len(m_names) + 1), loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png")
    plt.close(fig)


# ==========================================================================
# 5. SWEEP PLAN & SUMMARY AGGREGATION
# ==========================================================================

def build_sweep_configs(blocks):
    plan = []
    if "base" in blocks or "all" in blocks:
        plan.append(("base", "base_a4_w8_lam0.5_th45", dict(BASE_CFG)))
    if "theta" in blocks or "all" in blocks:
        for th in SWEEP_THETA:
            if th != BASE_CFG["theta"]:
                plan.append(("theta", f"theta_{th}", dict(BASE_CFG, theta=float(th))))
    if "lambda" in blocks or "all" in blocks:
        for lam in SWEEP_LAMBDA:
            if lam != BASE_CFG["lam"]:
                plan.append(("lambda", f"lambda_{lam:g}", dict(BASE_CFG, lam=float(lam))))
    if "alpha" in blocks or "all" in blocks:
        for tb in SWEEP_ALPHA_BITS:
            if tb != BASE_CFG["total_bits"]:
                plan.append(("alpha", f"alpha_{1 << (tb // BASE_CFG['k'])}_bits_{tb}", dict(BASE_CFG, total_bits=int(tb))))
    if "wordlen" in blocks or "all" in blocks:
        for w in SWEEP_WORDLEN:
            if w != BASE_CFG["k"]:
                plan.append(("wordlen", f"wordlen_{w}_bits_{2*w}", dict(BASE_CFG, k=int(w), total_bits=2 * int(w))))
    if "step" in blocks or "all" in blocks:
        for st in SWEEP_STEP:
            if st != BASE_CFG["step"]:
                plan.append(("step", f"step_{st:g}", dict(BASE_CFG, step=float(st))))
    if "ahead" in blocks:
        for fa in SWEEP_AHEAD:
            if fa != BASE_CFG["freeze_ahead"]:
                plan.append(("ahead", f"ahead_{fa:g}", dict(BASE_CFG, freeze_ahead=float(fa))))
    if "warmup" in blocks or "all" in blocks:
        for wc in SWEEP_WARMUP:
            if wc != BASE_CFG["warmup_cap"]:
                plan.append(("warmup", f"warmup_{wc}", dict(BASE_CFG, warmup_cap=int(wc))))
    return plan


def _wtl_p(a, b):
    w = int(np.sum(a > b + 1e-12))
    t = int(np.sum(np.abs(a - b) <= 1e-12))
    l = int(np.sum(a < b - 1e-12))
    if len(a) < 2 or np.all(np.abs(a - b) <= 1e-12):
        return w, t, l, 1.0
    try:
        _, p = wilcoxon(a, b, zero_method="zsplit")
    except ValueError:
        p = 1.0
    return w, t, l, float(p)


def _stats(rows, sfx):
    """Means, and online vs batch / offline, over the given rows. sfx "" is
    every dataset; "_stream" only those whose warm-up is SHORTER than their
    train, the ones where the online method actually streams."""
    out = {f"n{sfx}": len(rows)}
    if not rows:
        return out

    def arr(k):
        return np.array([float(r[k]) if r.get(k, "") != "" else np.nan for r in rows])

    for k in ("acc_l1_online", "acc_l1_batch", "acc_l1_warm", "acc_l1_offline",
              "acc_mindist_search", "acc_mindist_box", "acc_mindist_join",
              "acc_mindist_batch", "acc_mindist_warm", "acc_mindist_offline", "acc_ed",
              "tlb_search", "tlb_box", "tlb_join", "tlb_batch", "tlb_warm", "tlb_offline"):
        out[k + sfx] = float(np.nanmean(arr(k)))
    for metric, on in (("l1", "acc_l1_online"), ("mindist", "acc_mindist_search")):
        for ref in ("batch", "offline", "warm"):
            a, b = arr(on), arr(f"acc_{metric}_{ref}")
            keep = ~(np.isnan(a) | np.isnan(b))
            w, t, l, p = _wtl_p(a[keep], b[keep])
            out[f"wtl_{metric}_vs_{ref}{sfx}"] = f"{w}/{t}/{l}"
            out[f"p_{metric}_vs_{ref}{sfx}"] = p
    return out


def _extra_stats(ok):
    """The new columns, appended after the old ones: the offline column, the
    recut check, and every number again over only the datasets whose warm-up
    is shorter than their train (suffix _stream)."""
    stream = [r for r in ok if int(float(r["warmup"])) < int(float(r["n_train"]))]
    out = dict(total_recut_mismatch=int(sum(int(float(r.get("recut_mismatch") or 0)) for r in ok)))
    for k in ("acc_l1_offline", "acc_mindist_offline", "tlb_offline"):
        v = [float(r[k]) for r in ok if r.get(k, "") not in ("", "nan")]
        out[k] = float(np.mean(v)) if v else float("nan")
    out.update(_stats(stream, "_stream"))
    return out


def summarize_tag(block, tag, s, rows):
    ok = [r for r in rows if not r.get("error") and r.get("acc_l1_online") != ""]
    if not ok:
        return dict(block=block, tag=tag, n_datasets=0)

    def arr(k):
        return np.array([float(r[k]) for r in ok], dtype=float)

    def wtl_p(a, b):
        w = int(np.sum(a > b + 1e-12))
        t = int(np.sum(np.abs(a - b) <= 1e-12))
        l = int(np.sum(a < b - 1e-12))
        if np.all(np.abs(a - b) <= 1e-12):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(a, b, zero_method="zsplit")
            except Exception:
                p = 1.0
        return w, t, l, float(p)

    q_md, b_md, j_md, ba_md, wa_md = (
        arr("acc_mindist_search"),
        arr("acc_mindist_box"),
        arr("acc_mindist_join"),
        arr("acc_mindist_batch"),
        arr("acc_mindist_warm"),
    )
    q_l1, ba_l1, wa_l1 = arr("acc_l1_online"), arr("acc_l1_batch"), arr("acc_l1_warm")
    tlb_q, tlb_b, tlb_j, tlb_ba, tlb_wa = (
        arr("tlb_search"),
        arr("tlb_box"),
        arr("tlb_join"),
        arr("tlb_batch"),
        arr("tlb_warm"),
    )

    w_md, t_md, l_md, p_md = wtl_p(q_md, ba_md)
    w_l1, t_l1, l_l1, p_l1 = wtl_p(q_l1, ba_l1)

    return dict(
        block=block,
        tag=tag,
        n_datasets=len(ok),
        k=s["k"],
        total_bits=s["total_bits"],
        lam=s["lam"],
        theta=s["theta"],
        step=s["step"],
        freeze_ahead=s["freeze_ahead"],
        mean_t1=float(arr("t1").mean()),
        mean_t2=float(arr("t2").mean()),
        acc_l1_online=float(q_l1.mean()),
        acc_l1_batch=float(ba_l1.mean()),
        acc_l1_warm=float(wa_l1.mean()),
        wtl_l1_vs_batch=f"{w_l1}/{t_l1}/{l_l1}",
        p_l1_vs_batch=p_l1,
        acc_mindist_search=float(q_md.mean()),
        acc_mindist_box=float(b_md.mean()),
        acc_mindist_join=float(j_md.mean()),
        acc_mindist_batch=float(ba_md.mean()),
        acc_mindist_warm=float(wa_md.mean()),
        wtl_mindist_vs_batch=f"{w_md}/{t_md}/{l_md}",
        p_mindist_vs_batch=p_md,
        acc_ed=float(arr("acc_ed").mean()),
        tlb_search=float(tlb_q.mean()),
        tlb_box=float(tlb_b.mean()),
        tlb_join=float(tlb_j.mean()),
        tlb_batch=float(tlb_ba.mean()),
        tlb_warm=float(tlb_wa.mean()),
        total_viol=int(arr("viol").sum()),
        **_extra_stats(ok),
    )


def generate_all_figures(out_dir, summary_rows):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    base_csv = os.path.join(out_dir, "per_dataset", "base_a4_w8_lam0.5_th45.csv")
    if os.path.exists(base_csv):
        rows = [r for r in csv.DictReader(open(base_csv)) if not r.get("error") and r.get("acc_l1_online") != ""]
        if rows:
            f = lambda k: np.array([float(r[k]) for r in rows])
            plot_pairwise_scatter(
                f("acc_l1_batch"), f("acc_l1_online"),
                "Batch SPARTAN", "Online SPARTAN",
                "1-NN Accuracy (Symbolic L1, a=4, w=8)",
                os.path.join(fig_dir, "fig5_pairwise_acc_l1_vs_batch"),
            )
            plot_pairwise_scatter(
                f("acc_mindist_batch"), f("acc_mindist_search"),
                "Batch SPARTAN", "Online (Search)",
                "1-NN Accuracy (PCA-MINDIST, a=4, w=8)",
                os.path.join(fig_dir, "fig11_pairwise_acc_mindist_vs_batch"),
            )
            plot_pairwise_scatter(
                f("acc_mindist_warm"), f("acc_mindist_search"),
                "Batch (Warm-up)", "Online (Search)",
                "1-NN Accuracy vs. Warm-up Fit",
                os.path.join(fig_dir, "fig5_pairwise_acc_vs_warmup"),
            )
            plot_pairwise_scatter(
                f("tlb_batch"), f("tlb_search"),
                "Batch SPARTAN", "Online (Search)",
                "Test-Split TLB (a=4, w=8)",
                os.path.join(fig_dir, "fig7_pairwise_tlb_search_vs_batch"),
            )
            plot_pairwise_scatter(
                f("tlb_batch"), f("tlb_box"),
                "Batch SPARTAN", "Online (Box)",
                "Test-Split TLB: Box vs. Batch",
                os.path.join(fig_dir, "fig7_pairwise_tlb_box_vs_batch"),
            )
            plot_critical_diagram(
                {
                    "Online (Search)": f("acc_mindist_search"),
                    "Online (Box)": f("acc_mindist_box"),
                    "Batch SPARTAN": f("acc_mindist_batch"),
                    "Batch (Warm-up)": f("acc_mindist_warm"),
                },
                "Critical Diagram: 1-NN Accuracy (PCA-MINDIST)",
                os.path.join(fig_dir, "fig5_cd_accuracy_mindist"),
            )
            plot_critical_diagram(
                {
                    "Online (Search)": f("tlb_search"),
                    "Online (Box)": f("tlb_box"),
                    "Online (Join)": f("tlb_join"),
                    "Batch SPARTAN": f("tlb_batch"),
                },
                "Critical Diagram: Tightness of Lower Bound (TLB)",
                os.path.join(fig_dir, "fig7_cd_tlb"),
            )

    if not summary_rows:
        return
    base_r = next((r for r in summary_rows if r["block"] == "base" and int(r.get("n_datasets", 0)) > 0), None)

    lam_rows = [r for r in summary_rows if r["block"] == "lambda" and int(r.get("n_datasets", 0)) > 0]
    if base_r and lam_rows:
        pts = sorted(lam_rows + [base_r], key=lambda r: float(r["lam"]))
        xs = [float(r["lam"]) for r in pts]
        plot_sensitivity_curve(
            xs,
            {
                "Online SPARTAN (L1)": [float(r["acc_l1_online"]) for r in pts],
                "Batch SPARTAN (L1)": [float(r["acc_l1_batch"]) for r in pts],
                "Online Search (MINDIST)": [float(r["acc_mindist_search"]) for r in pts],
                "Batch (MINDIST)": [float(r["acc_mindist_batch"]) for r in pts],
            },
            "Regularization parameter $\\lambda$",
            "Average 1-NN Accuracy",
            "Accuracy vs. Regularization Parameter $\\lambda$",
            os.path.join(fig_dir, "fig6_sensitivity_lambda_acc"),
        )
        plot_sensitivity_curve(
            xs,
            {
                "Online (Search)": [float(r["tlb_search"]) for r in pts],
                "Online (Box)": [float(r["tlb_box"]) for r in pts],
                "Online (Join)": [float(r["tlb_join"]) for r in pts],
                "Batch SPARTAN": [float(r["tlb_batch"]) for r in pts],
            },
            "Regularization parameter $\\lambda$",
            "Mean TLB",
            "TLB vs. Regularization Parameter $\\lambda$",
            os.path.join(fig_dir, "fig6_sensitivity_lambda_tlb"),
        )

    th_rows = [r for r in summary_rows if r["block"] == "theta" and int(r.get("n_datasets", 0)) > 0]
    if base_r and th_rows:
        pts = sorted(th_rows + [base_r], key=lambda r: float(r["theta"]))
        xs = [float(r["theta"]) for r in pts]
        plot_sensitivity_curve(
            xs,
            {
                "Online (Search)": [float(r["tlb_search"]) for r in pts],
                "Online (Box)": [float(r["tlb_box"]) for r in pts],
                "Online (Join)": [float(r["tlb_join"]) for r in pts],
                "Batch SPARTAN": [float(r["tlb_batch"]) for r in pts],
            },
            "Angle threshold $\\theta$ (degrees)",
            "Mean TLB",
            "TLB vs. Drift Angle Threshold $\\theta$",
            os.path.join(fig_dir, "fig_sensitivity_theta_tlb"),
        )
        plot_sensitivity_curve(
            xs,
            {
                "Online (Search)": [float(r["acc_mindist_search"]) for r in pts],
                "Online (Box)": [float(r["acc_mindist_box"]) for r in pts],
                "Batch SPARTAN": [float(r["acc_mindist_batch"]) for r in pts],
            },
            "Angle threshold $\\theta$ (degrees)",
            "Average 1-NN Accuracy",
            "1-NN Accuracy vs. Drift Angle Threshold $\\theta$",
            os.path.join(fig_dir, "fig_sensitivity_theta_acc"),
        )

    al_rows = [r for r in summary_rows if r["block"] == "alpha" and int(r.get("n_datasets", 0)) > 0]
    if base_r and al_rows:
        pts = sorted(al_rows + [base_r], key=lambda r: int(r["total_bits"]))
        labels = [f"$\\alpha={1 << (int(r['total_bits']) // int(r['k']))}$" for r in pts]
        ed_v = float(pts[0]["acc_ed"])
        plot_grouped_bars(
            labels,
            {
                "Online (L1)": [float(r["acc_l1_online"]) for r in pts],
                "Batch (L1)": [float(r["acc_l1_batch"]) for r in pts],
                "Online (MINDIST)": [float(r["acc_mindist_search"]) for r in pts],
                "Batch (MINDIST)": [float(r["acc_mindist_batch"]) for r in pts],
            },
            ed_v,
            "Alphabet Size $\\alpha$ ($\\omega=8$)",
            "Accuracy",
            "(a.1) Avg. accuracy on varying alphabet sizes",
            os.path.join(fig_dir, "fig12_bars_alphabet_size"),
        )

    wl_rows = [r for r in summary_rows if r["block"] == "wordlen" and int(r.get("n_datasets", 0)) > 0]
    if base_r and wl_rows:
        pts = sorted(wl_rows + [base_r], key=lambda r: int(r["k"]))
        labels = [f"$\\omega={int(r['k'])}$" for r in pts]
        ed_v = float(pts[0]["acc_ed"])
        plot_grouped_bars(
            labels,
            {
                "Online (L1)": [float(r["acc_l1_online"]) for r in pts],
                "Batch (L1)": [float(r["acc_l1_batch"]) for r in pts],
                "Online (MINDIST)": [float(r["acc_mindist_search"]) for r in pts],
                "Batch (MINDIST)": [float(r["acc_mindist_batch"]) for r in pts],
            },
            ed_v,
            "Word Length $\\omega$ ($\\alpha=4$)",
            "Accuracy",
            "(a.2) Avg. accuracy on varying word lengths",
            os.path.join(fig_dir, "fig12_bars_word_length"),
        )


# ==========================================================================
# MAIN CLI
# ==========================================================================

CSV_COLS = [
    "dataset", "n_train", "n_test", "m", "warmup", "t1", "t2",
    "acc_mindist_search", "acc_mindist_box", "acc_mindist_join", "acc_mindist_batch", "acc_mindist_warm",
    "acc_l1_online", "acc_l1_batch", "acc_l1_warm", "acc_ed",
    "tlb_search", "tlb_box", "tlb_join", "tlb_batch", "tlb_warm",
    "viol", "sec_online", "sec_batch",
    "acc_mindist_offline", "acc_l1_offline", "tlb_offline",
    "bits_batch", "bits_online0", "bits_offline", "recut_mismatch", "error",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ucr-dir", default=DEFAULT_UCR_DIR)
    ap.add_argument("--spartan-repo", default=DEFAULT_SPARTAN_REPO)
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--blocks", default="base")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--no-resume", action="store_true", help="re-run even if per_dataset/<tag>.csv exists")
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "per_dataset"), exist_ok=True)
    sum_path = os.path.join(args.out, "ucr_summary.csv")

    if args.plots_only:
        rows = list(csv.DictReader(open(sum_path))) if os.path.exists(sum_path) else []
        generate_all_figures(args.out, rows)
        print(f"Rebuilt figures in {os.path.join(args.out, 'figures')}")
        return 0

    if not os.path.exists(args.bin):
        sys.exit(f"Cannot find compiled C binary at '{args.bin}'. Run 'make' first.")

    spartan_cls = load_spartan_class(args.spartan_repo)
    dist_mod = load_their_distances(args.spartan_repo)

    if args.datasets:
        datasets = sorted(args.datasets)
    else:
        datasets = sorted(
            d for d in os.listdir(args.ucr_dir)
            if os.path.isdir(os.path.join(args.ucr_dir, d))
        )

    blocks = {b.strip().lower() for b in args.blocks.split(",")}
    plan = build_sweep_configs(blocks)
    print(f"UCR Archive : {args.ucr_dir} ({len(datasets)} datasets)")
    print(f"Settings    : {len(plan)} configurations")
    print(f"Output Dir  : {args.out}\n")

    summary_rows = []
    for idx_cfg, (blk, tag, setting) in enumerate(plan, 1):
        t0 = time.perf_counter()
        ds_csv = os.path.join(args.out, "per_dataset", f"{tag}.csv")

        # Auto-resume if this configuration already finished all requested datasets
        if not args.no_resume and os.path.exists(ds_csv):
            existing = list(csv.DictReader(open(ds_csv)))
            if len(existing) == len(datasets):
                s_row = summarize_tag(blk, tag, setting, existing)
                s_row["elapsed_s"] = 0.0
                summary_rows.append(s_row)
                print(
                    f"  [{idx_cfg}/{len(plan)}] {tag:<26} | CACHED ({s_row.get('n_datasets', 0)} ds) | "
                    f"Acc(L1) On/Ba: {s_row.get('acc_l1_online', 0):.4f}/{s_row.get('acc_l1_batch', 0):.4f} "
                    f"({s_row.get('wtl_l1_vs_batch', '-')}, p={s_row.get('p_l1_vs_batch', 1):.3f}) | "
                    f"TLB(Q/B/Ba): {s_row.get('tlb_search', 0):.3f}/{s_row.get('tlb_box', 0):.3f}/{s_row.get('tlb_batch', 0):.3f}"
                )
                continue

        per_ds_rows = []
        for idx_d, dname in enumerate(datasets, 1):
            sys.stdout.write(
                f"\r  [{idx_cfg}/{len(plan)}] {tag:<26} dataset {idx_d:3d}/{len(datasets)} ({dname:<28})"
            )
            sys.stdout.flush()
            try:
                res = evaluate_dataset_setting(
                    args.ucr_dir, dname, setting, args.bin, spartan_cls, dist_mod, args.out,
                    args.spartan_repo,
                )
            except Exception as exc:
                res = dict(dataset=dname, error=f"{type(exc).__name__}: {exc}")
            res["dataset"] = dname
            per_ds_rows.append(res)
            gc.collect()

        with open(ds_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS, restval="")
            w.writeheader()
            w.writerows(per_ds_rows)

        s_row = summarize_tag(blk, tag, setting, per_ds_rows)
        s_row["elapsed_s"] = round(time.perf_counter() - t0, 1)
        summary_rows.append(s_row)

        with open(sum_path, "w", newline="") as f:
            keys = list(dict.fromkeys(k for r in summary_rows for k in r))
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(summary_rows)

        print(
            f"\r  [{idx_cfg}/{len(plan)}] {tag:<26} | "
            f"Acc(L1) On/Ba: {s_row.get('acc_l1_online', 0):.4f}/{s_row.get('acc_l1_batch', 0):.4f} "
            f"({s_row.get('wtl_l1_vs_batch', '-')}, p={s_row.get('p_l1_vs_batch', 1):.3f}) | "
            f"TLB(Q/B/Ba): {s_row.get('tlb_search', 0):.3f}/{s_row.get('tlb_box', 0):.3f}/{s_row.get('tlb_batch', 0):.3f} | "
            f"viol: {s_row.get('total_viol', 0)} | {s_row['elapsed_s']}s"
        )

    generate_all_figures(args.out, summary_rows)
    print(f"\nDone! Summary table: {sum_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
