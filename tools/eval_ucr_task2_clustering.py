#!/usr/bin/env python3
"""Task-II: Time-Series Clustering on the 128 UCR Archive Datasets.

Replicates Section 5.4 (Figure 8) of Yang & Paparrizos (SIGMOD 2025):
  - Evaluates k-Medoids (PAM on precomputed distance matrices) and
    Agglomerative Clustering using Adjusted Rand Index (ARI) and Normalized
    Mutual Information (NMI).
  - Compares Online SPARTAN (Search, Box, Join, and Symbolic L1) against
    Batch SPARTAN (Full fit and Warm-up fit) and Raw Euclidean Distance (ED).
  - Generates SIGMOD'25-style Pairwise Scatter Plots, Critical Difference (CD)
    Diagrams, and Parameter Sensitivity Curves in out/ucr_task2/figures/.

Place this file in:  tools/eval_ucr_task2_clustering.py
"""

import argparse
import csv
import gc
import math
import os
import subprocess
import sys
import time
import numpy as np

try:
    from scipy.stats import wilcoxon
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.cluster import AgglomerativeClustering, KMeans
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
)
from compare_stages import digitize
from eval_ucr_task1 import (
    recut_check,
    run_offline,
    write_cfg,
    write_frames,
    clean_run_dir,
    _wtl_p,
    DEFAULT_UCR_DIR,
    DEFAULT_SPARTAN_REPO,
    DEFAULT_BIN,
    WARMUP_CAP,
    SEED,
    BASE_CFG,
    load_spartan_class,
    load_ucr_dataset,
    build_stratified_stream,
    mindist_blocks,
    symbolic_l1_blocks,
    plot_pairwise_scatter,
    plot_critical_diagram,
    plot_sensitivity_curve,
    build_sweep_configs,
)

DEFAULT_OUT_TASK2 = "out/ucr_task2"
# Their eval_clustering.py clusters the WHOLE test set. Above this many series
# the n x n matrices (several at once) do not fit in memory; such a test set is
# subsampled, stratified by class, and the row says so (subsampled = 1)
MAX_CLUSTER_N = 4000
REPEATS = 10  # their -e/--repeat_num default: k-medoids and k-means seeds 0..9
LINKAGE = "complete"  # their main.py --linkage default


# ==========================================================================
# 1. THEIR CLUSTERING PROTOCOL (benchmark/eval_clustering.py)
# ==========================================================================


def cluster_metrics(D, y_true, n_clusters):
    """Their protocol on a precomputed distance matrix D (n, n).

    k-medoids: kmedoids.pam(diss=D, medoids=n_clusters, max_iter=100,
    init='random', random_state=r) for r = 0..REPEATS-1, ARI and NMI averaged,
    exactly as eval_clustering.py:206-216 and :263-265.
    Hierarchical: AgglomerativeClustering with LINKAGE linkage, once
    (eval_clustering.py:225-231). They pass the L1 metric on the words; on a
    precomputed L1 matrix the merges are the same, and only a precomputed
    matrix can carry our cross-epoch distances.
    Returns (ari_pam, nmi_pam, ari_hier).
    """
    import kmedoids  # their requirements.txt

    D = np.ascontiguousarray(np.nan_to_num(0.5 * (D + D.T), nan=0.0), dtype=float)
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)
    aris, nmis = [], []
    for r in range(REPEATS):
        lab = kmedoids.pam(
            diss=D, medoids=n_clusters, max_iter=100, init="random", random_state=r
        ).labels
        aris.append(adjusted_rand_score(y_true, lab))
        nmis.append(normalized_mutual_info_score(y_true, lab))
    try:
        agg = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage=LINKAGE
        )
    except TypeError:  # scikit-learn < 1.2
        agg = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage=LINKAGE
        )
    ari_h = adjusted_rand_score(y_true, agg.fit_predict(D))
    return float(np.mean(aris)), float(np.mean(nmis)), float(ari_h)


def ed_kmeans(Z, y_true, n_clusters):
    """Their Euclidean reference: KMeans on the z-normed series, seeds
    0..REPEATS-1, n_init='auto' (eval_clustering.py:242-248). (ari, nmi)."""
    aris, nmis = [], []
    for r in range(REPEATS):
        lab = (
            KMeans(n_clusters=n_clusters, random_state=r, n_init="auto").fit(Z).labels_
        )
        aris.append(adjusted_rand_score(y_true, lab))
        nmis.append(normalized_mutual_info_score(y_true, lab))
    return float(np.mean(aris)), float(np.mean(nmis))


# ==========================================================================
# 2. PAIRWISE SYMBOLIC L1 ACROSS EPOCHS (SEARCH MODE)
# ==========================================================================


def online_l1_pairwise(words, ep_of, eps, Z_sub, rows=100):
    """(n, n) symbolic L1 between the online words, the search form.

    Same epoch: L1 of the two stored words. Different epochs: the pair lives
    in the OLDER epoch. The later window is projected onto the older epoch's
    basis and cut with its dictionary, the older word is used as stored. That
    is the only order a stream has: the past is stored, the present asks. The
    older window's raw values are never touched, so nothing here needs data
    a stream no longer has. Filled symmetrically, no averaging.
    """
    n = len(words)
    D = np.zeros((n, n), dtype=float)
    by_ep = {e: np.flatnonzero(ep_of == e) for e in np.unique(ep_of)}
    for eq, iq in by_ep.items():  # the later epoch
        for ec, ic in by_ep.items():  # the older one
            if ec > eq:
                continue
            if ec == eq:
                D[np.ix_(iq, ic)] = symbolic_l1_blocks(words[iq], words[ic], rows=rows)
                continue
            Ec = eps[ec]
            wq = digitize(Z_sub[iq] @ Ec["basis"].T, Ec["bkpt"], Ec["alpha"])
            d = symbolic_l1_blocks(wq, words[ic], rows=rows)
            D[np.ix_(iq, ic)] = d
            D[np.ix_(ic, iq)] = d.T
    return D


# ==========================================================================
# 3. EVALUATING ONE DATASET FOR CLUSTERING
# ==========================================================================


def stratified_take(y, cap, seed):
    """Indices of at most cap series, every class kept in proportion."""
    n = len(y)
    if n <= cap:
        return np.arange(n), 0
    rng = np.random.default_rng(seed)
    take = []
    for c in np.unique(y):
        ix = np.flatnonzero(y == c)
        k = max(1, int(round(cap * len(ix) / n)))
        take.extend(rng.choice(ix, size=min(k, len(ix)), replace=False))
    return np.sort(np.array(take[:cap] if len(take) > cap else take, dtype=int)), 1


def evaluate_dataset_clustering(
    ucr_dir,
    name,
    setting,
    bin_path,
    spartan_cls,
    dist_mod,
    work_dir,
    frames_cache_dir,
    spartan_repo,
):
    loaded = load_ucr_dataset(ucr_dir, name, spartan_repo)
    if loaded is None:
        return dict(error="missing .ts files or unfilled gaps")
    X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = loaded
    m = X_tr_raw.shape[1]
    k = setting["k"]
    if k > m:
        return dict(error=f"k={k} > m={m}")
    # their eval_clustering.py:115: the number of clusters is the number of
    # classes in train
    n_clusters = len(np.unique(y_tr_raw))

    w_cap = setting.get("warmup_cap", WARMUP_CAP)
    stream, X_tr, y_tr, X_te, warmup = build_stratified_stream(
        X_tr_raw, y_tr_raw, X_te_raw, warmup_cap=w_cap
    )
    n_tr, n_te = len(X_tr), len(X_te)
    ring_w = max(16, warmup)

    os.makedirs(frames_cache_dir, exist_ok=True)
    frames_path = os.path.join(frames_cache_dir, f"{name}_w{warmup}.frames")
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
        return dict(
            error=f"C exit {p.returncode}: {p.stderr.splitlines()[-1] if p.stderr else ''}"
        )

    epochs = read_epochs(run_dir)
    eps = []
    for e in range(len(epochs)):
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
    types = [
        int(r["type"])
        for r in csv.DictReader(open(os.path.join(run_dir, "epochs.csv")))
    ]
    t1_cnt, t2_cnt = types.count(1), types.count(2)

    Z_all = znorm_rows(stream, 1, 1e-12)
    Z_scored = Z_all[warmup:]
    Z_tr = Z_scored[:n_tr]
    Z_te = Z_scored[n_tr:]

    firsts = np.array([e["first"] for e in epochs])
    ep_scored = np.searchsorted(firsts, widx_o, side="right") - 1
    recut_mismatch = 0
    for e in np.unique(ep_scored):
        ix = np.flatnonzero(ep_scored == e)
        recut_mismatch += recut_check(
            Z_scored[ix] @ eps[e]["basis"].T,
            words_o[ix],
            eps[e]["bkpt"],
            eps[e]["alpha"],
        )
    ep_te = ep_scored[n_tr:]
    wo_te = words_o[n_tr:]

    take, subsampled = stratified_take(y_te_raw, MAX_CLUSTER_N, SEED)
    Z_sub = Z_te[take]
    y_sub = y_te_raw[take]
    wo_sub = wo_te[take]
    ep_sub = ep_te[take]
    widx_sub = widx_o[n_tr + take]

    alpha_in = 2 ** (setting["total_bits"] // k)

    def their_fit(X):
        sp = spartan_cls(
            alphabet_size=alpha_in,
            word_length=k,
            window_size=0,
            binning_method="equi-depth",
            assignment_policy="DAA",
            bit_budget=setting["total_bits"],
            lamda=setting["lam"],
            pca_solver="full",
        )
        sp.fit(X)
        return sp

    t0 = time.perf_counter()
    sp_batch = their_fit(Z_tr)
    W_sub_batch = sp_batch.transform(Z_sub).astype(int)
    t_batch = time.perf_counter() - t0
    mdb_batch = [np.asarray(b, dtype=float) for b in sp_batch.mindist_breakpoints]
    sp_warm = their_fit(Z_all[:warmup])
    W_sub_warm = sp_warm.transform(Z_sub).astype(int)
    mdb_warm = [np.asarray(b, dtype=float) for b in sp_warm.mindist_breakpoints]

    off = run_offline(
        bin_path,
        setting,
        X_tr,
        m,
        k,
        os.path.join(work_dir, "runs_offline", name),
        frames_cache_dir,
        name,
    )

    res = dict(
        dataset=name,
        n_test=len(take),
        subsampled=subsampled,
        n_clusters=n_clusters,
        n_train=n_tr,
        m=m,
        warmup=warmup,
        t1=t1_cnt,
        t2=t2_cnt,
        recut_mismatch=recut_mismatch,
        sec_online=t_online,
        sec_batch=t_batch,
    )

    # one matrix at a time: compute, cluster, free
    def score(tag, D):
        a, n_, h = cluster_metrics(D, y_sub, n_clusters)
        res[f"ari_{tag}"], res[f"nmi_{tag}"], res[f"agg_ari_{tag}"] = a, n_, h

    with np.errstate(over="ignore", invalid="ignore"):
        score("l1_online", online_l1_pairwise(wo_sub, ep_sub, eps, Z_sub))
        score("l1_batch", symbolic_l1_blocks(W_sub_batch, W_sub_batch))
        score("l1_warm", symbolic_l1_blocks(W_sub_warm, W_sub_warm))
        score(
            "mindist_batch",
            mindist_blocks(dist_mod, W_sub_batch, W_sub_batch, mdb_batch),
        )
        score(
            "mindist_warm", mindist_blocks(dist_mod, W_sub_warm, W_sub_warm, mdb_warm)
        )
        if off is not None:
            B_off, E_off = off
            W_off = digitize(Z_sub @ B_off.T, E_off["bkpt"], E_off["alpha"])
            score("l1_offline", symbolic_l1_blocks(W_off, W_off))
            score(
                "mindist_offline", mindist_blocks(dist_mod, W_off, W_off, E_off["mdb"])
            )
        md_search, md_box, md_join, _, _ = online_distances(
            dist_mod, widx_sub, wo_sub, ep_sub, eps, Z_all, Z_sub, m, 1e-12
        )
        score("mindist_search", md_search)
        del md_search
        score("mindist_box", md_box)
        del md_box
        score("mindist_join", md_join)
        del md_join
    res["ari_ed"], res["nmi_ed"] = ed_kmeans(Z_sub, y_sub, n_clusters)
    _, _, res["agg_ari_ed"] = cluster_metrics(
        euclid_blocks(dist_mod, Z_sub), y_sub, n_clusters
    )
    res["error"] = ""
    return res


# ==========================================================================
# 4. SUMMARY & SIGMOD'25 FIGURE 8 GENERATION
# ==========================================================================

CSV_COLS_T2 = [
    "dataset",
    "n_test",
    "n_clusters",
    "m",
    "warmup",
    "t1",
    "t2",
    "ari_l1_online",
    "ari_l1_batch",
    "ari_l1_warm",
    "ari_mindist_search",
    "ari_mindist_box",
    "ari_mindist_join",
    "ari_mindist_batch",
    "ari_mindist_warm",
    "ari_ed",
    "nmi_l1_online",
    "nmi_l1_batch",
    "nmi_mindist_search",
    "nmi_mindist_box",
    "nmi_mindist_batch",
    "nmi_ed",
    "agg_ari_l1_online",
    "agg_ari_l1_batch",
    "agg_ari_mindist_search",
    "agg_ari_mindist_box",
    "agg_ari_mindist_batch",
    "agg_ari_ed",
    "sec_online",
    "sec_batch",
    "error",
]
# new columns, after the old ones
CSV_COLS_T2 += [
    "subsampled",
    "n_train",
    "ari_l1_offline",
    "ari_mindist_offline",
    "nmi_l1_warm",
    "nmi_l1_offline",
    "nmi_mindist_join",
    "nmi_mindist_warm",
    "nmi_mindist_offline",
    "agg_ari_l1_warm",
    "agg_ari_l1_offline",
    "agg_ari_mindist_join",
    "agg_ari_mindist_warm",
    "agg_ari_mindist_offline",
    "recut_mismatch",
]


def _clu_stats(rows, sfx):
    out = {f"n{sfx}": len(rows)}
    if not rows:
        return out

    def arr(k):
        return np.array([float(r[k]) if r.get(k, "") != "" else np.nan for r in rows])

    for pre in ("ari", "nmi", "agg_ari"):
        for mm in (
            "l1_online",
            "l1_batch",
            "l1_warm",
            "l1_offline",
            "mindist_search",
            "mindist_box",
            "mindist_join",
            "mindist_batch",
            "mindist_warm",
            "mindist_offline",
        ):
            out[f"{pre}_{mm}{sfx}"] = float(np.nanmean(arr(f"{pre}_{mm}")))
    out[f"ari_ed{sfx}"] = float(np.nanmean(arr("ari_ed")))
    for metric, on in (("l1", "ari_l1_online"), ("mindist", "ari_mindist_search")):
        for ref in ("batch", "offline", "warm"):
            a, b = arr(on), arr(f"ari_{metric}_{ref}")
            keep = ~(np.isnan(a) | np.isnan(b))
            w, t, l, p = _wtl_p(a[keep], b[keep])
            out[f"wtl_ari_{metric}_vs_{ref}{sfx}"] = f"{w}/{t}/{l}"
            out[f"p_ari_{metric}_vs_{ref}{sfx}"] = p
    return out


def _clu_extra(ok):
    """The new columns, after the old ones: offline, recut check, how many
    test sets were subsampled, and every number again over only the datasets
    whose warm-up is shorter than their train (suffix _stream)."""
    stream = [r for r in ok if int(float(r["warmup"])) < int(float(r["n_train"]))]
    out = dict(
        n_subsampled=int(sum(int(float(r.get("subsampled") or 0)) for r in ok)),
        total_recut_mismatch=int(
            sum(int(float(r.get("recut_mismatch") or 0)) for r in ok)
        ),
    )
    for k in (
        "ari_l1_offline",
        "ari_mindist_offline",
        "nmi_l1_offline",
        "nmi_mindist_offline",
    ):
        v = [float(r[k]) for r in ok if r.get(k, "") not in ("", "nan")]
        out[k] = float(np.mean(v)) if v else float("nan")
    out.update(_clu_stats(stream, "_stream"))
    return out


def summarize_clustering_tag(block, tag, s, rows):
    ok = [r for r in rows if not r.get("error") and r.get("ari_l1_online") != ""]
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

    l1_on, l1_ba, l1_wa = arr("ari_l1_online"), arr("ari_l1_batch"), arr("ari_l1_warm")
    md_q, md_b, md_j, md_ba, md_wa = (
        arr("ari_mindist_search"),
        arr("ari_mindist_box"),
        arr("ari_mindist_join"),
        arr("ari_mindist_batch"),
        arr("ari_mindist_warm"),
    )
    agg_l1_on, agg_l1_ba = arr("agg_ari_l1_online"), arr("agg_ari_l1_batch")
    agg_md_q, agg_md_ba = arr("agg_ari_mindist_search"), arr("agg_ari_mindist_batch")

    w_l1, t_l1, l_l1, p_l1 = wtl_p(l1_on, l1_ba)
    w_md, t_md, l_md, p_md = wtl_p(md_q, md_ba)
    w_agg, t_agg, l_agg, p_agg = wtl_p(agg_md_q, agg_md_ba)

    return dict(
        block=block,
        tag=tag,
        n_datasets=len(ok),
        k=s["k"],
        total_bits=s["total_bits"],
        lam=s["lam"],
        theta=s["theta"],
        step=s["step"],
        warmup_cap=s.get("warmup_cap", WARMUP_CAP),
        ari_l1_online=float(l1_on.mean()),
        ari_l1_batch=float(l1_ba.mean()),
        ari_l1_warm=float(l1_wa.mean()),
        wtl_ari_l1=f"{w_l1}/{t_l1}/{l_l1}",
        p_ari_l1=p_l1,
        ari_mindist_search=float(md_q.mean()),
        ari_mindist_box=float(md_b.mean()),
        ari_mindist_join=float(md_j.mean()),
        ari_mindist_batch=float(md_ba.mean()),
        ari_mindist_warm=float(md_wa.mean()),
        wtl_ari_mindist=f"{w_md}/{t_md}/{l_md}",
        p_ari_mindist=p_md,
        ari_ed=float(arr("ari_ed").mean()),
        nmi_l1_online=float(arr("nmi_l1_online").mean()),
        nmi_l1_batch=float(arr("nmi_l1_batch").mean()),
        nmi_mindist_search=float(arr("nmi_mindist_search").mean()),
        nmi_mindist_box=float(arr("nmi_mindist_box").mean()),
        nmi_mindist_batch=float(arr("nmi_mindist_batch").mean()),
        agg_ari_l1_online=float(agg_l1_on.mean()),
        agg_ari_l1_batch=float(agg_l1_ba.mean()),
        agg_ari_mindist_search=float(agg_md_q.mean()),
        agg_ari_mindist_batch=float(agg_md_ba.mean()),
        wtl_agg_mindist=f"{w_agg}/{t_agg}/{l_agg}",
        p_agg_mindist=p_agg,
        **_clu_extra(ok),
    )


def generate_clustering_figures(out_dir, summary_rows):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    base_csv = os.path.join(out_dir, "per_dataset", "base_a4_w8_lam0.5_th45.csv")
    if os.path.exists(base_csv):
        rows = [
            r
            for r in csv.DictReader(open(base_csv))
            if not r.get("error") and r.get("ari_l1_online") != ""
        ]
        if rows:
            f = lambda k: np.clip(np.array([float(r[k]) for r in rows]), 0.0, 1.0)
            plot_pairwise_scatter(
                f("ari_l1_batch"),
                f("ari_l1_online"),
                "Batch SPARTAN",
                "Online SPARTAN",
                "Clustering ARI (Symbolic L1, a=4, w=8)",
                os.path.join(fig_dir, "fig8_pairwise_ari_l1_vs_batch"),
            )
            plot_pairwise_scatter(
                f("ari_mindist_batch"),
                f("ari_mindist_search"),
                "Batch SPARTAN",
                "Online (Search)",
                "Clustering ARI (PCA-MINDIST, a=4, w=8)",
                os.path.join(fig_dir, "fig8_pairwise_ari_mindist_vs_batch"),
            )
            plot_critical_diagram(
                {
                    "Online (Search)": np.array(
                        [float(r["ari_mindist_search"]) for r in rows]
                    ),
                    "Online (Box)": np.array(
                        [float(r["ari_mindist_box"]) for r in rows]
                    ),
                    "Online (Join)": np.array(
                        [float(r["ari_mindist_join"]) for r in rows]
                    ),
                    "Batch SPARTAN": np.array(
                        [float(r["ari_mindist_batch"]) for r in rows]
                    ),
                },
                "Critical Diagram: Clustering ARI (PCA-MINDIST)",
                os.path.join(fig_dir, "fig8_cd_clustering_ari"),
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ucr-dir", default=DEFAULT_UCR_DIR)
    ap.add_argument("--spartan-repo", default=DEFAULT_SPARTAN_REPO)
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--out", default=DEFAULT_OUT_TASK2)
    ap.add_argument("--frames-cache", default="out/ucr_task1/_frames_cache")
    ap.add_argument("--blocks", default="base")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "per_dataset"), exist_ok=True)
    sum_path = os.path.join(args.out, "ucr_clustering_summary.csv")

    spartan_cls = load_spartan_class(args.spartan_repo)
    dist_mod = load_their_distances(args.spartan_repo)

    if args.datasets:
        datasets = sorted(args.datasets)
    else:
        datasets = sorted(
            d
            for d in os.listdir(args.ucr_dir)
            if os.path.isdir(os.path.join(args.ucr_dir, d))
        )

    blocks = {b.strip().lower() for b in args.blocks.split(",")}
    plan = build_sweep_configs(blocks)
    print(f"UCR Archive : {args.ucr_dir} ({len(datasets)} datasets)")
    print(f"Task-II     : {len(plan)} clustering configurations")
    print(f"Output Dir  : {args.out}\n")

    summary_rows = []
    for idx_cfg, (blk, tag, setting) in enumerate(plan, 1):
        t0 = time.perf_counter()
        ds_csv = os.path.join(args.out, "per_dataset", f"{tag}.csv")

        if not args.no_resume and os.path.exists(ds_csv):
            existing = list(csv.DictReader(open(ds_csv)))
            if len(existing) == len(datasets):
                s_row = summarize_clustering_tag(blk, tag, setting, existing)
                s_row["elapsed_s"] = 0.0
                summary_rows.append(s_row)
                print(
                    f"  [{idx_cfg}/{len(plan)}] {tag:<26} | CACHED | "
                    f"ARI(L1) On/Ba: {s_row.get('ari_l1_online', 0):.4f}/{s_row.get('ari_l1_batch', 0):.4f} "
                    f"({s_row.get('wtl_ari_l1', '-')}) | "
                    f"ARI(MD) Q/B/Ba: {s_row.get('ari_mindist_search', 0):.4f}/{s_row.get('ari_mindist_box', 0):.4f}/{s_row.get('ari_mindist_batch', 0):.4f} "
                    f"({s_row.get('wtl_ari_mindist', '-')}, p={s_row.get('p_ari_mindist', 1):.3f})"
                )
                continue

        per_ds_rows = []
        for idx_d, dname in enumerate(datasets, 1):
            sys.stdout.write(
                f"\r  [{idx_cfg}/{len(plan)}] {tag:<26} dataset {idx_d:3d}/{len(datasets)} ({dname:<28})"
            )
            sys.stdout.flush()
            try:
                res = evaluate_dataset_clustering(
                    args.ucr_dir,
                    dname,
                    setting,
                    args.bin,
                    spartan_cls,
                    dist_mod,
                    args.out,
                    args.frames_cache,
                    args.spartan_repo,
                )
            except Exception as exc:
                res = dict(dataset=dname, error=f"{type(exc).__name__}: {exc}")
            res["dataset"] = dname
            per_ds_rows.append(res)
            gc.collect()

        with open(ds_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS_T2, restval="")
            w.writeheader()
            w.writerows(per_ds_rows)

        s_row = summarize_clustering_tag(blk, tag, setting, per_ds_rows)
        s_row["elapsed_s"] = round(time.perf_counter() - t0, 1)
        summary_rows.append(s_row)

        with open(sum_path, "w", newline="") as f:
            keys = list(dict.fromkeys(k for r in summary_rows for k in r))
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(summary_rows)

        print(
            f"\r  [{idx_cfg}/{len(plan)}] {tag:<26} | "
            f"ARI(L1) On/Ba: {s_row.get('ari_l1_online', 0):.4f}/{s_row.get('ari_l1_batch', 0):.4f} "
            f"({s_row.get('wtl_ari_l1', '-')}) | "
            f"ARI(MD) Q/B/Ba: {s_row.get('ari_mindist_search', 0):.4f}/{s_row.get('ari_mindist_box', 0):.4f}/{s_row.get('ari_mindist_batch', 0):.4f} "
            f"({s_row.get('wtl_ari_mindist', '-')}, p={s_row.get('p_ari_mindist', 1):.3f}) | {s_row['elapsed_s']}s"
        )

    generate_clustering_figures(args.out, summary_rows)
    print(f"\nDone! Summary table: {sum_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
