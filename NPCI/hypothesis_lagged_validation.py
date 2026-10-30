# Hypothesis validation:Full LPCMCI vs Lagged Screening + Local LPCMCI
# 方法描述：为验证滞后筛选能否保留病灶相关核心因果结构，我们在 40、60、80 个 ROI 的小规模子系统上分别运行全量 LPCMCI 和“滞后筛选 + 局部 LPCMCI”。
# 对两种方法得到的结果，均提取病灶节点的一阶/二阶邻域边集，并计算 lesion-edge recall 与 Jaccard 相似度，同时记录运行时间。
# 评价标准：若 screening-local 方法在显著缩短运行时间的同时，对 full LPCMCI 的 lesion-centered edge set 保持较高 recall，
# 则说明滞后筛选可以作为全脑局部因果发现的有效前端压缩策略。
# 在 compare_lagged_hypothesis() 里默认hop=2，这表示比较 lesion 的二阶邻域边集。如果你只想看直接连接 lesion 的边，可以改成hop=1
# 建议正式报告两个都可以做：
# 一阶邻域：hypothesis_df_hop1, _ = compare_lagged_hypothesis(..., hop=1)
# 二阶邻域：hypothesis_df_hop2, _ = compare_lagged_hypothesis(..., hop=2)
# 主文放 hop=2，补充材料放 hop=1。

import time
import numpy as np
import pandas as pd


# 这里依赖你现有 Stage 2 里的 LPCMCI 接口
from local_lpcmci import (
    run_candidate_subnetwork_lpcmci,
    lpcmci_to_edge_df,
)


# =========================================================
# 1. Basic utilities
# =========================================================

def _as_signal_array(signals):
    """
    Convert signals to a 2D array.
    signals:
        np.ndarray, shape = (T, ROI)
        or list of arrays.
    """
    if isinstance(signals, list):
        return np.vstack([np.asarray(x, dtype=float) for x in signals])
    return np.asarray(signals, dtype=float)


def _safe_zscore(X, eps=1e-8):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < eps] = 1.0
    return (X - mu) / sd


def compute_lagged_score_matrix(signals, tau=1, standardize=True, eps=1e-8):
    """
    Compute lagged correlation score matrix.

    score[src, dst] = |corr(X_src(t-tau), X_dst(t))|

    Parameters
    ----------
    signals:
        shape = (T, ROI)
    tau:
        lag order. For your current Stage 2, tau=1.

    Returns
    -------
    score_mat:
        absolute lagged correlation, shape = (ROI, ROI)
    signed_mat:
        signed lagged correlation, shape = (ROI, ROI)
    """
    X = _as_signal_array(signals)

    if X.shape[0] <= tau:
        raise ValueError("signals length must be larger than tau.")

    X_lag = X[:-tau, :]
    X_cur = X[tau:, :]

    X_lag = np.nan_to_num(X_lag, nan=0.0, posinf=0.0, neginf=0.0)
    X_cur = np.nan_to_num(X_cur, nan=0.0, posinf=0.0, neginf=0.0)

    if standardize:
        X_lag = _safe_zscore(X_lag, eps=eps)
        X_cur = _safe_zscore(X_cur, eps=eps)

    n = X_lag.shape[0]
    signed_mat = (X_lag.T @ X_cur) / max(n - 1, 1)
    signed_mat = np.nan_to_num(signed_mat, nan=0.0, posinf=0.0, neginf=0.0)

    score_mat = np.abs(signed_mat)

    np.fill_diagonal(score_mat, 0.0)
    np.fill_diagonal(signed_mat, 0.0)

    return score_mat, signed_mat


def _extract_stage1_top_response_nodes(stage1_res, top_k=30, exclude=None):
    """
    Extract top lesion-response nodes from stage1_res.
    Compatible with:
        stage1_res["lesion_response_df"]
        stage1_res["response_df"]
    """
    exclude = set(int(x) for x in (exclude or []))

    if stage1_res is None:
        return []

    if isinstance(stage1_res, dict):
        if "lesion_response_df" in stage1_res:
            df = stage1_res["lesion_response_df"].copy()
        elif "response_df" in stage1_res:
            df = stage1_res["response_df"].copy()
        else:
            return []
    elif isinstance(stage1_res, pd.DataFrame):
        df = stage1_res.copy()
    else:
        return []

    if len(df) == 0:
        return []

    if "node" not in df.columns:
        if "roi" in df.columns:
            df = df.rename(columns={"roi": "node"})
        else:
            return []

    score_col = None
    for c in [
        "response_abs",
        "lesion_response",
        "abnormality_score",
        "spread_score",
        "damage_score",
        "candidate_score",
        "score",
    ]:
        if c in df.columns:
            score_col = c
            break

    if score_col is None:
        return []

    df = df[~df["node"].astype(int).isin(exclude)].copy()
    df = df.sort_values(score_col, ascending=False)

    return df["node"].astype(int).head(int(top_k)).tolist()


def _unique_keep_order(xs):
    out = []
    seen = set()
    for x in xs:
        x = int(x)
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# =========================================================
# 2. Build small ROI subset
# =========================================================

def build_small_roi_subset(
    signals,
    lesion_idx,
    n_roi_small=40,
    stage1_res=None,
    tau=1,
    top_response=20,
    top_lagged_direct=20,
    random_fill=True,
    random_state=123,
):
    """
    Build a lesion-aware small ROI subset for Full LPCMCI comparison.

    Subset = lesion node
             + Stage 1 response top nodes
             + direct lagged nodes around lesion
             + random fill

    This makes the small system lesion-relevant instead of purely random.

    Returns
    -------
    roi_subset:
        List[int], global ROI indices.
    info:
        Dict.
    """
    rng = np.random.default_rng(random_state)
    X = _as_signal_array(signals)
    n_roi_total = X.shape[1]

    lesion_idx = int(lesion_idx)

    if n_roi_small > n_roi_total:
        raise ValueError("n_roi_small cannot exceed total ROI number.")

    selected = [lesion_idx]

    # Stage 1 lesion response top nodes
    response_nodes = _extract_stage1_top_response_nodes(
        stage1_res,
        top_k=top_response,
        exclude=selected,
    )
    selected.extend(response_nodes)

    # Lagged direct nodes around lesion
    score_mat, _ = compute_lagged_score_matrix(
        X,
        tau=tau,
        standardize=True,
    )

    # nodes with strong i(t-1) -> lesion(t)
    parent_score = score_mat[:, lesion_idx].copy()
    parent_score[lesion_idx] = -np.inf

    # nodes with strong lesion(t-1) -> j(t)
    child_score = score_mat[lesion_idx, :].copy()
    child_score[lesion_idx] = -np.inf

    combined_score = parent_score + child_score
    order = np.argsort(-combined_score)

    lagged_nodes = []
    for node in order:
        node = int(node)
        if node in selected:
            continue
        lagged_nodes.append(node)
        if len(lagged_nodes) >= int(top_lagged_direct):
            break

    selected.extend(lagged_nodes)
    selected = _unique_keep_order(selected)

    # Random fill
    if len(selected) < n_roi_small and random_fill:
        remain = [i for i in range(n_roi_total) if i not in set(selected)]
        fill = rng.choice(
            remain,
            size=n_roi_small - len(selected),
            replace=False,
        ).astype(int).tolist()
        selected.extend(fill)

    roi_subset = _unique_keep_order(selected)[: int(n_roi_small)]

    if lesion_idx not in roi_subset:
        roi_subset[-1] = lesion_idx
        roi_subset = [lesion_idx] + [x for x in roi_subset if x != lesion_idx]

    info = {
        "n_roi_small": int(n_roi_small),
        "lesion_idx": lesion_idx,
        "n_response_nodes_used": len(response_nodes),
        "n_lagged_nodes_used": len(lagged_nodes),
        "n_final_subset": len(roi_subset),
        "roi_subset": roi_subset,
    }

    return roi_subset, info


# =========================================================
# 3. LPCMCI runners
# =========================================================

def run_full_lpcmci_small(
    signals,
    roi_subset,
    tau_min=1,
    tau_max=1,
    pc_alpha=0.01,
    alpha_level=0.01,
    verbosity=0,
    standardize=True,
    max_cond_px=1,
    max_p_global=1,
    max_p_non_ancestral=1,
    max_q_global=3,
    max_pds_set=5,
):
    """
    Run Full LPCMCI on the small ROI subset.

    Full here means:
        no lagged screening within this subset.
        LPCMCI is run on all nodes in roi_subset.

    Returns
    -------
    res:
        Dict with edge_df, elapsed_time, node_map, raw_lpcmci_res.
    """
    roi_subset = [int(x) for x in roi_subset]

    t0 = time.perf_counter()

    lpcmci_res, node_map = run_candidate_subnetwork_lpcmci(
        signals=signals,
        candidate_nodes=roi_subset,
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        verbosity=verbosity,
        standardize=standardize,
        max_cond_px=max_cond_px,
        max_p_global=max_p_global,
        max_p_non_ancestral=max_p_non_ancestral,
        max_q_global=max_q_global,
        max_pds_set=max_pds_set,
    )

    edge_df = lpcmci_to_edge_df(
        lpcmci_res,
        node_map=node_map,
        include_contemp=False,
        lag_only=True,
        alpha_level=alpha_level,
    )

    elapsed = time.perf_counter() - t0

    return {
        "method": "full_lpcmci_small",
        "edge_df": edge_df,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60.0,
        "node_map": node_map,
        "lpcmci_res": lpcmci_res,
        "roi_subset": roi_subset,
    }


def _lagged_screen_within_subset(
    signals,
    roi_subset,
    lesion_idx,
    tau=1,
    max_screen_nodes=30,
    min_screen_nodes=15,
    top_parents=15,
    top_children=15,
    stage1_res=None,
    top_response=10,
):
    """
    Lagged screening restricted within a given small ROI subset.

    It does not call run_lagged_candidate_screening because that function
    usually screens from the whole brain. Here we need a fair comparison
    within the same small ROI subset.

    Returns
    -------
    candidate_nodes:
        global ROI indices.
    score_df:
        dataframe of screening scores.
    """
    roi_subset = [int(x) for x in roi_subset]
    lesion_idx = int(lesion_idx)

    X = _as_signal_array(signals)
    score_mat, signed_mat = compute_lagged_score_matrix(
        X,
        tau=tau,
        standardize=True,
    )

    rows = []
    for node in roi_subset:
        node = int(node)
        if node == lesion_idx:
            parent_score = 0.0
            child_score = 0.0
        else:
            parent_score = float(score_mat[node, lesion_idx])
            child_score = float(score_mat[lesion_idx, node])

        direct_score = max(parent_score, child_score)

        rows.append({
            "node": node,
            "parent_proxy": parent_score,
            "child_proxy": child_score,
            "direct_lagged_score": direct_score,
        })

    score_df = pd.DataFrame(rows)

    # Add response score if available
    response_nodes = _extract_stage1_top_response_nodes(
        stage1_res,
        top_k=10_000,
        exclude=[],
    )
    response_rank = {int(n): i for i, n in enumerate(response_nodes)}

    # rank-based response score
    if len(response_rank) > 0:
        score_df["response_score"] = score_df["node"].apply(
            lambda x: 1.0 / (1.0 + response_rank[int(x)])
            if int(x) in response_rank else 0.0
        )
    else:
        score_df["response_score"] = 0.0

    # Combined screening score
    score_df["screening_score"] = (
        score_df["direct_lagged_score"]
        + 0.5 * score_df["response_score"]
    )

    # Force lesion node
    selected = [lesion_idx]

    # top parent/child proxy
    parent_nodes = (
        score_df[score_df["node"] != lesion_idx]
        .sort_values("parent_proxy", ascending=False)
        ["node"].astype(int).head(int(top_parents)).tolist()
    )

    child_nodes = (
        score_df[score_df["node"] != lesion_idx]
        .sort_values("child_proxy", ascending=False)
        ["node"].astype(int).head(int(top_children)).tolist()
    )

    response_top_nodes = (
        score_df[score_df["node"] != lesion_idx]
        .sort_values("response_score", ascending=False)
        ["node"].astype(int).head(int(top_response)).tolist()
    )

    selected.extend(parent_nodes)
    selected.extend(child_nodes)
    selected.extend(response_top_nodes)
    selected = _unique_keep_order(selected)

    # =====================================================
    # Fill to target max_screen_nodes, not only min_screen_nodes
    # =====================================================
    max_screen_nodes = min(int(max_screen_nodes), len(roi_subset))
    min_screen_nodes = min(int(min_screen_nodes), max_screen_nodes)

    # target number of screened nodes
    # Usually this is max_screen_nodes, e.g. 60% of the small ROI subset.
    target_screen_nodes = max(min_screen_nodes, max_screen_nodes)

    selected = _unique_keep_order(selected)

    # Fill by screening_score until target_screen_nodes
    if len(selected) < target_screen_nodes:
        fill_nodes = (
            score_df[~score_df["node"].astype(int).isin(selected)]
            .sort_values("screening_score", ascending=False)
            ["node"].astype(int)
            .head(target_screen_nodes - len(selected))
            .tolist()
        )
        selected.extend(fill_nodes)
        selected = _unique_keep_order(selected)

    # Truncate to target_screen_nodes while preserving lesion
    if len(selected) > target_screen_nodes:
        tmp = score_df[score_df["node"].astype(int).isin(selected)].copy()
        tmp = tmp[tmp["node"].astype(int) != lesion_idx]
        tmp = tmp.sort_values("screening_score", ascending=False)

        selected = [lesion_idx] + tmp["node"].astype(int).head(
            target_screen_nodes - 1
        ).tolist()

    selected = _unique_keep_order(selected)

    score_df = score_df.sort_values(
        "screening_score",
        ascending=False,
    ).reset_index(drop=True)

    return selected, score_df


def run_screen_local_lpcmci_small(
    signals,
    roi_subset,
    lesion_idx,
    stage1_res=None,
    tau=1,
    max_screen_nodes=None,
    min_screen_nodes=15,
    top_parents=15,
    top_children=15,
    top_response=10,
    tau_min=1,
    tau_max=1,
    pc_alpha=0.01,
    alpha_level=0.01,
    verbosity=0,
    standardize=True,
    max_cond_px=1,
    max_p_global=1,
    max_p_non_ancestral=1,
    max_q_global=3,
    max_pds_set=5,
):
    """
    Run lagged screening + local LPCMCI within the same small ROI subset.

    Returns
    -------
    res:
        Dict with candidate_nodes, edge_df, elapsed_time.
    """
    roi_subset = [int(x) for x in roi_subset]
    n_small = len(roi_subset)

    if max_screen_nodes is None:
        # default: screen to around 60% of the small subset
        max_screen_nodes = max(min_screen_nodes, int(0.6 * n_small))

    t0 = time.perf_counter()

    candidate_nodes, score_df = _lagged_screen_within_subset(
        signals=signals,
        roi_subset=roi_subset,
        lesion_idx=lesion_idx,
        tau=tau,
        max_screen_nodes=max_screen_nodes,
        min_screen_nodes=min_screen_nodes,
        top_parents=top_parents,
        top_children=top_children,
        stage1_res=stage1_res,
        top_response=top_response,
    )

    lpcmci_res, node_map = run_candidate_subnetwork_lpcmci(
        signals=signals,
        candidate_nodes=candidate_nodes,
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        verbosity=verbosity,
        standardize=standardize,
        max_cond_px=max_cond_px,
        max_p_global=max_p_global,
        max_p_non_ancestral=max_p_non_ancestral,
        max_q_global=max_q_global,
        max_pds_set=max_pds_set,
    )

    edge_df = lpcmci_to_edge_df(
        lpcmci_res,
        node_map=node_map,
        include_contemp=False,
        lag_only=True,
        alpha_level=alpha_level,
    )

    elapsed = time.perf_counter() - t0

    return {
        "method": "screen_local_lpcmci_small",
        "candidate_nodes": candidate_nodes,
        "score_df": score_df,
        "edge_df": edge_df,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60.0,
        "node_map": node_map,
        "lpcmci_res": lpcmci_res,
        "roi_subset": roi_subset,
        "max_screen_nodes": int(max_screen_nodes),
        "min_screen_nodes": int(min_screen_nodes),
    }


# =========================================================
# 4. Lesion-centered edge set extraction
# =========================================================

def extract_lesion_edge_set(
    edge_df,
    lesion_idx,
    hop=2,
    tau_filter=1,
    directed=True,
):
    """
    Extract lesion-centered edge set from an LPCMCI edge dataframe.

    Procedure:
        1. Build undirected neighborhood from lagged directed edges.
        2. Find nodes within `hop` steps from lesion.
        3. Extract all directed lagged edges inside this lesion neighborhood.

    Edge representation:
        directed=True:
            (src, dst, tau)
        directed=False:
            (min(src,dst), max(src,dst), tau)

    Returns
    -------
    edge_set:
        set of tuples.
    neighbor_nodes:
        set of nodes in lesion-centered neighborhood.
    """
    lesion_idx = int(lesion_idx)

    if edge_df is None or len(edge_df) == 0:
        return set(), set([lesion_idx])

    df = edge_df.copy()

    if "tau" in df.columns and tau_filter is not None:
        df = df[df["tau"].astype(int) == int(tau_filter)].copy()

    if len(df) == 0:
        return set(), set([lesion_idx])

    required = {"src", "dst"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"edge_df missing required columns: {missing}")

    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)

    if "tau" not in df.columns:
        df["tau"] = int(tau_filter) if tau_filter is not None else 1

    # Build undirected adjacency for hop search
    adj = {}
    for _, row in df.iterrows():
        s = int(row["src"])
        d = int(row["dst"])
        adj.setdefault(s, set()).add(d)
        adj.setdefault(d, set()).add(s)

    # BFS up to hop
    visited = set([lesion_idx])
    frontier = set([lesion_idx])

    for _ in range(int(hop)):
        new_frontier = set()
        for u in frontier:
            for v in adj.get(u, set()):
                if v not in visited:
                    new_frontier.add(v)
        visited |= new_frontier
        frontier = new_frontier

    neighbor_nodes = visited

    # Extract directed edges inside neighborhood
    edge_set = set()
    for _, row in df.iterrows():
        s = int(row["src"])
        d = int(row["dst"])
        tau = int(row["tau"])

        if s in neighbor_nodes and d in neighbor_nodes:
            if directed:
                edge_set.add((s, d, tau))
            else:
                edge_set.add((min(s, d), max(s, d), tau))

    return edge_set, neighbor_nodes


# =========================================================
# 5. Compare Full vs Screening+Local
# =========================================================
def _edge_key_from_row(row, tau_filter=1):
    src = int(row["src"])
    dst = int(row["dst"])
    tau = int(row["tau"]) if "tau" in row.index else int(tau_filter)
    return (src, dst, tau)


def _edge_score_from_row(row):
    """
    Robust edge score extractor.

    Priority:
        1. abs_val / abs_value / abs_weight
        2. abs(val / value / weight / coef)
        3. -log10(p / pval / p_value)
        4. 1.0
    """
    abs_cols = ["abs_val", "abs_value", "abs_weight", "score", "edge_score"]
    for c in abs_cols:
        if c in row.index and pd.notna(row[c]):
            return abs(float(row[c]))

    signed_cols = ["val", "value", "weight", "coef", "effect"]
    for c in signed_cols:
        if c in row.index and pd.notna(row[c]):
            return abs(float(row[c]))

    p_cols = ["p", "pval", "p_value", "pval_min"]
    for c in p_cols:
        if c in row.index and pd.notna(row[c]):
            p = max(float(row[c]), 1e-300)
            return float(-np.log10(p))

    return 1.0


def extract_scored_lesion_edge_df(
    edge_df,
    lesion_idx,
    hop=2,
    tau_filter=1,
):
    """
    Extract lesion-centered edges and attach edge scores.

    Returns
    -------
    scored_df:
        columns = ["edge_key", "src", "dst", "tau", "score"]
    """
    edge_set, neighbor_nodes = extract_lesion_edge_set(
        edge_df=edge_df,
        lesion_idx=lesion_idx,
        hop=hop,
        tau_filter=tau_filter,
        directed=True,
    )

    if edge_df is None or len(edge_df) == 0 or len(edge_set) == 0:
        return pd.DataFrame(columns=["edge_key", "src", "dst", "tau", "score"])

    df = edge_df.copy()

    if "tau" in df.columns and tau_filter is not None:
        df = df[df["tau"].astype(int) == int(tau_filter)].copy()

    rows = []
    for _, row in df.iterrows():
        key = _edge_key_from_row(row, tau_filter=tau_filter)
        if key not in edge_set:
            continue

        rows.append({
            "edge_key": key,
            "src": int(key[0]),
            "dst": int(key[1]),
            "tau": int(key[2]),
            "score": float(_edge_score_from_row(row)),
        })

    scored_df = pd.DataFrame(rows)

    if len(scored_df) == 0:
        return pd.DataFrame(columns=["edge_key", "src", "dst", "tau", "score"])

    # If duplicated edges exist, keep the maximum score.
    scored_df = (
        scored_df
        .sort_values("score", ascending=False)
        .groupby("edge_key", as_index=False)
        .agg({
            "src": "first",
            "dst": "first",
            "tau": "first",
            "score": "max",
        })
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    return scored_df


def compare_topk_strong_edges(
    full_edge_df,
    local_edge_df,
    lesion_idx,
    hop=2,
    tau_filter=1,
    topk_list=(50, 100, 200),
):
    """
    Compare whether screening-local preserves the strongest full-LPCMCI
    lesion-centered edges.

    Top-K Recall:
        |E_full_topK ∩ E_screen+local| / |E_full_topK|
    """
    full_scored = extract_scored_lesion_edge_df(
        edge_df=full_edge_df,
        lesion_idx=lesion_idx,
        hop=hop,
        tau_filter=tau_filter,
    )

    local_set, _ = extract_lesion_edge_set(
        edge_df=local_edge_df,
        lesion_idx=lesion_idx,
        hop=hop,
        tau_filter=tau_filter,
        directed=True,
    )

    out = {}

    for k in topk_list:
        k = int(k)

        if len(full_scored) == 0:
            out[f"top{k}_k_eff"] = 0
            out[f"top{k}_hit"] = 0
            out[f"top{k}_recall"] = np.nan
            continue

        k_eff = min(k, len(full_scored))
        full_topk_set = set(full_scored.head(k_eff)["edge_key"].tolist())

        hit = len(full_topk_set & set(local_set))
        recall = hit / k_eff if k_eff > 0 else np.nan

        out[f"top{k}_k_eff"] = int(k_eff)
        out[f"top{k}_hit"] = int(hit)
        out[f"top{k}_recall"] = float(recall)

    return out

def _compare_edge_sets(full_set, local_set):
    full_set = set(full_set)
    local_set = set(local_set)

    inter = full_set & local_set
    union = full_set | local_set

    recall = np.nan if len(full_set) == 0 else len(inter) / len(full_set)
    precision = np.nan if len(local_set) == 0 else len(inter) / len(local_set)
    jaccard = np.nan if len(union) == 0 else len(inter) / len(union)

    return {
        "n_full_edges": len(full_set),
        "n_local_edges": len(local_set),
        "n_intersection": len(inter),
        "n_union": len(union),
        "lesion_edge_recall": recall,
        "lesion_edge_precision": precision,
        "jaccard": jaccard,
    }


def compare_lagged_hypothesis(
    signals,
    lesion_idx,
    stage1_res=None,
    roi_sizes=(30, 45, 60),
    random_state=123,
    hop=2,
    tau=1,
    pc_alpha=0.01,
    alpha_level=0.01,
    screen_ratio=0.60,
    min_screen_ratio=0.30,
    topk_list=(50, 100, 200),
    full_lpcmci_kwargs=None,
    local_lpcmci_kwargs=None,
    save_csv_path=None,
):
    """
    Run hypothesis validation experiment.

    For each ROI size:
        1. Build lesion-aware small ROI subset.
        2. Run Full LPCMCI on this subset.
        3. Run lagged screening + local LPCMCI on the same subset.
        4. Extract lesion-centered edge sets.
        5. Compute recall / Jaccard / runtime speedup.

    Returns
    -------
    result_df:
        Summary table.
    details:
        Dict storing raw outputs for each ROI size.
    """
    full_lpcmci_kwargs = dict(full_lpcmci_kwargs or {})
    local_lpcmci_kwargs = dict(local_lpcmci_kwargs or {})

    rows = []
    details = {}

    for n_roi_small in roi_sizes:
        print("=" * 90)
        print(f"[Lagged hypothesis] ROI size = {n_roi_small}")
        print("=" * 90)

        roi_subset, subset_info = build_small_roi_subset(
            signals=signals,
            lesion_idx=lesion_idx,
            n_roi_small=n_roi_small,
            stage1_res=stage1_res,
            tau=tau,
            top_response=min(20, max(5, n_roi_small // 3)),
            top_lagged_direct=min(20, max(5, n_roi_small // 3)),
            random_fill=True,
            random_state=random_state + int(n_roi_small),
        )

        print("Small subset built:", subset_info)

        full_res = run_full_lpcmci_small(
            signals=signals,
            roi_subset=roi_subset,
            tau_min=tau,
            tau_max=tau,
            pc_alpha=pc_alpha,
            alpha_level=alpha_level,
            **full_lpcmci_kwargs,
        )

        # local screen size: target ratio of current ROI subset
        # Example:
        #   ROI=30 -> max_screen_nodes=18
        #   ROI=45 -> max_screen_nodes=27
        #   ROI=60 -> max_screen_nodes=36
        default_max_screen = min(
            int(n_roi_small),
            max(8, int(round(screen_ratio * n_roi_small)))
        )

        default_min_screen = min(
            default_max_screen,
            max(6, int(round(min_screen_ratio * n_roi_small)))
        )

        # 关键：复制一份临时参数，避免修改 local_lpcmci_kwargs 原字典
        local_kwargs_this = dict(local_lpcmci_kwargs)

        max_screen_nodes = local_kwargs_this.pop("max_screen_nodes", default_max_screen)
        min_screen_nodes = local_kwargs_this.pop("min_screen_nodes", default_min_screen)

        local_res = run_screen_local_lpcmci_small(
            signals=signals,
            roi_subset=roi_subset,
            lesion_idx=lesion_idx,
            stage1_res=stage1_res,
            tau=tau,
            max_screen_nodes=max_screen_nodes,
            min_screen_nodes=min_screen_nodes,
            tau_min=tau,
            tau_max=tau,
            pc_alpha=pc_alpha,
            alpha_level=alpha_level,
            **local_kwargs_this,
        )

        full_edge_set, full_neighbor_nodes = extract_lesion_edge_set(
            full_res["edge_df"],
            lesion_idx=lesion_idx,
            hop=hop,
            tau_filter=tau,
            directed=True,
        )

        local_edge_set, local_neighbor_nodes = extract_lesion_edge_set(
            local_res["edge_df"],
            lesion_idx=lesion_idx,
            hop=hop,
            tau_filter=tau,
            directed=True,
        )

        cmp_info = _compare_edge_sets(full_edge_set, local_edge_set)

        topk_info = compare_topk_strong_edges(
            full_edge_df=full_res["edge_df"],
            local_edge_df=local_res["edge_df"],
            lesion_idx=lesion_idx,
            hop=hop,
            tau_filter=tau,
            topk_list=topk_list,
        )

        speedup = (
            np.nan
            if local_res["elapsed_sec"] <= 0
            else full_res["elapsed_sec"] / local_res["elapsed_sec"]
        )

        row = {
            "roi_size": int(n_roi_small),

            "full_time_sec": float(full_res["elapsed_sec"]),
            "full_time_min": float(full_res["elapsed_min"]),
            "screen_local_time_sec": float(local_res["elapsed_sec"]),
            "screen_local_time_min": float(local_res["elapsed_min"]),
            "speedup_full_over_local": float(speedup),

            "n_subset_nodes": len(roi_subset),
            "n_screen_nodes": len(local_res["candidate_nodes"]),
            "n_full_all_edges": len(full_res["edge_df"]),
            "n_local_all_edges": len(local_res["edge_df"]),

            "n_full_lesion_neighbors": len(full_neighbor_nodes),
            "n_local_lesion_neighbors": len(local_neighbor_nodes),

            **cmp_info,
            **topk_info,
        }

        rows.append(row)

        details[int(n_roi_small)] = {
            "subset_info": subset_info,
            "roi_subset": roi_subset,
            "full_res": full_res,
            "local_res": local_res,
            "full_edge_set": full_edge_set,
            "local_edge_set": local_edge_set,
            "full_neighbor_nodes": full_neighbor_nodes,
            "local_neighbor_nodes": local_neighbor_nodes,
            "row": row,
        }

        print("Result row:")
        print(row)

        if save_csv_path is not None:
            pd.DataFrame(rows).to_csv(save_csv_path, index=False)

    result_df = pd.DataFrame(rows)

    if save_csv_path is not None:
        result_df.to_csv(save_csv_path, index=False)

    return result_df, details


# =========================================================
# 6. Optional judgement helper
# =========================================================

def judge_lagged_hypothesis_result(
    result_df,
    recall_threshold=0.30,
    precision_threshold=0.80,
    jaccard_threshold=0.25,
    speedup_threshold=3.0,
    topk_recall_thresholds=None,
):
    """
    Judge whether the lagged screening hypothesis is empirically supported.

    This version supports both:
        1. ordinary lesion-edge overlap metrics
        2. Top-K strong edge recall metrics

    Parameters
    ----------
    result_df:
        Output dataframe from compare_lagged_hypothesis()
        or recompute_lagged_hypothesis_from_details().

    recall_threshold:
        Threshold for ordinary lesion-edge recall.

    precision_threshold:
        Threshold for lesion-edge precision.

    jaccard_threshold:
        Threshold for lesion-edge Jaccard.

    speedup_threshold:
        Threshold for runtime speedup.

    topk_recall_thresholds:
        Dict such as:
        {
            "top50_recall": 0.50,
            "top100_recall": 0.40,
            "top200_recall": 0.30,
        }
    """
    df = result_df.copy()

    required_cols = [
        "lesion_edge_recall",
        "lesion_edge_precision",
        "jaccard",
        "speedup_full_over_local",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"result_df missing required columns: {missing}")

    mean_recall = float(df["lesion_edge_recall"].mean(skipna=True))
    mean_precision = float(df["lesion_edge_precision"].mean(skipna=True))
    mean_jaccard = float(df["jaccard"].mean(skipna=True))
    mean_speedup = float(df["speedup_full_over_local"].mean(skipna=True))

    pass_recall = mean_recall >= recall_threshold
    pass_precision = mean_precision >= precision_threshold
    pass_jaccard = mean_jaccard >= jaccard_threshold
    pass_speedup = mean_speedup >= speedup_threshold

    # -------------------------------------------------
    # Top-K strong edge recall judgement
    # -------------------------------------------------
    topk_recall_thresholds = topk_recall_thresholds or {}

    topk_summary = {}
    topk_pass_flags = {}

    for col, threshold in topk_recall_thresholds.items():
        if col not in df.columns:
            topk_summary[f"mean_{col}"] = float("nan")
            topk_pass_flags[f"pass_{col}"] = False
            continue

        mean_val = float(df[col].mean(skipna=True))
        topk_summary[f"mean_{col}"] = mean_val
        topk_summary[f"{col}_threshold"] = float(threshold)
        topk_pass_flags[f"pass_{col}"] = bool(mean_val >= threshold)

    if len(topk_pass_flags) > 0:
        pass_topk_any = any(topk_pass_flags.values())
        pass_topk_all = all(topk_pass_flags.values())
    else:
        pass_topk_any = False
        pass_topk_all = False

    # -------------------------------------------------
    # Final judgement
    # -------------------------------------------------
    if pass_precision and pass_speedup and pass_recall and pass_topk_any:
        judgement = "supported"
    elif pass_precision and pass_speedup and pass_topk_any:
        judgement = "supported_for_strong_edges"
    elif pass_precision and pass_speedup:
        judgement = "partially_supported_high_precision_high_speedup"
    elif pass_precision:
        judgement = "partially_supported_high_precision_only"
    else:
        judgement = "not_supported"

    return {
        "mean_lesion_edge_recall": mean_recall,
        "mean_lesion_edge_precision": mean_precision,
        "mean_jaccard": mean_jaccard,
        "mean_speedup": mean_speedup,

        "recall_threshold": float(recall_threshold),
        "precision_threshold": float(precision_threshold),
        "jaccard_threshold": float(jaccard_threshold),
        "speedup_threshold": float(speedup_threshold),

        "pass_recall": bool(pass_recall),
        "pass_precision": bool(pass_precision),
        "pass_jaccard": bool(pass_jaccard),
        "pass_speedup": bool(pass_speedup),

        "pass_topk_any": bool(pass_topk_any),
        "pass_topk_all": bool(pass_topk_all),

        **topk_summary,
        **topk_pass_flags,

        "judgement": judgement,
    }

def build_lagged_hypothesis_paper_table(
    result_df,
    decimals=3,
    speedup_decimals=2,
    save_csv_path=None,
):
    """
    Build a compact paper-ready table for the lagged screening hypothesis test.

    Main columns:
        ROI size, Screen nodes, Full edges, Local edges, Intersection,
        Recall, Precision, Jaccard, Top-K Recall, Speedup
    """
    df = result_df.copy()

    base_cols = [
        "roi_size",
        "n_screen_nodes",
        "n_full_edges",
        "n_local_edges",
        "n_intersection",
        "lesion_edge_recall",
        "lesion_edge_precision",
        "jaccard",
        "speedup_full_over_local",
    ]

    topk_cols = [
        c for c in df.columns
        if c.startswith("top") and c.endswith("_recall")
    ]

    required_cols = base_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"result_df missing required columns: {missing}")

    paper_df = df[base_cols + topk_cols].copy()

    rename_map = {
        "roi_size": "ROI size",
        "n_screen_nodes": "Screen nodes",
        "n_full_edges": "Full edges",
        "n_local_edges": "Local edges",
        "n_intersection": "Intersection",
        "lesion_edge_recall": "Recall",
        "lesion_edge_precision": "Precision",
        "jaccard": "Jaccard",
        "speedup_full_over_local": "Speedup",
    }

    for c in topk_cols:
        # top50_recall -> Top-50 Recall
        k = c.replace("top", "").replace("_recall", "")
        rename_map[c] = f"Top-{k} Recall"

    paper_df = paper_df.rename(columns=rename_map)

    metric_cols = [
        c for c in paper_df.columns
        if c not in ["ROI size", "Screen nodes", "Full edges", "Local edges", "Intersection", "Speedup"]
    ]

    for col in metric_cols:
        paper_df[col] = paper_df[col].astype(float).round(decimals)

    paper_df["Speedup"] = paper_df["Speedup"].astype(float).round(speedup_decimals)

    int_cols = ["ROI size", "Screen nodes", "Full edges", "Local edges", "Intersection"]
    for col in int_cols:
        paper_df[col] = paper_df[col].astype(int)

    if save_csv_path is not None:
        paper_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")

    return paper_df

def recompute_lagged_hypothesis_from_details(
    details,
    lesion_idx,
    hop=1,
    tau=1,
    topk_list=(50, 100, 200),
    save_csv_path=None,
):
    """
    Recompute lesion-centered comparison metrics from existing details.

    This avoids rerunning full LPCMCI.
    Only edge-set extraction and metric computation are repeated.
    """
    rows = []

    for n_roi_small, d in details.items():
        full_res = d["full_res"]
        local_res = d["local_res"]
        roi_subset = d["roi_subset"]

        full_edge_set, full_neighbor_nodes = extract_lesion_edge_set(
            full_res["edge_df"],
            lesion_idx=lesion_idx,
            hop=hop,
            tau_filter=tau,
            directed=True,
        )

        local_edge_set, local_neighbor_nodes = extract_lesion_edge_set(
            local_res["edge_df"],
            lesion_idx=lesion_idx,
            hop=hop,
            tau_filter=tau,
            directed=True,
        )

        cmp_info = _compare_edge_sets(full_edge_set, local_edge_set)

        topk_info = compare_topk_strong_edges(
            full_edge_df=full_res["edge_df"],
            local_edge_df=local_res["edge_df"],
            lesion_idx=lesion_idx,
            hop=hop,
            tau_filter=tau,
            topk_list=topk_list,
        )

        speedup = (
            np.nan
            if local_res["elapsed_sec"] <= 0
            else full_res["elapsed_sec"] / local_res["elapsed_sec"]
        )

        row = {
            "roi_size": int(n_roi_small),

            "full_time_sec": float(full_res["elapsed_sec"]),
            "full_time_min": float(full_res["elapsed_min"]),
            "screen_local_time_sec": float(local_res["elapsed_sec"]),
            "screen_local_time_min": float(local_res["elapsed_min"]),
            "speedup_full_over_local": float(speedup),

            "n_subset_nodes": len(roi_subset),
            "n_screen_nodes": len(local_res["candidate_nodes"]),
            "n_full_all_edges": len(full_res["edge_df"]),
            "n_local_all_edges": len(local_res["edge_df"]),

            "n_full_lesion_neighbors": len(full_neighbor_nodes),
            "n_local_lesion_neighbors": len(local_neighbor_nodes),

            **cmp_info,
            **topk_info,
        }

        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values("roi_size").reset_index(drop=True)

    if save_csv_path is not None:
        result_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")

    return result_df