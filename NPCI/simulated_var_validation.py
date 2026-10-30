# Simulated VAR validation:Full LPCMCI vs Lagged Screening + Local LPCMCI，如果结果：
# Screening + Local LPCMCI:
# lesion_neighbor_recall >= 0.80
# recall >= 0.75
# fpr 较低
# runtime 明显少于 Full LPCMCI
# 就可以说明：在已知真实因果图的 VAR 模拟数据中，滞后筛选 + 局部 LPCMCI 能够在压缩节点规模和降低计算代价的同时，保留大部分真实病灶邻域滞后因果边。
# 因此，滞后筛选作为 Stage 2 的前端压缩策略具有经验合理性。
# 如果全局 recall 不高，但 lesion_neighbor_recall 高，也仍然支持你的主线，因为你的主线不是恢复全图，而是恢复：lesion-centered causal subnetwork

import time
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:
    average_precision_score = None
    roc_auc_score = None


from local_lpcmci import (
    run_candidate_subnetwork_lpcmci,
    lpcmci_to_edge_df,
)


# =========================================================
# 1. VAR data simulation
# =========================================================

def generate_sparse_stable_var_adjacency(
    n_nodes: int,
    lesion_idx: int = 0,
    edge_prob: float = 0.03,
    n_lesion_parents: int = 5,
    n_lesion_children: int = 5,
    n_spouses_per_child: int = 1,
    weight_low: float = 0.15,
    weight_high: float = 0.45,
    spectral_radius: float = 0.75,
    random_state: int = 123,
):
    """
    Generate a sparse stable VAR(1) coefficient matrix A.

    Convention
    ----------
    X(t) = A X(t-1) + eps(t)

    A[dst, src] != 0 means:
        X_src(t-1) -> X_dst(t)

    Returns
    -------
    A:
        shape = (n_nodes, n_nodes)
    info:
        dict
    """
    rng = np.random.default_rng(random_state)

    lesion_idx = int(lesion_idx)
    A = np.zeros((n_nodes, n_nodes), dtype=float)

    # -----------------------------------------------------
    # 1. Random sparse directed edges
    # -----------------------------------------------------
    for dst in range(n_nodes):
        for src in range(n_nodes):
            if src == dst:
                continue
            if rng.random() < edge_prob:
                sign = rng.choice([-1.0, 1.0])
                A[dst, src] = sign * rng.uniform(weight_low, weight_high)

    # -----------------------------------------------------
    # 2. Force lesion-related structure
    # -----------------------------------------------------
    candidates = [i for i in range(n_nodes) if i != lesion_idx]

    n_parent = min(n_lesion_parents, len(candidates))
    parents = rng.choice(candidates, size=n_parent, replace=False).astype(int).tolist()

    remaining = [i for i in candidates if i not in parents]
    n_child = min(n_lesion_children, len(remaining))
    children = rng.choice(remaining, size=n_child, replace=False).astype(int).tolist()

    # parent -> lesion
    for src in parents:
        sign = rng.choice([-1.0, 1.0])
        A[lesion_idx, src] = sign * rng.uniform(weight_low, weight_high)

    # lesion -> child
    for dst in children:
        sign = rng.choice([-1.0, 1.0])
        A[dst, lesion_idx] = sign * rng.uniform(weight_low, weight_high)

    # spouses: some nodes jointly influence lesion's children
    spouses = []
    for child in children:
        available = [
            i for i in range(n_nodes)
            if i not in {lesion_idx, child}
        ]
        if len(available) == 0:
            continue

        n_sp = min(n_spouses_per_child, len(available))
        sp_nodes = rng.choice(available, size=n_sp, replace=False).astype(int).tolist()

        for src in sp_nodes:
            sign = rng.choice([-1.0, 1.0])
            A[child, src] = sign * rng.uniform(weight_low, weight_high)
            spouses.append(src)

    spouses = sorted(set(spouses))

    # -----------------------------------------------------
    # 3. Stabilize spectral radius
    # -----------------------------------------------------
    eig_radius = float(np.max(np.abs(np.linalg.eigvals(A)))) if n_nodes > 0 else 0.0

    if eig_radius > 0:
        A = A * (spectral_radius / max(eig_radius, spectral_radius))

    final_radius = float(np.max(np.abs(np.linalg.eigvals(A)))) if n_nodes > 0 else 0.0

    info = {
        "n_nodes": int(n_nodes),
        "lesion_idx": lesion_idx,
        "n_edges": int(np.sum(np.abs(A) > 0)),
        "n_lesion_parents": len(parents),
        "n_lesion_children": len(children),
        "n_spouses": len(spouses),
        "parents": parents,
        "children": children,
        "spouses": spouses,
        "initial_spectral_radius": eig_radius,
        "final_spectral_radius": final_radius,
    }

    return A, info


def simulate_var1(
    A: np.ndarray,
    T: int = 1200,
    burn_in: int = 200,
    noise_std: float = 1.0,
    random_state: int = 123,
):
    """
    Simulate VAR(1):

        X(t) = A X(t-1) + eps(t)

    Returns
    -------
    X:
        shape = (T, n_nodes)
    """
    rng = np.random.default_rng(random_state)
    A = np.asarray(A, dtype=float)

    n_nodes = A.shape[0]
    total_T = int(T) + int(burn_in)

    X = np.zeros((total_T, n_nodes), dtype=float)

    for t in range(1, total_T):
        eps = rng.normal(loc=0.0, scale=noise_std, size=n_nodes)
        X[t] = A @ X[t - 1] + eps

    return X[burn_in:]


def true_edge_set_from_A(A: np.ndarray, threshold: float = 1e-12, tau: int = 1):
    """
    Convert VAR coefficient matrix A into true directed lagged edge set.

    A[dst, src] != 0 means:
        src -> dst at lag tau

    Returns
    -------
    edge_set:
        set of (src, dst, tau)
    """
    A = np.asarray(A, dtype=float)
    n_nodes = A.shape[0]

    edges = set()
    for dst in range(n_nodes):
        for src in range(n_nodes):
            if src == dst:
                continue
            if abs(A[dst, src]) > threshold:
                edges.add((int(src), int(dst), int(tau)))

    return edges


# =========================================================
# 2. Lagged score matrix and screening
# =========================================================

def _safe_zscore(X, eps=1e-8):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < eps] = 1.0
    return (X - mu) / sd


def compute_lagged_score_matrix(
    signals,
    tau: int = 1,
    standardize: bool = True,
    eps: float = 1e-8,
):
    """
    score[src, dst] = |corr(X_src(t-tau), X_dst(t))|
    """
    X = np.asarray(signals, dtype=float)

    if X.shape[0] <= tau:
        raise ValueError("signals length must be larger than tau.")

    X_lag = X[:-tau]
    X_cur = X[tau:]

    X_lag = np.nan_to_num(X_lag, nan=0.0, posinf=0.0, neginf=0.0)
    X_cur = np.nan_to_num(X_cur, nan=0.0, posinf=0.0, neginf=0.0)

    if standardize:
        X_lag = _safe_zscore(X_lag, eps=eps)
        X_cur = _safe_zscore(X_cur, eps=eps)

    n = X_lag.shape[0]
    signed = (X_lag.T @ X_cur) / max(n - 1, 1)
    signed = np.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)

    score = np.abs(signed)

    np.fill_diagonal(score, 0.0)
    np.fill_diagonal(signed, 0.0)

    return score, signed


def lagged_screen_nodes_for_lesion(
    signals,
    lesion_idx: int,
    max_screen_nodes: int = 30,
    min_screen_nodes: int = 15,
    tau: int = 1,
    top_parents: int = 12,
    top_children: int = 12,
    top_spouses: int = 8,
    standardize: bool = True,

    # 新增：lesion-centered screening 权重
    parent_weight: float = 1.5,
    child_weight: float = 1.5,
    spouse_weight: float = 0.5,

    # 新增：是否强制填充到 max_screen_nodes
    fill_to_max: bool = True,
):
    """
    Simulated-data version of lesion-centered lagged screening.

    This version strengthens lesion-centered screening:

        score_i =
            parent_weight * S_{i -> lesion}
          + child_weight  * S_{lesion -> i}
          + spouse_weight * S_{i -> proxy_child}

    The selected candidate set is filled to max_screen_nodes by default,
    instead of stopping at min_screen_nodes.
    """
    X = np.asarray(signals, dtype=float)
    n_nodes = X.shape[1]
    lesion_idx = int(lesion_idx)

    max_screen_nodes = min(int(max_screen_nodes), int(n_nodes))
    min_screen_nodes = min(int(min_screen_nodes), max_screen_nodes)

    S, signed = compute_lagged_score_matrix(
        X,
        tau=tau,
        standardize=standardize,
    )

    def top_by_score(scores, k, exclude):
        exclude = set(int(x) for x in exclude)
        order = np.argsort(-scores)
        out = []
        for node in order:
            node = int(node)
            if node in exclude:
                continue
            if not np.isfinite(scores[node]):
                continue
            out.append(node)
            if len(out) >= int(k):
                break
        return out

    selected = [lesion_idx]

    # =====================================================
    # 1) Lesion parent proxy: X_i(t-1) -> X_l(t)
    # =====================================================
    parent_scores = S[:, lesion_idx].copy()
    parent_scores[lesion_idx] = -np.inf

    parents = top_by_score(
        parent_scores,
        top_parents,
        exclude=selected,
    )
    selected.extend(parents)

    # =====================================================
    # 2) Lesion child proxy: X_l(t-1) -> X_j(t)
    # =====================================================
    child_scores = S[lesion_idx, :].copy()
    child_scores[lesion_idx] = -np.inf

    children = top_by_score(
        child_scores,
        top_children,
        exclude=selected,
    )
    selected.extend(children)

    # =====================================================
    # 3) Spouse proxy: X_i(t-1) -> proxy_child(t)
    #    用 max_i max_child S[i, child] 表示
    # =====================================================
    spouse_scores = np.zeros(n_nodes, dtype=float)

    if len(children) > 0:
        for node in range(n_nodes):
            if node == lesion_idx:
                spouse_scores[node] = -np.inf
                continue

            vals = []
            for child in children:
                if node == child:
                    continue
                vals.append(float(S[node, child]))

            spouse_scores[node] = max(vals) if len(vals) > 0 else 0.0
    else:
        spouse_scores[:] = 0.0
        spouse_scores[lesion_idx] = -np.inf

    spouses = top_by_score(
        spouse_scores,
        top_spouses,
        exclude=selected,
    )
    selected.extend(spouses)

    # =====================================================
    # 4) Weighted lesion-centered combined score
    # =====================================================
    combined_score = (
        parent_weight * np.nan_to_num(parent_scores, nan=0.0, posinf=0.0, neginf=0.0)
        + child_weight * np.nan_to_num(child_scores, nan=0.0, posinf=0.0, neginf=0.0)
        + spouse_weight * np.nan_to_num(spouse_scores, nan=0.0, posinf=0.0, neginf=0.0)
    )

    combined_score[lesion_idx] = np.inf

    # score dataframe
    rows = []
    for node in range(n_nodes):
        rows.append({
            "node": int(node),
            "parent_proxy": float(S[node, lesion_idx]) if node != lesion_idx else 0.0,
            "child_proxy": float(S[lesion_idx, node]) if node != lesion_idx else 0.0,
            "spouse_proxy": float(spouse_scores[node]) if np.isfinite(spouse_scores[node]) else 0.0,
            "direct_score": float(max(S[node, lesion_idx], S[lesion_idx, node])) if node != lesion_idx else 0.0,
            "combined_score": float(combined_score[node]) if np.isfinite(combined_score[node]) else 0.0,
        })

    score_df = pd.DataFrame(rows)

    # =====================================================
    # 5) 关键修改：填充到 max_screen_nodes，而不是只到 min_screen_nodes
    # =====================================================
    selected = list(dict.fromkeys(int(x) for x in selected))

    if fill_to_max:
        target_nodes = max_screen_nodes
    else:
        target_nodes = min_screen_nodes

    target_nodes = min(max(target_nodes, min_screen_nodes), max_screen_nodes)

    if len(selected) < target_nodes:
        fill_nodes = (
            score_df[~score_df["node"].astype(int).isin(selected)]
            .sort_values("combined_score", ascending=False)
            ["node"].astype(int)
            .head(target_nodes - len(selected))
            .tolist()
        )
        selected.extend(fill_nodes)

    selected = list(dict.fromkeys(int(x) for x in selected))

    # 如果超过 max_screen_nodes，则按 combined_score 截断，保留 lesion
    if len(selected) > max_screen_nodes:
        tmp = score_df[score_df["node"].astype(int).isin(selected)].copy()
        tmp = tmp[tmp["node"].astype(int) != lesion_idx]
        tmp = tmp.sort_values("combined_score", ascending=False)

        selected = [lesion_idx] + tmp["node"].astype(int).head(
            max_screen_nodes - 1
        ).tolist()

    selected = list(dict.fromkeys(int(x) for x in selected))

    # 更新 selected 标记
    score_df["selected"] = score_df["node"].astype(int).isin(selected).astype(int)
    score_df = score_df.sort_values(
        "combined_score",
        ascending=False,
    ).reset_index(drop=True)

    info = {
        "lesion_idx": lesion_idx,
        "n_screen_nodes": len(selected),
        "max_screen_nodes": int(max_screen_nodes),
        "min_screen_nodes": int(min_screen_nodes),
        "screen_ratio_actual": len(selected) / n_nodes,
        "parents_proxy": parents,
        "children_proxy": children,
        "spouses_proxy": spouses,
        "screen_nodes": selected,
        "parent_weight": float(parent_weight),
        "child_weight": float(child_weight),
        "spouse_weight": float(spouse_weight),
        "fill_to_max": bool(fill_to_max),
    }

    return selected, score_df, info


# =========================================================
# 3. LPCMCI runners
# =========================================================

def _run_lpcmci_on_nodes(
    signals,
    candidate_nodes,
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
    Wrapper around your existing run_candidate_subnetwork_lpcmci.
    """
    t0 = time.perf_counter()

    try:
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
    except TypeError:
        # fallback if your function signature is simpler
        lpcmci_res, node_map = run_candidate_subnetwork_lpcmci(
            signals=signals,
            candidate_nodes=candidate_nodes,
            tau_min=tau_min,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            verbosity=verbosity,
            standardize=standardize,
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
        "edge_df": edge_df,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60.0,
        "candidate_nodes": [int(x) for x in candidate_nodes],
        "node_map": node_map,
        "lpcmci_res": lpcmci_res,
    }


def run_full_lpcmci_sim(
    signals,
    tau_min=1,
    tau_max=1,
    pc_alpha=0.01,
    alpha_level=0.01,
    **kwargs,
):
    """
    Full LPCMCI on all simulated nodes.
    """
    n_nodes = np.asarray(signals).shape[1]
    candidate_nodes = list(range(n_nodes))

    res = _run_lpcmci_on_nodes(
        signals=signals,
        candidate_nodes=candidate_nodes,
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        alpha_level=alpha_level,
        **kwargs,
    )

    res["method"] = "full_lpcmci"
    return res


def run_screen_local_lpcmci_sim(
    signals,
    lesion_idx,
    max_screen_nodes=30,
    min_screen_nodes=15,
    tau=1,
    top_parents=12,
    top_children=12,
    top_spouses=8,

    # 新增
    parent_weight=1.5,
    child_weight=1.5,
    spouse_weight=0.5,
    fill_to_max=True,

    tau_min=1,
    tau_max=1,
    pc_alpha=0.01,
    alpha_level=0.01,
    **kwargs,
):
    """
    Lagged screening + local LPCMCI on simulated data.
    """
    t0 = time.perf_counter()

    candidate_nodes, score_df, screen_info = lagged_screen_nodes_for_lesion(
        signals=signals,
        lesion_idx=lesion_idx,
        max_screen_nodes=max_screen_nodes,
        min_screen_nodes=min_screen_nodes,
        tau=tau,
        top_parents=top_parents,
        top_children=top_children,
        top_spouses=top_spouses,
        standardize=True,

        parent_weight=parent_weight,
        child_weight=child_weight,
        spouse_weight=spouse_weight,
        fill_to_max=fill_to_max,
    )

    lpcmci_res = _run_lpcmci_on_nodes(
        signals=signals,
        candidate_nodes=candidate_nodes,
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        alpha_level=alpha_level,
        **kwargs,
    )

    elapsed_total = time.perf_counter() - t0

    lpcmci_res["method"] = "lagged_screening_local_lpcmci"
    lpcmci_res["screen_score_df"] = score_df
    lpcmci_res["screen_info"] = screen_info
    lpcmci_res["elapsed_sec"] = elapsed_total
    lpcmci_res["elapsed_min"] = elapsed_total / 60.0

    return lpcmci_res


# =========================================================
# 4. Edge set and metrics
# =========================================================

def edge_df_to_edge_set(edge_df, tau_filter=1, directed=True):
    """
    Convert LPCMCI edge dataframe to set of edges.
    """
    if edge_df is None or len(edge_df) == 0:
        return set()

    df = edge_df.copy()

    if "tau" in df.columns and tau_filter is not None:
        df = df[df["tau"].astype(int) == int(tau_filter)].copy()

    if len(df) == 0:
        return set()

    edges = set()

    for _, row in df.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])
        tau = int(row["tau"]) if "tau" in df.columns else int(tau_filter)

        if src == dst:
            continue

        if directed:
            edges.add((src, dst, tau))
        else:
            edges.add((min(src, dst), max(src, dst), tau))

    return edges


def edge_df_to_score_map(edge_df, tau_filter=1):
    """
    Convert edge_df to score map for AUPRC.
    score = abs_val if available, else |val|, else -log10(p), else 1.
    """
    score_map = {}

    if edge_df is None or len(edge_df) == 0:
        return score_map

    df = edge_df.copy()

    if "tau" in df.columns and tau_filter is not None:
        df = df[df["tau"].astype(int) == int(tau_filter)].copy()

    if len(df) == 0:
        return score_map

    for _, row in df.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])
        tau = int(row["tau"]) if "tau" in df.columns else int(tau_filter)

        if "abs_val" in row:
            score = float(row["abs_val"])
        elif "val" in row:
            score = abs(float(row["val"]))
        elif "p" in row:
            p = max(float(row["p"]), 1e-300)
            score = float(-np.log10(p))
        else:
            score = 1.0

        key = (src, dst, tau)
        score_map[key] = max(score_map.get(key, 0.0), score)

    return score_map


def extract_lesion_edge_set_from_edges(
    edge_set,
    lesion_idx,
    hop=2,
):
    """
    Extract lesion-centered edge set from a directed edge set.

    BFS is performed on the undirected skeleton.
    Then all directed edges inside the lesion-centered neighborhood are retained.
    """
    lesion_idx = int(lesion_idx)

    edge_set = set(edge_set)
    adj = {}

    for src, dst, tau in edge_set:
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)

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

    lesion_edges = set()
    for src, dst, tau in edge_set:
        if src in visited and dst in visited:
            lesion_edges.add((src, dst, tau))

    return lesion_edges, visited

def compute_candidate_screening_metrics(
    true_edges,
    candidate_nodes,
    n_nodes,
    lesion_idx,
    tau=1,
    lesion_hop=2,
):
    """
    Layer 1: evaluate lagged screening itself.

    This does not evaluate LPCMCI output.
    It only checks whether the candidate nodes selected by lagged screening
    cover the true lesion-centered VAR structure.
    """
    true_edges = set(true_edges)
    candidate_nodes = set(int(x) for x in candidate_nodes)
    all_nodes = set(range(int(n_nodes)))
    lesion_idx = int(lesion_idx)

    true_lesion_edges, true_lesion_nodes = extract_lesion_edge_set_from_edges(
        true_edges,
        lesion_idx=lesion_idx,
        hop=lesion_hop,
    )

    # A true edge is screenable only if both endpoints are retained.
    covered_true_edges = {
        e for e in true_edges
        if e[0] in candidate_nodes and e[1] in candidate_nodes
    }

    covered_true_lesion_edges = {
        e for e in true_lesion_edges
        if e[0] in candidate_nodes and e[1] in candidate_nodes
    }

    global_true_edge_coverage = (
        np.nan
        if len(true_edges) == 0
        else len(covered_true_edges) / len(true_edges)
    )

    lesion_true_edge_coverage = (
        np.nan
        if len(true_lesion_edges) == 0
        else len(covered_true_lesion_edges) / len(true_lesion_edges)
    )

    # Exclude lesion itself because it is forced into the candidate set.
    true_lesion_nodes_no_self = set(true_lesion_nodes) - {lesion_idx}
    covered_true_lesion_nodes = true_lesion_nodes_no_self & candidate_nodes

    lesion_true_node_coverage = (
        np.nan
        if len(true_lesion_nodes_no_self) == 0
        else len(covered_true_lesion_nodes) / len(true_lesion_nodes_no_self)
    )

    # Node-level FPR of screening.
    non_target_nodes = all_nodes - set(true_lesion_nodes)
    selected_non_target_nodes = candidate_nodes & non_target_nodes

    screening_node_fpr = (
        np.nan
        if len(non_target_nodes) == 0
        else len(selected_non_target_nodes) / len(non_target_nodes)
    )

    return {
        "n_candidate_nodes": len(candidate_nodes),
        "candidate_compression_ratio": len(candidate_nodes) / int(n_nodes),

        "n_true_edges_screenable": len(covered_true_edges),
        "global_true_edge_coverage": global_true_edge_coverage,

        "n_true_lesion_edges_screenable": len(covered_true_lesion_edges),
        "lesion_true_edge_coverage": lesion_true_edge_coverage,

        "n_true_lesion_nodes_screened": len(covered_true_lesion_nodes),
        "lesion_true_node_coverage": lesion_true_node_coverage,

        "screening_node_fpr": screening_node_fpr,
    }

def compute_edge_metrics(
    true_edges,
    pred_edges,
    n_nodes,
    lesion_idx,
    tau=1,
    pred_score_map=None,
    lesion_hop=2,
):
    """
    Compute global edge recovery metrics and lesion-neighborhood recall.
    """
    true_edges = set(true_edges)
    pred_edges = set(pred_edges)

    universe = set(
        (src, dst, tau)
        for src in range(n_nodes)
        for dst in range(n_nodes)
        if src != dst
    )

    tp = true_edges & pred_edges
    fp = pred_edges - true_edges
    fn = true_edges - pred_edges
    tn = universe - true_edges - pred_edges

    recall = np.nan if len(true_edges) == 0 else len(tp) / len(true_edges)
    precision = np.nan if len(pred_edges) == 0 else len(tp) / len(pred_edges)
    fpr = np.nan if (len(fp) + len(tn)) == 0 else len(fp) / (len(fp) + len(tn))
    jaccard = np.nan if len(true_edges | pred_edges) == 0 else len(tp) / len(true_edges | pred_edges)
    f1 = np.nan

    if precision == precision and recall == recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    # Lesion-neighborhood recall
    true_lesion_edges, true_lesion_nodes = extract_lesion_edge_set_from_edges(
        true_edges,
        lesion_idx=lesion_idx,
        hop=lesion_hop,
    )

    pred_lesion_edges, pred_lesion_nodes = extract_lesion_edge_set_from_edges(
        pred_edges,
        lesion_idx=lesion_idx,
        hop=lesion_hop,
    )

    lesion_inter = true_lesion_edges & pred_lesion_edges

    lesion_recall = (
        np.nan
        if len(true_lesion_edges) == 0
        else len(lesion_inter) / len(true_lesion_edges)
    )

    lesion_jaccard = (
        np.nan
        if len(true_lesion_edges | pred_lesion_edges) == 0
        else len(lesion_inter) / len(true_lesion_edges | pred_lesion_edges)
    )

    # AUPRC / AUROC
    auprc = np.nan
    auroc = np.nan

    if pred_score_map is not None:
        y_true = []
        y_score = []

        for e in sorted(universe):
            y_true.append(1 if e in true_edges else 0)
            y_score.append(float(pred_score_map.get(e, 0.0)))

        if len(set(y_true)) > 1:
            if average_precision_score is not None:
                auprc = float(average_precision_score(y_true, y_score))

            if roc_auc_score is not None:
                auroc = float(roc_auc_score(y_true, y_score))

    return {
        "n_true_edges": len(true_edges),
        "n_pred_edges": len(pred_edges),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "tn": len(tn),
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "jaccard": jaccard,
        "f1": f1,
        "auprc": auprc,
        "auroc": auroc,
        "n_true_lesion_edges": len(true_lesion_edges),
        "n_pred_lesion_edges": len(pred_lesion_edges),
        "lesion_neighbor_recall": lesion_recall,
        "lesion_neighbor_jaccard": lesion_jaccard,
        "n_true_lesion_nodes": len(true_lesion_nodes),
        "n_pred_lesion_nodes": len(pred_lesion_nodes),
    }


# =========================================================
# 5. Main experiment
# =========================================================

def run_simulated_var_validation_once(
    n_nodes=40,
    T=1200,
    burn_in=200,
    lesion_idx=0,
    edge_prob=0.03,
    n_lesion_parents=5,
    n_lesion_children=5,
    noise_std=1.0,
    random_state=123,
    tau=1,
    pc_alpha=0.01,
    alpha_level=0.01,

    # 修改：默认提高筛选比例
    max_screen_ratio=0.70,
    min_screen_nodes=15,

    lesion_hop=2,
    lpcmci_kwargs=None,

    # 新增：lesion-centered screening 权重
    parent_weight=1.5,
    child_weight=1.5,
    spouse_weight=0.5,
    fill_to_max=True,
):
    """
    One repeat of simulated VAR validation.
    """
    lpcmci_kwargs = lpcmci_kwargs or {}

    A, A_info = generate_sparse_stable_var_adjacency(
        n_nodes=n_nodes,
        lesion_idx=lesion_idx,
        edge_prob=edge_prob,
        n_lesion_parents=n_lesion_parents,
        n_lesion_children=n_lesion_children,
        random_state=random_state,
    )

    signals = simulate_var1(
        A,
        T=T,
        burn_in=burn_in,
        noise_std=noise_std,
        random_state=random_state + 1000,
    )

    true_edges = true_edge_set_from_A(
        A,
        threshold=1e-12,
        tau=tau,
    )

    # Full LPCMCI
    full_res = run_full_lpcmci_sim(
        signals=signals,
        tau_min=tau,
        tau_max=tau,
        pc_alpha=pc_alpha,
        alpha_level=alpha_level,
        **lpcmci_kwargs,
    )

    full_edges = edge_df_to_edge_set(
        full_res["edge_df"],
        tau_filter=tau,
        directed=True,
    )

    full_scores = edge_df_to_score_map(
        full_res["edge_df"],
        tau_filter=tau,
    )

    full_metrics = compute_edge_metrics(
        true_edges=true_edges,
        pred_edges=full_edges,
        n_nodes=n_nodes,
        lesion_idx=lesion_idx,
        tau=tau,
        pred_score_map=full_scores,
        lesion_hop=lesion_hop,
    )

    full_screening_metrics = compute_candidate_screening_metrics(
        true_edges=true_edges,
        candidate_nodes=list(range(n_nodes)),
        n_nodes=n_nodes,
        lesion_idx=lesion_idx,
        tau=tau,
        lesion_hop=lesion_hop,
    )

    # Screening + Local LPCMCI
    max_screen_nodes = max(
        min_screen_nodes,
        int(round(max_screen_ratio * n_nodes)),
    )
    max_screen_nodes = min(max_screen_nodes, n_nodes)

    local_res = run_screen_local_lpcmci_sim(
        signals=signals,
        lesion_idx=lesion_idx,
        max_screen_nodes=max_screen_nodes,
        min_screen_nodes=min_screen_nodes,
        tau=tau,

        # parent / child 给更多预算，spouse 少一点
        top_parents=max(5, int(round(0.35 * max_screen_nodes))),
        top_children=max(5, int(round(0.35 * max_screen_nodes))),
        top_spouses=max(2, int(round(0.15 * max_screen_nodes))),

        # 新增：lesion-centered screening 权重
        parent_weight=parent_weight,
        child_weight=child_weight,
        spouse_weight=spouse_weight,
        fill_to_max=fill_to_max,

        tau_min=tau,
        tau_max=tau,
        pc_alpha=pc_alpha,
        alpha_level=alpha_level,
        **lpcmci_kwargs,
    )

    local_edges = edge_df_to_edge_set(
        local_res["edge_df"],
        tau_filter=tau,
        directed=True,
    )

    local_scores = edge_df_to_score_map(
        local_res["edge_df"],
        tau_filter=tau,
    )

    local_metrics = compute_edge_metrics(
        true_edges=true_edges,
        pred_edges=local_edges,
        n_nodes=n_nodes,
        lesion_idx=lesion_idx,
        tau=tau,
        pred_score_map=local_scores,
        lesion_hop=lesion_hop,
    )

    local_screening_metrics = compute_candidate_screening_metrics(
        true_edges=true_edges,
        candidate_nodes=local_res["candidate_nodes"],
        n_nodes=n_nodes,
        lesion_idx=lesion_idx,
        tau=tau,
        lesion_hop=lesion_hop,
    )

    rows = []

    rows.append({
        "method": "Full LPCMCI",
        "n_nodes": int(n_nodes),
        "random_state": int(random_state),
        "runtime_sec": float(full_res["elapsed_sec"]),
        "runtime_min": float(full_res["elapsed_min"]),
        "n_screen_nodes": int(n_nodes),

        # Layer 1
        **full_screening_metrics,

        # Layer 2
        **full_metrics,
    })

    rows.append({
        "method": "Screening + Local LPCMCI",
        "n_nodes": int(n_nodes),
        "random_state": int(random_state),
        "runtime_sec": float(local_res["elapsed_sec"]),
        "runtime_min": float(local_res["elapsed_min"]),
        "n_screen_nodes": int(len(local_res["candidate_nodes"])),

        # Layer 1: lagged screening itself
        **local_screening_metrics,

        # Layer 2: screening + local LPCMCI edge recovery
        **local_metrics,
    })

    detail = {
        "A": A,
        "A_info": A_info,
        "signals": signals,
        "true_edges": true_edges,
        "full_res": full_res,
        "local_res": local_res,
        "full_edges": full_edges,
        "local_edges": local_edges,
        "full_metrics": full_metrics,
        "local_metrics": local_metrics,
    }

    return pd.DataFrame(rows), detail


def run_simulated_var_validation_experiment(
    n_nodes_list=(40, 60, 80),
    n_repeats=3,
    base_seed=123,
    T=1200,
    burn_in=200,
    lesion_idx=0,
    edge_prob=0.03,
    n_lesion_parents=5,
    n_lesion_children=5,
    noise_std=1.0,
    tau=1,
    pc_alpha=0.01,
    alpha_level=0.01,

    # 修改：默认提高筛选比例
    max_screen_ratio=0.70,
    min_screen_nodes=15,

    lesion_hop=2,
    lpcmci_kwargs=None,
    save_csv_path=None,

    # 新增
    parent_weight=1.5,
    child_weight=1.5,
    spouse_weight=0.5,
    fill_to_max=True,
):
    """
    Run simulated VAR validation across multiple node sizes and repeats.
    """
    all_rows = []
    details = {}

    for n_nodes in n_nodes_list:
        for rep in range(n_repeats):
            seed = int(base_seed + 1000 * n_nodes + rep)

            print("=" * 90)
            print(f"[Simulated VAR validation] n_nodes={n_nodes}, repeat={rep+1}/{n_repeats}, seed={seed}")
            print("=" * 90)

            rows, detail = run_simulated_var_validation_once(
                n_nodes=n_nodes,
                T=T,
                burn_in=burn_in,
                lesion_idx=lesion_idx,
                edge_prob=edge_prob,
                n_lesion_parents=n_lesion_parents,
                n_lesion_children=n_lesion_children,
                noise_std=noise_std,
                random_state=seed,
                tau=tau,
                pc_alpha=pc_alpha,
                alpha_level=alpha_level,
                max_screen_ratio=max_screen_ratio,
                min_screen_nodes=min_screen_nodes,
                lesion_hop=lesion_hop,
                lpcmci_kwargs=lpcmci_kwargs,

                parent_weight=parent_weight,
                child_weight=child_weight,
                spouse_weight=spouse_weight,
                fill_to_max=fill_to_max,
            )

            rows["repeat"] = rep
            all_rows.append(rows)

            details[(int(n_nodes), int(rep))] = detail

            result_df_tmp = pd.concat(all_rows, ignore_index=True)

            if save_csv_path is not None:
                result_df_tmp.to_csv(save_csv_path, index=False)

    result_df = pd.concat(all_rows, ignore_index=True)

    if save_csv_path is not None:
        result_df.to_csv(save_csv_path, index=False)

    return result_df, details


def summarize_simulated_var_results(result_df):
    """
    Summarize simulation results by n_nodes and method.
    """
    metrics = [
        "runtime_min",
        "n_screen_nodes",

        # Layer 1: screening itself
        "candidate_compression_ratio",
        "global_true_edge_coverage",
        "lesion_true_edge_coverage",
        "lesion_true_node_coverage",
        "screening_node_fpr",

        # Layer 2: final edge recovery
        "n_true_edges",
        "n_pred_edges",
        "recall",
        "precision",
        "fpr",
        "jaccard",
        "f1",
        "auprc",
        "auroc",
        "lesion_neighbor_recall",
        "lesion_neighbor_jaccard",
    ]

    summary = (
        result_df
        .groupby(["n_nodes", "method"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    # flatten columns
    summary.columns = [
        "_".join([str(x) for x in col if x != ""])
        for col in summary.columns.values
    ]

    return summary


def judge_simulated_var_validation(
    summary_df,
    method_name="Screening + Local LPCMCI",
    lesion_candidate_coverage_threshold=0.80,
    lesion_recall_threshold=0.80,
    fpr_threshold=0.15,
    auroc_threshold=0.70,
):
    """
    Two-layer judgement for simulated VAR validation.

    Layer 1:
        Whether lagged screening keeps enough true lesion-neighborhood edges.

    Layer 2:
        Whether screening + local LPCMCI recovers true VAR edges.
    """
    df = summary_df.copy()
    sub = df[df["method"] == method_name].copy()

    if len(sub) == 0:
        return {
            "judgement": "method_not_found",
            "method": method_name,
        }

    mean_lesion_candidate_coverage = float(
        sub["lesion_true_edge_coverage_mean"].mean()
    )
    mean_lesion_recall = float(
        sub["lesion_neighbor_recall_mean"].mean()
    )
    mean_fpr = float(sub["fpr_mean"].mean())

    if "auroc_mean" in sub.columns:
        mean_auroc = float(sub["auroc_mean"].mean())
    else:
        mean_auroc = np.nan

    pass_layer1 = (
        mean_lesion_candidate_coverage >= lesion_candidate_coverage_threshold
    )
    pass_lesion_recall = mean_lesion_recall >= lesion_recall_threshold
    pass_fpr = mean_fpr <= fpr_threshold

    pass_auroc = (
        mean_auroc >= auroc_threshold
        if mean_auroc == mean_auroc
        else False
    )

    if pass_layer1 and pass_lesion_recall and pass_fpr:
        judgement = "supported"
    elif pass_layer1 and pass_lesion_recall:
        judgement = "partially_supported_fpr_or_auroc_weak"
    elif pass_layer1:
        judgement = "partially_supported_screening_good_lpcmci_recovery_weak"
    else:
        judgement = "not_supported_screening_coverage_insufficient"

    return {
        "method": method_name,

        "mean_lesion_true_edge_coverage": mean_lesion_candidate_coverage,
        "mean_lesion_neighbor_recall": mean_lesion_recall,
        "mean_fpr": mean_fpr,
        "mean_auroc": mean_auroc,

        "pass_layer1_screening_coverage": bool(pass_layer1),
        "pass_lesion_recall": bool(pass_lesion_recall),
        "pass_fpr": bool(pass_fpr),
        "pass_auroc": bool(pass_auroc),

        "judgement": judgement,
    }

def build_simulated_var_two_layer_tables(
    result_df,
    decimals=3,
    speedup_decimals=2,
    save_prefix=None,
):
    """
    Build two paper-ready tables.

    Table 1:
        Lagged screening candidate coverage.

    Table 2:
        Final edge recovery after screening + local LPCMCI.
    """
    df = result_df.copy()

    full_runtime = (
        df[df["method"] == "Full LPCMCI"]
        .set_index(["n_nodes", "random_state"])["runtime_sec"]
    )

    df["speedup_full_over_method"] = np.nan

    for idx, row in df.iterrows():
        key = (row["n_nodes"], row["random_state"])
        if key in full_runtime.index and row["runtime_sec"] > 0:
            df.loc[idx, "speedup_full_over_method"] = (
                float(full_runtime.loc[key]) / float(row["runtime_sec"])
            )

    local = df[df["method"] == "Screening + Local LPCMCI"].copy()

    layer1 = (
        local
        .groupby("n_nodes")[
            [
                "n_screen_nodes",
                "candidate_compression_ratio",
                "lesion_true_node_coverage",
                "lesion_true_edge_coverage",
                "global_true_edge_coverage",
                "screening_node_fpr",
            ]
        ]
        .mean()
        .reset_index()
    )

    layer2 = (
        local
        .groupby("n_nodes")[
            [
                "n_pred_edges",
                "recall",
                "precision",
                "fpr",
                "auprc",
                "auroc",
                "lesion_neighbor_recall",
                "lesion_neighbor_jaccard",
                "speedup_full_over_method",
            ]
        ]
        .mean()
        .reset_index()
    )

    layer1 = layer1.rename(columns={
        "n_nodes": "Nodes",
        "n_screen_nodes": "Screen nodes",
        "candidate_compression_ratio": "Compression ratio",
        "lesion_true_node_coverage": "Lesion node coverage",
        "lesion_true_edge_coverage": "Lesion edge coverage",
        "global_true_edge_coverage": "Global edge coverage",
        "screening_node_fpr": "Screening node FPR",
    })

    layer2 = layer2.rename(columns={
        "n_nodes": "Nodes",
        "n_pred_edges": "Pred edges",
        "recall": "Recall",
        "precision": "Precision",
        "fpr": "FPR",
        "auprc": "AUPRC",
        "auroc": "AUROC",
        "lesion_neighbor_recall": "Lesion recall",
        "lesion_neighbor_jaccard": "Lesion Jaccard",
        "speedup_full_over_method": "Speedup",
    })

    for table in [layer1, layer2]:
        for col in table.columns:
            if col in ["Nodes", "Screen nodes"]:
                table[col] = table[col].round(0).astype(int)
            elif col == "Pred edges":
                table[col] = table[col].round(1)
            elif col == "Speedup":
                table[col] = table[col].round(speedup_decimals)
            elif col != "Nodes":
                table[col] = table[col].astype(float).round(decimals)

    if save_prefix is not None:
        layer1.to_csv(f"{save_prefix}_layer1_screening.csv", index=False)
        layer2.to_csv(f"{save_prefix}_layer2_recovery.csv", index=False)

    return layer1, layer2

def build_screen_ratio_comparison_table(
    result_dict,
    decimals=3,
    speedup_decimals=2,
    save_csv_path=None,
):
    """
    Build comparison table across different max_screen_ratio settings.

    Parameters
    ----------
    result_dict:
        dict, e.g.
        {
            "ratio=0.55": sim_result_055,
            "ratio=0.70": sim_result_070,
            "ratio=0.75": sim_result_075,
        }
    """
    rows = []

    for config_name, df in result_dict.items():
        local = df[df["method"] == "Screening + Local LPCMCI"].copy()
        full = df[df["method"] == "Full LPCMCI"].copy()

        if len(local) == 0:
            continue

        # compute speedup from paired rows
        speedups = []
        for _, row in local.iterrows():
            sub_full = full[
                (full["n_nodes"] == row["n_nodes"])
                & (full["random_state"] == row["random_state"])
            ]
            if len(sub_full) > 0 and row["runtime_sec"] > 0:
                speedups.append(
                    float(sub_full.iloc[0]["runtime_sec"]) / float(row["runtime_sec"])
                )

        rows.append({
            "Config": config_name,
            "Screen nodes": local["n_screen_nodes"].mean(),
            "Compression ratio": local["candidate_compression_ratio"].mean(),

            # Layer 1
            "Lesion node coverage": local["lesion_true_node_coverage"].mean(),
            "Lesion edge coverage": local["lesion_true_edge_coverage"].mean(),
            "Global edge coverage": local["global_true_edge_coverage"].mean(),
            "Screening node FPR": local["screening_node_fpr"].mean(),

            # Layer 2
            "Recall": local["recall"].mean(),
            "Precision": local["precision"].mean(),
            "FPR": local["fpr"].mean(),
            "AUPRC": local["auprc"].mean(),
            "AUROC": local["auroc"].mean() if "auroc" in local.columns else np.nan,
            "Lesion recall": local["lesion_neighbor_recall"].mean(),
            "Lesion Jaccard": local["lesion_neighbor_jaccard"].mean(),

            "Speedup": np.mean(speedups) if len(speedups) > 0 else np.nan,
        })

    out = pd.DataFrame(rows)

    int_cols = ["Screen nodes"]
    for col in int_cols:
        if col in out.columns:
            out[col] = out[col].round(1)

    metric_cols = [
        c for c in out.columns
        if c not in ["Config", "Screen nodes", "Speedup"]
    ]
    for col in metric_cols:
        out[col] = out[col].astype(float).round(decimals)

    if "Speedup" in out.columns:
        out["Speedup"] = out["Speedup"].astype(float).round(speedup_decimals)

    if save_csv_path is not None:
        out.to_csv(save_csv_path, index=False, encoding="utf-8-sig")

    return out