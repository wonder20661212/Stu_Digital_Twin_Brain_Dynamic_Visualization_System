"""
New main line:
    Stage 1 lesion response provides lesion-related dynamic abnormality prior.
    Stage 2.1 whole-brain lagged screening keeps potential time-lagged influences.
    Stage 2.2 LPCMCI is run only on the screened lesion-centered candidate subnetwork.
    Stage 2.3 FC-constrained pruning only compresses dense non-protected edges.

Important design rules:
    1. Use lagged edge Xi(t-1) -> Xj(t) as the main causal layer.
    2. Screening is a high-recall candidate generator, not the final causal discovery.
    3. LPCMCI is the formal causal discovery step.
    4. FC is only used to compress excessive edges and is not allowed to reject
       direct lesion edges or strong significant LPCMCI edges.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd



# ArrayLike = Union[np.ndarray, Sequence[float]]


def normalize01(x: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    """把一个分数向量做 min-max 归一化到 [0, 1]."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _standardize_columns(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """对每个 ROI 的时间序列做 z-score 标准化."""
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < eps] = 1.0
    return (X - mu) / sd


def _is_already_standardized(
    X: np.ndarray,
    mean_tol: float = 0.10,
    std_tol: float = 0.10,
) -> bool:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)

    return (
        np.all(np.abs(mu) < mean_tol) and
        np.all(np.abs(sd - 1.0) < std_tol)
    )


def _prepare_signals(
    signals: np.ndarray,
    standardize: Union[bool, str] = "auto",
) -> np.ndarray:
    """
    standardize:
        True   : 强制 z-score
        False  : 不做 z-score，只清理 NaN/inf 和检查维度
        "auto" : 检查是否已标准化；未标准化才 z-score
    """
    X = np.asarray(signals, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if X.ndim != 2:
        raise ValueError(f"signals must be 2-D with shape (T, ROI_num), got {X.shape}")

    if standardize == "auto":
        if not _is_already_standardized(X):
            X = _standardize_columns(X)
    elif standardize is True:
        X = _standardize_columns(X)
    elif standardize is False:
        pass
    else:
        raise ValueError("standardize must be True, False, or 'auto'")

    return X


def compute_lagged_score_matrix(
    signals: np.ndarray,
    tau: int = 1,
    standardize: bool = True,
    exclude_self: bool = True,
) -> np.ndarray:
    """
    Compute whole-brain lagged association scores for Xi(t-tau) -> Xj(t).
    计算全脑所有 ROI 对之间的滞后相关分数
    Score is absolute lagged Pearson correlation:
        S_ij = |corr(X_i(t-tau), X_j(t))|.
    它衡量第 i 个 ROI 在过去时刻的活动，和第 j 个 ROI 当前时刻的活动有多相关。
    This is a fast high-recall screening statistic. It is not used as the final
    causal evidence; LPCMCI is run afterwards on the screened subnetwork.
    Returns
    score_matrix:
        shape (ROI_num, ROI_num), where score_matrix[src, dst] is the score for
        ROI_src(t-tau) -> ROI_dst(t).
        表示：ROI_src(t - tau) 和 ROI_dst(t) 的绝对滞后相关，也就是：src 过去的变化，是否能对应 dst 当前的变化
    """
    if tau < 1:
        raise ValueError("tau must be >= 1 for lagged candidate screening")

    X = _prepare_signals(signals, standardize=standardize)
    T, n_roi = X.shape
    if T <= tau + 2:
        raise ValueError(f"time length T={T} is too short for tau={tau}")

    X_past = X[:-tau, :]
    X_future = X[tau:, :]

    # Re-standardize the aligned blocks to get correlation by matrix product.
    X_past = _standardize_columns(X_past)
    X_future = _standardize_columns(X_future)

    denom = max(X_past.shape[0], 1)
    corr = (X_past.T @ X_future) / denom
    score = np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))

    if exclude_self:
        np.fill_diagonal(score, 0.0)

    return score


def lagged_score_matrix_to_edge_df(
    score_matrix: np.ndarray,
    tau: int = 1,
    top_k_edges: Optional[int] = None,
    min_score: Optional[float] = None,
) -> pd.DataFrame:
    """把矩阵形式的 score_matrix 转成边表."""
    S = np.asarray(score_matrix, dtype=float)
    n = S.shape[0]
    rows: List[Dict] = []

    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            score = float(S[src, dst])
            if min_score is not None and score < float(min_score):
                continue
            rows.append({
                "src": int(src),
                "dst": int(dst),
                "tau": int(tau),  # 滞后阶数
                "screen_score": score,   # 滞后相关分数
            })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("screen_score", ascending=False).reset_index(drop=True)
        if top_k_edges is not None:
            df = df.head(int(top_k_edges)).reset_index(drop=True)
    return df


def _extract_response_prior(
    lesion_response_prior: Optional[Union[Dict, pd.DataFrame, np.ndarray]],
    roi_num: int,
) -> np.ndarray:
    """
    stage 1 先验融合部分：把 Stage 1 的 lesion response 输出转换成一个长度为 ROI_num 的向量。.
    Accepted inputs:
        - stage1_res dict containing "response_vector" or "lesion_response_df";
        - DataFrame with columns ["roi", "response_abs"];
        - 1-D numpy array/list with length ROI_num;
        - None, returning zeros.
    """
    if lesion_response_prior is None:
        return np.zeros(int(roi_num), dtype=float)

    if isinstance(lesion_response_prior, dict):
        if "response_vector" in lesion_response_prior:
            v = np.asarray(lesion_response_prior["response_vector"], dtype=float)
            if v.shape[0] != roi_num:
                raise ValueError("response_vector length does not match ROI_num")
            return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if "lesion_response_df" in lesion_response_prior:
            lesion_response_prior = lesion_response_prior["lesion_response_df"]
        else:
            return np.zeros(int(roi_num), dtype=float)

    if isinstance(lesion_response_prior, pd.DataFrame):
        if not {"roi", "response_abs"}.issubset(set(lesion_response_prior.columns)):
            raise ValueError("lesion_response_df must contain columns ['roi', 'response_abs']")
        v = np.zeros(int(roi_num), dtype=float)
        for _, row in lesion_response_prior.iterrows():
            roi = int(row["roi"])
            if 0 <= roi < roi_num:
                v[roi] = float(row["response_abs"])
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    # 输出每个 ROI 对当前 lesion 的动态响应强度，是 Stage 1 和 Stage 2 的关键接口
    v = np.asarray(lesion_response_prior, dtype=float)
    if v.ndim != 1 or v.shape[0] != roi_num:
        raise ValueError("lesion_response_prior array must be 1-D with length ROI_num")
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def run_lagged_candidate_screening(
    signals: np.ndarray,
    lesion_idx: int,
    lesion_response_prior: Optional[Union[Dict, pd.DataFrame, np.ndarray]] = None,
    empirical_FC: Optional[np.ndarray] = None,
    roi_names: Optional[Sequence[str]] = None,
    tau: int = 1,
    max_candidate_nodes: int = 80,
    min_candidate_nodes: int = 30,
    top_k_direct_lagged: int = 40,
    top_k_response: int = 30,
    top_k_fc: int = 0,
    w_lagged: float = 1.0,
    w_response: float = 0.6,
    w_fc: float = 0.0,
    standardize: Union[bool, str] = "auto",
) -> Dict:
    """
    从全脑 360 个 ROI 里筛出一个 lesion-centered 候选节点集合，供后续 LPCMCI 使用。
    不跑 LPCMCI，只负责从全脑中筛出候选节点
    Candidate node score:
        final_score(node)
            = w_lagged  * max(S_lesion,node, S_node,lesion)
            + w_response * lesion_response_prior(node)
            + w_fc       * |FC_lesion,node|.
    In the recommended main pipeline, set w_fc=0 and top_k_fc=0 so that FC is
    used only in the final pruning stage. The FC option is kept for ablation or
    emergency fallback, but is disabled by default.

    The final candidate set is the union of:
        - lesion node;
        - top nodes by direct lagged relation with lesion;
        - top nodes by Stage 1 lesion response prior;
        - top nodes by FC with lesion;
        - top nodes by final combined score until min_candidate_nodes is reached;
      then truncated to max_candidate_nodes by final_score.
    Returns
    dict with:
        score_matrix, screening_edge_df, node_score_df, candidate_nodes
    """
    X = _prepare_signals(signals, standardize=standardize)
    roi_num = X.shape[1]
    lesion_idx = int(lesion_idx)
    if not (0 <= lesion_idx < roi_num):
        raise ValueError(f"lesion_idx={lesion_idx} out of range for ROI_num={roi_num}")

    S = compute_lagged_score_matrix(X, tau=tau, standardize=False, exclude_self=True)

    # --------------------------------------------------
    # Lagged parent / child proxy around lesion
    # Convention:
    #   S[src, dst] = score of X_src(t-tau) -> X_dst(t)
    # --------------------------------------------------
    parent_proxy = S[:, lesion_idx].copy()  # X_node(t-1) -> X_lesion(t)
    child_proxy = S[lesion_idx, :].copy()  # X_lesion(t-1) -> X_node(t)

    parent_proxy[lesion_idx] = 0.0
    child_proxy[lesion_idx] = 0.0

    # direct lagged relation with lesion
    lag_direct = np.maximum(parent_proxy, child_proxy)
    lag_direct[lesion_idx] = 0.0

    response_prior = _extract_response_prior(lesion_response_prior, roi_num)
    response_prior = np.asarray(response_prior, dtype=float)
    response_prior[lesion_idx] = 0.0

    if empirical_FC is None:
        fc_lesion = np.zeros(roi_num, dtype=float)
    else:
        FC = np.asarray(empirical_FC, dtype=float)
        if FC.shape[0] != roi_num or FC.shape[1] != roi_num:
            raise ValueError("empirical_FC shape must be (ROI_num, ROI_num)")
        fc_lesion = np.abs(np.nan_to_num(FC[lesion_idx, :], nan=0.0, posinf=0.0, neginf=0.0))
        fc_lesion[lesion_idx] = 0.0

    lag_norm = normalize01(lag_direct)
    parent_proxy_norm = normalize01(parent_proxy)
    child_proxy_norm = normalize01(child_proxy)

    resp_norm = normalize01(response_prior)
    fc_norm = normalize01(fc_lesion)

    final_score = w_lagged * lag_norm + w_response * resp_norm + w_fc * fc_norm
    final_score[lesion_idx] = np.inf

    def top_indices(values: np.ndarray, k: int) -> List[int]:
        order = np.argsort(-np.asarray(values, dtype=float))
        return [int(i) for i in order if i != lesion_idx][: max(int(k), 0)]

    candidate_set: Set[int] = {lesion_idx}
    candidate_set |= set(top_indices(lag_direct, top_k_direct_lagged))
    candidate_set |= set(top_indices(response_prior, top_k_response))
    candidate_set |= set(top_indices(fc_lesion, top_k_fc))

    # Fill up to min_candidate_nodes by the combined screening score.
    for idx in top_indices(final_score, roi_num):
        if len(candidate_set) >= int(min_candidate_nodes):
            break
        candidate_set.add(int(idx))

    # Truncate to max_candidate_nodes by final score, lesion always kept.
    if len(candidate_set) > int(max_candidate_nodes):
        ranked = sorted(candidate_set, key=lambda i: final_score[i], reverse=True)
        keep = ranked[: int(max_candidate_nodes)]
        if lesion_idx not in keep:
            keep[-1] = lesion_idx
        candidate_set = set(keep)

    # Build node score table.
    candidate_nodes = sorted(candidate_set, key=lambda i: final_score[i], reverse=True)
    if lesion_idx in candidate_nodes:
        candidate_nodes.remove(lesion_idx)
    candidate_nodes = [lesion_idx] + candidate_nodes

    node_score_df = pd.DataFrame({
        "node": np.arange(roi_num, dtype=int),

        # --------------------------------------------------
        # Lagged screening scores
        # --------------------------------------------------
        "parent_proxy": parent_proxy,  # X_node(t-1) -> X_lesion(t)
        "child_proxy": child_proxy,  # X_lesion(t-1) -> X_node(t)
        "parent_proxy_norm": parent_proxy_norm,
        "child_proxy_norm": child_proxy_norm,

        "lagged_direct_score": lag_direct,
        "lagged_direct_norm": lag_norm,

        # --------------------------------------------------
        # Stage 1 prior / FC auxiliary scores
        # --------------------------------------------------
        "lesion_response_prior": response_prior,
        "lesion_response_norm": resp_norm,
        "fc_lesion_abs": fc_lesion,
        "fc_norm": fc_norm,

        # --------------------------------------------------
        # Final screening score
        # --------------------------------------------------
        "final_screen_score": np.where(
            np.isinf(final_score),
            np.nanmax(final_score[np.isfinite(final_score)]) + 1.0
            if np.any(np.isfinite(final_score))
            else 1.0,
            final_score
        ),
    })
    node_score_df["selected"] = node_score_df["node"].isin(candidate_set)

    def _lagged_role(row):
        node = int(row["node"])

        if node == lesion_idx:
            return "lesion node"

        p = float(row["parent_proxy"])
        c = float(row["child_proxy"])

        if p > 0 and c > 0:
            ratio = min(p, c) / (max(p, c) + 1e-12)
            if ratio >= 0.65:
                return "mixed lagged proxy"

        if p >= c:
            return "lagged parent proxy"
        else:
            return "lagged child proxy"

    node_score_df["lagged_role"] = node_score_df.apply(_lagged_role, axis=1)
    if roi_names is not None:
        node_score_df.insert(
            node_score_df.columns.get_loc("node") + 1,
            "roi_name",
            node_score_df["node"].astype(int).map(
                lambda i: roi_names[int(i)] if int(i) < len(roi_names) else f"ROI_{int(i)}"
            )
        )
    node_score_df = node_score_df.sort_values(
        ["selected", "final_screen_score"], ascending=[False, False]
    ).reset_index(drop=True)

    screening_edge_df = lagged_score_matrix_to_edge_df(S, tau=tau)

    info = {
        "lesion_idx": lesion_idx,
        "roi_num": roi_num,
        "tau": tau,
        "n_candidate_nodes": len(candidate_nodes),
        "max_candidate_nodes": int(max_candidate_nodes),
        "min_candidate_nodes": int(min_candidate_nodes),
        "screening_role": "high-recall candidate generation before LPCMCI",
    }

    return {
        "score_matrix": S,
        "screening_edge_df": screening_edge_df,
        "node_score_df": node_score_df,
        "candidate_nodes": candidate_nodes,
        "info": info,
    }

def compute_stage1_screening_overlap(
    stage1_res,
    screen_res,
    stage1_top_k: int = 30,
    screening_top_k: int = 30,
    stage1_score_col: str = None,
    screening_score_col: str = "final_screen_score",
):
    """
    Compute overlap between Stage 1 top lesion-response nodes
    and Stage 2 lagged-screening selected nodes.

    Returns
    -------
    overlap_info : dict
    """
    # 1. Stage 1 top response nodes
    if isinstance(stage1_res, dict):
        if "lesion_response_df" in stage1_res:
            stage1_df = stage1_res["lesion_response_df"].copy()
        elif "response_df" in stage1_res:
            stage1_df = stage1_res["response_df"].copy()
        else:
            raise KeyError("stage1_res must contain 'lesion_response_df' or 'response_df'.")
    elif isinstance(stage1_res, pd.DataFrame):
        stage1_df = stage1_res.copy()
    else:
        raise TypeError("stage1_res must be dict or pd.DataFrame.")

    if "node" not in stage1_df.columns:
        if "roi" in stage1_df.columns:
            stage1_df = stage1_df.rename(columns={"roi": "node"})
        else:
            raise KeyError("Stage 1 dataframe must contain 'node' or 'roi' column.")

    if stage1_score_col is None:
        for c in [
            "response_abs",
            "lesion_response",
            "abnormality_score",
            "spread_score",
            "damage_score",
            "candidate_score",
            "score",
        ]:
            if c in stage1_df.columns:
                stage1_score_col = c
                break

    if stage1_score_col is None:
        raise KeyError(
            "Cannot find Stage 1 score column. "
            "Expected one of response_abs / lesion_response / abnormality_score / score."
        )

    stage1_top_nodes = (
        stage1_df
        .sort_values(stage1_score_col, ascending=False)
        ["node"]
        .astype(int)
        .head(int(stage1_top_k))
        .tolist()
    )

    stage1_set = set(stage1_top_nodes)

    # --------------------------------------------------
    # 2. Stage 2 screening selected nodes
    # --------------------------------------------------
    if isinstance(screen_res, dict):
        if "node_score_df" in screen_res:
            screen_df = screen_res["node_score_df"].copy()
        else:
            raise KeyError("screen_res must contain 'node_score_df'.")
    elif isinstance(screen_res, pd.DataFrame):
        screen_df = screen_res.copy()
    else:
        raise TypeError("screen_res must be dict or pd.DataFrame.")

    if "node" not in screen_df.columns:
        if "roi" in screen_df.columns:
            screen_df = screen_df.rename(columns={"roi": "node"})
        else:
            raise KeyError("Screening dataframe must contain 'node' or 'roi' column.")

    if screening_score_col not in screen_df.columns:
        for c in [
            "candidate_score",
            "screening_score",
            "final_score",
            "score",
            "lagged_direct_score",
        ]:
            if c in screen_df.columns:
                screening_score_col = c
                break

    if screening_score_col not in screen_df.columns:
        raise KeyError("Cannot find screening score column.")

    if "selected" in screen_df.columns:
        screen_df = screen_df[screen_df["selected"].astype(bool)].copy()

    screening_top_nodes = (
        screen_df
        .sort_values(screening_score_col, ascending=False)
        ["node"]
        .astype(int)
        .head(int(screening_top_k))
        .tolist()
    )

    screening_set = set(screening_top_nodes)

    # --------------------------------------------------
    # 3. Overlap statistics
    # --------------------------------------------------
    overlap_set = stage1_set & screening_set
    union_set = stage1_set | screening_set

    n_stage1 = len(stage1_set)
    n_screening = len(screening_set)
    n_overlap = len(overlap_set)
    n_union = len(union_set)

    stage1_recall = n_overlap / n_stage1 if n_stage1 > 0 else np.nan
    screening_precision = n_overlap / n_screening if n_screening > 0 else np.nan
    jaccard = n_overlap / n_union if n_union > 0 else np.nan

    if stage1_recall >= 0.80:
        judgement = "good"
    elif stage1_recall >= 0.50:
        judgement = "moderate"
    else:
        judgement = "weak"

    overlap_info = {
        "stage1_top_k": int(stage1_top_k),
        "screening_top_k": int(screening_top_k),

        "stage1_nodes": sorted(stage1_set),
        "screening_nodes": sorted(screening_set),
        "overlap_nodes": sorted(overlap_set),

        "n_stage1_nodes": n_stage1,
        "n_screening_nodes": n_screening,
        "n_overlap_nodes": n_overlap,
        "n_union_nodes": n_union,

        "stage1_screen_overlap_topK": stage1_recall,
        "screening_overlap_precision": screening_precision,
        "jaccard": jaccard,
        "overlap_judgement": judgement,

        "stage1_score_col": stage1_score_col,
        "screening_score_col": screening_score_col,
    }

    return overlap_info


def run_candidate_subnetwork_lpcmci(
    signals: np.ndarray,
    candidate_nodes: Sequence[int],
    tau_min: int = 1,
    tau_max: int = 1,
    pc_alpha: float = 0.05,
    verbosity: int = 0,
    standardize: Union[bool, str] = "auto",
    max_cond_px: int = 1,
    max_p_global: int = 2,
    max_p_non_ancestral: int = 2,
    max_q_global: int = 5,
    max_pds_set: int = 8,
) -> Tuple[Dict, List[int]]:
    """
    Run LPCMCI on the screened candidate subnetwork, not on all 360 ROIs.

    Parameters are intentionally lighter than full whole-brain LPCMCI. Increase
    them only after the main pipeline has been verified.

    Returns
    -------
    results, node_map
        node_map[local_idx] = global ROI index.
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.lpcmci import LPCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except ImportError as exc:
        raise ImportError("Please install tigramite first: pip install tigramite") from exc

    X_full = _prepare_signals(signals, standardize=standardize)
    roi_num = X_full.shape[1]

    node_map = [int(x) for x in candidate_nodes]
    node_map = list(dict.fromkeys(node_map))  # preserve order and remove duplicates
    if len(node_map) < 2:
        raise ValueError("candidate_nodes must contain at least two unique nodes")
    if min(node_map) < 0 or max(node_map) >= roi_num:
        raise ValueError("candidate_nodes contains ROI index outside signals.shape[1]")

    X = X_full[:, node_map]
    var_names = [f"ROI_{idx}" for idx in node_map]

    dataframe = pp.DataFrame(X, var_names=var_names)
    cond_ind_test = ParCorr(significance="analytic")
    lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=verbosity)

    print(f"[Stage2] Running LPCMCI on screened subnetwork: n_nodes={len(node_map)}, tau_max={tau_max}")

    results = lpcmci.run_lpcmci(
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        max_cond_px=max_cond_px,
        max_p_global=max_p_global,
        max_p_non_ancestral=max_p_non_ancestral,
        max_q_global=max_q_global,
        max_pds_set=max_pds_set,
    )
    return results, node_map


def lpcmci_to_edge_df(
    results: Dict,
    node_map: Optional[Sequence[int]] = None,
    include_contemp: bool = False,
    lag_only: bool = True,
    alpha_level: Optional[float] = None,
) -> pd.DataFrame:
    """
    Convert LPCMCI graph, val_matrix and p_matrix to an edge table.
    把 LPCMCI 的三维结果矩阵转成普通 DataFrame 边表。
    Interpretation:
        graph[i, j, tau] corresponds to ROI_i(t-tau) -> ROI_j(t).
        因为后面所有操作都更适合用 DataFrame：提取 lesion 邻域、计算 edge density、FC pruning、构建 adjacency、生成 intervention candidates、画网络图
    If node_map is supplied, local LPCMCI indices are mapped back to global ROI
    indices. This is the expected use for screened subnetwork LPCMCI.
    """
    graph = results["graph"]
    val_matrix = results["val_matrix"]
    p_matrix = results["p_matrix"]

    n_roi = graph.shape[0]
    tau_dim = graph.shape[2]

    if node_map is None:
        node_map = list(range(n_roi))
    node_map = [int(x) for x in node_map]
    if len(node_map) != n_roi:
        raise ValueError("len(node_map) must match LPCMCI graph.shape[0]")

    rows: List[Dict] = []
    for src_local in range(n_roi):
        for dst_local in range(n_roi):
            if src_local == dst_local:
                continue
            for tau in range(tau_dim):
                if tau == 0 and not include_contemp:
                    continue
                # 默认只保留滞后边：Xi(t-1) → Xj(t)，不保留同时刻边：Xi(t) — Xj(t)，当前主线以 lagged causal edge 作为主要因果层。
                if lag_only and tau < 1:
                    continue

                link = str(graph[src_local, dst_local, tau]).strip()
                if link == "":
                    continue

                val = float(val_matrix[src_local, dst_local, tau])
                p = float(p_matrix[src_local, dst_local, tau])
                if alpha_level is not None and p > float(alpha_level):
                    continue

                rows.append({
                    "src": int(node_map[src_local]),
                    "dst": int(node_map[dst_local]),
                    "src_local": int(src_local),
                    "dst_local": int(dst_local),
                    "tau": int(tau),
                    "link": link,
                    "val": val,
                    "abs_val": abs(val),
                    "p": p,
                })

    edge_df = pd.DataFrame(rows)
    if len(edge_df) > 0:
        edge_df = edge_df.sort_values(["abs_val", "p"], ascending=[False, True]).reset_index(drop=True)
    return edge_df


def _neighbors(edge_df: pd.DataFrame, nodes: Set[int], direction: str = "both") -> Set[int]:
    """
    One-hop neighbors from an edge table.
    从边表里找某些节点的一跳邻居
    """
    out: Set[int] = set()
    if len(edge_df) == 0 or len(nodes) == 0:
        return out
    if direction in {"out", "both"}:
        out |= set(edge_df.loc[edge_df["src"].isin(nodes), "dst"].astype(int).tolist())
    if direction in {"in", "both"}:
        out |= set(edge_df.loc[edge_df["dst"].isin(nodes), "src"].astype(int).tolist())
    return out


def extract_lesion_neighborhood(
    edge_df: pd.DataFrame,
    lesion_idx: int,
    min_nodes: int = 8,
    max_nodes: int = 30,
    direction: str = "both",
    expand_to_second_order: bool = True,
) -> Tuple[List[int], pd.DataFrame, Dict]:
    """
    Extract lesion-centered first-order, optionally second-order, LPCMCI subnetwork.
    从 LPCMCI 的边表中提取 lesion-centered 子图，也就是：以 lesion_idx 为中心，先找一阶邻居，如果节点太少，再扩展到二阶邻居，如果节点太多，再按边强度截断
    This is applied AFTER subnetwork LPCMCI. It does not re-run screening.
    """
    if direction not in {"out", "in", "both"}:
        raise ValueError("direction must be one of {'out', 'in', 'both'}")
    if len(edge_df) == 0:
        raise ValueError("edge_df is empty. LPCMCI did not return valid lagged edges.")
    # 一阶邻域：lesion 自己 + 与 lesion 直接相连的节点
    lesion_idx = int(lesion_idx)
    hop0 = {lesion_idx}
    hop1 = _neighbors(edge_df, hop0, direction=direction)
    nodes = hop0 | hop1
    used_hop = 1
    # 二阶扩展：如果只看 lesion 一阶邻居太少，就把邻居的邻居也纳入
    hop2: Set[int] = set()
    if expand_to_second_order and len(nodes) < int(min_nodes):
        hop2 = _neighbors(edge_df, hop1, direction=direction)
        nodes = nodes | hop2
        used_hop = 2

    sub_edges = edge_df[
        edge_df["src"].isin(nodes) & edge_df["dst"].isin(nodes)
    ].copy()
    # 代码会根据节点参与边的强度打分，保留最重要的节点
    if len(nodes) > int(max_nodes):
        # 保证 lesion 自己永远不会被删掉
        score: Dict[int, float] = {lesion_idx: 1e9}
        for _, row in sub_edges.iterrows():
            s, d, w = int(row["src"]), int(row["dst"]), float(row["abs_val"])
            score[s] = score.get(s, 0.0) + w
            score[d] = score.get(d, 0.0) + w
            if s == lesion_idx or d == lesion_idx:
                score[s] = score.get(s, 0.0) + w
                score[d] = score.get(d, 0.0) + w
        keep = sorted(score, key=lambda x: score[x], reverse=True)[: int(max_nodes)]
        nodes = set(keep)
        sub_edges = sub_edges[
            sub_edges["src"].isin(nodes) & sub_edges["dst"].isin(nodes)
        ].copy()
    # 输出病灶中心子图节点、子图内部边、一阶/二阶邻居节点数和边数
    info = {
        "lesion_idx": lesion_idx,
        "hop1_nodes": sorted(hop1),
        "hop2_nodes": sorted(hop2),
        "used_hop": used_hop,
        "n_nodes": len(nodes),
        "n_edges": len(sub_edges),
    }
    return sorted(nodes), sub_edges.reset_index(drop=True), info


def prune_edges_by_lpcmci_fc_constraints(
    sub_edges,
    empirical_FC,
    lesion_idx,
    max_edges=80,
    alpha_level=0.01,
    strong_quantile=0.85,
    causal_weight=0.7,
    fc_weight=0.3,
    max_direct_lesion_edges=35,
):
    """
    FC-constrained pruning v2.

    原则：
    1. direct lesion edge 优先，但不再全部无条件保护；
    2. direct + strong + significant 的边最优先保留；
    3. strong + significant 的非 direct 边也优先保留；
    4. weak direct edge 按 fused_score 保留一部分；
    5. FC 只参与压缩弱边和普通边，不删除强 LPCMCI 边。
    """

    df = sub_edges.copy()

    if len(df) == 0:
        return df, {
            "reason": "empty_sub_edges",
            "n_before": 0,
            "n_after": 0,
        }

    def normalize01(x, eps=1e-12):
        x = np.asarray(x, dtype=float)
        lo = np.min(x)
        hi = np.max(x)
        if hi - lo < eps:
            return np.zeros_like(x, dtype=float)
        return (x - lo) / (hi - lo)

    # FC strength
    fc_abs = []
    for _, row in df.iterrows():
        i = int(row["src"])
        j = int(row["dst"])
        fc_abs.append(abs(float(empirical_FC[i, j])))

    df["fc_abs"] = fc_abs

    # normalize
    df["causal_norm"] = normalize01(df["abs_val"].values)
    df["fc_norm"] = normalize01(df["fc_abs"].values)

    df["fused_score"] = (
        causal_weight * df["causal_norm"] +
        fc_weight * df["fc_norm"]
    )

    # direct lesion edge
    df["direct_lesion_edge"] = (
        (df["src"].astype(int) == int(lesion_idx)) |
        (df["dst"].astype(int) == int(lesion_idx))
    )

    # strong significant edge
    strong_thr = float(np.quantile(df["abs_val"].values, strong_quantile))

    df["strong_significant_edge"] = (
        (df["p"].astype(float) <= float(alpha_level)) &
        (df["abs_val"].astype(float) >= strong_thr)
    )

    # tier 1: direct + strong
    tier1 = df[
        df["direct_lesion_edge"] &
        df["strong_significant_edge"]
    ].copy()

    # tier 2: non-direct + strong
    tier2 = df[
        (~df["direct_lesion_edge"]) &
        df["strong_significant_edge"]
    ].copy()

    # tier 3: direct but not strong
    tier3 = df[
        df["direct_lesion_edge"] &
        (~df["strong_significant_edge"])
    ].copy()

    # tier 4: other weak edges
    tier4 = df[
        (~df["direct_lesion_edge"]) &
        (~df["strong_significant_edge"])
    ].copy()

    # 排序规则
    sort_cols = ["abs_val", "fused_score", "fc_abs"]
    ascending = [False, False, False]

    tier1 = tier1.sort_values(sort_cols, ascending=ascending)
    tier2 = tier2.sort_values(sort_cols, ascending=ascending)
    tier3 = tier3.sort_values(["fused_score", "abs_val", "fc_abs"], ascending=False)
    tier4 = tier4.sort_values(["fused_score", "abs_val", "fc_abs"], ascending=False)

    selected_parts = []

    # 先保留 direct + strong
    selected_parts.append(tier1)

    # 再保留 strong non-direct
    selected_parts.append(tier2)

    selected = pd.concat(selected_parts, axis=0).drop_duplicates(
        subset=["src", "dst", "tau"]
    )

    # 如果 strong edge 已经超过 max_edges，就只取最强的 max_edges 条
    if len(selected) >= max_edges:
        out = selected.sort_values(
            ["strong_significant_edge", "direct_lesion_edge", "abs_val", "fused_score"],
            ascending=[False, False, False, False]
        ).head(max_edges).reset_index(drop=True)

        info = {
            "reason": "strong_edges_exceed_limit",
            "n_before": len(df),
            "n_tier1_direct_strong": len(tier1),
            "n_tier2_strong": len(tier2),
            "n_after": len(out),
            "strong_abs_val_threshold": strong_thr,
            "fc_used": False,
        }
        return out, info

    # direct weak edge 只补到 max_direct_lesion_edges
    current_direct = int(selected["direct_lesion_edge"].sum()) if len(selected) else 0
    remain_direct = max(0, max_direct_lesion_edges - current_direct)

    if remain_direct > 0:
        add_tier3 = tier3.head(remain_direct)
        selected = pd.concat([selected, add_tier3], axis=0).drop_duplicates(
            subset=["src", "dst", "tau"]
        )

    # 剩余名额用 fused_score 从普通边中补
    remain_total = max_edges - len(selected)

    if remain_total > 0:
        add_tier4 = tier4.head(remain_total)
        selected = pd.concat([selected, add_tier4], axis=0).drop_duplicates(
            subset=["src", "dst", "tau"]
        )

    out = selected.sort_values(
        ["direct_lesion_edge", "strong_significant_edge", "fused_score", "abs_val"],
        ascending=[False, False, False, False]
    ).head(max_edges).reset_index(drop=True)

    info = {
        "reason": "tiered_lpcmci_fc_pruning",
        "n_before": len(df),
        "n_tier1_direct_strong": len(tier1),
        "n_tier2_strong": len(tier2),
        "n_tier3_direct_weak": len(tier3),
        "n_tier4_other_weak": len(tier4),
        "n_after": len(out),
        "strong_abs_val_threshold": strong_thr,
        "max_direct_lesion_edges": max_direct_lesion_edges,
        "fc_used": True,
    }

    return out, info


def build_intervention_candidates_from_core(
    core_edges: pd.DataFrame,
    lesion_idx: int,
    exclude_lesion: bool = True,
) -> pd.DataFrame:
    """
    Rank intervention candidates from the lesion-centered causal core.
    从最终保留下来的 causal core 边表里，给每个节点打分，生成后续 intervention candidates。
    Candidate score combines:
        - causal/fused edge involvement;
        - direct relation with lesion;
        - upstream/downstream role flags.
    """
    if len(core_edges) == 0:
        return pd.DataFrame(columns=["node", "candidate_score", "is_upstream_of_lesion", "is_downstream_of_lesion"])

    lesion_idx = int(lesion_idx)
    node_score: Dict[int, float] = {}
    upstream: Set[int] = set(core_edges.loc[core_edges["dst"] == lesion_idx, "src"].astype(int).tolist())
    downstream: Set[int] = set(core_edges.loc[core_edges["src"] == lesion_idx, "dst"].astype(int).tolist())

    score_col = "fused_score" if "fused_score" in core_edges.columns else "abs_val"
    for _, row in core_edges.iterrows():
        s, d = int(row["src"]), int(row["dst"])
        w = float(row[score_col])
        node_score[s] = node_score.get(s, 0.0) + w
        node_score[d] = node_score.get(d, 0.0) + w
        if s == lesion_idx or d == lesion_idx:
            other = d if s == lesion_idx else s
            node_score[other] = node_score.get(other, 0.0) + 1.0

    nodes = sorted(node_score, key=lambda x: node_score[x], reverse=True)
    if exclude_lesion:
        nodes = [x for x in nodes if x != lesion_idx]

    return pd.DataFrame({
        "node": nodes,
        "candidate_score": [node_score[x] for x in nodes],
        "is_upstream_of_lesion": [x in upstream for x in nodes],
        "is_downstream_of_lesion": [x in downstream for x in nodes],
    })


def build_adjacency_from_edges(edge_df: pd.DataFrame, roi_num: int, weight_col: str = "abs_val") -> np.ndarray:
    """把边表转成加权邻接矩阵：."""
    A = np.zeros((int(roi_num), int(roi_num)), dtype=float)
    if len(edge_df) == 0:
        return A
    for _, row in edge_df.iterrows():
        A[int(row["src"]), int(row["dst"])] = float(row[weight_col])
    return A


def check_stage2_quality(
    stage1_res,
    screen_res,
    local_edge_df,
    lesion_sub_edges,
    lesion_core_edges,
    lesion_idx,
    top_k=30
):
    #质控函数:检查Stage 1 和 Stage 2 是否一致？检查LPCMCI 边密度是否过高？检查core edges 是否足够？
    # Stage1 top nodes
    stage1_top = set(
        stage1_res["lesion_response_df"]
        .head(top_k)["roi"]
        .astype(int)
        .tolist()
    )

    # Screening selected nodes
    screen_nodes = set(
        screen_res["candidate_nodes"]
    )

    overlap = stage1_top & screen_nodes
    response_screen_overlap = len(overlap) / max(1, len(stage1_top))

    # Edge density
    n_candidate = len(screen_res["candidate_nodes"])
    possible_edges = n_candidate * (n_candidate - 1)
    edge_density = len(local_edge_df) / max(1, possible_edges)

    # Lesion direct edges
    n_direct_sub = int(
        ((lesion_sub_edges["src"] == lesion_idx) |
         (lesion_sub_edges["dst"] == lesion_idx)).sum()
    )

    n_direct_core = int(
        ((lesion_core_edges["src"] == lesion_idx) |
         (lesion_core_edges["dst"] == lesion_idx)).sum()
    )

    out = {
        "stage1_screen_overlap_topK": response_screen_overlap,
        "n_screen_nodes": n_candidate,
        "n_local_edges": len(local_edge_df),
        "local_edge_density": edge_density,
        "n_lesion_sub_edges": len(lesion_sub_edges),
        "n_direct_lesion_edges_before_prune": n_direct_sub,
        "n_core_edges": len(lesion_core_edges),
        "n_direct_lesion_edges_after_prune": n_direct_core,
        "overlap_nodes": sorted(list(overlap)),
    }

    if response_screen_overlap >= 0.5:
        out["overlap_judgement"] = "good"
    else:
        out["overlap_judgement"] = "weak"

    if edge_density <= 0.25:
        out["density_judgement"] = "acceptable"
    else:
        out["density_judgement"] = "too_dense"

    if len(lesion_core_edges) >= 20:
        out["core_judgement"] = "enough_for_intervention"
    else:
        out["core_judgement"] = "too_few_edges"

    return out


