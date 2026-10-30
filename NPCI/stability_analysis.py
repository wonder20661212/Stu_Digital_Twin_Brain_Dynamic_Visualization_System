#做多 seed / 多 NPI checkpoint下结果是否稳定
# Stability analysis: Top-K / pruning-threshold sensitivity


import time
import numpy as np
import pandas as pd


from local_lpcmci import (
    run_lagged_candidate_screening,
    run_candidate_subnetwork_lpcmci,
    lpcmci_to_edge_df,
    extract_lesion_neighborhood,
)

try:
    from local_lpcmci import prune_edges_by_lpcmci_fc_constraints_v2 as prune_edges_by_lpcmci_fc_constraints
except Exception:
    from local_lpcmci import prune_edges_by_lpcmci_fc_constraints


from intervention_optimization import run_single_node_search
from recovery_evaluation import summarize_recovery


# =========================================================
# 1. Small helpers
# =========================================================

def _safe_get_response_df(stage1_res):
    if isinstance(stage1_res, dict):
        if "lesion_response_df" in stage1_res:
            return stage1_res["lesion_response_df"].copy()
        if "response_df" in stage1_res:
            return stage1_res["response_df"].copy()
    elif isinstance(stage1_res, pd.DataFrame):
        return stage1_res.copy()
    return None


def _get_stage1_response_score_map(stage1_res):
    df = _safe_get_response_df(stage1_res)
    if df is None or len(df) == 0:
        return {}

    if "node" not in df.columns:
        if "roi" in df.columns:
            df = df.rename(columns={"roi": "node"})
        else:
            return {}

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
        return {}

    return dict(zip(df["node"].astype(int), df[score_col].astype(float)))


def _get_screen_score_map(screen_res):
    if not isinstance(screen_res, dict):
        return {}

    score_df = None
    for key in ["node_score_df", "candidate_df", "screen_df"]:
        if key in screen_res:
            score_df = screen_res[key].copy()
            break

    if score_df is None or len(score_df) == 0:
        return {}

    if "node" not in score_df.columns:
        if "roi" in score_df.columns:
            score_df = score_df.rename(columns={"roi": "node"})
        else:
            return {}

    score_col = None
    for c in [
        "candidate_score",
        "screening_score",
        "final_screen_score",
        "final_score",
        "score",
    ]:
        if c in score_df.columns:
            score_col = c
            break

    if score_col is None:
        return {}

    return dict(zip(score_df["node"].astype(int), score_df[score_col].astype(float)))


def _minmax_series(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + eps)


def build_candidate_df_from_core_edges_simple(
    core_edges,
    lesion_idx,
    screen_res=None,
    stage1_res=None,
    roi_names=None,
    top_k=None,
):
    """
    Fallback candidate builder.

    If your project already has a candidate_df builder, you can ignore this.
    This version ranks nodes by:
        1. incident causal strength around lesion-centered core graph
        2. upstream/downstream relation to lesion
        3. stage1 response score
        4. lagged screening score
    """
    lesion_idx = int(lesion_idx)

    if core_edges is None or len(core_edges) == 0:
        return pd.DataFrame(columns=[
            "node",
            "candidate_score",
            "is_upstream_of_lesion",
            "is_downstream_of_lesion",
        ])

    df = core_edges.copy()

    if "abs_val" not in df.columns:
        if "val" in df.columns:
            df["abs_val"] = np.abs(df["val"].astype(float))
        else:
            df["abs_val"] = 1.0

    response_map = _get_stage1_response_score_map(stage1_res)
    screen_map = _get_screen_score_map(screen_res)

    node_stats = {}

    def init_node(n):
        n = int(n)
        if n not in node_stats:
            node_stats[n] = {
                "node": n,
                "causal_strength": 0.0,
                "incident_edges": 0,
                "is_upstream_of_lesion": False,
                "is_downstream_of_lesion": False,
            }

    for _, row in df.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])
        w = float(row["abs_val"])

        if src == lesion_idx and dst == lesion_idx:
            continue

        for n in [src, dst]:
            if n == lesion_idx:
                continue
            init_node(n)
            node_stats[n]["causal_strength"] += w
            node_stats[n]["incident_edges"] += 1

        # src -> lesion means src is upstream of lesion
        if dst == lesion_idx and src != lesion_idx:
            init_node(src)
            node_stats[src]["is_upstream_of_lesion"] = True
            node_stats[src]["causal_strength"] += 0.5 * w

        # lesion -> dst means dst is downstream of lesion
        if src == lesion_idx and dst != lesion_idx:
            init_node(dst)
            node_stats[dst]["is_downstream_of_lesion"] = True
            node_stats[dst]["causal_strength"] += 0.5 * w

    rows = []
    for n, st in node_stats.items():
        response_score = float(response_map.get(n, 0.0))
        screen_score = float(screen_map.get(n, 0.0))

        rows.append({
            "node": int(n),
            "causal_strength": float(st["causal_strength"]),
            "incident_edges": int(st["incident_edges"]),
            "is_upstream_of_lesion": bool(st["is_upstream_of_lesion"]),
            "is_downstream_of_lesion": bool(st["is_downstream_of_lesion"]),
            "response_score": response_score,
            "screen_score": screen_score,
        })

    cand = pd.DataFrame(rows)

    if len(cand) == 0:
        return cand

    cand["causal_norm"] = _minmax_series(cand["causal_strength"].values)
    cand["response_norm"] = _minmax_series(cand["response_score"].values)
    cand["screen_norm"] = _minmax_series(cand["screen_score"].values)

    cand["role_bonus"] = (
        cand["is_upstream_of_lesion"].astype(float) * 0.20
        + cand["is_downstream_of_lesion"].astype(float) * 0.10
    )

    cand["candidate_score"] = (
        0.55 * cand["causal_norm"]
        + 0.25 * cand["response_norm"]
        + 0.15 * cand["screen_norm"]
        + cand["role_bonus"]
    )

    if roi_names is not None:
        cand["roi_name"] = cand["node"].apply(lambda x: roi_names[int(x)])

    cand = cand.sort_values("candidate_score", ascending=False).reset_index(drop=True)

    if top_k is not None:
        cand = cand.head(int(top_k)).reset_index(drop=True)

    return cand


def _get_best_from_stage3(stage3_res):
    """
    Compatible with your current run_single_node_search output.
    """
    single_df = stage3_res["single_df"].copy()
    best_row = single_df.sort_values("objective", ascending=False).iloc[0]
    best_node = int(best_row["node"])

    detail = stage3_res.get("best_details", {}).get(best_node, None)

    if detail is not None:
        policy = detail.get("policy", None)
        policy_traj = detail.get("policy_traj", None)
        metrics = detail.get("metrics", None)
    else:
        policy = None
        policy_traj = None
        metrics = None

    return best_row, policy, policy_traj, metrics


def _edge_density(n_edges, n_nodes):
    n_nodes = int(n_nodes)
    if n_nodes <= 1:
        return np.nan
    return float(n_edges) / float(n_nodes * (n_nodes - 1))


def _jaccard_set(a, b):
    a, b = set(a), set(b)
    if len(a | b) == 0:
        return np.nan
    return len(a & b) / len(a | b)


# =========================================================
# 2. One full Stage 2 + Stage 3 + Stage 4 run
# =========================================================

def run_one_topk_threshold_setting(
    signals,
    model,
    stage1_res,
    lesion_idx,
    roi_num,
    using_steps,
    empirical_FC,
    roi_names=None,

    top_k=60,
    strong_quantile=0.90,

    # Stage 2 lagged screening params
    tau=1,
    min_candidate_nodes=25,
    top_k_response=None,
    top_k_fc=0,
    w_lagged=1.0,
    w_response=0.6,
    w_fc=0.0,

    # LPCMCI params
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

    # pruning params
    max_core_edges=80,
    max_direct_lesion_edges=35,
    causal_weight=0.75,
    fc_weight=0.25,

    # candidate/stage3 params
    top_k_stage3_candidates=20,
    strengths=(0.05, 0.1, 0.2, 0.3, 0.5, 0.6),
    schedules=("constant", "early_pulse", "ramp_up", "ramp_decay"),
    alpha_traj=1.0,
    beta_fc=0.8,
    lambda_strength=0.05,
    lambda_temporal=0.03,
    lesion_strength=0.5,
    lesion_mode="subtract",
    batch_size=256,

    candidate_builder=None,
):
    """
    Run one Top-K / threshold setting through:
        Stage 2: lagged screening + LPCMCI + FC pruning
        Stage 3: single-node intervention search
        Stage 4: recovery evaluation
    """
    lesion_idx = int(lesion_idx)

    if top_k_response is None:
        top_k_response = max(10, int(0.4 * top_k))

    t0 = time.perf_counter()

    # -----------------------------------------------------
    # Stage 2.1 lagged screening
    # -----------------------------------------------------
    screen_res = run_lagged_candidate_screening(
        signals=signals,
        lesion_idx=lesion_idx,
        roi_names=roi_names,
        lesion_response_prior=stage1_res,
        empirical_FC=None if top_k_fc == 0 else empirical_FC,
        tau=tau,
        max_candidate_nodes=int(top_k),
        min_candidate_nodes=int(min_candidate_nodes),
        top_k_direct_lagged=int(top_k),
        top_k_response=int(top_k_response),
        top_k_fc=int(top_k_fc),
        w_lagged=float(w_lagged),
        w_response=float(w_response),
        w_fc=float(w_fc),
        standardize=standardize,
    )

    screen_nodes = [int(x) for x in screen_res["candidate_nodes"]]

    # -----------------------------------------------------
    # Stage 2.2 LPCMCI
    # -----------------------------------------------------
    try:
        local_lpcmci_res, node_map = run_candidate_subnetwork_lpcmci(
            signals=signals,
            candidate_nodes=screen_nodes,
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
        local_lpcmci_res, node_map = run_candidate_subnetwork_lpcmci(
            signals=signals,
            candidate_nodes=screen_nodes,
            tau_min=tau_min,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            verbosity=verbosity,
            standardize=standardize,
        )

    local_edge_df = lpcmci_to_edge_df(
        local_lpcmci_res,
        node_map=node_map,
        include_contemp=False,
        lag_only=True,
        alpha_level=alpha_level,
    )

    # -----------------------------------------------------
    # Stage 2.3 lesion neighborhood
    # -----------------------------------------------------
    lesion_nodes, lesion_sub_edges, subnet_info = extract_lesion_neighborhood(
        edge_df=local_edge_df,
        lesion_idx=lesion_idx,
        min_nodes=8,
        max_nodes=int(top_k),
        direction="both",
        expand_to_second_order=True,
    )

    # -----------------------------------------------------
    # Stage 2.4 FC-constrained pruning
    # -----------------------------------------------------
    try:
        lesion_core_edges, prune_info = prune_edges_by_lpcmci_fc_constraints(
            sub_edges=lesion_sub_edges,
            empirical_FC=empirical_FC,
            lesion_idx=lesion_idx,
            max_edges=int(max_core_edges),
            alpha_level=alpha_level,
            strong_quantile=float(strong_quantile),
            causal_weight=float(causal_weight),
            fc_weight=float(fc_weight),
            max_direct_lesion_edges=int(max_direct_lesion_edges),
        )
    except TypeError:
        lesion_core_edges, prune_info = prune_edges_by_lpcmci_fc_constraints(
            lesion_sub_edges,
            empirical_FC,
            lesion_idx=lesion_idx,
            max_edges=int(max_core_edges),
            strong_quantile=float(strong_quantile),
        )

    # -----------------------------------------------------
    # Stage 2.5 candidate df
    # -----------------------------------------------------
    if candidate_builder is not None:
        candidate_df = candidate_builder(
            lesion_core_edges=lesion_core_edges,
            screen_res=screen_res,
            stage1_res=stage1_res,
            empirical_FC=empirical_FC,
            lesion_idx=lesion_idx,
            roi_names=roi_names,
        )
    else:
        candidate_df = build_candidate_df_from_core_edges_simple(
            core_edges=lesion_core_edges,
            lesion_idx=lesion_idx,
            screen_res=screen_res,
            stage1_res=stage1_res,
            roi_names=roi_names,
        )

    if candidate_df is None or len(candidate_df) == 0:
        raise RuntimeError(
            f"No candidate nodes generated for top_k={top_k}, strong_quantile={strong_quantile}."
        )

    candidate_nodes = (
        candidate_df["node"]
        .head(int(top_k_stage3_candidates))
        .astype(int)
        .tolist()
    )

    # -----------------------------------------------------
    # Stage 3 single-node search
    # -----------------------------------------------------
    rollout_horizon = int(stage1_res["healthy_traj"].shape[1])

    stage3_res = run_single_node_search(
        model=model,
        candidate_nodes=candidate_nodes,
        roi_num=roi_num,
        using_steps=using_steps,
        lesion_idx=lesion_idx,
        horizon=rollout_horizon,
        lesion_strength=lesion_strength,
        lesion_mode=lesion_mode,
        seed_inputs=stage1_res["seed_inputs"],
        healthy_traj=stage1_res["healthy_traj"],
        lesion_traj=stage1_res["lesion_traj"],
        signs=(-1, +1),
        strengths=strengths,
        schedules=schedules,
        alpha_traj=alpha_traj,
        beta_fc=beta_fc,
        lambda_strength=lambda_strength,
        lambda_temporal=lambda_temporal,
        batch_size=batch_size,
    )

    best_row, best_policy, best_policy_traj, best_metrics = _get_best_from_stage3(stage3_res)

    # -----------------------------------------------------
    # Stage 4 recovery
    # -----------------------------------------------------
    if best_metrics is None or best_policy_traj is None:
        stage4_res = summarize_recovery(
            healthy_traj=stage3_res["healthy_traj"],
            lesion_traj=stage3_res["lesion_traj"],
            policy_traj=best_policy_traj,
            policy=best_policy,
            alpha_traj=alpha_traj,
            beta_fc=beta_fc,
            lambda_strength=lambda_strength,
            lambda_temporal=lambda_temporal,
        )
        metrics = stage4_res["metrics"]
    else:
        metrics = best_metrics
        stage4_res = {"metrics": metrics}

    elapsed = time.perf_counter() - t0

    n_local_nodes = len(set(screen_nodes))
    n_local_edges = len(local_edge_df)
    n_core_edges = len(lesion_core_edges)
    n_direct_lesion_edges = int(
        (
            (lesion_core_edges["src"].astype(int) == lesion_idx)
            | (lesion_core_edges["dst"].astype(int) == lesion_idx)
        ).sum()
    ) if len(lesion_core_edges) > 0 else 0

    summary = {
        "top_k": int(top_k),
        "strong_quantile": float(strong_quantile),
        "runtime_sec": float(elapsed),
        "runtime_min": float(elapsed / 60.0),

        "n_screen_nodes": int(len(screen_nodes)),
        "n_local_edges": int(n_local_edges),
        "local_edge_density": _edge_density(n_local_edges, n_local_nodes),
        "n_lesion_sub_edges": int(len(lesion_sub_edges)),
        "n_core_edges": int(n_core_edges),
        "n_direct_lesion_edges": int(n_direct_lesion_edges),

        "top_candidate_nodes": candidate_nodes,
        "best_node": int(best_row["node"]),
        "best_sign": int(best_row["sign"]),
        "best_strength": float(best_row["strength"]),
        "best_schedule": str(best_row["schedule"]),

        "objective": float(metrics["objective"]),
        "traj_recovery": float(metrics["traj_recovery"]),
        "fc_recovery": float(metrics["fc_recovery"]),
        "D_lesion_traj": float(metrics["D_lesion_traj"]),
        "D_policy_traj": float(metrics["D_policy_traj"]),
        "D_lesion_fc": float(metrics["D_lesion_fc"]),
        "D_policy_fc": float(metrics["D_policy_fc"]),
        "strength_cost": float(metrics.get("strength_cost", np.nan)),
        "temporal_cost": float(metrics.get("temporal_cost", np.nan)),
    }

    details = {
        "screen_res": screen_res,
        "local_lpcmci_res": local_lpcmci_res,
        "node_map": node_map,
        "local_edge_df": local_edge_df,
        "lesion_nodes": lesion_nodes,
        "lesion_sub_edges": lesion_sub_edges,
        "subnet_info": subnet_info,
        "lesion_core_edges": lesion_core_edges,
        "prune_info": prune_info,
        "candidate_df": candidate_df,
        "stage3_res": stage3_res,
        "stage4_res": stage4_res,
        "summary": summary,
    }

    return summary, details


# =========================================================
# 3. Main Top-K / threshold sensitivity runner
# =========================================================

def run_topk_threshold_sensitivity(
    signals,
    model,
    stage1_res,
    lesion_idx,
    roi_num,
    using_steps,
    empirical_FC,
    roi_names=None,

    top_k_grid=(20, 40, 60, 80, 100),
    strong_quantile_grid=(0.90, 0.85, 0.80, 0.75),

    save_csv_path=None,
    candidate_builder=None,
    **kwargs,
):
    """
    Run Top-K / threshold sensitivity analysis.

    Parameters
    ----------
    top_k_grid:
        Different max_candidate_nodes for lagged screening.

    strong_quantile_grid:
        Different strong-edge quantile used in FC-constrained pruning.
        Lower value means more edges are treated as strong/protected.

    Returns
    -------
    result_df:
        One row per Top-K / threshold setting.
    details:
        Dict storing full outputs for each setting.
    """
    rows = []
    details = {}

    for top_k in top_k_grid:
        for q in strong_quantile_grid:
            key = (int(top_k), float(q))

            print("=" * 90)
            print(f"[TopK sensitivity] top_k={top_k}, strong_quantile={q}")
            print("=" * 90)

            try:
                summary, detail = run_one_topk_threshold_setting(
                    signals=signals,
                    model=model,
                    stage1_res=stage1_res,
                    lesion_idx=lesion_idx,
                    roi_num=roi_num,
                    using_steps=using_steps,
                    empirical_FC=empirical_FC,
                    roi_names=roi_names,
                    top_k=top_k,
                    strong_quantile=q,
                    candidate_builder=candidate_builder,
                    **kwargs,
                )

                summary["status"] = "ok"

            except Exception as e:
                summary = {
                    "top_k": int(top_k),
                    "strong_quantile": float(q),
                    "status": "failed",
                    "error": repr(e),
                }
                detail = {"error": e, "summary": summary}
                print("[FAILED]", repr(e))

            rows.append(summary)
            details[key] = detail

            result_df_tmp = pd.DataFrame(rows)
            if save_csv_path is not None:
                result_df_tmp.to_csv(save_csv_path, index=False)

    result_df = pd.DataFrame(rows)

    if save_csv_path is not None:
        result_df.to_csv(save_csv_path, index=False)

    return result_df, details


# =========================================================
# 4. Stability summary
# =========================================================

def summarize_topk_threshold_sensitivity(result_df):
    """
    Summarize selected node stability, candidate-set stability, and metric stability.
    """
    df = result_df.copy()
    df_ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()

    if len(df_ok) == 0:
        return {
            "summary": pd.DataFrame(),
            "node_frequency": pd.DataFrame(),
            "pairwise_candidate_jaccard": np.nan,
            "judgement": {
                "status": "no_successful_runs",
            },
        }

    # best node frequency
    node_freq = (
        df_ok["best_node"]
        .astype(int)
        .value_counts()
        .rename_axis("best_node")
        .reset_index(name="count")
    )
    node_freq["present_rate"] = node_freq["count"] / len(df_ok)

    # metric summary
    metric_cols = [
        "objective",
        "traj_recovery",
        "fc_recovery",
        "n_screen_nodes",
        "n_local_edges",
        "local_edge_density",
        "n_core_edges",
        "n_direct_lesion_edges",
    ]

    metric_summary_rows = []
    for c in metric_cols:
        if c in df_ok.columns:
            vals = pd.to_numeric(df_ok[c], errors="coerce")
            metric_summary_rows.append({
                "metric": c,
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
            })

    metric_summary = pd.DataFrame(metric_summary_rows)

    # pairwise candidate jaccard
    cand_sets = []
    for _, row in df_ok.iterrows():
        nodes = row.get("top_candidate_nodes", [])
        if isinstance(nodes, str):
            # If loaded from csv, list may be string; skip parsing aggressively.
            continue
        cand_sets.append(set(int(x) for x in nodes))

    jaccards = []
    for i in range(len(cand_sets)):
        for j in range(i + 1, len(cand_sets)):
            jaccards.append(_jaccard_set(cand_sets[i], cand_sets[j]))

    pairwise_jaccard = float(np.nanmean(jaccards)) if len(jaccards) > 0 else np.nan

    # judgement
    top_present_rate = float(node_freq["present_rate"].iloc[0]) if len(node_freq) else 0.0
    objective_mean = float(df_ok["objective"].mean()) if "objective" in df_ok.columns else np.nan
    traj_mean = float(df_ok["traj_recovery"].mean()) if "traj_recovery" in df_ok.columns else np.nan
    fc_mean = float(df_ok["fc_recovery"].mean()) if "fc_recovery" in df_ok.columns else np.nan

    if top_present_rate >= 0.60 and traj_mean > 0 and fc_mean > 0:
        judgement = "stable"
    elif top_present_rate >= 0.40 and traj_mean > 0:
        judgement = "partially_stable"
    else:
        judgement = "unstable"

    judgement_info = {
        "n_successful_runs": int(len(df_ok)),
        "top_best_node": int(node_freq["best_node"].iloc[0]) if len(node_freq) else None,
        "top_best_node_present_rate": top_present_rate,
        "mean_candidate_jaccard": pairwise_jaccard,
        "mean_objective": objective_mean,
        "mean_traj_recovery": traj_mean,
        "mean_fc_recovery": fc_mean,
        "judgement": judgement,
    }

    return {
        "summary": metric_summary,
        "node_frequency": node_freq,
        "pairwise_candidate_jaccard": pairwise_jaccard,
        "judgement": judgement_info,
    }


# =========================================================
# 5. Optional plotting helpers
# =========================================================

def plot_topk_sensitivity_metrics(result_df):
    """
    Simple plots:
        top_k vs objective / traj_recovery / fc_recovery
    """
    import matplotlib.pyplot as plt

    df = result_df.copy()
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    if len(df) == 0:
        raise ValueError("No successful rows to plot.")

    for metric in ["objective", "traj_recovery", "fc_recovery"]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))

        for q, sub in df.groupby("strong_quantile"):
            sub = sub.sort_values("top_k")
            ax.plot(
                sub["top_k"],
                sub[metric],
                marker="o",
                linewidth=2.0,
                label=f"threshold={q}",
            )

        ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Top-K screening size")
        ax.set_ylabel(metric)
        ax.set_title(f"Top-K sensitivity: {metric}")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25, linestyle="--")
        plt.tight_layout()
        plt.show()


def plot_best_node_frequency(node_frequency_df, top_k=15):
    import matplotlib.pyplot as plt

    df = node_frequency_df.head(int(top_k)).copy()

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(df["best_node"].astype(str), df["present_rate"])
    ax.set_xlabel("Best intervention node")
    ax.set_ylabel("Present rate")
    ax.set_title("Best-node stability across Top-K / threshold settings")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lesion_response import rollout_npi


def compute_lesion_response_given_seeds(
    model,
    seed_inputs,
    roi_num,
    using_steps,
    lesion_idx,
    roi_names=None,
    horizon=20,
    lesion_strength=0.5,
    lesion_mode="subtract",
    batch_size=256,
):
    """
    使用已经指定好的 seed_inputs 计算 lesion response。
    这个函数和 compute_lesion_response(...) 类似，
    但它不再随机抽 seed，而是直接使用外部传入的 seed_inputs。
    这样可以保证不同 n_seed_windows 之间是嵌套比较。
    """

    healthy_traj = rollout_npi(
        model=model,
        seed_inputs=seed_inputs,
        roi_num=roi_num,
        using_steps=using_steps,
        horizon=horizon,
        lesion_node=None,
        lesion_strength=0.0,
        batch_size=batch_size,
    )

    lesion_traj = rollout_npi(
        model=model,
        seed_inputs=seed_inputs,
        roi_num=roi_num,
        using_steps=using_steps,
        horizon=horizon,
        lesion_node=lesion_idx,
        lesion_strength=lesion_strength,
        lesion_mode=lesion_mode,
        batch_size=batch_size,
    )

    diff = lesion_traj - healthy_traj

    response_vector = np.mean(np.abs(diff), axis=(0, 1))
    signed_response_vector = np.mean(diff, axis=(0, 1))
    temporal_response = np.mean(np.abs(diff), axis=(0, 2))
    roi_temporal_response = np.mean(np.abs(diff), axis=0)

    lesion_response_df = pd.DataFrame({
        "roi": np.arange(roi_num, dtype=int),
        "response_abs": response_vector,
        "response_signed": signed_response_vector,
    })

    if roi_names is not None:
        lesion_response_df.insert(
            lesion_response_df.columns.get_loc("roi") + 1,
            "roi_name",
            lesion_response_df["roi"].astype(int).map(
                lambda i: roi_names[int(i)] if int(i) < len(roi_names) else f"ROI_{int(i)}"
            )
        )

    lesion_response_df = (
        lesion_response_df
        .sort_values("response_abs", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "lesion_idx": int(lesion_idx),
        "seed_inputs": seed_inputs,
        "healthy_traj": healthy_traj,
        "lesion_traj": lesion_traj,
        "response_vector": response_vector,
        "signed_response_vector": signed_response_vector,
        "temporal_response": temporal_response,
        "roi_temporal_response": roi_temporal_response,
        "lesion_response_df": lesion_response_df,
    }


def topk_set(response_vector, k=20, exclude=None):
    """
    返回 response_vector 中 top-k ROI 集合。
    """
    exclude = set(exclude or [])
    order = np.argsort(-np.asarray(response_vector, dtype=float))

    out = []
    for idx in order:
        idx = int(idx)
        if idx in exclude:
            continue
        out.append(idx)
        if len(out) >= k:
            break

    return set(out)


def topk_list(response_vector, k=20, exclude=None):
    """
    返回 response_vector 中 top-k ROI 列表。
    """
    exclude = set(exclude or [])
    order = np.argsort(-np.asarray(response_vector, dtype=float))

    out = []
    for idx in order:
        idx = int(idx)
        if idx in exclude:
            continue
        out.append(idx)
        if len(out) >= k:
            break

    return out


def pearson_corr_safe(x, y, eps=1e-12):
    """
    安全计算 Pearson 相关。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))

    if denom < eps:
        return np.nan

    return float(np.sum(x * y) / denom)


def spearman_corr_by_rank(x, y):
    """
    不依赖 scipy，用 pandas 计算 Spearman 相关。
    """
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def mean_abs_rank_change(vec_a, vec_b, top_k=20):
    """
    计算两个 response_vector 在 top-k union 节点上的平均排名变化。
    数值越小，说明排名越稳定。
    """
    vec_a = np.asarray(vec_a, dtype=float)
    vec_b = np.asarray(vec_b, dtype=float)

    order_a = np.argsort(-vec_a)
    order_b = np.argsort(-vec_b)

    rank_a = np.empty_like(order_a)
    rank_b = np.empty_like(order_b)

    rank_a[order_a] = np.arange(len(order_a))
    rank_b[order_b] = np.arange(len(order_b))

    top_a = set(order_a[:top_k].astype(int).tolist())
    top_b = set(order_b[:top_k].astype(int).tolist())
    union_nodes = sorted(top_a | top_b)

    changes = [
        abs(int(rank_a[i]) - int(rank_b[i]))
        for i in union_nodes
    ]

    return float(np.mean(changes)) if len(changes) > 0 else np.nan


def run_n_seed_windows_convergence_check(
    model,
    inputs,
    roi_num,
    using_steps,
    lesion_idx,
    roi_names=None,
    n_seed_windows_list=(8, 16, 32, 64, 128),
    horizon=20,
    lesion_strength=0.5,
    lesion_mode="subtract",
    random_state=123,
    batch_size=256,
    top_k=20,
    exclude_lesion=True,
):
    """
    对不同 n_seed_windows 做收敛性检查。

    输出：
        results_by_B:
            每个 B 对应的 lesion response 结果。

        summary_df:
            每个 B 相对于最大 B 的稳定性指标。

        adjacent_df:
            相邻 B 之间的变化，例如 8->16, 16->32, 32->64, 64->128。

        topk_df:
            每个 B 的 top-k ROI 列表。
    """

    inputs = np.asarray(inputs, dtype=float)
    n_total = inputs.shape[0]

    n_seed_windows_list = [int(x) for x in n_seed_windows_list]
    n_seed_windows_list = sorted([x for x in n_seed_windows_list if x > 0])

    max_B = min(max(n_seed_windows_list), n_total)

    # 固定一个最大 seed 池，然后不同 B 使用前 B 个。
    rng = np.random.default_rng(random_state)
    seed_indices_pool = rng.choice(n_total, size=max_B, replace=False)

    results_by_B = {}

    for B in n_seed_windows_list:
        B_eff = min(B, max_B)
        seed_indices = seed_indices_pool[:B_eff]
        seed_inputs = inputs[seed_indices]

        print(f"[Convergence] Running n_seed_windows={B_eff}")

        res = compute_lesion_response_given_seeds(
            model=model,
            seed_inputs=seed_inputs,
            roi_num=roi_num,
            using_steps=using_steps,
            lesion_idx=lesion_idx,
            roi_names=roi_names,
            horizon=horizon,
            lesion_strength=lesion_strength,
            lesion_mode=lesion_mode,
            batch_size=batch_size,
        )

        res["seed_indices"] = seed_indices
        res["n_seed_windows"] = B_eff
        results_by_B[B_eff] = res

    # 以最大 B 作为近似参考。
    ref_B = max(results_by_B.keys())
    ref_vec = results_by_B[ref_B]["response_vector"]

    exclude = [lesion_idx] if exclude_lesion else []

    ref_topk = topk_set(ref_vec, k=top_k, exclude=exclude)

    summary_rows = []

    for B, res in results_by_B.items():
        vec = res["response_vector"]

        cur_topk = topk_set(vec, k=top_k, exclude=exclude)

        overlap = len(cur_topk & ref_topk)
        overlap_ratio = overlap / max(1, top_k)

        jaccard = len(cur_topk & ref_topk) / max(1, len(cur_topk | ref_topk))

        summary_rows.append({
            "n_seed_windows": int(B),
            "reference_B": int(ref_B),
            "pearson_to_ref": pearson_corr_safe(vec, ref_vec),
            "spearman_to_ref": spearman_corr_by_rank(vec, ref_vec),
            f"top{top_k}_overlap_count": int(overlap),
            f"top{top_k}_overlap_ratio": float(overlap_ratio),
            f"top{top_k}_jaccard": float(jaccard),
            f"top{top_k}_mean_abs_rank_change_to_ref": mean_abs_rank_change(
                vec,
                ref_vec,
                top_k=top_k,
            ),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("n_seed_windows").reset_index(drop=True)

    # 相邻 B 之间比较。
    adjacent_rows = []
    Bs = sorted(results_by_B.keys())

    for b1, b2 in zip(Bs[:-1], Bs[1:]):
        vec1 = results_by_B[b1]["response_vector"]
        vec2 = results_by_B[b2]["response_vector"]

        top1 = topk_set(vec1, k=top_k, exclude=exclude)
        top2 = topk_set(vec2, k=top_k, exclude=exclude)

        overlap = len(top1 & top2)

        adjacent_rows.append({
            "B_from": int(b1),
            "B_to": int(b2),
            "pearson": pearson_corr_safe(vec1, vec2),
            "spearman": spearman_corr_by_rank(vec1, vec2),
            f"top{top_k}_overlap_count": int(overlap),
            f"top{top_k}_overlap_ratio": float(overlap / max(1, top_k)),
            f"top{top_k}_jaccard": float(len(top1 & top2) / max(1, len(top1 | top2))),
            f"top{top_k}_mean_abs_rank_change": mean_abs_rank_change(
                vec1,
                vec2,
                top_k=top_k,
            ),
        })

    adjacent_df = pd.DataFrame(adjacent_rows)

    # 每个 B 的 top-k 列表。
    topk_rows = []

    for B, res in results_by_B.items():
        vec = res["response_vector"]
        nodes = topk_list(vec, k=top_k, exclude=exclude)

        for rank, node in enumerate(nodes, start=1):
            row = {
                "n_seed_windows": int(B),
                "rank": int(rank),
                "roi": int(node),
                "response_abs": float(vec[node]),
            }

            if roi_names is not None:
                row["roi_name"] = roi_names[int(node)] if int(node) < len(roi_names) else f"ROI_{int(node)}"

            topk_rows.append(row)

    topk_df = pd.DataFrame(topk_rows)

    return {
        "results_by_B": results_by_B,
        "summary_df": summary_df,
        "adjacent_df": adjacent_df,
        "topk_df": topk_df,
        "reference_B": ref_B,
        "seed_indices_pool": seed_indices_pool,
    }