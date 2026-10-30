import os
import copy
import itertools
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.ioff()
from lesion_response import compute_lesion_response
from local_lpcmci import (run_lagged_candidate_screening,
                          run_candidate_subnetwork_lpcmci,
                          lpcmci_to_edge_df,
                          extract_lesion_neighborhood,
                          prune_edges_by_lpcmci_fc_constraints,
                          build_intervention_candidates_from_core,
                          check_stage2_quality
                          )
from intervention_optimization import run_single_node_search
from recovery_evaluation import summarize_recovery


def _setup_paper_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.facecolor": "#f1f2f6",
        "figure.facecolor": "white",
        "grid.color": "white",
        "grid.linewidth": 1.6,
    })
    sns.set_theme(style="whitegrid")


def _beautify_axis(ax):
    ax.set_facecolor("#f1f2f6")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#cfcfcf")
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=1.4, color="white", alpha=1.0)
    ax.grid(True, which="major", axis="x", linestyle="--", linewidth=1.2, color="white", alpha=1.0)
    ax.tick_params(axis="both", direction="out", length=4.5, width=1.0)

# ============================================================
# 1. Base config
# ============================================================

BASE_CONFIG = {
    # -------------------------
    # Stage 1: lesion response
    # -------------------------
    "n_seed_windows": 64,
    "lesion_strength": 0.5,
    "horizon": 20,
    "lesion_mode": "subtract",

    # -------------------------
    # Stage 2: lagged screening
    # -------------------------
    "tau": 1,
    "max_candidate_nodes": 80,
    "min_candidate_nodes": 30,
    "top_k_direct_lagged": 40,
    "top_k_response": 30,
    "top_k_fc": 0,
    "w_lagged": 1.0,
    "w_response": 0.6,
    "w_fc": 0.0,

    # -------------------------
    # Stage 2: LPCMCI
    # -------------------------
    "tau_min": 1,
    "tau_max": 1,
    "pc_alpha": 0.05,
    "alpha_level": None,

    # -------------------------
    # Stage 2: lesion neighborhood
    # -------------------------
    "min_lesion_sub_nodes": 8,
    "max_lesion_sub_nodes": 30,
    "expand_to_second_order": True,

    # -------------------------
    # Stage 2: pruning
    # -------------------------
    "max_edges": 80,
    "prune_alpha_level": 0.01,
    "strong_quantile": 0.85,
    "causal_weight": 0.7,
    "fc_weight": 0.3,
    "max_direct_lesion_edges": 35,

    # -------------------------
    # Stage 3: intervention search
    # -------------------------
    "alpha_traj": 1.0,
    "beta_fc": 0.8,
    "lambda_node": 0.01,
    "lambda_strength": 0.05,
    "lambda_temporal": 0.03,

    "signs": (-1, +1),
    "strengths": (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    "schedules": (
        "constant",
        "early_pulse",
        "middle_pulse",
        "late_pulse",
        "ramp_up",
        "ramp_decay",
        "exponential_decay",
    ),

    "run_greedy": True,
    "max_interventions": 3,
    "top_m_per_step": 20,
    "min_marginal_gain": 1e-4,
    "force_accept_first": True,

    # -------------------------
    # Runtime
    # -------------------------
    "random_state": 123,
    "batch_size": 256,
    "standardize": "auto",
}


# ============================================================
# 2. Search grids
# ============================================================

STAGE1_SEED_GRID = {
    "n_seed_windows": [8, 16, 32, 64, 128],
}

STAGE1_SENSITIVITY_GRID = {
    "lesion_strength": [0.3, 0.5, 0.8],
    "horizon": [10, 20, 30, 40],
}

# STAGE2_GRID = {
#     "pc_alpha": [0.01, 0.05, 0.10],
#     "max_candidate_nodes": [50, 80],  # 跑通再用[50, 80, 120]
#     "max_edges": [40, 80],    # 跑通再用[40, 80, 120]
#     "strong_quantile": [0.75, 0.85],    # 跑通再用[0.75, 0.85, 0.90]
#     "prune_weight_pair": [
#         (1.0, 0.0),
#         (0.8, 0.2),
#         (0.7, 0.3),
#     ],
#     #  (1.0, 0.0),
#     #  (0.9, 0.1),
#     #  (0.8, 0.2),
#     #  (0.7, 0.3),
#     #  (0.6, 0.4),跑通再用
# }
STAGE2_CAUSAL_GRID = {
    "pc_alpha": [0.005, 0.01, 0.02, 0.03],
    "max_candidate_nodes": [20, 30, 40, 50],
}

STAGE2_PRUNE_GRID = {
    "max_edges": [10, 20, 30, 40],
    "strong_quantile": [0.80, 0.85, 0.90],
    "prune_weight_pair": [
        (1.0, 0.0),
        (0.8, 0.2),
        (0.7, 0.3),
    ],
}

STAGE3_GRID = {
    "lambda_strength": [0.01, 0.02, 0.03, 0.04, 0.05],   # 跑通再用[0.01, 0.03, 0.05, 0.10]
    "lambda_temporal": [0.00, 0.01, 0.02, 0.03],   # 跑通再用[0.00, 0.01, 0.03, 0.05, 0.10]
    "lambda_node": [0.00, 0.01, 0.02, 0.03],    # 跑通再用[0.00, 0.02, 0.05, 0.10]
}


# ============================================================
# 3. Output directory
# ============================================================

OUTPUT_DIR = "./results/grid_search"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 通用工具函数
def save_fig(fig, path, dpi=300):
    """
    Save matplotlib figure and close it.
    This avoids displaying figures in Notebook.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved figure] {path}")

def update_config(base_config, **kwargs):
    """
    返回一个新的 config，不修改原始 base_config。
    """
    cfg = copy.deepcopy(base_config)
    cfg.update(kwargs)
    return cfg


def iter_grid(grid_dict):
    """
    把参数网格 dict 转成逐组 config。
    例如：
        {"a": [1,2], "b": [3,4]}
    生成：
        {"a":1,"b":3}, {"a":1,"b":4}, ...
    """
    keys = list(grid_dict.keys())
    values = [grid_dict[k] for k in keys]

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def pearson_corr_safe(x, y, eps=1e-12):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))

    if denom < eps:
        return np.nan

    return float(np.sum(x * y) / denom)


def spearman_corr_safe(x, y):
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def topk_nodes_from_vector(vec, k=20, exclude=None):
    exclude = set(exclude or [])
    order = np.argsort(-np.asarray(vec, dtype=float))

    out = []
    for idx in order:
        idx = int(idx)
        if idx in exclude:
            continue
        out.append(idx)
        if len(out) >= k:
            break

    return out


def topk_overlap_ratio(vec_a, vec_b, k=20, exclude=None):
    a = set(topk_nodes_from_vector(vec_a, k=k, exclude=exclude))
    b = set(topk_nodes_from_vector(vec_b, k=k, exclude=exclude))

    inter = len(a & b)
    union = len(a | b)

    return {
        "topk_overlap_count": inter,
        "topk_overlap_ratio": inter / max(1, k),
        "topk_jaccard": inter / max(1, union),
    }


def mean_abs_rank_change(vec_a, vec_b, top_k=20):
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

    if len(union_nodes) == 0:
        return np.nan

    changes = [
        abs(int(rank_a[i]) - int(rank_b[i]))
        for i in union_nodes
    ]

    return float(np.mean(changes))


def safe_policy_nodes(policy):
    if policy is None:
        return []
    if isinstance(policy, dict):
        policy = [policy]
    return [int(item["node"]) for item in policy if "node" in item]


def save_df(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {path}")

# Stage 1 病灶响应稳定性搜索
# 分两部分：n_seed_windows 收敛性搜索；lesion_strength / horizon 敏感性检查
def run_stage1_once(
    config,
    model,
    inputs,
    roi_num,
    using_steps,
    lesion_idx,
    roi_names=None,
):
    """
    用一组 config 运行 Stage 1 lesion response。
    """
    res = compute_lesion_response(
        model=model,
        inputs=inputs,
        roi_num=roi_num,
        using_steps=using_steps,
        lesion_idx=lesion_idx,
        roi_names=roi_names,
        horizon=config["horizon"],
        n_seed_windows=config["n_seed_windows"],
        lesion_strength=config["lesion_strength"],
        lesion_mode=config["lesion_mode"],
        random_state=config["random_state"],
        batch_size=config["batch_size"],
    )

    return res


def run_stage1_seed_grid(
    base_config,
    model,
    inputs,
    roi_num,
    using_steps,
    lesion_idx,
    roi_names=None,
    top_k=20,
):
    """
    Stage 1-A:
    搜索 n_seed_windows，检查 response_vector 是否收敛。
    """
    rows = []
    results = {}

    B_list = STAGE1_SEED_GRID["n_seed_windows"]
    ref_B = max(B_list)

    # 先跑所有 B
    for B in B_list:
        cfg = update_config(base_config, n_seed_windows=B)
        print(f"[Stage1 seed grid] n_seed_windows={B}")

        res = run_stage1_once(
            config=cfg,
            model=model,
            inputs=inputs,
            roi_num=roi_num,
            using_steps=using_steps,
            lesion_idx=lesion_idx,
            roi_names=roi_names,
        )

        results[B] = res

    # 以最大 B 作为 reference
    ref_vec = results[ref_B]["response_vector"]
    exclude = [lesion_idx]

    for B in B_list:
        vec = results[B]["response_vector"]

        overlap = topk_overlap_ratio(
            vec_a=vec,
            vec_b=ref_vec,
            k=top_k,
            exclude=exclude,
        )

        rows.append({
            "n_seed_windows": B,
            "reference_B": ref_B,
            "pearson_to_ref": pearson_corr_safe(vec, ref_vec),
            "spearman_to_ref": spearman_corr_safe(vec, ref_vec),
            f"top{top_k}_overlap_count": overlap["topk_overlap_count"],
            f"top{top_k}_overlap_ratio": overlap["topk_overlap_ratio"],
            f"top{top_k}_jaccard": overlap["topk_jaccard"],
            f"top{top_k}_mean_abs_rank_change_to_ref": mean_abs_rank_change(
                vec,
                ref_vec,
                top_k=top_k,
            ),
        })

    df = pd.DataFrame(rows)

    # 选择规则：最小 B 满足约束
    stable = df[
        (df["pearson_to_ref"] >= 0.95) &
        (df["spearman_to_ref"] >= 0.90) &
        (df[f"top{top_k}_overlap_ratio"] >= 0.80)
    ].copy()

    if len(stable) > 0:
        selected_B = int(stable.sort_values("n_seed_windows").iloc[0]["n_seed_windows"])
    else:
        selected_B = int(ref_B)

    print(f"[Stage1 seed grid] selected n_seed_windows = {selected_B}")

    return {
        "stage1_seed_df": df,
        "stage1_seed_results": results,
        "selected_n_seed_windows": selected_B,
        "reference_B": ref_B,
    }


def run_stage1_strength_horizon_sensitivity(
    base_config,
    model,
    inputs,
    roi_num,
    using_steps,
    lesion_idx,
    roi_names=None,
    top_k=20,
):
    """
    Stage 1-B:
    检查 lesion_strength 和 horizon 对 response_vector / Top-K ROI 的影响。
    """
    rows = []
    results = {}

    # reference config
    ref_cfg = copy.deepcopy(base_config)
    ref_key = (
        ref_cfg["lesion_strength"],
        ref_cfg["horizon"],
    )

    print(f"[Stage1 sensitivity] reference={ref_key}")

    ref_res = run_stage1_once(
        config=ref_cfg,
        model=model,
        inputs=inputs,
        roi_num=roi_num,
        using_steps=using_steps,
        lesion_idx=lesion_idx,
        roi_names=roi_names,
    )

    ref_vec = ref_res["response_vector"]
    exclude = [lesion_idx]

    for grid_cfg in iter_grid(STAGE1_SENSITIVITY_GRID):
        cfg = update_config(base_config, **grid_cfg)

        key = (
            cfg["lesion_strength"],
            cfg["horizon"],
        )

        print(f"[Stage1 sensitivity] lesion_strength={cfg['lesion_strength']}, horizon={cfg['horizon']}")

        res = run_stage1_once(
            config=cfg,
            model=model,
            inputs=inputs,
            roi_num=roi_num,
            using_steps=using_steps,
            lesion_idx=lesion_idx,
            roi_names=roi_names,
        )

        results[key] = res

        vec = res["response_vector"]
        overlap = topk_overlap_ratio(
            vec_a=vec,
            vec_b=ref_vec,
            k=top_k,
            exclude=exclude,
        )

        temporal_response = res["temporal_response"]

        rows.append({
            "lesion_strength": cfg["lesion_strength"],
            "horizon": cfg["horizon"],
            "pearson_to_base": pearson_corr_safe(vec, ref_vec),
            "spearman_to_base": spearman_corr_safe(vec, ref_vec),
            f"top{top_k}_overlap_count": overlap["topk_overlap_count"],
            f"top{top_k}_overlap_ratio": overlap["topk_overlap_ratio"],
            f"top{top_k}_jaccard": overlap["topk_jaccard"],
            "mean_response": float(np.mean(vec)),
            "max_response": float(np.max(vec)),
            "temporal_peak_step": int(np.argmax(temporal_response)),
            "temporal_peak_value": float(np.max(temporal_response)),
        })

    df = pd.DataFrame(rows)

    return {
        "stage1_sensitivity_df": df,
        "stage1_sensitivity_results": results,
        "reference_result": ref_res,
    }


def plot_stage1_seed_convergence(stage1_seed_df, top_k=20, output_dir="./results/grid_search"):
    df = stage1_seed_df.copy()
    x = df["n_seed_windows"].values

    # --------------------------------------------------
    # Figure 1: response vector convergence + top-k stability
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(x, df["pearson_to_ref"], marker="o", label="Pearson")
    ax.plot(x, df["spearman_to_ref"], marker="s", label="Spearman")
    ax.plot(x, df[f"top{top_k}_overlap_ratio"], marker="^", label=f"Top-{top_k} overlap")
    ax.set_xlabel("n_seed_windows")
    ax.set_ylabel("Similarity to reference")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()

    response_path = os.path.join(output_dir, "stage1_response_vector_convergence.png")
    stability_path = os.path.join(output_dir, f"stage1_top{top_k}_roi_stability.png")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(response_path, dpi=300, bbox_inches="tight")
    fig.savefig(stability_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved figure] {response_path}")
    print(f"[Saved figure] {stability_path}")

    # --------------------------------------------------
    # Figure 3: rank change
    # --------------------------------------------------
    rank_col = f"top{top_k}_mean_abs_rank_change_to_ref"

    if rank_col in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(x, df[rank_col], marker="o")
        ax.set_xlabel("n_seed_windows")
        ax.set_ylabel("Mean absolute rank change")
        ax.set_title(f"Stage 1 Top-{top_k} Rank Stability")
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, f"stage1_top{top_k}_rank_stability.png")
        )

def plot_stage1_strength_horizon_sensitivity(
    stage1_sens_df,
    top_k=20,
    output_dir="./results/grid_search",
):
    """
    画 Stage 1 lesion_strength / horizon 敏感性分析图。
    不在 Notebook 中显示，直接保存到 output_dir。
    """

    df = stage1_sens_df.copy()

    overlap_col = f"top{top_k}_overlap_ratio"

    # ============================================================
    # 1. Top-K overlap heatmap
    # ============================================================

    if overlap_col in df.columns:
        pivot = df.pivot(
            index="lesion_strength",
            columns="horizon",
            values=overlap_col,
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("horizon")
        ax.set_ylabel("lesion_strength")
        ax.set_title(f"Stage 1 Top-{top_k} Overlap Sensitivity")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")

        fig.colorbar(im, ax=ax, label=f"Top-{top_k} overlap ratio")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, f"stage1_strength_horizon_top{top_k}_overlap.png"),
        )

    # ============================================================
    # 2. Pearson stability heatmap
    # ============================================================

    if "pearson_to_base" in df.columns:
        pivot = df.pivot(
            index="lesion_strength",
            columns="horizon",
            values="pearson_to_base",
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("horizon")
        ax.set_ylabel("lesion_strength")
        ax.set_title("Stage 1 Response Vector Pearson Stability")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")

        fig.colorbar(im, ax=ax, label="Pearson to base")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage1_strength_horizon_pearson.png"),
        )

    # ============================================================
    # 3. Spearman stability heatmap
    # ============================================================

    if "spearman_to_base" in df.columns:
        pivot = df.pivot(
            index="lesion_strength",
            columns="horizon",
            values="spearman_to_base",
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("horizon")
        ax.set_ylabel("lesion_strength")
        ax.set_title("Stage 1 Response Ranking Spearman Stability")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")

        fig.colorbar(im, ax=ax, label="Spearman to base")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage1_strength_horizon_spearman.png"),
        )

    # ============================================================
    # 4. Mean response vs horizon under different lesion strengths
    # ============================================================

    if "mean_response" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))

        for strength, sub in df.groupby("lesion_strength"):
            sub = sub.sort_values("horizon")
            ax.plot(
                sub["horizon"].values,
                sub["mean_response"].values,
                marker="o",
                label=f"strength={strength}",
            )

        ax.set_xlabel("horizon")
        ax.set_ylabel("Mean response")
        ax.set_title("Stage 1 Mean Lesion Response Sensitivity")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage1_mean_response_vs_horizon.png"),
        )

    # ============================================================
    # 5. Max response vs horizon under different lesion strengths
    # ============================================================

    if "max_response" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))

        for strength, sub in df.groupby("lesion_strength"):
            sub = sub.sort_values("horizon")
            ax.plot(
                sub["horizon"].values,
                sub["max_response"].values,
                marker="o",
                label=f"strength={strength}",
            )

        ax.set_xlabel("horizon")
        ax.set_ylabel("Max response")
        ax.set_title("Stage 1 Max Lesion Response Sensitivity")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage1_max_response_vs_horizon.png"),
        )

    # ============================================================
    # 6. Temporal peak step heatmap
    # ============================================================

    if "temporal_peak_step" in df.columns:
        pivot = df.pivot(
            index="lesion_strength",
            columns="horizon",
            values="temporal_peak_step",
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("horizon")
        ax.set_ylabel("lesion_strength")
        ax.set_title("Stage 1 Temporal Peak Step")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{int(val)}", ha="center", va="center")

        fig.colorbar(im, ax=ax, label="Peak step")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage1_temporal_peak_step.png"),
        )

# Stage 2 因果图质量网格搜索
# 这一部分会循环：pc_alpha、max_candidate_nodes、max_edges、strong_quantile并记录candidate 数量、LPCMCI 边数量、edge density、core edge 数量、direct lesion edge 数量、Stage1-Screening overlap
def run_stage2_once(
    config,
    signals,
    lesion_idx,
    stage1_res,
    empirical_FC,
    roi_names=None,
):
    """
    用一组 config 运行 Stage 2：
    screening -> LPCMCI -> edge_df -> lesion neighborhood -> pruning -> candidates
    """

    # --------------------------------------------------
    # Stage 2.1: lagged candidate screening
    # --------------------------------------------------
    screen_res = run_lagged_candidate_screening(
        signals=signals,
        lesion_idx=lesion_idx,
        lesion_response_prior=stage1_res,
        empirical_FC=empirical_FC,
        roi_names=roi_names,
        tau=config["tau"],
        max_candidate_nodes=config["max_candidate_nodes"],
        min_candidate_nodes=config["min_candidate_nodes"],
        top_k_direct_lagged=config["top_k_direct_lagged"],
        top_k_response=config["top_k_response"],
        top_k_fc=config["top_k_fc"],
        w_lagged=config["w_lagged"],
        w_response=config["w_response"],
        w_fc=config["w_fc"],
        standardize=config["standardize"],
    )

    # --------------------------------------------------
    # Stage 2.2: LPCMCI
    # --------------------------------------------------
    lpcmci_results, node_map = run_candidate_subnetwork_lpcmci(
        signals=signals,
        candidate_nodes=screen_res["candidate_nodes"],
        tau_min=config["tau_min"],
        tau_max=config["tau_max"],
        pc_alpha=config["pc_alpha"],
        standardize=config["standardize"],
    )

    local_edge_df = lpcmci_to_edge_df(
        results=lpcmci_results,
        node_map=node_map,
        include_contemp=False,
        lag_only=True,
        alpha_level=config["alpha_level"],
    )

    # --------------------------------------------------
    # Stage 2.3: lesion-centered neighborhood
    # --------------------------------------------------
    lesion_nodes, lesion_sub_edges, lesion_info = extract_lesion_neighborhood(
        edge_df=local_edge_df,
        lesion_idx=lesion_idx,
        min_nodes=config["min_lesion_sub_nodes"],
        max_nodes=config["max_lesion_sub_nodes"],
        direction="both",
        expand_to_second_order=config["expand_to_second_order"],
    )

    # --------------------------------------------------
    # Stage 2.4: FC-constrained pruning
    # --------------------------------------------------
    lesion_core_edges, prune_info = prune_edges_by_lpcmci_fc_constraints(
        sub_edges=lesion_sub_edges,
        empirical_FC=empirical_FC,
        lesion_idx=lesion_idx,
        max_edges=config["max_edges"],
        alpha_level=config["prune_alpha_level"],
        strong_quantile=config["strong_quantile"],
        causal_weight=config["causal_weight"],
        fc_weight=config["fc_weight"],
        max_direct_lesion_edges=config["max_direct_lesion_edges"],
    )

    # --------------------------------------------------
    # Intervention candidates
    # --------------------------------------------------
    candidate_df = build_intervention_candidates_from_core(
        core_edges=lesion_core_edges,
        lesion_idx=lesion_idx,
        exclude_lesion=True,
    )

    # --------------------------------------------------
    # Quality check
    # --------------------------------------------------
    quality = check_stage2_quality(
        stage1_res=stage1_res,
        screen_res=screen_res,
        local_edge_df=local_edge_df,
        lesion_sub_edges=lesion_sub_edges,
        lesion_core_edges=lesion_core_edges,
        lesion_idx=lesion_idx,
        top_k=30,
    )

    return {
        "screen_res": screen_res,
        "lpcmci_results": lpcmci_results,
        "node_map": node_map,
        "local_edge_df": local_edge_df,
        "lesion_nodes": lesion_nodes,
        "lesion_sub_edges": lesion_sub_edges,
        "lesion_info": lesion_info,
        "lesion_core_edges": lesion_core_edges,
        "prune_info": prune_info,
        "candidate_df": candidate_df,
        "quality": quality,
    }


def run_stage2_causal_once(
    config,
    signals,
    lesion_idx,
    stage1_res,
    empirical_FC,
    roi_names=None,
):
    """
    Stage 2 causal layer:
    screening -> LPCMCI -> edge_df -> lesion-centered neighborhood

    注意：这个函数不做 FC-constrained pruning。
    """

    # --------------------------------------------------
    # Stage 2.1: lagged candidate screening
    # --------------------------------------------------
    screen_res = run_lagged_candidate_screening(
        signals=signals,
        lesion_idx=lesion_idx,
        lesion_response_prior=stage1_res,
        empirical_FC=empirical_FC,
        roi_names=roi_names,
        tau=config["tau"],
        max_candidate_nodes=config["max_candidate_nodes"],
        min_candidate_nodes=config["min_candidate_nodes"],
        top_k_direct_lagged=config["top_k_direct_lagged"],
        top_k_response=config["top_k_response"],
        top_k_fc=config["top_k_fc"],
        w_lagged=config["w_lagged"],
        w_response=config["w_response"],
        w_fc=config["w_fc"],
        standardize=config["standardize"],
    )

    # --------------------------------------------------
    # Stage 2.2: LPCMCI
    # --------------------------------------------------
    lpcmci_results, node_map = run_candidate_subnetwork_lpcmci(
        signals=signals,
        candidate_nodes=screen_res["candidate_nodes"],
        tau_min=config["tau_min"],
        tau_max=config["tau_max"],
        pc_alpha=config["pc_alpha"],
        standardize=config["standardize"],
    )

    local_edge_df = lpcmci_to_edge_df(
        results=lpcmci_results,
        node_map=node_map,
        include_contemp=False,
        lag_only=True,
        alpha_level=config["alpha_level"],
    )

    # --------------------------------------------------
    # Stage 2.3: lesion-centered neighborhood
    # --------------------------------------------------
    lesion_nodes, lesion_sub_edges, lesion_info = extract_lesion_neighborhood(
        edge_df=local_edge_df,
        lesion_idx=lesion_idx,
        min_nodes=config["min_lesion_sub_nodes"],
        max_nodes=config["max_lesion_sub_nodes"],
        direction="both",
        expand_to_second_order=config["expand_to_second_order"],
    )

    return {
        "screen_res": screen_res,
        "lpcmci_results": lpcmci_results,
        "node_map": node_map,
        "local_edge_df": local_edge_df,
        "lesion_nodes": lesion_nodes,
        "lesion_sub_edges": lesion_sub_edges,
        "lesion_info": lesion_info,
    }

def run_stage2_prune_once(
    config,
    lesion_idx,
    stage1_res,
    empirical_FC,
    causal_res,
):
    """
    Stage 2 pruning layer:
    在已经得到的 lesion_sub_edges 上执行 FC-constrained pruning。
    不重新跑 screening / LPCMCI。
    """

    lesion_sub_edges = causal_res["lesion_sub_edges"]
    screen_res = causal_res["screen_res"]
    local_edge_df = causal_res["local_edge_df"]

    # --------------------------------------------------
    # Stage 2.4: FC-constrained pruning
    # --------------------------------------------------
    lesion_core_edges, prune_info = prune_edges_by_lpcmci_fc_constraints(
        sub_edges=lesion_sub_edges,
        empirical_FC=empirical_FC,
        lesion_idx=lesion_idx,
        max_edges=config["max_edges"],
        alpha_level=config["prune_alpha_level"],
        strong_quantile=config["strong_quantile"],
        causal_weight=config["causal_weight"],
        fc_weight=config["fc_weight"],
        max_direct_lesion_edges=config["max_direct_lesion_edges"],
    )

    # --------------------------------------------------
    # Intervention candidates
    # --------------------------------------------------
    candidate_df = build_intervention_candidates_from_core(
        core_edges=lesion_core_edges,
        lesion_idx=lesion_idx,
        exclude_lesion=True,
    )

    # --------------------------------------------------
    # Quality check
    # --------------------------------------------------
    quality = check_stage2_quality(
        stage1_res=stage1_res,
        screen_res=screen_res,
        local_edge_df=local_edge_df,
        lesion_sub_edges=lesion_sub_edges,
        lesion_core_edges=lesion_core_edges,
        lesion_idx=lesion_idx,
        top_k=30,
    )

    out = dict(causal_res)
    out.update({
        "lesion_core_edges": lesion_core_edges,
        "prune_info": prune_info,
        "candidate_df": candidate_df,
        "quality": quality,
    })

    return out

def stage2_selection_score(row):
    """
    Stage 2 配置选择分数。
    不是纯最大化，而是满足质量约束后选择稳定、适中、可解释的图。
    """

    score = 0.0

    # Stage1-Screening overlap 越高越好
    score += 2.0 * float(row.get("stage1_screen_overlap_topK", 0.0))

    # 边密度不要太高
    density = float(row.get("local_edge_density", np.nan))
    if np.isfinite(density):
        if density <= 0.25:
            score += 1.0
        else:
            score -= 2.0 * (density - 0.25)

    # core edges 要足够，但也不能鼓励无限多
    n_core = float(row.get("n_core_edges", 0.0))
    if n_core >= 20:
        score += 1.0
    else:
        score -= 1.0

    # lesion direct edges 保留
    n_direct = float(row.get("n_direct_lesion_edges_after_prune", 0.0))
    if n_direct > 0:
        score += 1.0

    # 过多边略惩罚
    score -= 0.002 * n_core

    return float(score)


def run_stage2_grid_search(
    base_config,
    signals,
    lesion_idx,
    stage1_res,
    empirical_FC,
    roi_names=None,
):
    """
    Two-level Stage 2 grid search.

    Level 1:
        pc_alpha / max_candidate_nodes
        需要重新跑 screening + LPCMCI。

    Level 2:
        max_edges / strong_quantile / prune_weight_pair
        只需要在已有 lesion_sub_edges 上重新剪枝。
    """

    rows = []
    results = {}

    run_id = 0
    causal_run_id = 0

    for causal_cfg in iter_grid(STAGE2_CAUSAL_GRID):

        cfg_causal = update_config(base_config, **causal_cfg)

        print(
            f"[Stage2 causal grid] causal_run_id={causal_run_id}, "
            f"pc_alpha={cfg_causal['pc_alpha']}, "
            f"max_candidate_nodes={cfg_causal['max_candidate_nodes']}"
        )

        try:
            causal_res = run_stage2_causal_once(
                config=cfg_causal,
                signals=signals,
                lesion_idx=lesion_idx,
                stage1_res=stage1_res,
                empirical_FC=empirical_FC,
                roi_names=roi_names,
            )

        except Exception as e:
            print(f"[Stage2 causal grid] failed: {repr(e)}")

            # 如果 causal 层失败，为每个 prune cfg 记录 failed，方便追踪
            for prune_cfg in iter_grid(STAGE2_PRUNE_GRID):
                prune_cfg = dict(prune_cfg)

                if "prune_weight_pair" in prune_cfg:
                    causal_weight, fc_weight = prune_cfg.pop("prune_weight_pair")
                    prune_cfg["causal_weight"] = float(causal_weight)
                    prune_cfg["fc_weight"] = float(fc_weight)

                row = {
                    "run_id": run_id,
                    "causal_run_id": causal_run_id,
                    **causal_cfg,
                    **prune_cfg,
                    "status": "failed",
                    "error": repr(e),
                    "selection_score": -np.inf,
                }

                rows.append(row)
                results[run_id] = None
                run_id += 1

            causal_run_id += 1
            continue

        # --------------------------------------------------
        # 对同一个 causal_res，搜索多个剪枝配置
        # --------------------------------------------------
        for prune_cfg in iter_grid(STAGE2_PRUNE_GRID):

            prune_cfg = dict(prune_cfg)

            if "prune_weight_pair" in prune_cfg:
                causal_weight, fc_weight = prune_cfg.pop("prune_weight_pair")
                prune_cfg["causal_weight"] = float(causal_weight)
                prune_cfg["fc_weight"] = float(fc_weight)

            cfg = update_config(cfg_causal, **prune_cfg)

            print(
                f"[Stage2 prune grid] run_id={run_id}, "
                f"causal_run_id={causal_run_id}, "
                f"max_edges={cfg['max_edges']}, "
                f"strong_quantile={cfg['strong_quantile']}, "
                f"causal_weight={cfg['causal_weight']}, "
                f"fc_weight={cfg['fc_weight']}"
            )

            try:
                res = run_stage2_prune_once(
                    config=cfg,
                    lesion_idx=lesion_idx,
                    stage1_res=stage1_res,
                    empirical_FC=empirical_FC,
                    causal_res=causal_res,
                )

                q = res["quality"]
                prune_info = res["prune_info"]

                row = {
                    "run_id": run_id,
                    "causal_run_id": causal_run_id,

                    **causal_cfg,
                    **prune_cfg,

                    "n_screen_nodes": q.get("n_screen_nodes", np.nan),
                    "n_local_edges": q.get("n_local_edges", np.nan),
                    "local_edge_density": q.get("local_edge_density", np.nan),
                    "n_lesion_sub_edges": q.get("n_lesion_sub_edges", np.nan),
                    "n_direct_lesion_edges_before_prune": q.get("n_direct_lesion_edges_before_prune", np.nan),
                    "n_core_edges": q.get("n_core_edges", np.nan),
                    "n_direct_lesion_edges_after_prune": q.get("n_direct_lesion_edges_after_prune", np.nan),
                    "stage1_screen_overlap_topK": q.get("stage1_screen_overlap_topK", np.nan),

                    "overlap_judgement": q.get("overlap_judgement", None),
                    "density_judgement": q.get("density_judgement", None),
                    "core_judgement": q.get("core_judgement", None),

                    "prune_reason": prune_info.get("reason", None),
                    "n_tier1_direct_strong": prune_info.get("n_tier1_direct_strong", np.nan),
                    "n_tier2_strong": prune_info.get("n_tier2_strong", np.nan),
                    "n_tier3_direct_weak": prune_info.get("n_tier3_direct_weak", np.nan),
                    "n_tier4_other_weak": prune_info.get("n_tier4_other_weak", np.nan),

                    "status": "ok",
                    "error": "",
                }

                row["selection_score"] = stage2_selection_score(row)

                results[run_id] = res

            except Exception as e:
                row = {
                    "run_id": run_id,
                    "causal_run_id": causal_run_id,
                    **causal_cfg,
                    **prune_cfg,
                    "status": "failed",
                    "error": repr(e),
                    "selection_score": -np.inf,
                }

                results[run_id] = None
                print(f"[Stage2 prune grid] failed: {repr(e)}")

            rows.append(row)
            run_id += 1

        causal_run_id += 1

    df = pd.DataFrame(rows)

    # --------------------------------------------------
    # Hierarchical Stage 2 selection
    # --------------------------------------------------
    ok_df = df[df["status"] == "ok"].copy()

    cond1 = (
            (ok_df["local_edge_density"] <= 0.25) &
            (ok_df["n_core_edges"] >= 20) &
            (ok_df["n_direct_lesion_edges_after_prune"] > 0)
    )

    cond2 = (
            (ok_df["local_edge_density"] <= 0.35) &
            (ok_df["n_core_edges"] >= 10) &
            (ok_df["n_direct_lesion_edges_after_prune"] > 0)
    )

    cond3 = (
            (ok_df["n_core_edges"] >= 10) &
            (ok_df["n_direct_lesion_edges_after_prune"] > 0)
    )

    if len(ok_df) == 0:
        print("[Stage2 grid] no ok rows. Fallback to all rows.")
        valid = df.copy()
        selection_level = "failed_fallback"

    elif cond1.sum() > 0:
        print("[Stage2 grid] Use Level 1 strict criteria.")
        valid = ok_df[cond1].copy()
        selection_level = "level1_strict"

    elif cond2.sum() > 0:
        print("[Stage2 grid] Use Level 2 relaxed density/core criteria.")
        valid = ok_df[cond2].copy()
        selection_level = "level2_relaxed"

    elif cond3.sum() > 0:
        print("[Stage2 grid] Use Level 3 core/direct criteria.")
        valid = ok_df[cond3].copy()
        selection_level = "level3_core_direct"

    else:
        print("[Stage2 grid] Use fallback: status == ok only.")
        valid = ok_df.copy()
        selection_level = "ok_fallback"

    best_row = valid.sort_values("selection_score", ascending=False).iloc[0]
    best_run_id = int(best_row["run_id"])

    print(f"[Stage2 grid] selected run_id = {best_run_id}")
    print(f"[Stage2 grid] selection_level = {selection_level}")

    print(f"[Stage2 grid] selected run_id = {best_run_id}")

    return {
        "stage2_grid_df": df,
        "stage2_grid_results": results,
        "best_stage2_run_id": best_run_id,
        "best_stage2_row": best_row,
        "best_stage2_result": results[best_run_id],
        "stage2_selection_level": selection_level,
    }


def plot_stage2_grid_summary(stage2_grid_df, output_dir="./results/grid_search"):
    """
    Save Stage 2 grid-search figures to output_dir.
    Figures are not shown in Notebook.
    """

    df = stage2_grid_df.copy()
    df = df[df["status"] == "ok"].copy()

    if len(df) == 0:
        print("[Stage2 plot] No valid Stage 2 grid result to plot.")
        return

    # ============================================================
    # 1. Overall graph quality: density vs core edges
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(df["local_edge_density"], df["n_core_edges"])
    ax.axvline(0.25, linestyle="--", linewidth=1)
    ax.axhline(20, linestyle="--", linewidth=1)
    ax.set_xlabel("Local edge density")
    ax.set_ylabel("Number of core edges")
    ax.set_title("Stage 2 Graph Quality Grid")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()

    save_fig(
        fig,
        os.path.join(output_dir, "stage2_graph_quality_density_core_edges.png")
    )

    # ============================================================
    # 2. Stage1-Screening overlap vs selection score
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(df["stage1_screen_overlap_topK"], df["selection_score"])
    ax.set_xlabel("Stage1-Screening overlap")
    ax.set_ylabel("Stage2 selection score")
    ax.set_title("Stage 2 Selection Score")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()

    save_fig(
        fig,
        os.path.join(output_dir, "stage2_selection_score.png")
    )

    # ============================================================
    # 3. pc_alpha sensitivity
    # ============================================================
    if "pc_alpha" in df.columns:
        grouped = (
            df.groupby("pc_alpha")
              .agg({
                  "local_edge_density": "mean",
                  "n_local_edges": "mean",
                  "n_core_edges": "mean",
                  "selection_score": "mean",
              })
              .reset_index()
              .sort_values("pc_alpha")
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grouped["pc_alpha"], grouped["local_edge_density"], marker="o", label="edge density")
        ax.plot(grouped["pc_alpha"], grouped["n_core_edges"], marker="s", label="core edges")
        ax.set_xlabel("pc_alpha")
        ax.set_ylabel("Mean value")
        ax.set_title("Stage 2 pc_alpha Sensitivity")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage2_pc_alpha_sensitivity.png")
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grouped["pc_alpha"], grouped["selection_score"], marker="o")
        ax.set_xlabel("pc_alpha")
        ax.set_ylabel("Mean selection score")
        ax.set_title("Stage 2 pc_alpha vs Selection Score")
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage2_pc_alpha_selection_score.png")
        )

    # ============================================================
    # 4. max_candidate_nodes sensitivity
    # ============================================================
    if "max_candidate_nodes" in df.columns:
        grouped = (
            df.groupby("max_candidate_nodes")
              .agg({
                  "n_screen_nodes": "mean",
                  "n_local_edges": "mean",
                  "local_edge_density": "mean",
                  "n_core_edges": "mean",
                  "selection_score": "mean",
              })
              .reset_index()
              .sort_values("max_candidate_nodes")
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grouped["max_candidate_nodes"], grouped["n_screen_nodes"], marker="o", label="screen nodes")
        ax.plot(grouped["max_candidate_nodes"], grouped["n_local_edges"], marker="s", label="local edges")
        ax.plot(grouped["max_candidate_nodes"], grouped["n_core_edges"], marker="^", label="core edges")
        ax.set_xlabel("max_candidate_nodes")
        ax.set_ylabel("Mean count")
        ax.set_title("Stage 2 Candidate Subnetwork Size Sensitivity")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage2_max_candidate_nodes_sensitivity.png")
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grouped["max_candidate_nodes"], grouped["local_edge_density"], marker="o")
        ax.set_xlabel("max_candidate_nodes")
        ax.set_ylabel("Mean local edge density")
        ax.set_title("Stage 2 Candidate Size vs Edge Density")
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage2_max_candidate_nodes_density.png")
        )

    # ============================================================
    # 5. max_edges sensitivity
    # ============================================================
    if "max_edges" in df.columns:
        grouped = (
            df.groupby("max_edges")
              .agg({
                  "n_core_edges": "mean",
                  "n_direct_lesion_edges_after_prune": "mean",
                  "selection_score": "mean",
              })
              .reset_index()
              .sort_values("max_edges")
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grouped["max_edges"], grouped["n_core_edges"], marker="o", label="core edges")
        ax.plot(grouped["max_edges"], grouped["n_direct_lesion_edges_after_prune"], marker="s", label="direct lesion edges")
        ax.set_xlabel("max_edges")
        ax.set_ylabel("Mean count")
        ax.set_title("Stage 2 Pruning Edge Budget Sensitivity")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage2_max_edges_sensitivity.png")
        )

    # ============================================================
    # 6. strong_quantile sensitivity
    # ============================================================
    if "strong_quantile" in df.columns:
        grouped = (
            df.groupby("strong_quantile")
              .agg({
                  "n_core_edges": "mean",
                  "n_tier1_direct_strong": "mean",
                  "n_tier2_strong": "mean",
                  "selection_score": "mean",
              })
              .reset_index()
              .sort_values("strong_quantile")
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grouped["strong_quantile"], grouped["n_core_edges"], marker="o", label="core edges")
        ax.plot(grouped["strong_quantile"], grouped["n_tier1_direct_strong"], marker="s", label="tier1 direct strong")
        ax.plot(grouped["strong_quantile"], grouped["n_tier2_strong"], marker="^", label="tier2 strong")
        ax.set_xlabel("strong_quantile")
        ax.set_ylabel("Mean count")
        ax.set_title("Stage 2 Strong Edge Threshold Sensitivity")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage2_strong_quantile_sensitivity.png")
        )

    # ============================================================
    # 7. causal_weight / fc_weight sensitivity
    # ============================================================
    if "causal_weight" in df.columns and "fc_weight" in df.columns:
        grouped = (
            df.groupby(["causal_weight", "fc_weight"])
              .agg({
                  "selection_score": "mean",
                  "n_core_edges": "mean",
                  "n_direct_lesion_edges_after_prune": "mean",
              })
              .reset_index()
              .sort_values("causal_weight")
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grouped["causal_weight"], grouped["selection_score"], marker="o")
        ax.set_xlabel("causal_weight")
        ax.set_ylabel("Mean selection score")
        ax.set_title("Stage 2 Causal/FC Weight Sensitivity")
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage2_causal_weight_sensitivity.png")
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grouped["causal_weight"], grouped["n_core_edges"], marker="o", label="core edges")
        ax.plot(grouped["causal_weight"], grouped["n_direct_lesion_edges_after_prune"], marker="s", label="direct lesion edges")
        ax.set_xlabel("causal_weight")
        ax.set_ylabel("Mean count")
        ax.set_title("Stage 2 Causal Weight vs Core Structure")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage2_causal_weight_core_structure.png")
        )

    # ============================================================
    # 8. Heatmap: pc_alpha × max_candidate_nodes -> edge density
    # ============================================================
    if "pc_alpha" in df.columns and "max_candidate_nodes" in df.columns:
        pivot = (
            df.groupby(["pc_alpha", "max_candidate_nodes"])["local_edge_density"]
              .mean()
              .reset_index()
              .pivot(index="pc_alpha", columns="max_candidate_nodes", values="local_edge_density")
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("max_candidate_nodes")
        ax.set_ylabel("pc_alpha")
        ax.set_title("Stage 2 Edge Density Heatmap")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")

        fig.colorbar(im, ax=ax, label="Mean edge density")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage2_pc_alpha_candidate_density_heatmap.png")
        )

    # ============================================================
    # 9. Heatmap: max_edges × strong_quantile -> core edges
    # ============================================================
    if "max_edges" in df.columns and "strong_quantile" in df.columns:
        pivot = (
            df.groupby(["strong_quantile", "max_edges"])["n_core_edges"]
              .mean()
              .reset_index()
              .pivot(index="strong_quantile", columns="max_edges", values="n_core_edges")
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("max_edges")
        ax.set_ylabel("strong_quantile")
        ax.set_title("Stage 2 Core Edge Count Heatmap")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center")

        fig.colorbar(im, ax=ax, label="Mean core edges")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage2_max_edges_strong_quantile_core_heatmap.png")
        )


# Stage 3 干预代价权重网格搜索
# 这一阶段固定 Stage 1 和 Stage 2 结果，搜索lambda_strength、lambda_temporal、lambda_node
def stage3_selection_score(row):
    """
    Stage 3 选择分数：
    不单纯选 objective，而是考虑恢复效果和代价之间的折中。
    """

    objective = float(row.get("objective", 0.0))
    traj_recovery = float(row.get("traj_recovery", 0.0))
    fc_recovery = float(row.get("fc_recovery", 0.0))
    strength_cost = float(row.get("strength_cost", 0.0))
    temporal_cost = float(row.get("temporal_cost", 0.0))
    node_cost = float(row.get("node_cost", 0.0))

    score = (
        objective
        + 0.2 * traj_recovery
        + 0.2 * fc_recovery
        - 0.1 * strength_cost
        - 0.1 * temporal_cost
        - 0.1 * node_cost
    )

    return float(score)


def run_stage3_once(
    config,
    model,
    roi_num,
    using_steps,
    lesion_idx,
    stage1_res,
    candidate_df,
):
    """
    用一组 lambda 配置运行 Stage 3。
    """

    if candidate_df is None or len(candidate_df) == 0:
        raise ValueError("candidate_df is empty. Cannot run Stage 3.")

    candidate_nodes = (
        candidate_df["node"]
        .astype(int)
        .drop_duplicates()
        .tolist()
    )

    stage3_res = run_single_node_search(
        model=model,
        candidate_nodes=candidate_nodes,
        roi_num=roi_num,
        using_steps=using_steps,
        lesion_idx=lesion_idx,

        horizon=config["horizon"],
        lesion_strength=config["lesion_strength"],
        lesion_mode=config["lesion_mode"],

        seed_inputs=stage1_res["seed_inputs"],
        healthy_traj=stage1_res["healthy_traj"],
        lesion_traj=stage1_res["lesion_traj"],

        signs=config["signs"],
        strengths=config["strengths"],
        schedules=config["schedules"],

        alpha_traj=config["alpha_traj"],
        beta_fc=config["beta_fc"],
        lambda_node=config["lambda_node"],
        lambda_strength=config["lambda_strength"],
        lambda_temporal=config["lambda_temporal"],

        batch_size=config["batch_size"],

        run_greedy=config["run_greedy"],
        max_interventions=config["max_interventions"],
        top_m_per_step=config["top_m_per_step"],
        min_marginal_gain=config["min_marginal_gain"],
        force_accept_first=config["force_accept_first"],
    )

    return stage3_res


def run_stage3_lambda_grid_search(
    base_config,
    model,
    roi_num,
    using_steps,
    lesion_idx,
    stage1_res,
    candidate_df,
):
    """
    Stage 3 网格搜索：
    lambda_strength / lambda_temporal / lambda_node
    """
    rows = []
    results = {}

    run_id = 0

    for grid_cfg in iter_grid(STAGE3_GRID):
        cfg = update_config(base_config, **grid_cfg)

        print(
            f"[Stage3 grid] run_id={run_id}, "
            f"lambda_strength={cfg['lambda_strength']}, "
            f"lambda_temporal={cfg['lambda_temporal']}, "
            f"lambda_node={cfg['lambda_node']}"
        )

        try:
            stage3_res = run_stage3_once(
                config=cfg,
                model=model,
                roi_num=roi_num,
                using_steps=using_steps,
                lesion_idx=lesion_idx,
                stage1_res=stage1_res,
                candidate_df=candidate_df,
            )

            metrics = stage3_res["best_metrics"]
            policy = stage3_res["best_policy"]

            row = {
                "run_id": run_id,
                **grid_cfg,

                "objective": float(metrics.get("objective", stage3_res.get("best_objective", np.nan))),
                "traj_recovery": float(metrics.get("traj_recovery", np.nan)),
                "fc_recovery": float(metrics.get("fc_recovery", np.nan)),
                "D_lesion_traj": float(metrics.get("D_lesion_traj", np.nan)),
                "D_policy_traj": float(metrics.get("D_policy_traj", np.nan)),
                "D_lesion_fc": float(metrics.get("D_lesion_fc", np.nan)),
                "D_policy_fc": float(metrics.get("D_policy_fc", np.nan)),

                "node_cost": float(metrics.get("node_cost", np.nan)),
                "strength_cost": float(metrics.get("strength_cost", np.nan)),
                "temporal_cost": float(metrics.get("temporal_cost", np.nan)),
                "policy_size": int(metrics.get("policy_size", len(policy))),
                "policy_nodes": safe_policy_nodes(policy),

                "status": "ok",
                "error": "",
            }

            row["selection_score"] = stage3_selection_score(row)

            results[run_id] = stage3_res

        except Exception as e:
            row = {
                "run_id": run_id,
                **grid_cfg,
                "status": "failed",
                "error": repr(e),
                "selection_score": -np.inf,
            }
            results[run_id] = None
            print(f"[Stage3 grid] failed: {repr(e)}")

        rows.append(row)
        run_id += 1

    df = pd.DataFrame(rows)

    valid = df[
        (df["status"] == "ok") &
        (df["traj_recovery"] > 0) &
        (df["fc_recovery"] > 0) &
        (df["policy_size"] <= base_config["max_interventions"])
    ].copy()

    if len(valid) > 0:
        best_row = valid.sort_values("selection_score", ascending=False).iloc[0]
    else:
        best_row = df.sort_values("selection_score", ascending=False).iloc[0]

    best_run_id = int(best_row["run_id"])

    print(f"[Stage3 grid] selected run_id = {best_run_id}")

    return {
        "stage3_grid_df": df,
        "stage3_grid_results": results,
        "best_stage3_run_id": best_run_id,
        "best_stage3_row": best_row,
        "best_stage3_result": results[best_run_id],
    }


def plot_stage3_grid_summary(stage3_grid_df, output_dir="./results/grid_search"):
    """
    Save Stage 3 grid-search figures to output_dir.
    Figures are not shown in Notebook.
    """

    df = stage3_grid_df.copy()
    df = df[df["status"] == "ok"].copy()

    if len(df) == 0:
        print("[Stage3 plot] No valid Stage 3 grid result to plot.")
        return

    # ============================================================
    # 1. Strength cost vs trajectory recovery
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(df["strength_cost"], df["traj_recovery"])
    ax.set_xlabel("Strength cost")
    ax.set_ylabel("Trajectory recovery")
    ax.set_title("Stage 3 Strength-Recovery Tradeoff")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()

    save_fig(
        fig,
        os.path.join(output_dir, "stage3_strength_recovery_tradeoff.png")
    )

    # ============================================================
    # 2. Temporal cost vs FC recovery
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(df["temporal_cost"], df["fc_recovery"])
    ax.set_xlabel("Temporal cost")
    ax.set_ylabel("FC recovery")
    ax.set_title("Stage 3 Temporal Cost vs FC Recovery")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()

    save_fig(
        fig,
        os.path.join(output_dir, "stage3_temporal_cost_fc_recovery.png")
    )

    # ============================================================
    # 3. Policy size vs objective
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(df["policy_size"], df["objective"])
    ax.set_xlabel("Policy size")
    ax.set_ylabel("Objective")
    ax.set_title("Stage 3 Policy Size vs Objective")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()

    save_fig(
        fig,
        os.path.join(output_dir, "stage3_policy_size_objective.png")
    )

    # ============================================================
    # 4. lambda_strength vs objective
    # ============================================================
    if "lambda_strength" in df.columns:
        grouped = (
            df.groupby("lambda_strength")
              .agg({
                  "objective": "mean",
                  "traj_recovery": "mean",
                  "fc_recovery": "mean",
                  "strength_cost": "mean",
              })
              .reset_index()
              .sort_values("lambda_strength")
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(grouped["lambda_strength"], grouped["objective"], marker="o", label="objective")
        ax.plot(grouped["lambda_strength"], grouped["traj_recovery"], marker="s", label="traj recovery")
        ax.plot(grouped["lambda_strength"], grouped["fc_recovery"], marker="^", label="FC recovery")

        ax.set_xlabel("lambda_strength")
        ax.set_ylabel("Mean value")
        ax.set_title("Stage 3 lambda_strength Sensitivity")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage3_lambda_strength_objective_recovery.png")
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(grouped["lambda_strength"], grouped["strength_cost"], marker="o")
        ax.set_xlabel("lambda_strength")
        ax.set_ylabel("Mean strength cost")
        ax.set_title("Stage 3 lambda_strength vs Strength Cost")
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage3_lambda_strength_cost.png")
        )

    # ============================================================
    # 5. lambda_temporal vs temporal cost / objective
    # ============================================================
    if "lambda_temporal" in df.columns:
        grouped = (
            df.groupby("lambda_temporal")
              .agg({
                  "objective": "mean",
                  "temporal_cost": "mean",
                  "traj_recovery": "mean",
                  "fc_recovery": "mean",
              })
              .reset_index()
              .sort_values("lambda_temporal")
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(grouped["lambda_temporal"], grouped["temporal_cost"], marker="o")
        ax.set_xlabel("lambda_temporal")
        ax.set_ylabel("Mean temporal cost")
        ax.set_title("Stage 3 lambda_temporal vs Temporal Cost")
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage3_lambda_temporal_cost.png")
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(grouped["lambda_temporal"], grouped["objective"], marker="o", label="objective")
        ax.plot(grouped["lambda_temporal"], grouped["traj_recovery"], marker="s", label="traj recovery")
        ax.plot(grouped["lambda_temporal"], grouped["fc_recovery"], marker="^", label="FC recovery")

        ax.set_xlabel("lambda_temporal")
        ax.set_ylabel("Mean value")
        ax.set_title("Stage 3 lambda_temporal Sensitivity")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage3_lambda_temporal_objective_recovery.png")
        )

    # ============================================================
    # 6. lambda_node vs policy size / objective
    # ============================================================
    if "lambda_node" in df.columns:
        grouped = (
            df.groupby("lambda_node")
              .agg({
                  "objective": "mean",
                  "policy_size": "mean",
                  "node_cost": "mean",
                  "traj_recovery": "mean",
                  "fc_recovery": "mean",
              })
              .reset_index()
              .sort_values("lambda_node")
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(grouped["lambda_node"], grouped["policy_size"], marker="o", label="policy size")
        ax.plot(grouped["lambda_node"], grouped["node_cost"], marker="s", label="node cost")

        ax.set_xlabel("lambda_node")
        ax.set_ylabel("Mean node count")
        ax.set_title("Stage 3 lambda_node vs Policy Size")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage3_lambda_node_policy_size.png")
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(grouped["lambda_node"], grouped["objective"], marker="o", label="objective")
        ax.plot(grouped["lambda_node"], grouped["traj_recovery"], marker="s", label="traj recovery")
        ax.plot(grouped["lambda_node"], grouped["fc_recovery"], marker="^", label="FC recovery")

        ax.set_xlabel("lambda_node")
        ax.set_ylabel("Mean value")
        ax.set_title("Stage 3 lambda_node Sensitivity")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage3_lambda_node_objective_recovery.png")
        )

    # ============================================================
    # 7. Heatmap: lambda_strength × lambda_temporal -> objective
    # ============================================================
    if "lambda_strength" in df.columns and "lambda_temporal" in df.columns:
        pivot = (
            df.groupby(["lambda_strength", "lambda_temporal"])["objective"]
              .mean()
              .reset_index()
              .pivot(index="lambda_strength", columns="lambda_temporal", values="objective")
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("lambda_temporal")
        ax.set_ylabel("lambda_strength")
        ax.set_title("Stage 3 Objective Heatmap")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")

        fig.colorbar(im, ax=ax, label="Mean objective")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage3_lambda_strength_temporal_objective_heatmap.png")
        )

    # ============================================================
    # 8. Heatmap: lambda_node × lambda_strength -> policy size
    # ============================================================
    if "lambda_node" in df.columns and "lambda_strength" in df.columns:
        pivot = (
            df.groupby(["lambda_node", "lambda_strength"])["policy_size"]
              .mean()
              .reset_index()
              .pivot(index="lambda_node", columns="lambda_strength", values="policy_size")
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("lambda_strength")
        ax.set_ylabel("lambda_node")
        ax.set_title("Stage 3 Policy Size Heatmap")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center")

        fig.colorbar(im, ax=ax, label="Mean policy size")
        fig.tight_layout()

        save_fig(
            fig,
            os.path.join(output_dir, "stage3_lambda_node_strength_policy_size_heatmap.png")
        )

# 总控算法：这个函数把 Stage 1、Stage 2、Stage 3 串起来，最后得到 BEST_CONFIG
def run_stagewise_grid_search_with_constraints(
    base_config,
    model,
    inputs,
    signals,
    roi_num,
    using_steps,
    lesion_idx,
    empirical_FC,
    roi_names=None,
    top_k=20,
    output_dir="./stagewise_grid_outputs",
):
    """
    总算法：
    Stage-wise Grid Search with Stability Constraints
    """

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # Stage 1-A: n_seed_windows convergence
    # ============================================================
    print("\n" + "=" * 80)
    print("Stage 1-A: n_seed_windows convergence grid")
    print("=" * 80)

    stage1_seed_res = run_stage1_seed_grid(
        base_config=base_config,
        model=model,
        inputs=inputs,
        roi_num=roi_num,
        using_steps=using_steps,
        lesion_idx=lesion_idx,
        roi_names=roi_names,
        top_k=top_k,
    )

    stage1_seed_df = stage1_seed_res["stage1_seed_df"]
    selected_B = stage1_seed_res["selected_n_seed_windows"]

    save_df(
        stage1_seed_df,
        os.path.join(output_dir, "stage1_seed_grid_summary.csv")
    )

    plot_stage1_seed_convergence(
        stage1_seed_df,
        top_k=top_k,
        output_dir=output_dir,
    )

    # 更新 config
    cfg_stage1 = update_config(
        base_config,
        n_seed_windows=selected_B,
    )

    # 如果你想强制论文主图用 64，可以取消下面注释：
    cfg_stage1["n_seed_windows"] = max(cfg_stage1["n_seed_windows"], 64)

    # ============================================================
    # Stage 1-B: lesion_strength / horizon sensitivity
    # ============================================================
    print("\n" + "=" * 80)
    print("Stage 1-B: lesion_strength / horizon sensitivity")
    print("=" * 80)

    stage1_sens_res = run_stage1_strength_horizon_sensitivity(
        base_config=cfg_stage1,
        model=model,
        inputs=inputs,
        roi_num=roi_num,
        using_steps=using_steps,
        lesion_idx=lesion_idx,
        roi_names=roi_names,
        top_k=top_k,
    )

    stage1_sens_df = stage1_sens_res["stage1_sensitivity_df"]

    save_df(
        stage1_sens_df,
        os.path.join(output_dir, "stage1_strength_horizon_sensitivity.csv")
    )
    plot_stage1_strength_horizon_sensitivity(
        stage1_sens_df,
        top_k=top_k,
        output_dir=output_dir,
    )

    # 正式主线仍用 base lesion_strength/horizon
    stage1_main_res = run_stage1_once(
        config=cfg_stage1,
        model=model,
        inputs=inputs,
        roi_num=roi_num,
        using_steps=using_steps,
        lesion_idx=lesion_idx,
        roi_names=roi_names,
    )

    # ============================================================
    # Stage 2: causal graph quality grid
    # ============================================================
    print("\n" + "=" * 80)
    print("Stage 2: causal graph quality grid")
    print("=" * 80)

    stage2_grid_res = run_stage2_grid_search(
        base_config=cfg_stage1,
        signals=signals,
        lesion_idx=lesion_idx,
        stage1_res=stage1_main_res,
        empirical_FC=empirical_FC,
        roi_names=roi_names,
    )

    stage2_grid_df = stage2_grid_res["stage2_grid_df"]
    best_stage2_row = stage2_grid_res["best_stage2_row"]
    best_stage2_result = stage2_grid_res["best_stage2_result"]

    save_df(
        stage2_grid_df,
        os.path.join(output_dir, "stage2_graph_quality_grid_summary.csv")
    )

    plot_stage2_grid_summary(
        stage2_grid_df,
        output_dir=output_dir,
    )
    stage2_selection_level = stage2_grid_res.get("stage2_selection_level", "unknown")
    cfg_stage2 = update_config(
        cfg_stage1,
        pc_alpha=float(best_stage2_row["pc_alpha"]),
        max_candidate_nodes=int(best_stage2_row["max_candidate_nodes"]),
        max_edges=int(best_stage2_row["max_edges"]),
        strong_quantile=float(best_stage2_row["strong_quantile"]),

        # selected FC-constrained pruning weights
        causal_weight=float(best_stage2_row["causal_weight"]),
        fc_weight=float(best_stage2_row["fc_weight"]),
        # record how Stage 2 config was selected
        stage2_selection_level=stage2_selection_level,
    )

    candidate_df = best_stage2_result["candidate_df"]

    # ============================================================
    # Stage 3: intervention-cost grid
    # ============================================================
    print("\n" + "=" * 80)
    print("Stage 3: intervention-cost weight grid")
    print("=" * 80)

    stage3_grid_res = run_stage3_lambda_grid_search(
        base_config=cfg_stage2,
        model=model,
        roi_num=roi_num,
        using_steps=using_steps,
        lesion_idx=lesion_idx,
        stage1_res=stage1_main_res,
        candidate_df=candidate_df,
    )

    stage3_grid_df = stage3_grid_res["stage3_grid_df"]
    best_stage3_row = stage3_grid_res["best_stage3_row"]
    best_stage3_result = stage3_grid_res["best_stage3_result"]

    save_df(
        stage3_grid_df,
        os.path.join(output_dir, "stage3_lambda_grid_summary.csv")
    )

    plot_stage3_grid_summary(
        stage3_grid_df,
        output_dir=output_dir,
    )

    cfg_stage3 = update_config(
        cfg_stage2,
        lambda_strength=float(best_stage3_row["lambda_strength"]),
        lambda_temporal=float(best_stage3_row["lambda_temporal"]),
        lambda_node=float(best_stage3_row["lambda_node"]),
    )

    BEST_CONFIG = cfg_stage3

    # 保存 BEST_CONFIG
    best_config_df = pd.DataFrame([
        {"parameter": k, "value": str(v)}
        for k, v in BEST_CONFIG.items()
    ])

    save_df(
        best_config_df,
        os.path.join(output_dir, "BEST_CONFIG.csv")
    )

    print("\n" + "=" * 80)
    print("Selected BEST_CONFIG")
    print("=" * 80)
    for k, v in BEST_CONFIG.items():
        print(f"{k}: {v}")

    return {
        "BEST_CONFIG": BEST_CONFIG,

        "stage1_seed_res": stage1_seed_res,
        "stage1_sensitivity_res": stage1_sens_res,
        "stage1_main_res": stage1_main_res,

        "stage2_grid_res": stage2_grid_res,
        "best_stage2_result": best_stage2_result,

        "stage3_grid_res": stage3_grid_res,
        "best_stage3_result": best_stage3_result,
    }

# 用 BEST_CONFIG 重新跑正式完整主线：用于最后正式跑一次完整 pipeline。
def run_full_pipeline_with_config(
    config,
    model,
    inputs,
    signals,
    roi_num,
    using_steps,
    lesion_idx,
    empirical_FC,
    roi_names=None,
):
    """
    使用 BEST_CONFIG 重新运行完整四阶段主线。
    """

    # ============================================================
    # Stage 1
    # ============================================================
    stage1_res = run_stage1_once(
        config=config,
        model=model,
        inputs=inputs,
        roi_num=roi_num,
        using_steps=using_steps,
        lesion_idx=lesion_idx,
        roi_names=roi_names,
    )

    # ============================================================
    # Stage 2
    # ============================================================
    stage2_res = run_stage2_once(
        config=config,
        signals=signals,
        lesion_idx=lesion_idx,
        stage1_res=stage1_res,
        empirical_FC=empirical_FC,
        roi_names=roi_names,
    )

    candidate_df = stage2_res["candidate_df"]

    # ============================================================
    # Stage 3
    # ============================================================
    stage3_res = run_stage3_once(
        config=config,
        model=model,
        roi_num=roi_num,
        using_steps=using_steps,
        lesion_idx=lesion_idx,
        stage1_res=stage1_res,
        candidate_df=candidate_df,
    )

    # ============================================================
    # Stage 4
    # ============================================================
    recovery_res = summarize_recovery(
        healthy_traj=stage1_res["healthy_traj"],
        lesion_traj=stage1_res["lesion_traj"],
        policy_traj=stage3_res["best_policy_traj"],
        policy=stage3_res["best_policy"],
        roi_names=roi_names,
        distance_metric="rmse",

        alpha_traj=config["alpha_traj"],
        beta_fc=config["beta_fc"],
        lambda_node=config["lambda_node"],
        lambda_strength=config["lambda_strength"],
        lambda_temporal=config["lambda_temporal"],
    )

    return {
        "config": config,
        "stage1_res": stage1_res,
        "stage2_res": stage2_res,
        "stage3_res": stage3_res,
        "recovery_res": recovery_res,

        "final_policy": stage3_res["best_policy"],
        "final_policy_traj": stage3_res["best_policy_traj"],
        "final_metrics": recovery_res["metrics"],
        "candidate_df": candidate_df,
        "lesion_core_edges": stage2_res["lesion_core_edges"],
    }


def summarize_final_pipeline_result(final_res):
    metrics = final_res["final_metrics"]
    policy = final_res["final_policy"]

    print("\n" + "=" * 80)
    print("Final Policy")
    print("=" * 80)
    for i, item in enumerate(policy, start=1):
        print(
            f"[{i}] node={item.get('node')}, "
            f"sign={item.get('sign')}, "
            f"strength={item.get('strength')}, "
            f"schedule={item.get('schedule')}"
        )

    print("\n" + "=" * 80)
    print("Final Recovery Metrics")
    print("=" * 80)
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            print(f"{k}: {v}")
        elif k in ["policy_nodes"]:
            print(f"{k}: {v}")


# ============================================================
# Lesion-node robustness analysis
# ============================================================
def safe_save_df(df, path):
    """
    Save DataFrame safely.
    If the target file is locked, save to a timestamped fallback file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[Saved] {path}")
        return str(path)

    except PermissionError:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")

        print(f"[Warning] Permission denied: {path}")
        print(f"[Saved fallback] {fallback_path}")
        return str(fallback_path)


def make_lesion_list_from_stage1_response(
    stage1_res,
    primary_lesion,
    roi_num,
    roi_names=None,
    top_n=3,
    mid_n=2,
    random_n=2,
    random_state=123,
    output_dir="./results/grid_search",
):
    """
    Construct lesion-node robustness list from Stage 1 response ranking.

    Components:
        1. primary lesion node
        2. top-response ROIs
        3. middle-response ROIs
        4. random-control ROIs

    Returns
    -------
    lesion_list : list[int]
    lesion_list_df : pd.DataFrame
    """

    os.makedirs(output_dir, exist_ok=True)

    primary_lesion = int(primary_lesion)

    if "lesion_response_df" not in stage1_res:
        raise KeyError("stage1_res must contain 'lesion_response_df'.")

    response_df = stage1_res["lesion_response_df"].copy()
    response_df["roi"] = response_df["roi"].astype(int)

    # Exclude primary lesion itself
    response_df_no_self = response_df[
        response_df["roi"] != primary_lesion
    ].copy()

    # ------------------------------------------------------------
    # Top-response lesion candidates
    # ------------------------------------------------------------
    top_lesions = (
        response_df_no_self["roi"]
        .drop_duplicates()
        .head(top_n)
        .astype(int)
        .tolist()
    )

    # ------------------------------------------------------------
    # Middle-response lesion candidates
    # ------------------------------------------------------------
    mid_start = len(response_df_no_self) // 2

    mid_lesions = (
        response_df_no_self["roi"]
        .drop_duplicates()
        .iloc[mid_start: mid_start + mid_n]
        .astype(int)
        .tolist()
    )

    # ------------------------------------------------------------
    # Random-control lesion candidates
    # ------------------------------------------------------------
    rng = np.random.default_rng(random_state)

    exclude_nodes = set([primary_lesion] + top_lesions + mid_lesions)

    candidate_random_pool = [
        int(x) for x in range(int(roi_num))
        if int(x) not in exclude_nodes
    ]

    if random_n > 0 and len(candidate_random_pool) > 0:
        random_n_actual = min(random_n, len(candidate_random_pool))
        random_lesions = rng.choice(
            candidate_random_pool,
            size=random_n_actual,
            replace=False,
        ).astype(int).tolist()
    else:
        random_lesions = []

    # ------------------------------------------------------------
    # Merge lesion list
    # ------------------------------------------------------------
    lesion_list = sorted(set(
        [primary_lesion]
        + top_lesions
        + mid_lesions
        + random_lesions
    ))

    def _source_of_node(node):
        if node == primary_lesion:
            return "primary"
        if node in top_lesions:
            return "top_response"
        if node in mid_lesions:
            return "mid_response"
        if node in random_lesions:
            return "random_control"
        return "unknown"

    lesion_list_df = pd.DataFrame({
        "lesion_idx": lesion_list,
        "lesion_name": [
            roi_names[i] if roi_names is not None and i < len(roi_names) else str(i)
            for i in lesion_list
        ],
        "source": [
            _source_of_node(i)
            for i in lesion_list
        ],
    })

    safe_save_df(
        lesion_list_df,
        os.path.join(output_dir, "lesion_node_list.csv")
    )

    return lesion_list, lesion_list_df


def run_lesion_node_robustness(
    lesion_list,
    config,
    model,
    inputs,
    signals,
    roi_num,
    using_steps,
    empirical_FC,
    roi_names=None,
    output_dir="./results/grid_search",
):
    """
    Fixed-BEST_CONFIG lesion-node robustness test.

    Only lesion_idx changes.
    Hyperparameters are fixed to avoid lesion-specific tuning.
    """

    os.makedirs(output_dir, exist_ok=True)

    rows = []
    results = {}

    for lesion_idx_i in lesion_list:
        lesion_idx_i = int(lesion_idx_i)

        lesion_name = (
            roi_names[lesion_idx_i]
            if roi_names is not None and lesion_idx_i < len(roi_names)
            else str(lesion_idx_i)
        )

        print("\n" + "=" * 80)
        print(f"[Lesion robustness] lesion_idx={lesion_idx_i}, lesion_name={lesion_name}")
        print("=" * 80)

        try:
            final_res_i = run_full_pipeline_with_config(
                config=config,
                model=model,
                inputs=inputs,
                signals=signals,
                roi_num=roi_num,
                using_steps=using_steps,
                lesion_idx=lesion_idx_i,
                empirical_FC=empirical_FC,
                roi_names=roi_names,
            )

            metrics = final_res_i.get("final_metrics", {})
            stage2_res = final_res_i.get("stage2_res", {})
            quality = stage2_res.get("quality", {})

            lesion_core_edges = final_res_i.get("lesion_core_edges", None)
            candidate_df = final_res_i.get("candidate_df", None)
            final_policy = final_res_i.get("final_policy", [])

            n_core_edges = (
                len(lesion_core_edges)
                if lesion_core_edges is not None
                else quality.get("n_core_edges", np.nan)
            )

            n_candidates = (
                len(candidate_df)
                if candidate_df is not None
                else np.nan
            )

            n_direct_edges = quality.get(
                "n_direct_lesion_edges_after_prune",
                np.nan
            )

            policy_nodes = []
            if isinstance(final_policy, list):
                for p in final_policy:
                    if isinstance(p, dict) and "node" in p:
                        policy_nodes.append(int(p["node"]))

            traj_recovery = metrics.get("traj_recovery", np.nan)
            fc_recovery = metrics.get("fc_recovery", np.nan)
            objective = metrics.get("objective", np.nan)

            success = (
                pd.notna(traj_recovery)
                and pd.notna(fc_recovery)
                and pd.notna(n_core_edges)
                and pd.notna(n_direct_edges)
                and traj_recovery > 0
                and fc_recovery > 0
                and n_core_edges >= 10
                and n_direct_edges > 0
            )

            row = {
                "lesion_idx": lesion_idx_i,
                "lesion_name": lesion_name,
                "status": "ok",

                "n_core_edges": n_core_edges,
                "n_direct_lesion_edges_after_prune": n_direct_edges,
                "n_candidates": n_candidates,

                "policy_size": len(final_policy) if isinstance(final_policy, list) else np.nan,
                "policy_nodes": str(policy_nodes),

                "objective": objective,
                "traj_recovery": traj_recovery,
                "fc_recovery": fc_recovery,

                "D_lesion_traj": metrics.get("D_lesion_traj", np.nan),
                "D_policy_traj": metrics.get("D_policy_traj", np.nan),
                "D_lesion_fc": metrics.get("D_lesion_fc", np.nan),
                "D_policy_fc": metrics.get("D_policy_fc", np.nan),

                "success": bool(success),
            }

            results[lesion_idx_i] = final_res_i

        except Exception as e:
            print("[Failed]", repr(e))
            print(traceback.format_exc())

            row = {
                "lesion_idx": lesion_idx_i,
                "lesion_name": lesion_name,
                "status": "failed",

                "n_core_edges": np.nan,
                "n_direct_lesion_edges_after_prune": np.nan,
                "n_candidates": np.nan,

                "policy_size": np.nan,
                "policy_nodes": "[]",

                "objective": np.nan,
                "traj_recovery": np.nan,
                "fc_recovery": np.nan,

                "D_lesion_traj": np.nan,
                "D_policy_traj": np.nan,
                "D_lesion_fc": np.nan,
                "D_policy_fc": np.nan,

                "success": False,
            }

            results[lesion_idx_i] = None

        rows.append(row)

    robustness_df = pd.DataFrame(rows)

    safe_save_df(
        robustness_df,
        os.path.join(output_dir, "lesion_node_robustness_summary.csv")
    )

    return {
        "robustness_df": robustness_df,
        "robustness_results": results,
    }


def summarize_lesion_node_robustness(
    robustness_df,
    output_dir="./results/grid_search",
):
    """
    Summarize lesion-node robustness results.
    """

    os.makedirs(output_dir, exist_ok=True)

    ok_df = robustness_df[robustness_df["status"] == "ok"].copy()

    if len(ok_df) == 0:
        overall_summary = {
            "n_lesions": len(robustness_df),
            "n_ok": 0,
            "n_success": int(robustness_df["success"].sum()) if "success" in robustness_df else 0,
            "success_rate": 0.0,
            "mean_traj_recovery": np.nan,
            "std_traj_recovery": np.nan,
            "mean_fc_recovery": np.nan,
            "std_fc_recovery": np.nan,
            "mean_n_core_edges": np.nan,
            "std_n_core_edges": np.nan,
            "mean_direct_lesion_edges": np.nan,
            "std_direct_lesion_edges": np.nan,
            "mean_n_candidates": np.nan,
            "mean_policy_size": np.nan,
        }
    else:
        overall_summary = {
            "n_lesions": len(robustness_df),
            "n_ok": int((robustness_df["status"] == "ok").sum()),
            "n_success": int(robustness_df["success"].sum()),
            "success_rate": float(robustness_df["success"].mean()),

            "mean_traj_recovery": ok_df["traj_recovery"].mean(),
            "std_traj_recovery": ok_df["traj_recovery"].std(),

            "mean_fc_recovery": ok_df["fc_recovery"].mean(),
            "std_fc_recovery": ok_df["fc_recovery"].std(),

            "mean_n_core_edges": ok_df["n_core_edges"].mean(),
            "std_n_core_edges": ok_df["n_core_edges"].std(),

            "mean_direct_lesion_edges": ok_df["n_direct_lesion_edges_after_prune"].mean(),
            "std_direct_lesion_edges": ok_df["n_direct_lesion_edges_after_prune"].std(),

            "mean_n_candidates": ok_df["n_candidates"].mean(),
            "mean_policy_size": ok_df["policy_size"].mean(),
        }

    overall_summary_df = pd.DataFrame([overall_summary])

    safe_save_df(
        overall_summary_df,
        os.path.join(output_dir, "lesion_node_robustness_overall_summary.csv")
    )

    return overall_summary_df


def plot_lesion_node_robustness(
    robustness_df,
    output_dir="./results/grid_search",
):
    """
    Plot lesion-node robustness recovery figure.
    Figure is saved to output_dir and not displayed.
    """

    _setup_paper_style()
    os.makedirs(output_dir, exist_ok=True)

    plot_df = robustness_df[robustness_df["status"] == "ok"].copy()

    if len(plot_df) == 0:
        print("[Robustness plot] No ok lesion result to plot.")
        return None

    labels = [
        f"{int(row['lesion_idx'])}\n{row['lesion_name']}"
        for _, row in plot_df.iterrows()
    ]

    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")

    ax.bar(
        x - 0.2,
        plot_df["traj_recovery"],
        width=0.4,
        label="Trajectory recovery",
        color="#1f77b4",
    )
    ax.bar(
        x + 0.2,
        plot_df["fc_recovery"],
        width=0.4,
        label="FC recovery",
        color="#ff7f0e",
    )

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylabel("Recovery ratio")
    ax.set_title("Lesion-node Robustness Analysis")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18))
    _beautify_axis(ax)

    fig.tight_layout()

    save_path = os.path.join(output_dir, "lesion_node_robustness_recovery.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved figure] {save_path}")

    return save_path


def print_lesion_node_robustness_conclusion(robustness_df):
    """
    Print simple conclusion for lesion-node robustness.
    """

    n_lesions = len(robustness_df)
    n_ok = int((robustness_df["status"] == "ok").sum())
    n_success = int(robustness_df["success"].sum())
    success_rate = robustness_df["success"].mean()

    print("=" * 80)
    print("Lesion-node robustness conclusion")
    print("=" * 80)
    print(f"Total lesion nodes tested: {n_lesions}")
    print(f"Successfully executed: {n_ok}")
    print(f"Successful robustness cases: {n_success}")
    print(f"Success rate: {success_rate:.2%}")

    if success_rate >= 0.7:
        print("Conclusion: The pipeline shows good lesion-node robustness.")
    elif success_rate >= 0.5:
        print("Conclusion: The pipeline shows moderate lesion-node robustness.")
    else:
        print("Conclusion: The pipeline is sensitive to lesion-node choice and needs further analysis.")
