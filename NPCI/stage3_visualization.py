import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from typing import Optional, Sequence, Dict
from lesion_response import build_schedule_vector
from typing import Optional, Sequence, Union

def plot_stage3_objective_ranking_bar(
    single_df: pd.DataFrame,
    top_k: int = 10,
    roi_names: Optional[Sequence[str]] = None,
    mode: str = "grouped",   # "grouped" or "objective_only"
    figsize=(10.8, 4.8),
    show_cost: bool = False,
    cost_mode: str = "inset",  # "none", "line", or "inset"
    x_rotation: int = 45,
    mark_best: bool = True,
    legend_outside: bool = True,
    ax=None,
    save_path: Optional[str] = None,
):
    """
    Paper-style ranking plot for Stage 3 candidate intervention nodes.

    Purpose
    -------
    Show which candidate intervention node is best and why:
      - objective
      - traj_recovery
      - fc_recovery
      - optional strength_cost / temporal_cost

    Recommended for paper main figure:
      top_k=10
      mode="grouped"
      show_cost=False or show_cost=True with cost_mode="inset"

    Parameters
    ----------
    single_df:
        Output table from stage3_res["single_df"] or attach_candidate_scores(...).

        Required columns:
            node, objective, traj_recovery, fc_recovery

        Optional columns:
            strength_cost, temporal_cost, roi_name, candidate_score

    top_k:
        Number of top nodes to show. Recommended: 10 or 15.

    mode:
        "objective_only":
            only show objective bars.
        "grouped":
            show objective / traj_recovery / fc_recovery grouped bars.

    show_cost:
        Whether to visualize strength_cost and temporal_cost.

    cost_mode:
        "none":
            do not show cost.
        "line":
            overlay cost as faint lines using a second y-axis.
        "inset":
            show cost in a small inset panel on the upper-right.

    mark_best:
        Mark the best objective node.

    legend_outside:
        Put legend outside the right side.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    required_cols = ["node", "objective", "traj_recovery", "fc_recovery"]

    for c in required_cols:
        if c not in single_df.columns:
            raise KeyError(f"single_df must contain column: {c}")

    df = (
        single_df.copy()
        .sort_values("objective", ascending=False)
        .head(int(top_k))
        .reset_index(drop=True)
    )

    if len(df) == 0:
        raise ValueError("No candidate nodes to plot.")

    # -------------------------
    # label selection
    # -------------------------
    name_col = None
    for c in ["roi_name", "roi_name_stage1", "roi_name_std", "node_label"]:
        if c in df.columns:
            name_col = c
            break

    if name_col is not None:
        labels = df[name_col].astype(str).tolist()
    elif roi_names is not None:
        labels = [roi_names[int(x)] for x in df["node"].values]
    else:
        labels = [str(int(x)) for x in df["node"].values]

    x = np.arange(len(df))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    # 给右侧 legend 留空间
    if created_fig and legend_outside:
        fig.subplots_adjust(right=0.78, bottom=0.22, top=0.88)
    elif created_fig:
        fig.subplots_adjust(bottom=0.22, top=0.88)

    # -------------------------
    # mode 1: objective only
    # -------------------------
    if mode == "objective_only":
        bars = ax.bar(
            x,
            df["objective"].astype(float).values,
            alpha=0.90,
            edgecolor="black",
            linewidth=0.45,
            label="objective",
            color="#1f77b4",
        )

        ax.set_ylabel("Objective", fontsize=11)
        ax.set_title(f"Top {top_k} candidate intervention nodes", fontsize=13)

        if mark_best and len(bars) > 0:
            bars[0].set_edgecolor("black")
            bars[0].set_linewidth(1.8)

    # -------------------------
    # mode 2: grouped bars
    # -------------------------
    elif mode == "grouped":
        width = 0.25

        bars_obj = ax.bar(
            x - width,
            df["objective"].astype(float).values,
            width=width,
            alpha=0.92,
            edgecolor="black",
            linewidth=0.40,
            label="objective",
            color="#1f77b4",
        )

        bars_traj = ax.bar(
            x,
            df["traj_recovery"].astype(float).values,
            width=width,
            alpha=0.90,
            edgecolor="black",
            linewidth=0.35,
            label="traj_recovery",
            color="#ff7f0e",
        )

        bars_fc = ax.bar(
            x + width,
            df["fc_recovery"].astype(float).values,
            width=width,
            alpha=0.90,
            edgecolor="black",
            linewidth=0.35,
            label="fc_recovery",
            color="#2ca02c",
        )

        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"Top {top_k} candidate intervention nodes", fontsize=13)

        # 高亮 best node 对应的 objective bar
        if mark_best and len(bars_obj) > 0:
            bars_obj[0].set_edgecolor("black")
            bars_obj[0].set_linewidth(1.8)

    else:
        raise ValueError("mode must be one of {'objective_only', 'grouped'}")

    # -------------------------
    # mark best node
    # -------------------------
    if mark_best:
        best_node = int(df.loc[0, "node"])
        best_label = labels[0]
        ymax = float(max(df["objective"].max(), df["traj_recovery"].max(), df["fc_recovery"].max()))

        ax.annotate(
            f"best node\n{best_label}",
            xy=(0, df.loc[0, "objective"]),
            xytext=(0, ymax * 1.10),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="black",
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.0,
                color="black",
                shrinkA=2,
                shrinkB=3,
            ),
        )

    # -------------------------
    # optional cost display
    # -------------------------
    has_strength_cost = "strength_cost" in df.columns
    has_temporal_cost = "temporal_cost" in df.columns
    should_show_cost = show_cost and cost_mode != "none" and (has_strength_cost or has_temporal_cost)

    if should_show_cost and cost_mode == "line":
        ax2 = ax.twinx()

        if has_strength_cost:
            ax2.plot(
                x,
                df["strength_cost"].astype(float).values,
                marker="o",
                linestyle="--",
                linewidth=1.2,
                alpha=0.45,
                label="strength_cost",
                color="#1f77b4",
            )

        if has_temporal_cost:
            ax2.plot(
                x,
                df["temporal_cost"].astype(float).values,
                marker="s",
                linestyle=":",
                linewidth=1.2,
                alpha=0.45,
                label="temporal_cost",
                color="#ff7f0e",
            )

        ax2.set_ylabel("Cost", fontsize=10)
        ax2.tick_params(labelsize=9)

    elif should_show_cost and cost_mode == "inset":
        # 小图放在右上角，不干扰主图
        inset_ax = ax.inset_axes([0.66, 0.58, 0.30, 0.32])

        if has_strength_cost:
            inset_ax.plot(
                x,
                df["strength_cost"].astype(float).values,
                marker="o",
                linestyle="--",
                linewidth=1.1,
                alpha=0.70,
                label="strength",
                color="#1f77b4",
            )

        if has_temporal_cost:
            inset_ax.plot(
                x,
                df["temporal_cost"].astype(float).values,
                marker="s",
                linestyle=":",
                linewidth=1.1,
                alpha=0.70,
                label="temporal",
                color="#ff7f0e",
            )

        inset_ax.set_title("Cost", fontsize=8, pad=2)
        inset_ax.set_xticks([])
        inset_ax.tick_params(axis="y", labelsize=7)
        inset_ax.grid(axis="y", alpha=0.20, linestyle="--")
        inset_ax.spines["top"].set_visible(False)
        inset_ax.spines["right"].set_visible(False)
        inset_ax.legend(frameon=False, fontsize=7, loc="upper right")

    # -------------------------
    # common style
    # -------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=x_rotation, ha="right", fontsize=9)

    ax.tick_params(axis="y", labelsize=9)
    ax.grid(alpha=0.22, linestyle="--", axis="y")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 适当提高 y 上界，给 best node 标注留空间
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)

    # node id 放在 x 轴下方，字体小一点
    for i, node in enumerate(df["node"].values):
        ax.text(
            i,
            ax.get_ylim()[0] - 0.055 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f"{int(node)}",
            ha="center",
            va="top",
            fontsize=7.5,
            color="dimgray",
            clip_on=False,
        )

    # -------------------------
    # legend
    # -------------------------
    handles, legend_labels = ax.get_legend_handles_labels()

    if should_show_cost and cost_mode == "line":
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        legend_labels += l2

    if legend_outside:
        ax.legend(
            handles,
            legend_labels,
            frameon=False,
            fontsize=9,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.00),
            borderaxespad=0.0,
        )
    else:
        ax.legend(
            handles,
            legend_labels,
            frameon=False,
            fontsize=9,
            ncol=3,
            loc="upper right",
        )

    # -------------------------
    # save
    # -------------------------
    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    return fig, ax


def plot_stage3_best_policy_card(
    best_row: pd.Series,
    title: str = "Stage 3 best policy",
    figsize=(6.8, 3.8),
    save_path: Optional[str] = None,
):
    """
    回答的是：最终选中的最优单节点干预策略到底是什么？包括：node、sign、strength、schedule、objective、traj_recovery、fc_recovery
    Parameters
    best_row :
        Usually single_df.iloc[0]
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    node = best_row.get("node", "NA")
    sign = best_row.get("sign", "NA")
    strength = best_row.get("strength", "NA")
    schedule = best_row.get("schedule", "NA")
    objective = best_row.get("objective", np.nan)
    traj_recovery = best_row.get("traj_recovery", np.nan)
    fc_recovery = best_row.get("fc_recovery", np.nan)
    strength_cost = best_row.get("strength_cost", np.nan)
    temporal_cost = best_row.get("temporal_cost", np.nan)

    lines = [
        f"node            : {node}",
        f"sign            : {sign}",
        f"strength        : {strength}",
        f"schedule        : {schedule}",
        "",
        f"objective       : {objective:.6f}" if pd.notna(objective) else "objective       : NA",
        f"traj_recovery   : {traj_recovery:.6f}" if pd.notna(traj_recovery) else "traj_recovery   : NA",
        f"fc_recovery     : {fc_recovery:.6f}" if pd.notna(fc_recovery) else "fc_recovery     : NA",
        f"strength_cost   : {strength_cost:.6f}" if pd.notna(strength_cost) else "strength_cost   : NA",
        f"temporal_cost   : {temporal_cost:.6f}" if pd.notna(temporal_cost) else "temporal_cost   : NA",
    ]

    ax.text(
        0.03, 0.95, title,
        ha="left", va="top",
        fontsize=14, fontweight="bold",
        transform=ax.transAxes
    )

    ax.text(
        0.05, 0.82, "\n".join(lines),
        ha="left", va="top",
        fontsize=12, family="monospace",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="lightgray")
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax


def plot_stage3_best_node_grid_heatmap(
    grid_df: pd.DataFrame,
    metric: str = "objective",
    title: Optional[str] = None,
    figsize=(10.8, 4.8),
    annotate: bool = True,
    cmap: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot policy grid heatmap for the best intervention node.

    x-axis: strength
    y-axis: schedule
    one panel per sign (-1 / +1)

    Main purpose:
        1. Check whether sign = +1 or sign = -1 is better.
        2. Check which strength works best.
        3. Check which temporal schedule is more stable.
        4. Explain why the best node / policy is selected.

    Parameters
    ----------
    grid_df:
        Usually stage3_res["all_grid"][best_node]

    metric:
        "objective", "traj_recovery", or "fc_recovery"

    cmap:
        If None:
            - use "RdBu_r" when metric contains both positive and negative values
            - use "Reds" otherwise
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    required_cols = ["sign", "strength", "schedule", metric]
    for c in required_cols:
        if c not in grid_df.columns:
            raise KeyError(f"grid_df must contain column: {c}")

    df = grid_df.copy()

    signs = sorted(df["sign"].dropna().unique().tolist())
    if len(signs) == 0:
        raise ValueError("grid_df has no sign values.")

    # --------------------------------------------------
    # 1. 固定 schedule / strength 顺序
    # --------------------------------------------------
    schedule_order = list(dict.fromkeys(df["schedule"].tolist()))
    strength_order = sorted(df["strength"].dropna().unique().tolist())

    # --------------------------------------------------
    # 2. 全局颜色范围，保证不同 sign 面板可比
    # --------------------------------------------------
    vals = df[metric].astype(float).values
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        raise ValueError(f"No valid numeric values for metric: {metric}")

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    # 如果有正有负，以 0 为中心，更适合 objective 这种指标
    use_diverging = (vmin < 0) and (vmax > 0)

    if cmap is None:
        cmap = "RdBu_r" if use_diverging else "Reds"

    if use_diverging:
        absmax = max(abs(vmin), abs(vmax))
        norm = mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # --------------------------------------------------
    # 3. 画布布局：右侧预留 colorbar 空间
    # --------------------------------------------------
    ncols = len(signs)

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=figsize,
        squeeze=False
    )

    axes = axes.ravel()

    fig.subplots_adjust(
        left=0.08,
        right=0.88,     # 给外置 colorbar 留位置
        bottom=0.18,
        top=0.84,
        wspace=0.18
    )

    im_ref = None

    # --------------------------------------------------
    # 4. 逐个 sign 画 heatmap
    # --------------------------------------------------
    for ax, s in zip(axes, signs):
        sub = df[df["sign"] == s].copy()

        pivot = sub.pivot_table(
            index="schedule",
            columns="strength",
            values=metric,
            aggfunc="mean"
        )

        pivot = pivot.reindex(index=schedule_order)
        pivot = pivot.reindex(columns=strength_order)

        im = ax.imshow(
            pivot.values.astype(float),
            aspect="auto",
            cmap=cmap,
            norm=norm
        )
        im_ref = im

        ax.set_title(f"sign = {int(s)}", fontsize=12, pad=8)

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(
            [str(x) for x in pivot.columns],
            rotation=0,
            fontsize=9
        )

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(
            pivot.index,
            fontsize=9
        )

        ax.set_xlabel("strength", fontsize=10)
        ax.set_ylabel("schedule", fontsize=10)

        # 轻网格线，增强表格感
        ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
        ax.grid(
            which="minor",
            color="#e5e5e5",
            linestyle="-",
            linewidth=0.6
        )
        ax.tick_params(which="minor", bottom=False, left=False)

        # --------------------------------------------------
        # 5. 数值标注
        # --------------------------------------------------
        if annotate:
            finite_vals = pivot.values[np.isfinite(pivot.values)]
            local_mid = np.nanmean(finite_vals) if len(finite_vals) > 0 else 0.0

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.iloc[i, j]

                    if pd.notna(val):
                        # 背景较深时用白字，否则用黑字
                        text_color = "white" if float(val) > local_mid else "black"

                        ax.text(
                            j,
                            i,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color=text_color
                        )

    # --------------------------------------------------
    # 6. 外置共享 colorbar
    # --------------------------------------------------
    cax = fig.add_axes([0.905, 0.20, 0.018, 0.60])
    cbar = fig.colorbar(im_ref, cax=cax)
    cbar.set_label(metric, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # --------------------------------------------------
    # 7. 总标题
    # --------------------------------------------------
    if title is None:
        title = f"Best-node policy grid heatmap ({metric})"

    fig.suptitle(
        title,
        fontsize=13,
        y=0.96
    )

    # --------------------------------------------------
    # 8. 保存
    # --------------------------------------------------
    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    return fig, axes


def make_stage3_topk_table(
    single_df: pd.DataFrame,
    top_k: int = 20,
    roi_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a clean Top-K candidate table for Stage 3.
    """
    required_cols = ["node", "objective", "traj_recovery", "fc_recovery"]
    for c in required_cols:
        if c not in single_df.columns:
            raise KeyError(f"single_df must contain column: {c}")

    df = single_df.copy().sort_values("objective", ascending=False).head(int(top_k)).reset_index(drop=True)

    name_col = None
    for c in ["roi_name", "roi_name_stage1", "roi_name_std", "node_label"]:
        if c in df.columns:
            name_col = c
            break

    if name_col is not None:
        roi_label = df[name_col].astype(str).tolist()
    elif roi_names is not None:
        roi_label = [roi_names[int(x)] for x in df["node"].values]
    else:
        roi_label = [str(int(x)) for x in df["node"].values]

    out = pd.DataFrame({
        "rank": np.arange(1, len(df) + 1),
        "node": df["node"].astype(int).values,
        "roi_label": roi_label,
        "sign": df["sign"].values if "sign" in df.columns else np.nan,
        "strength": df["strength"].values if "strength" in df.columns else np.nan,
        "schedule": df["schedule"].values if "schedule" in df.columns else "",
        "objective": np.round(df["objective"].values, 6),
        "traj_recovery": np.round(df["traj_recovery"].values, 6),
        "fc_recovery": np.round(df["fc_recovery"].values, 6),
    })

    if "strength_cost" in df.columns:
        out["strength_cost"] = np.round(df["strength_cost"].values, 6)

    if "temporal_cost" in df.columns:
        out["temporal_cost"] = np.round(df["temporal_cost"].values, 6)

    return out


def plot_stage3_topk_table(
    table_df: pd.DataFrame,
    title: str = "Top-K candidate intervention nodes",
    figsize: Optional[tuple] = None,
    font_size: int = 9,
    save_path: Optional[str] = None,
):
    """
    回答：除了最优节点之外，其他候选节点表现怎么样？
    你可以比较：objective、traj_recovery、fc_recovery、策略参数
    这很适合放补充材料。
    """
    df = table_df.copy()

    n_rows, n_cols = df.shape
    if figsize is None:
        figsize = (min(16, 1.45 * n_cols), max(3.2, 0.42 * n_rows + 1.6))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1.0, 1.25)

    # 表头加粗
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eaeaea")
        else:
            if row % 2 == 0:
                cell.set_facecolor("#f9f9f9")

    ax.set_title(title, fontsize=13, pad=12)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax


def plot_stage3_best_policy_schedule(
    best_row,
    horizon: int,
    schedules=("constant", "early_pulse", "ramp_up", "ramp_decay"),
    show_all_schedules: bool = True,
    title: str = "Best intervention schedule",
    figsize=(7.4, 4.4),
    background_alpha: float = 0.28,
    save_path: str = None,
):
    """
    Plot the temporal schedule of the best intervention policy.

    This figure answers:
        How is the selected intervention applied over rollout time?

    x-axis:
        rollout step

    left y-axis:
        schedule weight

    right y-axis:
        actual intervention amplitude = sign × strength × schedule weight

    best_row should contain:
        node, sign, strength, schedule
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    node = int(best_row["node"])
    sign = int(best_row["sign"])
    strength = float(best_row["strength"])
    best_schedule = str(best_row["schedule"])

    x = np.arange(int(horizon))

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.74, bottom=0.16, top=0.86)

    # --------------------------------------------------
    # 1. Candidate schedules as background references
    # --------------------------------------------------
    if show_all_schedules:
        for sch in schedules:
            v = build_schedule_vector(str(sch), int(horizon))

            is_best = str(sch) == best_schedule

            ax.plot(
                x,
                v,
                linewidth=2.4 if is_best else 1.2,
                alpha=0.95 if is_best else background_alpha,
                linestyle="-" if is_best else "--",
                marker="o" if is_best else None,
                markersize=3.8 if is_best else 0,
                label=f"Best schedule: {sch}" if is_best else str(sch),
                zorder=3 if is_best else 1,
            )
    else:
        best_v = build_schedule_vector(best_schedule, int(horizon))
        ax.plot(
            x,
            best_v,
            linewidth=2.6,
            marker="o",
            markersize=3.8,
            label=f"Best schedule: {best_schedule}",
            zorder=3,
        )

    # --------------------------------------------------
    # 2. Actual intervention amplitude
    # --------------------------------------------------
    best_v = build_schedule_vector(best_schedule, int(horizon))
    actual_amp = sign * strength * best_v

    ax2 = ax.twinx()

    ax2.plot(
        x,
        actual_amp,
        linewidth=2.0,
        marker="s",
        markersize=3.2,
        linestyle=":",
        label="Actual intervention amplitude",
        zorder=4,
    )

    # --------------------------------------------------
    # 3. Axes
    # --------------------------------------------------
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Rollout step", fontsize=10)
    ax.set_ylabel("Schedule weight", fontsize=10)
    ax2.set_ylabel("Intervention amplitude", fontsize=10)

    ax.set_ylim(-0.05, 1.05)

    amp_abs_max = max(float(np.max(np.abs(actual_amp))), 1e-6)
    ax2.set_ylim(-1.15 * amp_abs_max, 1.15 * amp_abs_max)

    ax.tick_params(labelsize=9)
    ax2.tick_params(labelsize=9)

    ax.grid(alpha=0.22, linestyle="--", linewidth=0.7)

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # --------------------------------------------------
    # 4. Policy parameter box
    # --------------------------------------------------
    policy_text = (
        f"node = {node}\n"
        f"sign = {sign:+d}\n"
        f"strength = {strength:g}\n"
        f"schedule = {best_schedule}"
    )

    ax.text(
        0.03,
        0.06,
        policy_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="lightgray",
            alpha=0.92,
        ),
    )

    # --------------------------------------------------
    # 5. Legend outside
    # --------------------------------------------------
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax.legend(
        h1 + h2,
        l1 + l2,
        frameon=False,
        fontsize=8.5,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.00),
        borderaxespad=0.0,
    )

    # --------------------------------------------------
    # 6. Save
    # --------------------------------------------------
    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    return fig, ax, ax2


def plot_stage3_objective_decomposition(
    best_row,
    alpha_traj: float = 1.0,
    beta_fc: float = 0.8,
    lambda_strength: float = 0.05,
    lambda_temporal: float = 0.03,
    title: str = "Best-policy objective decomposition",
    figsize=(7.4, 4.8),
    x_rotation: int = 45,
    show_formula: bool = True,
    show_policy_box: bool = True,
    save_path: str = None,
):
    """
    Paper-style objective decomposition for the best Stage 3 policy.

    Objective:
        objective = α * traj_recovery
                  + β * fc_recovery
                  - λs * strength_cost
                  - λt * temporal_cost

    Positive contribution:
        α * traj_recovery
        β * fc_recovery

    Negative contribution:
        -λs * strength_cost
        -λt * temporal_cost

    Final bar:
        objective

    best_row must contain:
        traj_recovery, fc_recovery, strength_cost, temporal_cost, objective

    Optional:
        node, sign, strength, schedule
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    required_cols = [
        "traj_recovery",
        "fc_recovery",
        "strength_cost",
        "temporal_cost",
        "objective",
    ]

    for c in required_cols:
        if c not in best_row.index:
            raise KeyError(f"best_row must contain column/key: {c}")

    traj_recovery = float(best_row["traj_recovery"])
    fc_recovery = float(best_row["fc_recovery"])
    strength_cost = float(best_row["strength_cost"])
    temporal_cost = float(best_row["temporal_cost"])
    objective = float(best_row["objective"])

    # --------------------------------------------------
    # 1. Weighted contributions
    # --------------------------------------------------
    contrib_traj = alpha_traj * traj_recovery
    contrib_fc = beta_fc * fc_recovery
    contrib_strength = -lambda_strength * strength_cost
    contrib_temporal = -lambda_temporal * temporal_cost

    names = [
        r"$\alpha$ · traj",
        r"$\beta$ · FC",
        r"$-\lambda_s$ · strength",
        r"$-\lambda_t$ · temporal",
        "objective",
    ]

    values = np.array(
        [
            contrib_traj,
            contrib_fc,
            contrib_strength,
            contrib_temporal,
            objective,
        ],
        dtype=float,
    )

    # --------------------------------------------------
    # 2. Colors
    # --------------------------------------------------
    pos_color = "#4C84B1"       # positive recovery contribution
    neg_color = "#C95F5F"       # negative cost contribution
    obj_color = "#666666"       # final objective

    colors = []

    for i, v in enumerate(values):
        if i == len(values) - 1:
            colors.append(obj_color)
        elif v >= 0:
            colors.append(pos_color)
        else:
            colors.append(neg_color)

    # --------------------------------------------------
    # 3. Figure
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(names))

    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.65,
        alpha=0.92,
    )

    # objective bar: slightly emphasized
    bars[-1].set_linewidth(1.2)
    bars[-1].set_alpha(0.96)

    # zero line
    ax.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        alpha=0.85,
    )

    # --------------------------------------------------
    # 4. Value labels
    # --------------------------------------------------
    y_span = max(1e-6, float(np.max(np.abs(values))))
    offset = 0.035 * y_span

    for rect, v in zip(bars, values):
        xpos = rect.get_x() + rect.get_width() / 2

        if abs(v) < 5e-4:
            label = "0.000"
        else:
            label = f"{v:.3f}"

        if v >= 0:
            ypos = v + offset
            va = "bottom"
        else:
            ypos = v - offset
            va = "top"

        ax.text(
            xpos,
            ypos,
            label,
            ha="center",
            va=va,
            fontsize=8.5,
        )

    # --------------------------------------------------
    # 5. Axes style
    # --------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(
        names,
        rotation=x_rotation,
        ha="right",
        fontsize=9,
    )

    ax.set_ylabel("Contribution to objective", fontsize=10)
    ax.set_title(title, fontsize=13, pad=10)

    ax.grid(
        alpha=0.22,
        linestyle="--",
        axis="y",
        linewidth=0.7,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.04 * (ymax - ymin), ymax + 0.12 * (ymax - ymin))

    # --------------------------------------------------
    # 6. Formula box, smaller and cleaner
    # --------------------------------------------------
    if show_formula:
        formula_text = (
            "objective = α·traj + β·FC\n"
            "            − λs·strength − λt·temporal"
        )

        ax.text(
            0.02,
            0.97,
            formula_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.90,
            ),
        )

    # --------------------------------------------------
    # 7. Optional policy box
    # --------------------------------------------------
    if show_policy_box and all(
        k in best_row.index for k in ["node", "sign", "strength", "schedule"]
    ):
        policy_text = (
            f"node = {int(best_row['node'])}\n"
            f"sign = {int(best_row['sign']):+d}\n"
            f"strength = {float(best_row['strength']):g}\n"
            f"schedule = {str(best_row['schedule'])}"
        )

        ax.text(
            0.98,
            0.05,
            policy_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.90,
            ),
        )

    plt.tight_layout()

    # --------------------------------------------------
    # 8. Save
    # --------------------------------------------------
    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    return fig, ax


def plot_stage3_topk_objective_decomposition(
    single_df: pd.DataFrame,
    top_k: int = 10,
    alpha_traj: float = 1.0,
    beta_fc: float = 0.8,
    lambda_strength: float = 0.05,
    lambda_temporal: float = 0.03,
    label_col: str = "node",
    figsize=(10.8, 5.2),
    show_objective_line: bool = True,
    x_rotation: int = 45,
    legend_outside: bool = True,
    title: str = None,
    save_path: str = None,
):
    """
    Plot Top-K candidate objective decomposition as stacked bars.

    作用：
    - 比较多个候选节点的 objective 组成
    - 看哪些节点是 traj_recovery 驱动更强
    - 看哪些节点是 fc_recovery 驱动更强
    - 看哪些节点虽然有 recovery，但被 cost 吃掉了

    Parameters
    ----------
    single_df :
        Stage 3 result table, usually stage3_res["single_df"]
        Must contain columns:
            node, objective, traj_recovery, fc_recovery,
            strength_cost, temporal_cost

    top_k :
        Number of top candidates to display.
        推荐：
            主补图可用 10
            如果太挤可改成 8

    label_col :
        x-axis label column, usually "node" or "roi_name"
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    required_cols = [
        "objective",
        "traj_recovery",
        "fc_recovery",
        "strength_cost",
        "temporal_cost",
    ]
    for c in required_cols:
        if c not in single_df.columns:
            raise KeyError(f"single_df must contain column: {c}")

    if label_col not in single_df.columns:
        raise KeyError(f"single_df must contain label column: {label_col}")

    # --------------------------------------------------
    # 1. 排序并截取 Top-K
    # --------------------------------------------------
    df = (
        single_df.copy()
        .sort_values("objective", ascending=False)
        .head(int(top_k))
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 2. 加权贡献
    # --------------------------------------------------
    df["traj_contrib"] = alpha_traj * df["traj_recovery"].astype(float)
    df["fc_contrib"] = beta_fc * df["fc_recovery"].astype(float)
    df["strength_penalty"] = -lambda_strength * df["strength_cost"].astype(float)
    df["temporal_penalty"] = -lambda_temporal * df["temporal_cost"].astype(float)

    labels = [str(x) for x in df[label_col].values]
    x = np.arange(len(df))

    # --------------------------------------------------
    # 3. 颜色统一
    # --------------------------------------------------
    color_traj = "#2C7BB6"       # 蓝
    color_fc = "#FF8C00"         # 橙
    color_strength = "#2CA02C"   # 绿
    color_temporal = "#D62728"   # 红
    color_objective = "black"    # 黑色折线

    # --------------------------------------------------
    # 4. 画图
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    if legend_outside:
        fig.subplots_adjust(right=0.80, bottom=0.22, top=0.88)
    else:
        fig.subplots_adjust(bottom=0.22, top=0.88)

    # 正向堆叠
    pos_bottom = np.zeros(len(df))

    b1 = ax.bar(
        x,
        df["traj_contrib"].values,
        bottom=pos_bottom,
        label=r"$\alpha$ · traj_recovery",
        color=color_traj,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
    )
    pos_bottom += df["traj_contrib"].values

    b2 = ax.bar(
        x,
        df["fc_contrib"].values,
        bottom=pos_bottom,
        label=r"$\beta$ · fc_recovery",
        color=color_fc,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
    )
    pos_bottom += df["fc_contrib"].values

    # 负向堆叠
    neg_bottom = np.zeros(len(df))

    b3 = ax.bar(
        x,
        df["strength_penalty"].values,
        bottom=neg_bottom,
        label=r"$-\lambda_s$ · strength_cost",
        color=color_strength,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
    )
    neg_bottom += df["strength_penalty"].values

    b4 = ax.bar(
        x,
        df["temporal_penalty"].values,
        bottom=neg_bottom,
        label=r"$-\lambda_t$ · temporal_cost",
        color=color_temporal,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
    )
    neg_bottom += df["temporal_penalty"].values

    # --------------------------------------------------
    # 5. objective 折线：更突出
    # --------------------------------------------------
    if show_objective_line:
        ax.plot(
            x,
            df["objective"].values,
            marker="o",
            linestyle="--",
            linewidth=2.0,
            markersize=5.5,
            color=color_objective,
            label="objective",
            zorder=6,
        )

    # 每个点上标 objective 数值
    for i, v in enumerate(df["objective"].values):
        ax.text(
            i,
            v + 0.012,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="dimgray",
        )

    # --------------------------------------------------
    # 6. 坐标轴与标题
    # --------------------------------------------------
    ax.axhline(0.0, color="black", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=x_rotation, ha="right", fontsize=9)

    ax.set_ylabel("Contribution to objective", fontsize=10)
    ax.set_xlabel("Candidate node", fontsize=10)

    if title is None:
        title = f"Top {top_k} candidate objective decomposition"
    ax.set_title(title, fontsize=13, pad=10)

    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    # --------------------------------------------------
    # 7. legend 放右侧
    # --------------------------------------------------
    handles, legend_labels = ax.get_legend_handles_labels()

    if legend_outside:
        ax.legend(
            handles,
            legend_labels,
            frameon=False,
            fontsize=9,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.00),
            borderaxespad=0.0,
        )
    else:
        ax.legend(
            handles,
            legend_labels,
            frameon=False,
            fontsize=9,
            ncol=2,
            loc="upper right",
        )

    # --------------------------------------------------
    # 8. 保存
    # --------------------------------------------------
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()
    return fig, ax, df

def plot_stage3_bestnode_policy_grid_heatmap(
    grid_df: pd.DataFrame,
    node: int = None,
    metric_col: str = "objective",
    schedule_order=("constant", "early_pulse", "middle_pulse", "late_pulse", "ramp_up", "ramp_decay", "exponential_decay"),
    figsize=(12, 4.8),
    annot=True,
    fmt=".3f",
    cmap="RdYlBu_r",
    save_path=None,
):
    """
    Plot best-node policy grid heatmap.
    Split into two panels:
        sign = -1
        sign = +1

    Parameters
    ----------
    grid_df :
        Usually stage3_res["all_grid"][best_node]
    node :
        node id, only used for title
    metric_col :
        usually "objective"
    """

    required_cols = ["sign", "strength", "schedule", metric_col]
    for c in required_cols:
        if c not in grid_df.columns:
            raise KeyError(f"grid_df must contain column: {c}")

    df = grid_df.copy()

    # strength 排序
    strengths_sorted = sorted(df["strength"].astype(float).unique().tolist())

    # schedule 排序：只保留实际出现的
    used_schedules = df["schedule"].astype(str).unique().tolist()
    sched_order = [s for s in schedule_order if s in used_schedules]
    # 如果有未包含在预设顺序中的 schedule，就补到后面
    for s in used_schedules:
        if s not in sched_order:
            sched_order.append(s)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    sign_list = [-1, 1]

    # 为了两个子图颜色一致
    vmin = float(df[metric_col].min())
    vmax = float(df[metric_col].max())

    for ax, sign_val in zip(axes, sign_list):
        sub = df[df["sign"].astype(int) == sign_val].copy()

        pivot = sub.pivot_table(
            index="schedule",
            columns="strength",
            values=metric_col,
            aggfunc="mean"
        )

        # 统一顺序
        pivot = pivot.reindex(index=sched_order)
        pivot = pivot.reindex(columns=strengths_sorted)

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            cbar=(sign_val == 1),   # 只在右图显示一个 colorbar
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
        )

        ax.set_title(f"sign = {sign_val}", fontsize=12)
        ax.set_xlabel("strength")
        ax.set_ylabel("schedule")

    if node is None and "node" in df.columns and len(df) > 0:
        node = int(df["node"].iloc[0])

    fig.suptitle(
        f"Best-node policy grid heatmap (node {node}, color = {metric_col})",
        fontsize=13,
        y=1.02
    )

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, axes

def build_stage3_topk_candidate_score_table(
    single_df: pd.DataFrame,
    top_k: int = 10,
    include_roi_name: bool = True,
    roi_name_col: str = "roi_name",
    round_digits: int = 3,
    save_csv_path: str = None,
):
    """
    Build a paper-style Top-K candidate score table.

    Parameters
    ----------
    single_df :
        Usually stage3_res["single_df"] after attach_candidate_scores(...)
    """

    required_cols = ["node", "objective", "traj_recovery", "fc_recovery", "strength", "schedule"]
    for c in required_cols:
        if c not in single_df.columns:
            raise KeyError(f"single_df must contain column: {c}")

    df = single_df.copy().sort_values("objective", ascending=False).head(int(top_k)).reset_index(drop=True)

    keep_cols = ["node"]
    if include_roi_name and roi_name_col in df.columns:
        keep_cols.append(roi_name_col)

    keep_cols += ["objective", "traj_recovery", "fc_recovery", "strength", "schedule"]

    out = df[keep_cols].copy()

    for c in ["objective", "traj_recovery", "fc_recovery", "strength"]:
        if c in out.columns:
            out[c] = out[c].astype(float).round(round_digits)

    if save_csv_path is not None:
        save_dir = os.path.dirname(save_csv_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        out.to_csv(save_csv_path, index=False, encoding="utf-8-sig")

    return out


def plot_stage3_intervention_search_path_graph(
    stage3_res: dict,
    roi_names=None,
    top_k_show: int = 10,
    figsize=(13.5, 5.8),
    title: str = "Stage 3 intervention search path",
    save_path: str = None,
):
    """
    Stage 3 干预搜索路径图。

    图的含义：
        candidate nodes
            -> single-node search
            -> best node
            -> best sign
            -> best strength
            -> best schedule
            -> objective

    适合论文 / 答辩中解释：
        最优干预策略是如何从候选节点中搜索出来的。

    Required:
        stage3_res["single_df"]

    Expected columns in single_df:
        node
        objective
        traj_recovery
        fc_recovery
        sign
        strength
        schedule

    如果部分列不存在，会自动降级显示。
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    if "single_df" not in stage3_res:
        raise KeyError("stage3_res must contain key: 'single_df'.")

    single_df = stage3_res["single_df"].copy()

    if "node" not in single_df.columns:
        raise KeyError("stage3_res['single_df'] must contain column: 'node'.")

    if "objective" not in single_df.columns:
        raise KeyError("stage3_res['single_df'] must contain column: 'objective'.")

    single_df = single_df.sort_values("objective", ascending=False).reset_index(drop=True)

    best_row = single_df.iloc[0]
    best_node = int(best_row["node"])

    def _roi_label(node):
        node = int(node)
        if roi_names is not None and 0 <= node < len(roi_names):
            return str(roi_names[node])
        return str(node)

    def _safe_get(row, col, default="N/A"):
        if col in row.index and pd.notna(row[col]):
            return row[col]
        return default

    best_label = _roi_label(best_node)
    best_objective = float(best_row["objective"])

    best_sign = _safe_get(best_row, "sign", "N/A")
    best_strength = _safe_get(best_row, "strength", "N/A")
    best_schedule = _safe_get(best_row, "schedule", "N/A")
    traj_recovery = _safe_get(best_row, "traj_recovery", None)
    fc_recovery = _safe_get(best_row, "fc_recovery", None)

    n_candidates = len(single_df)

    # Top-K candidates text
    top_df = single_df.head(int(top_k_show)).copy()
    top_lines = []
    for i, r in top_df.iterrows():
        node = int(r["node"])
        obj = float(r["objective"])
        top_lines.append(f"{i+1}. {_roi_label(node)}  obj={obj:.3f}")

    top_text = "\n".join(top_lines)

    # --------------------------------------------------
    # Build figure
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def draw_box(
        xy,
        width,
        height,
        text,
        fc="#F7F7F7",
        ec="#555555",
        fontsize=10,
        lw=1.2,
        bold_title=False,
    ):
        x, y = xy
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            alpha=0.96,
            zorder=2,
        )
        ax.add_patch(box)

        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold" if bold_title else "normal",
            color="black",
            zorder=3,
        )
        return box

    def draw_arrow(start, end, color="#555555", lw=1.4, rad=0.0):
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
        ax.add_patch(arrow)

    # --------------------------------------------------
    # Main path boxes
    # --------------------------------------------------
    y_mid = 0.57
    box_w = 0.125
    box_h = 0.16

    xs = [0.04, 0.20, 0.36, 0.52, 0.68, 0.84]
    labels = [
        f"Candidate pool\nn = {n_candidates}",
        "Single-node\nsearch",
        f"Best node\n{best_label}",
        f"Direction\nsign = {best_sign}",
        f"Intensity\nstrength = {best_strength}",
        f"Timing\nschedule = {best_schedule}",
    ]

    colors = [
        "#EEF2FF",
        "#EAF7EA",
        "#FFF4C2",
        "#F4E8FF",
        "#E8F4FF",
        "#FFF0E6",
    ]

    edges = [
        "#5B6FA8",
        "#4C8B4A",
        "#D8B100",
        "#805AD5",
        "#3182CE",
        "#DD6B20",
    ]

    boxes = []
    for x0, lab, fc, ec in zip(xs, labels, colors, edges):
        boxes.append(
            draw_box(
                (x0, y_mid),
                box_w,
                box_h,
                lab,
                fc=fc,
                ec=ec,
                fontsize=9.5,
                lw=1.4,
                bold_title=False,
            )
        )

    # arrows between boxes
    for i in range(len(xs) - 1):
        start = (xs[i] + box_w, y_mid + box_h / 2)
        end = (xs[i + 1], y_mid + box_h / 2)
        draw_arrow(start, end, color="#666666", lw=1.5)

    # --------------------------------------------------
    # Objective box
    # --------------------------------------------------
    obj_lines = [f"Final objective\n{best_objective:.4f}"]

    if traj_recovery is not None:
        obj_lines.append(f"traj = {float(traj_recovery):.3f}")
    if fc_recovery is not None:
        obj_lines.append(f"fc = {float(fc_recovery):.3f}")

    obj_text = "\n".join(obj_lines)

    draw_box(
        (0.40, 0.18),
        0.22,
        0.18,
        obj_text,
        fc="#F7FBFF",
        ec="#2B6CB0",
        fontsize=10,
        lw=1.5,
        bold_title=False,
    )

    # arrow from schedule to objective
    draw_arrow(
        (xs[-1] + box_w / 2, y_mid),
        (0.51, 0.36),
        color="#2B6CB0",
        lw=1.6,
        rad=-0.16,
    )

    # --------------------------------------------------
    # Top-K candidate list box
    # --------------------------------------------------
    draw_box(
        (0.04, 0.10),
        0.27,
        0.34,
        f"Top-{top_k_show} candidate nodes\n\n{top_text}",
        fc="#FFFFFF",
        ec="#CCCCCC",
        fontsize=8.2,
        lw=1.0,
    )

    # arrow from Top-K list to single-node search
    draw_arrow(
        (0.31, 0.28),
        (0.20, y_mid),
        color="#888888",
        lw=1.1,
        rad=0.12,
    )

    # --------------------------------------------------
    # Explanation note
    # --------------------------------------------------
    note = (
        "The optimized intervention policy is selected by evaluating candidate nodes, "
        "direction, strength, and temporal schedule under the objective function."
    )

    ax.text(
        0.50,
        0.035,
        note,
        ha="center",
        va="center",
        fontsize=9,
        color="dimgray",
    )

    ax.set_title(title, fontsize=14, pad=14)

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    summary = {
        "n_candidates": int(n_candidates),
        "best_node": int(best_node),
        "best_node_label": best_label,
        "best_sign": best_sign,
        "best_strength": best_strength,
        "best_schedule": best_schedule,
        "best_objective": float(best_objective),
    }

    return fig, ax, summary

def plot_stage3_greedy_policy_construction_path(
    greedy_history: pd.DataFrame,
    roi_names=None,
    figsize=(12.5, 5.2),
    title="Stage 3 greedy policy construction path",
    save_path=None,
):
    """
    Greedy policy construction path.

    Required columns:
        step
        node
        objective

    Optional columns:
        sign
        strength
        schedule
        marginal_gain
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    df = greedy_history.copy()

    if "step" not in df.columns:
        df["step"] = np.arange(1, len(df) + 1)

    if "node" not in df.columns:
        raise KeyError("greedy_history must contain column: node")

    if "objective" not in df.columns:
        raise KeyError("greedy_history must contain column: objective")

    df = df.sort_values("step").reset_index(drop=True)

    def _roi_label(node):
        node = int(node)
        if roi_names is not None and 0 <= node < len(roi_names):
            return str(roi_names[node])
        return str(node)

    def _safe(row, col, default="N/A"):
        if col in row.index and pd.notna(row[col]):
            return row[col]
        return default

    n_steps = len(df)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def draw_box(x, y, w, h, text, fc, ec, fontsize=9.2):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.35,
            edgecolor=ec,
            facecolor=fc,
            alpha=0.96,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=3,
        )
        return box

    def draw_arrow(start, end):
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
            color="#666666",
            zorder=1,
        )
        ax.add_patch(arrow)

    # x positions
    total_boxes = n_steps + 2
    xs = np.linspace(0.04, 0.84, total_boxes)
    box_w = 0.13
    box_h = 0.23
    y = 0.52

    # initial
    draw_box(
        xs[0],
        y,
        box_w,
        box_h,
        "Initial\nempty policy",
        fc="#F7F7F7",
        ec="#999999",
    )

    previous_center = (xs[0] + box_w, y + box_h / 2)

    # steps
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        node = int(r["node"])
        node_label = _roi_label(node)

        sign = _safe(r, "sign")
        strength = _safe(r, "strength")
        schedule = _safe(r, "schedule")
        objective = float(r["objective"])

        marginal_gain = _safe(r, "marginal_gain", None)

        lines = [
            f"Greedy step {int(r['step'])}",
            f"node: {node_label}",
            f"sign: {sign}",
            f"strength: {strength}",
            f"schedule: {schedule}",
            f"obj: {objective:.3f}",
        ]

        if marginal_gain is not None:
            lines.append(f"gain: {float(marginal_gain):.3f}")

        text = "\n".join(lines)

        draw_box(
            xs[i],
            y,
            box_w,
            box_h,
            text,
            fc="#EAF7EA",
            ec="#4C8B4A",
            fontsize=8.2,
        )

        draw_arrow(
            previous_center,
            (xs[i], y + box_h / 2),
        )
        previous_center = (xs[i] + box_w, y + box_h / 2)

    # best policy
    final_obj = float(df.iloc[-1]["objective"])
    final_nodes = ", ".join([_roi_label(n) for n in df["node"].astype(int).values])

    final_text = (
        "Best policy\n"
        f"nodes: {final_nodes}\n"
        f"objective: {final_obj:.3f}"
    )

    draw_box(
        xs[-1],
        y,
        box_w,
        box_h,
        final_text,
        fc="#FFF4C2",
        ec="#D8B100",
        fontsize=8.8,
    )

    draw_arrow(
        previous_center,
        (xs[-1], y + box_h / 2),
    )

    # objective trend below
    ax.plot(
        xs[1:1 + n_steps] + box_w / 2,
        0.23 + 0.12 * (
            (df["objective"].values - df["objective"].min())
            / (df["objective"].max() - df["objective"].min() + 1e-12)
        ),
        marker="o",
        linewidth=1.8,
    )

    ax.text(
        0.50,
        0.16,
        "Objective trend during greedy construction",
        ha="center",
        va="center",
        fontsize=9,
        color="dimgray",
    )

    ax.set_title(title, fontsize=14, pad=14)

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    summary = {
        "n_steps": int(n_steps),
        "final_nodes": final_nodes,
        "final_objective": final_obj,
    }

    return fig, ax, summary


# helpers
def _to_df(x):
    if isinstance(x, pd.DataFrame):
        return x.copy()
    return pd.DataFrame(x).copy()


def _node_to_name(node, roi_names=None):
    try:
        node_int = int(node)
    except Exception:
        return str(node)

    if roi_names is not None and 0 <= node_int < len(roi_names):
        return str(roi_names[node_int])
    return str(node_int)


def _pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Cannot find any of columns: {candidates}")
    return None


def _ensure_dir(save_path):
    if save_path is None:
        return
    folder = os.path.dirname(save_path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)


# =========================================================
# 1) Objective + marginal gain combo
# =========================================================
def plot_stage3_objective_marginal_gain_combo(
    greedy_history: Union[pd.DataFrame, list],
    roi_names: Optional[Sequence[str]] = None,
    baseline_objective: float = 0.0,
    figsize=(12.8, 4.8),
    title="Stage 3 objective and marginal gain",
    save_path: Optional[str] = None,
):
    """
    左图：objective 随 greedy step 增长
    右图：每一步 marginal gain 柱状图
    """
    gh = _to_df(greedy_history)

    step_col = _pick_col(gh, ["step"])
    node_col = _pick_col(gh, ["selected_node", "node"], required=False)
    obj_col = _pick_col(gh, ["objective"])
    gain_col = _pick_col(gh, ["marginal_gain", "gain"])

    gh = gh.sort_values(step_col).reset_index(drop=True)

    if node_col is not None:
        gh["node_label"] = [_node_to_name(x, roi_names) for x in gh[node_col].values]
    else:
        gh["node_label"] = [f"step{i+1}" for i in range(len(gh))]

    x_step = gh[step_col].astype(int).values
    obj_vals = gh[obj_col].astype(float).values
    gain_vals = gh[gain_col].astype(float).values

    # objective 曲线包含 baseline
    x_line = np.concatenate([[0], x_step])
    y_line = np.concatenate([[baseline_objective], obj_vals])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # -------------------------
    # panel A: objective curve
    # -------------------------
    ax = axes[0]
    ax.plot(x_line, y_line, marker="o", linewidth=2.2)
    ax.axhline(baseline_objective, linestyle="--", linewidth=1.2, alpha=0.7)

    for i, row in gh.iterrows():
        ax.text(
            row[step_col],
            row[obj_col],
            f"{row['node_label']}",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_title("Objective across greedy steps")
    ax.set_xlabel("Greedy step")
    ax.set_ylabel("Objective")
    ax.set_xticks(x_line)
    ax.grid(alpha=0.25, linestyle="--")

    # -------------------------
    # panel B: marginal gain
    # -------------------------
    ax = axes[1]
    bars = ax.bar(x_step, gain_vals, alpha=0.9)
    for rect, val, lab in zip(bars, gain_vals, gh["node_label"].values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{lab}\n{val:.3f}",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_title("Marginal gain per greedy step")
    ax.set_xlabel("Greedy step")
    ax.set_ylabel("Marginal gain")
    ax.set_xticks(x_step)
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    summary = {
        "n_steps": int(len(gh)),
        "final_objective": float(obj_vals[-1]) if len(obj_vals) else None,
        "total_gain": float(obj_vals[-1] - baseline_objective) if len(obj_vals) else None,
        "selected_nodes": gh["node_label"].tolist(),
    }
    return fig, axes, summary


# =========================================================
# 2) Objective decomposition stacked bar
# =========================================================
def plot_stage3_objective_decomposition_stackedbar(
    greedy_history: Union[pd.DataFrame, list],
    roi_names: Optional[Sequence[str]] = None,
    alpha_traj: float = 1.0,
    beta_fc: float = 0.8,
    lambda_node: float = 0.01,
    lambda_strength: float = 0.05,
    lambda_temporal: float = 0.03,
    figsize=(10.5, 5.4),
    title="Stage 3 objective decomposition",
    save_path: Optional[str] = None,
):
    """
    每个 greedy step 的 objective 分解：
        + α * traj_recovery
        + β * fc_recovery
        - λ_node * node_cost
        - λ_strength * strength_cost
        - λ_temporal * temporal_cost
    """
    gh = _to_df(greedy_history)

    step_col = _pick_col(gh, ["step"])
    node_col = _pick_col(gh, ["selected_node", "node"], required=False)
    traj_col = _pick_col(gh, ["traj_recovery"])
    fc_col = _pick_col(gh, ["fc_recovery"])
    obj_col = _pick_col(gh, ["objective"])

    gh = gh.sort_values(step_col).reset_index(drop=True)

    if node_col is not None:
        gh["node_label"] = [_node_to_name(x, roi_names) for x in gh[node_col].values]
    else:
        gh["node_label"] = [f"step{i+1}" for i in range(len(gh))]

    # 成本项：优先用表里的列，否则自己推
    if "node_cost" in gh.columns:
        node_cost = gh["node_cost"].astype(float).values
    else:
        # 默认 step=1 表示 1 个节点, step=2 表示 2 个节点
        node_cost = gh[step_col].astype(float).values

    if "strength_cost" in gh.columns:
        strength_cost = gh["strength_cost"].astype(float).values
    else:
        strength_cost = np.zeros(len(gh))

    if "temporal_cost" in gh.columns:
        temporal_cost = gh["temporal_cost"].astype(float).values
    else:
        temporal_cost = np.zeros(len(gh))

    traj_term = alpha_traj * gh[traj_col].astype(float).values
    fc_term = beta_fc * gh[fc_col].astype(float).values
    node_term = -lambda_node * node_cost
    strength_term = -lambda_strength * strength_cost
    temporal_term = -lambda_temporal * temporal_cost
    obj_vals = gh[obj_col].astype(float).values

    x = np.arange(len(gh))
    labels = [f"S{int(s)}\n{lab}" for s, lab in zip(gh[step_col].values, gh["node_label"].values)]

    fig, ax = plt.subplots(figsize=figsize)

    # positive stack
    pos_base = np.zeros(len(gh))
    b1 = ax.bar(x, traj_term, bottom=pos_base, label=r"$\alpha \cdot$traj recovery")
    pos_base += traj_term
    b2 = ax.bar(x, fc_term, bottom=pos_base, label=r"$\beta \cdot$fc recovery")
    pos_base += fc_term

    # negative stack
    neg_base = np.zeros(len(gh))
    b3 = ax.bar(x, node_term, bottom=neg_base, label=r"$-\lambda_n \cdot$node cost")
    neg_base += node_term
    b4 = ax.bar(x, strength_term, bottom=neg_base, label=r"$-\lambda_s \cdot$strength cost")
    neg_base += strength_term
    b5 = ax.bar(x, temporal_term, bottom=neg_base, label=r"$-\lambda_t \cdot$temporal cost")
    neg_base += temporal_term

    # objective dots
    ax.plot(x, obj_vals, marker="o", linewidth=2.0, label="Objective")

    for xi, yi in zip(x, obj_vals):
        ax.text(xi, yi, f"{yi:.3f}", fontsize=8, ha="center", va="bottom")

    ax.axhline(0.0, linestyle="--", linewidth=1.1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Objective term value")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--", axis="y")
    ax.legend(frameon=False, ncol=2)

    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    decomp_df = pd.DataFrame({
        "step": gh[step_col].values,
        "node_label": gh["node_label"].values,
        "traj_term": traj_term,
        "fc_term": fc_term,
        "node_term": node_term,
        "strength_term": strength_term,
        "temporal_term": temporal_term,
        "objective": obj_vals,
    })
    return fig, ax, decomp_df


# =========================================================
# 3) Candidate marginal gain heatmap / ranking
# =========================================================
def plot_stage3_candidate_marginal_gain_map(
    greedy_trial_df: Union[pd.DataFrame, list],
    roi_names: Optional[Sequence[str]] = None,
    mode: str = "heatmap",   # "heatmap" or "ranking"
    top_show_nodes: int = 18,
    top_k_per_step: int = 8,
    figsize=None,
    title_prefix="Stage 3 candidate marginal gain",
    save_path: Optional[str] = None,
):
    """
    mode = "heatmap":
        x轴=候选节点, y轴=greedy step, 颜色=marginal_gain

    mode = "ranking":
        每个 step 一张横向柱状图, 展示 Top-K marginal gain candidates
    """
    df = _to_df(greedy_trial_df)

    step_col = _pick_col(df, ["step"])
    node_col = _pick_col(df, ["candidate_node", "node", "selected_node"])
    gain_col = _pick_col(df, ["marginal_gain", "gain"])
    obj_col = _pick_col(df, ["objective"], required=False)

    df[step_col] = df[step_col].astype(int)
    df[node_col] = df[node_col].astype(int)
    df[gain_col] = df[gain_col].astype(float)

    df["node_label"] = [_node_to_name(x, roi_names) for x in df[node_col].values]

    if mode == "heatmap":
        # 选最值得展示的节点：按 max marginal gain 排序
        node_order = (
            df.groupby(["node_label"], as_index=False)[gain_col]
              .max()
              .sort_values(gain_col, ascending=False)["node_label"]
              .head(top_show_nodes)
              .tolist()
        )
        sub = df[df["node_label"].isin(node_order)].copy()

        heat = sub.pivot_table(
            index=step_col,
            columns="node_label",
            values=gain_col,
            aggfunc="max",
            fill_value=0.0,
        )

        # 按上面的 node_order 重排
        heat = heat.reindex(columns=[c for c in node_order if c in heat.columns])

        if figsize is None:
            figsize = (max(8.5, 0.45 * heat.shape[1] + 2.5), 4.8)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(heat.values, aspect="auto")

        ax.set_xticks(np.arange(heat.shape[1]))
        ax.set_xticklabels(heat.columns, rotation=60, ha="right")
        ax.set_yticks(np.arange(heat.shape[0]))
        ax.set_yticklabels([f"Step {int(x)}" for x in heat.index])
        ax.set_xlabel("Candidate node")
        ax.set_ylabel("Greedy step")
        ax.set_title(f"{title_prefix} heatmap")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Marginal gain")

        # 可选：写数值
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                val = heat.values[i, j]
                if abs(val) > 1e-12:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

        plt.tight_layout()

        _ensure_dir(save_path)
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        info = {
            "mode": "heatmap",
            "heatmap_df": heat,
            "top_nodes": list(heat.columns),
        }
        return fig, ax, info

    elif mode == "ranking":
        steps = sorted(df[step_col].unique().tolist())
        n_steps = len(steps)

        if figsize is None:
            figsize = (9.5, 3.0 * n_steps)

        fig, axes = plt.subplots(n_steps, 1, figsize=figsize, squeeze=False)
        axes = axes.ravel()

        ranking_info = {}

        for ax, st in zip(axes, steps):
            sub = df[df[step_col] == st].copy()
            sub = sub.sort_values(gain_col, ascending=False).head(top_k_per_step).copy()
            sub = sub.iloc[::-1]  # 让最大值在上面更好看

            y = np.arange(len(sub))
            ax.barh(y, sub[gain_col].values)
            ax.set_yticks(y)
            ax.set_yticklabels(sub["node_label"].values)
            ax.set_xlabel("Marginal gain")
            ax.set_title(f"Top {top_k_per_step} candidates at Step {st}")
            ax.grid(alpha=0.25, linestyle="--", axis="x")

            for yi, val in zip(y, sub[gain_col].values):
                ax.text(val, yi, f" {val:.3f}", va="center", ha="left", fontsize=8)

            ranking_info[int(st)] = sub[
                [node_col, "node_label", gain_col] + ([obj_col] if obj_col is not None else [])
            ].copy()

        fig.suptitle(f"{title_prefix} ranking", fontsize=14)
        plt.tight_layout()

        _ensure_dir(save_path)
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return fig, axes, {
            "mode": "ranking",
            "ranking_info": ranking_info,
        }

    else:
        raise ValueError("mode must be 'heatmap' or 'ranking'")