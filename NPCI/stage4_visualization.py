from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def draw_scatter_callout_labels(
    ax,
    label_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str = "_callout_label",
    all_points_df: pd.DataFrame = None,
    roi_col: str = "roi",
    fontsize: int = 8,
    offset_frac=(0.075, 0.070),
    line_color="gray",
    box_alpha=0.92,
    point_pad_frac=(0.012, 0.018),
    label_char_width_frac=0.0105,
    label_line_height_frac=0.045,
    pref_col="_label_pref",
):
    """
    支持按标签自定义优先方向的 scatter callout label：
    - 标签放在方框内
    - 连线指向节点
    - 支持每个标签自己的优先放置方向
    - 自动避让其它标签 / 其它散点
    """

    import numpy as np
    import pandas as pd

    if label_df is None or len(label_df) == 0:
        return

    label_df = label_df.copy()

    if all_points_df is None:
        all_points_df = label_df.copy()
    else:
        all_points_df = all_points_df.copy()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]

    dx_base = offset_frac[0] * x_span
    dy_base = offset_frac[1] * y_span

    point_pad_x = point_pad_frac[0] * x_span
    point_pad_y = point_pad_frac[1] * y_span

    point_xy = all_points_df[[x_col, y_col]].astype(float).values
    if roi_col in all_points_df.columns:
        point_roi = all_points_df[roi_col].astype(int).values
    else:
        point_roi = np.full(len(all_points_df), -999999, dtype=int)

    placed_rects = []

    def _estimate_label_size(label: str):
        lines = str(label).split("\n")
        max_chars = max(len(line) for line in lines)
        n_lines = len(lines)

        w = max(0.050 * x_span, max_chars * label_char_width_frac * x_span)
        h = max(0.040 * y_span, n_lines * label_line_height_frac * y_span)
        return w, h

    def _rect_from_anchor(tx, ty, w, h, ha, va):
        if ha == "left":
            x0, x1 = tx, tx + w
        elif ha == "right":
            x0, x1 = tx - w, tx
        else:
            x0, x1 = tx - w / 2, tx + w / 2

        if va == "bottom":
            y0, y1 = ty, ty + h
        elif va == "top":
            y0, y1 = ty - h, ty
        else:
            y0, y1 = ty - h / 2, ty + h / 2

        return (x0, y0, x1, y1)

    def _rect_overlap(r1, r2, pad_x=0.0, pad_y=0.0):
        a0, b0, a1, b1 = r1
        c0, d0, c1, d1 = r2

        a0 -= pad_x
        a1 += pad_x
        b0 -= pad_y
        b1 += pad_y

        c0 -= pad_x
        c1 += pad_x
        d0 -= pad_y
        d1 += pad_y

        return not (a1 < c0 or c1 < a0 or b1 < d0 or d1 < b0)

    def _count_points_inside(rect, current_roi=None):
        x0, y0, x1, y1 = rect
        inside = (
            (point_xy[:, 0] >= x0 - point_pad_x) &
            (point_xy[:, 0] <= x1 + point_pad_x) &
            (point_xy[:, 1] >= y0 - point_pad_y) &
            (point_xy[:, 1] <= y1 + point_pad_y)
        )
        if current_roi is not None:
            inside = inside & (point_roi != int(current_roi))
        return int(np.sum(inside))

    def _out_of_bounds_penalty(rect):
        x0, y0, x1, y1 = rect
        penalty = 0.0
        if x0 < xlim[0]:
            penalty += abs(xlim[0] - x0) / x_span
        if x1 > xlim[1]:
            penalty += abs(x1 - xlim[1]) / x_span
        if y0 < ylim[0]:
            penalty += abs(ylim[0] - y0) / y_span
        if y1 > ylim[1]:
            penalty += abs(y1 - ylim[1]) / y_span
        return penalty

    # 不同“优先方向组”的候选方向
    # (sx, sy, ha, va)
    direction_presets = {
        "up": [
            (0, 1, "center", "bottom"),
            (1, 1, "left", "bottom"),
            (-1, 1, "right", "bottom"),
            (0, -1, "center", "top"),
        ],
        "right_down": [
            (1, -1, "left", "top"),
            (1, 0, "left", "center"),
            (1, 1, "left", "bottom"),
            (0, -1, "center", "top"),
            (-1, -1, "right", "top"),
        ],
        "right_up": [
            (1, 1, "left", "bottom"),
            (1, 0, "left", "center"),
            (0, 1, "center", "bottom"),
            (1, -1, "left", "top"),
            (-1, 1, "right", "bottom"),
        ],
        "right_mid": [
            (1, 0, "left", "center"),
            (1, 1, "left", "bottom"),
            (1, -1, "left", "top"),
            (0, 1, "center", "bottom"),
            (0, -1, "center", "top"),
        ],
        "default": [
            (1, 0, "left", "center"),
            (1, 1, "left", "bottom"),
            (1, -1, "left", "top"),
            (0, 1, "center", "bottom"),
            (0, -1, "center", "top"),
            (-1, 0, "right", "center"),
        ]
    }

    distance_scales = [1.00, 1.25, 1.55, 1.90, 2.30, 2.80]

    if "_label_priority" in label_df.columns:
        label_df = label_df.sort_values("_label_priority").reset_index(drop=True)

    for _, r in label_df.iterrows():
        x = float(r[x_col])
        y = float(r[y_col])

        current_roi = int(r[roi_col]) if roi_col in r and pd.notna(r[roi_col]) else None

        label = (
            str(r[label_col])
            if label_col in r and pd.notna(r[label_col])
            else str(current_roi)
        )

        pref = str(r[pref_col]) if pref_col in r and pd.notna(r[pref_col]) else "default"
        base_directions = direction_presets.get(pref, direction_presets["default"])

        w, h = _estimate_label_size(label)

        best = None
        best_score = np.inf

        for scale in distance_scales:
            for sx, sy, ha, va in base_directions:
                tx = x + sx * dx_base * scale
                ty = y + sy * dy_base * scale

                rect = _rect_from_anchor(tx, ty, w, h, ha, va)

                distance_score = scale
                point_overlap = _count_points_inside(rect, current_roi=current_roi)
                label_overlap = sum(
                    _rect_overlap(
                        rect,
                        old_rect,
                        pad_x=0.006 * x_span,
                        pad_y=0.010 * y_span,
                    )
                    for old_rect in placed_rects
                )
                outside_penalty = _out_of_bounds_penalty(rect)

                score = (
                    distance_score
                    + 12.0 * point_overlap
                    + 30.0 * label_overlap
                    + 80.0 * outside_penalty
                )

                if score < best_score:
                    best_score = score
                    best = (tx, ty, ha, va, rect)

        if best is None:
            continue

        tx, ty, ha, va, rect = best
        placed_rects.append(rect)

        ax.annotate(
            label,
            xy=(x, y),
            xytext=(tx, ty),
            textcoords="data",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="lightgray",
                alpha=box_alpha,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=line_color,
                lw=0.8,
                shrinkA=2,
                shrinkB=3,
                alpha=0.85,
            ),
            zorder=30,
        )



def plot_time_recovery(
    time_df: pd.DataFrame,
    ax=None,
    figsize=(12.5, 7.2),
    save_path=None,
):
    """
    Plot Stage 4 time-resolved recovery as a paper-style combo panel.

    Panels:
        A. Time-resolved trajectory recovery
        B. Cumulative mean recovery
        C. Phase-wise recovery bar: Early / Middle / Late
        D. Summary recovery bar: Initial / Mean / Peak / Final

    Required columns:
        step
        traj_recovery
    """
    df = time_df.copy()

    if "step" not in df.columns:
        df["step"] = np.arange(len(df))

    if "traj_recovery" not in df.columns:
        raise KeyError("time_df must contain column: traj_recovery")

    df = df.sort_values("step").reset_index(drop=True)

    step = df["step"].to_numpy()
    recovery = df["traj_recovery"].astype(float).to_numpy()

    # --------------------------------------------------
    # Summary statistics
    # --------------------------------------------------
    initial_recovery = float(recovery[0])
    mean_recovery = float(np.mean(recovery))
    peak_recovery = float(np.max(recovery))
    final_recovery = float(recovery[-1])
    peak_step = int(step[np.argmax(recovery)])

    cumulative_mean = np.cumsum(recovery) / np.arange(1, len(recovery) + 1)

    # --------------------------------------------------
    # Phase-wise recovery
    # --------------------------------------------------
    phase_names = ["Early", "Middle", "Late"]
    phase_indices = np.array_split(np.arange(len(recovery)), 3)

    phase_values = []
    for idx in phase_indices:
        phase_values.append(float(np.mean(recovery[idx])))

    phase_df = pd.DataFrame({
        "phase": phase_names,
        "mean_recovery": phase_values,
    })

    summary_df = pd.DataFrame({
        "metric": ["Initial", "Mean", "Peak", "Final"],
        "value": [
            initial_recovery,
            mean_recovery,
            peak_recovery,
            final_recovery,
        ],
    })

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # ==================================================
    # A. Time-resolved recovery curve
    # ==================================================
    ax_a.plot(
        step,
        recovery,
        linewidth=2.3,
        label="Trajectory recovery",
    )

    ax_a.axhline(
        0.0,
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
        label="No recovery",
    )

    ax_a.axhline(
        mean_recovery,
        linestyle=":",
        linewidth=1.4,
        alpha=0.85,
        label=f"Mean = {mean_recovery:.3f}",
    )

    ax_a.scatter(
        [peak_step],
        [peak_recovery],
        s=48,
        zorder=5,
        label=f"Peak step = {peak_step}",
    )

    ax_a.fill_between(
        step,
        0,
        recovery,
        alpha=0.14,
    )

    ax_a.set_title("Time-resolved trajectory recovery")
    ax_a.set_xlabel("Rollout step")
    ax_a.set_ylabel("Trajectory recovery")
    ax_a.grid(alpha=0.25, linestyle="--")
    ax_a.legend(frameon=False, fontsize=8)

    # ==================================================
    # B. Cumulative mean recovery
    # ==================================================
    ax_b.plot(
        step,
        cumulative_mean,
        linewidth=2.3,
        label="Cumulative mean recovery",
    )

    ax_b.axhline(
        0.0,
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
    )

    ax_b.axhline(
        mean_recovery,
        linestyle=":",
        linewidth=1.4,
        alpha=0.85,
        label=f"Overall mean = {mean_recovery:.3f}",
    )

    ax_b.set_title("Cumulative recovery stability")
    ax_b.set_xlabel("Rollout step")
    ax_b.set_ylabel("Cumulative mean recovery")
    ax_b.grid(alpha=0.25, linestyle="--")
    ax_b.legend(frameon=False, fontsize=8)

    # ==================================================
    # C. Phase-wise recovery bar
    # ==================================================
    bars_c = ax_c.bar(
        phase_df["phase"],
        phase_df["mean_recovery"],
        width=0.62,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    ax_c.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
    )

    for bar, val in zip(bars_c, phase_df["mean_recovery"]):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
        )

    ax_c.set_title("Phase-wise trajectory recovery")
    ax_c.set_ylabel("Mean recovery")
    ax_c.grid(axis="y", alpha=0.25, linestyle="--")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ==================================================
    # D. Summary bar
    # ==================================================
    bars_d = ax_d.bar(
        summary_df["metric"],
        summary_df["value"],
        width=0.62,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    ax_d.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
    )

    for bar, val in zip(bars_d, summary_df["value"]):
        ax_d.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
        )

    final_gain = final_recovery - initial_recovery
    stability_gap = peak_recovery - final_recovery

    summary_text = (
        f"Final - initial = {final_gain:.3f}\n"
        f"Peak - final = {stability_gap:.3f}\n"
        f"Peak step = {peak_step}"
    )

    ax_d.text(
        0.03,
        0.95,
        summary_text,
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="lightgray",
            alpha=0.92,
        ),
    )

    ax_d.set_title("Recovery summary")
    ax_d.set_ylabel("Recovery value")
    ax_d.grid(axis="y", alpha=0.25, linestyle="--")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    # --------------------------------------------------
    # Panel labels
    # --------------------------------------------------
    for ax_i, lab in zip([ax_a, ax_b, ax_c, ax_d], ["A", "B", "C", "D"]):
        ax_i.text(
            -0.10,
            1.08,
            lab,
            transform=ax_i.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.suptitle(
        "Stage 4 time-resolved recovery analysis",
        fontsize=14,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    if ax is not None:
        # 兼容旧接口：如果用户传入 ax，只把主曲线画到 ax 上
        ax.plot(step, recovery, linewidth=2.2)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("Trajectory recovery")
        ax.set_title("Time-resolved recovery")
        return ax

    plt.show()

    return {
        "fig": fig,
        "axes": {
            "time_recovery": ax_a,
            "cumulative_recovery": ax_b,
            "phase_bar": ax_c,
            "summary_bar": ax_d,
        },
        "time_df": df,
        "phase_df": phase_df,
        "summary_df": summary_df,
        "diagnostics": {
            "initial_recovery": initial_recovery,
            "mean_recovery": mean_recovery,
            "peak_recovery": peak_recovery,
            "peak_step": peak_step,
            "final_recovery": final_recovery,
            "final_minus_initial": final_gain,
            "peak_minus_final": stability_gap,
        },
    }


def plot_fc_triplet(FC_healthy: np.ndarray, FC_lesion: np.ndarray, FC_policy: np.ndarray, neworder=None, vmin=-1, vmax=1):
    """Plot healthy, lesion, and policy FC matrices."""
    mats = [FC_healthy, FC_lesion, FC_policy]
    if neworder is not None:
        mats = [m[neworder].T[neworder].T for m in mats]
    titles = ["Healthy FC", "Lesion FC", "Policy FC"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, mat, title in zip(axes, mats, titles):
        sns.heatmap(mat, ax=ax, vmin=vmin, vmax=vmax, cmap="RdBu_r", cbar=False, square=True, xticklabels=False, yticklabels=False)
        ax.set_title(title)
    plt.tight_layout()
    return fig, axes


def plot_top_roi_recovery(roi_df: pd.DataFrame, top_k: int = 20, label_col: str = "roi", ax=None):
    """Plot top ROI-wise recovery."""
    df = roi_df.head(int(top_k)).copy()
    labels = df[label_col].astype(str).tolist()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(df)), df["roi_recovery"].values)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("ROI-wise recovery")
    ax.set_title(f"Top {top_k} ROI-wise recovery")
    return ax



def plot_three_trajectory_distance(
    dist_df,
    title: str = "Three-trajectory distance over time",
    ylabel: str = "Distance to healthy trajectory",
    ax=None,
):
    """
    Plot:
        D(healthy, lesion)
        D(healthy, policy)

    If D(healthy, policy) is lower than D(healthy, lesion),
    it means the policy trajectory is closer to the healthy trajectory.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
    else:
        fig = ax.figure

    x = dist_df["step"].values

    ax.plot(
        x,
        dist_df["D_healthy_lesion"].values,
        marker="o",
        linewidth=2.2,
        label="D(healthy, lesion)",
    )

    ax.plot(
        x,
        dist_df["D_healthy_policy"].values,
        marker="s",
        linewidth=2.2,
        label="D(healthy, policy)",
    )

    # 填充 policy 比 lesion 更接近 healthy 的区域
    better = dist_df["D_healthy_policy"].values < dist_df["D_healthy_lesion"].values

    ax.fill_between(
        x,
        dist_df["D_healthy_policy"].values,
        dist_df["D_healthy_lesion"].values,
        where=better,
        alpha=0.18,
        interpolate=True,
        label="policy closer to healthy",
    )

    ax.set_xlabel("Rollout step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linestyle="--")

    plt.tight_layout()

    return ax


def plot_stage4_fc_difference_maps(
    FC_healthy,
    FC_lesion,
    FC_policy,
    neworder=None,
    figsize=(11.2, 4.8),
    cmap="Reds",
    title="FC deviation from healthy reference",
    save_path=None,
):
    """
    Plot Stage 4 FC deviation maps:

        |FC_lesion - FC_healthy|
        |FC_policy - FC_healthy|

    Smaller value means closer to healthy FC.
    """
    FC_healthy = np.asarray(FC_healthy, dtype=float)
    FC_lesion = np.asarray(FC_lesion, dtype=float)
    FC_policy = np.asarray(FC_policy, dtype=float)

    D_lh = np.abs(FC_lesion - FC_healthy)
    D_ph = np.abs(FC_policy - FC_healthy)

    np.fill_diagonal(D_lh, 0.0)
    np.fill_diagonal(D_ph, 0.0)

    if neworder is not None:
        neworder = np.asarray(neworder, dtype=int)
        D_lh_plot = D_lh[np.ix_(neworder, neworder)]
        D_ph_plot = D_ph[np.ix_(neworder, neworder)]
    else:
        D_lh_plot = D_lh
        D_ph_plot = D_ph

    vmax = max(float(np.nanmax(D_lh_plot)), float(np.nanmax(D_ph_plot)))
    if vmax <= 1e-12:
        vmax = 1.0

    # --------------------------------------------------
    # 1. 画布
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 手动调整，不再用 tight_layout
    fig.subplots_adjust(
        left=0.06,
        right=0.88,   # 给右侧 colorbar 留空间
        bottom=0.08,
        top=0.82,
        wspace=0.18
    )

    # --------------------------------------------------
    # 2. 两张 deviation heatmap
    # --------------------------------------------------
    im0 = axes[0].imshow(D_lh_plot, cmap=cmap, vmin=0.0, vmax=vmax)
    axes[0].set_title("Lesion deviation", fontsize=12, pad=8)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(D_ph_plot, cmap=cmap, vmin=0.0, vmax=vmax)
    axes[1].set_title("Policy deviation", fontsize=12, pad=8)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # --------------------------------------------------
    # 3. 外置 colorbar
    # --------------------------------------------------
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.60])  # [left, bottom, width, height]
    cbar = fig.colorbar(im1, cax=cax)
    cbar.set_label("Absolute FC deviation from healthy", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # --------------------------------------------------
    # 4. 总标题
    # --------------------------------------------------
    fig.suptitle(title, fontsize=13, y=0.94)

    # --------------------------------------------------
    # 5. 保存
    # --------------------------------------------------
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    return fig, axes, {
        "D_lesion_healthy": D_lh,
        "D_policy_healthy": D_ph,
    }


def plot_stage4_fc_recovery_delta_map(
    FC_healthy,
    FC_lesion,
    FC_policy,
    neworder=None,
    figsize=(5.8, 5.2),
    cmap="RdBu_r",
    save_path=None,
):
    """
    Plot Stage 4 FC recovery delta map:

        ΔFC recovery = |FC_lesion - FC_healthy| - |FC_policy - FC_healthy|

    Positive:
        policy is closer to healthy than lesion.

    Negative:
        policy is farther from healthy than lesion.
    """
    FC_healthy = np.asarray(FC_healthy, dtype=float)
    FC_lesion = np.asarray(FC_lesion, dtype=float)
    FC_policy = np.asarray(FC_policy, dtype=float)

    D_lh = np.abs(FC_lesion - FC_healthy)
    D_ph = np.abs(FC_policy - FC_healthy)

    delta_recovery = D_lh - D_ph
    np.fill_diagonal(delta_recovery, 0.0)

    if neworder is not None:
        neworder = np.asarray(neworder, dtype=int)
        plot_mat = delta_recovery[np.ix_(neworder, neworder)]
    else:
        plot_mat = delta_recovery

    vmax = float(np.nanmax(np.abs(plot_mat)))
    if vmax <= 1e-12:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        plot_mat,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_title(r"$\Delta FC$ recovery: $|lesion-healthy|-|policy-healthy|$")
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("FC recovery gain")

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax, {
        "delta_fc_recovery": delta_recovery,
        "D_lesion_healthy": D_lh,
        "D_policy_healthy": D_ph,
    }


def plot_stage4_recovery_scatter(
    roi_recovery_df: pd.DataFrame,
    roi_names: Optional[Sequence[str]] = None,
    lesion_idx: Optional[int] = None,
    top_k_label: int = 5,
    color_by: str = "improvement",   # "roi_recovery" / "improvement" / None
    figsize=(6.0, 5.6),
    cmap="viridis",
    alpha=0.80,
    point_size=46,
    save_path: Optional[str] = None,
):
    """
    Stage 4 ROI-wise recovery scatter plot.

    x-axis: D_lesion_roi
    y-axis: D_policy_roi

    Interpretation:
        points below diagonal y=x  => policy is closer to healthy than lesion
        points above diagonal y=x  => policy is worse than lesion for that ROI
    """
    df = roi_recovery_df.copy()

    required_cols = ["roi", "D_lesion_roi", "D_policy_roi"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"roi_recovery_df must contain column: {c}")

    # 如果没有 roi_recovery，就自动补
    if "roi_recovery" not in df.columns:
        df["roi_recovery"] = 1.0 - df["D_policy_roi"] / (df["D_lesion_roi"] + 1e-12)

    # 如果没有 improvement，就自动补
    if "improvement" not in df.columns:
        df["improvement"] = df["D_lesion_roi"] - df["D_policy_roi"]

    # 标签列
    if roi_names is not None:
        df["_label_name"] = [
            roi_names[int(r)] if int(r) < len(roi_names) else str(int(r))
            for r in df["roi"].astype(int).values
        ]
    else:
        df["_label_name"] = df["roi"].astype(str)

    x = df["D_lesion_roi"].astype(float).values
    y = df["D_policy_roi"].astype(float).values

    # 去掉非有限值，避免 np.max 报错或图形异常
    valid = np.isfinite(x) & np.isfinite(y)
    if color_by is not None and color_by in df.columns:
        color_raw = df[color_by].astype(float).values
        valid = valid & np.isfinite(color_raw)

    df = df.loc[valid].copy()
    x = df["D_lesion_roi"].astype(float).values
    y = df["D_policy_roi"].astype(float).values

    if len(df) == 0:
        raise ValueError("No valid finite points to plot after filtering NaN/Inf values.")

    # 颜色
    if color_by is None:
        color_values = None
    else:
        if color_by not in df.columns:
            raise KeyError(f"color_by={color_by} not found in roi_recovery_df")
        color_values = df[color_by].astype(float).values

    fig, ax = plt.subplots(figsize=figsize)

    if color_values is None:
        sc = ax.scatter(
            x, y,
            s=point_size,
            alpha=alpha,
            edgecolor="k",
            linewidth=0.25,
        )
    else:
        sc = ax.scatter(
            x, y,
            c=color_values,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            edgecolor="k",
            linewidth=0.25,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.040, pad=0.03)
        cbar.set_label(color_by)

    # 对角线 y = x
    vmax = float(max(np.max(x), np.max(y)))
    vmax = vmax * 1.05 if vmax > 0 else 1.0

    ax.plot(
        [0, vmax],
        [0, vmax],
        linestyle="--",
        linewidth=1.5,
        color="red",
        alpha=0.9,
    )

    # 统计 below-diagonal 比例
    below_mask = y < x
    n_total = len(df)
    n_below = int(np.sum(below_mask))
    n_above = int(np.sum(y > x))
    frac_below = n_below / max(1, n_total)

    # 关键修改：x/y 使用同一范围，便于判断点在 y=x 上方还是下方
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22, linestyle="--")
    # --------------------------------------------------
    # Callout 标签：Top-K improvement + lesion ROI
    # --------------------------------------------------
    callout_parts = []

    # 1) lesion ROI 优先标注
    if lesion_idx is not None:
        lesion_rows = df[df["roi"].astype(int) == int(lesion_idx)]
        if len(lesion_rows) > 0:
            r = lesion_rows.iloc[0]

            ax.scatter(
                [float(r["D_lesion_roi"])],
                [float(r["D_policy_roi"])],
                s=180,
                facecolors="none",
                edgecolors="blue",
                linewidths=2.0,
                zorder=10,
            )

            lesion_label_df = lesion_rows.copy()
            lesion_label_df["_callout_label"] = [
                f"Lesion ROI\n{str(r['_label_name'])}"
                for _, r in lesion_label_df.iterrows()
            ]
            lesion_label_df["_label_priority"] = 0
            lesion_label_df["_label_pref"] = "up"
            callout_parts.append(lesion_label_df)

    # 2) Top-K improvement 节点
    if top_k_label is not None and top_k_label > 0:
        top_df = df.nlargest(int(top_k_label), "improvement").copy()
        top_df["_callout_label"] = top_df["_label_name"].astype(str)
        top_df["_label_priority"] = 1
        top_df["_label_pref"] = "right_down"
        callout_parts.append(top_df)

    # 3) 合并标签，避免 lesion 重复标
    if len(callout_parts) > 0:
        callout_df = (
            pd.concat(callout_parts, axis=0)
            .drop_duplicates("roi", keep="first")
            .reset_index(drop=True)
        )

        draw_scatter_callout_labels(
            ax=ax,
            label_df=callout_df,
            all_points_df=df,
            x_col="D_lesion_roi",
            y_col="D_policy_roi",
            label_col="_callout_label",
            roi_col="roi",
            fontsize=8,
            offset_frac=(0.085, 0.075),
            pref_col="_label_pref",
        )

    ax.set_xlabel(r"$D(\mathrm{healthy}, \mathrm{lesion})$ per ROI")
    ax.set_ylabel(r"$D(\mathrm{healthy}, \mathrm{policy})$ per ROI")
    ax.set_title("Stage 4 ROI-wise recovery scatter")

    # 图上写结论
    judgement = "good" if frac_below >= 0.6 else ("moderate" if frac_below >= 0.5 else "weak")
    txt = (
        f"Below diagonal: {n_below}/{n_total} ({frac_below:.1%})\n"
        f"Above diagonal: {n_above}/{n_total}\n"
        f"ROI-wise recovery: {judgement}"
    )
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.85,
            edgecolor="gray",
        ),
    )

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    summary = {
        "n_total": n_total,
        "n_below_diagonal": n_below,
        "n_above_diagonal": n_above,
        "fraction_below_diagonal": frac_below,
        "judgement": judgement,
    }

    return fig, ax, summary


def plot_lesion_response_vs_recovery_scatter(
    lesion_response_df: pd.DataFrame,
    roi_recovery_df: pd.DataFrame,
    roi_names: Optional[Sequence[str]] = None,
    lesion_idx: Optional[int] = None,
    response_col: str = "response_abs",
    recovery_col: str = "roi_recovery",
    label_top_k: int = 8,
    color_by: str = "improvement",   # "improvement" / "roi_recovery" / None
    figsize=(6.2, 5.4),
    cmap="viridis",
    alpha=0.80,
    point_size=48,
    show_reg_line: bool = True,
    save_path: Optional[str] = None,
):
    """
    Stage 1 lesion response vs Stage 4 ROI recovery scatter.

    x-axis:
        Stage 1 lesion response_abs

    y-axis:
        Stage 4 ROI-wise recovery

    标签规则：
        1. 红线以上的节点：标签优先放在右上角或右侧中部空白处
        2. 红线以下的节点：标签优先放在右下角空白处
        3. 标签使用方框 + 引导线
        4. 尽量避免标签之间、标签和散点之间重叠
    """
    from scipy.stats import pearsonr, spearmanr

    # ==================================================
    # 1. Basic checks
    # ==================================================
    if "roi" not in lesion_response_df.columns:
        raise KeyError("lesion_response_df must contain column: 'roi'")
    if response_col not in lesion_response_df.columns:
        raise KeyError(f"lesion_response_df must contain column: '{response_col}'")

    if "roi" not in roi_recovery_df.columns:
        raise KeyError("roi_recovery_df must contain column: 'roi'")
    if recovery_col not in roi_recovery_df.columns:
        raise KeyError(f"roi_recovery_df must contain column: '{recovery_col}'")

    # ==================================================
    # 2. Merge Stage 1 response and Stage 4 recovery
    # ==================================================
    left_cols = ["roi", response_col]
    if "roi_name" in lesion_response_df.columns:
        left_cols.append("roi_name")
    elif "roi_name_stage1" in lesion_response_df.columns:
        left_cols.append("roi_name_stage1")

    df1 = lesion_response_df[left_cols].copy()

    df2_cols = ["roi", recovery_col]
    for c in ["improvement", "D_lesion_roi", "D_policy_roi"]:
        if c in roi_recovery_df.columns:
            df2_cols.append(c)

    df2 = roi_recovery_df[df2_cols].copy()

    df = pd.merge(df1, df2, on="roi", how="inner")

    if len(df) == 0:
        raise ValueError(
            "No overlapping ROI found between lesion_response_df and roi_recovery_df."
        )

    df["roi"] = df["roi"].astype(int)

    # ==================================================
    # 3. ROI labels
    # ==================================================
    if roi_names is not None:
        df["_label_name"] = [
            roi_names[int(r)] if int(r) < len(roi_names) else str(int(r))
            for r in df["roi"].astype(int).values
        ]
    else:
        name_col = None
        for c in ["roi_name", "roi_name_stage1"]:
            if c in df.columns:
                name_col = c
                break

        if name_col is not None:
            df["_label_name"] = df[name_col].astype(str)
        else:
            df["_label_name"] = df["roi"].astype(str)

    x = df[response_col].astype(float).values
    y = df[recovery_col].astype(float).values

    # ==================================================
    # 4. Color
    # ==================================================
    if color_by is None:
        color_values = None
    else:
        if color_by not in df.columns:
            raise KeyError(f"color_by='{color_by}' not found in merged dataframe.")
        color_values = df[color_by].astype(float).values

    # ==================================================
    # 5. Correlation and regression line
    # ==================================================
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    coef = np.polyfit(x, y, deg=1)
    slope, intercept = float(coef[0]), float(coef[1])

    # ==================================================
    # 6. Helper: smart right-side callout labels
    # ==================================================
    def _draw_region_callout_labels(
        ax,
        full_df: pd.DataFrame,
        callout_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        label_col: str = "_callout_label",
        region_col: str = "_label_region",
        priority_col: str = "_label_priority",
        fontsize: int = 8,
    ):
        """
        按区域放置标签：
            upper/right_mid: 右上或右侧中部
            lower: 右下

        会估计标签框大小，并惩罚：
            1. 标签挡住散点
            2. 标签之间重叠
            3. 标签出界
        """
        if callout_df is None or len(callout_df) == 0:
            return

        callout_df = callout_df.copy()
        full_df = full_df.copy()

        if priority_col in callout_df.columns:
            callout_df = callout_df.sort_values(priority_col).reset_index(drop=True)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]

        points = full_df[[x_col, y_col]].astype(float).values
        point_roi = full_df["roi"].astype(int).values if "roi" in full_df.columns else None

        placed_rects = []

        def _label_size(label: str):
            lines = str(label).split("\n")
            max_chars = max(len(s) for s in lines)
            n_lines = len(lines)

            w = max(0.055 * x_span, max_chars * 0.0105 * x_span)
            h = max(0.040 * y_span, n_lines * 0.045 * y_span)
            return w, h

        def _rect(tx, ty, w, h, ha, va):
            if ha == "left":
                x0, x1 = tx, tx + w
            elif ha == "right":
                x0, x1 = tx - w, tx
            else:
                x0, x1 = tx - w / 2, tx + w / 2

            if va == "bottom":
                y0, y1 = ty, ty + h
            elif va == "top":
                y0, y1 = ty - h, ty
            else:
                y0, y1 = ty - h / 2, ty + h / 2

            return x0, y0, x1, y1

        def _overlap(r1, r2, pad_x=0.0, pad_y=0.0):
            a0, b0, a1, b1 = r1
            c0, d0, c1, d1 = r2

            a0 -= pad_x
            a1 += pad_x
            b0 -= pad_y
            b1 += pad_y

            c0 -= pad_x
            c1 += pad_x
            d0 -= pad_y
            d1 += pad_y

            return not (a1 < c0 or c1 < a0 or b1 < d0 or d1 < b0)

        def _point_overlap_count(rect, current_roi=None):
            x0, y0, x1, y1 = rect

            pad_x = 0.010 * x_span
            pad_y = 0.014 * y_span

            inside = (
                (points[:, 0] >= x0 - pad_x) &
                (points[:, 0] <= x1 + pad_x) &
                (points[:, 1] >= y0 - pad_y) &
                (points[:, 1] <= y1 + pad_y)
            )

            if current_roi is not None and point_roi is not None:
                inside = inside & (point_roi != int(current_roi))

            return int(np.sum(inside))

        def _outside_penalty(rect):
            x0, y0, x1, y1 = rect
            p = 0.0
            if x0 < xlim[0]:
                p += abs(xlim[0] - x0) / x_span
            if x1 > xlim[1]:
                p += abs(x1 - xlim[1]) / x_span
            if y0 < ylim[0]:
                p += abs(ylim[0] - y0) / y_span
            if y1 > ylim[1]:
                p += abs(y1 - ylim[1]) / y_span
            return p

        # 右侧候选槽位：用轴坐标比例转成数据坐标
        # upper: 右上 / 中上空白
        # lower: 右下空白
        upper_slots_frac = [
            (0.88, 0.86),
            (0.82, 0.78),
            (0.92, 0.72),
            (0.78, 0.66),
            (0.90, 0.60),
            (0.74, 0.55),
        ]

        lower_slots_frac = [
            (0.90, 0.28),
            (0.82, 0.22),
            (0.94, 0.16),
            (0.76, 0.34),
            (0.88, 0.40),
            (0.72, 0.26),
        ]

        mid_slots_frac = [
            (0.90, 0.50),
            (0.80, 0.48),
            (0.94, 0.44),
            (0.76, 0.55),
        ]

        def _frac_to_data(fx, fy):
            return (
                xlim[0] + fx * x_span,
                ylim[0] + fy * y_span,
            )

        for _, r in callout_df.iterrows():
            px = float(r[x_col])
            py = float(r[y_col])
            roi = int(r["roi"]) if "roi" in r and pd.notna(r["roi"]) else None

            label = (
                str(r[label_col])
                if label_col in r and pd.notna(r[label_col])
                else str(roi)
            )

            region = (
                str(r[region_col])
                if region_col in r and pd.notna(r[region_col])
                else "upper"
            )

            if region == "lower":
                candidate_fracs = lower_slots_frac + mid_slots_frac + upper_slots_frac
            elif region == "upper":
                candidate_fracs = upper_slots_frac + mid_slots_frac + lower_slots_frac
            else:
                candidate_fracs = mid_slots_frac + upper_slots_frac + lower_slots_frac

            w, h = _label_size(label)

            best = None
            best_score = np.inf

            for j, (fx, fy) in enumerate(candidate_fracs):
                tx, ty = _frac_to_data(fx, fy)

                # 右侧标签一般让文字向左贴齐更稳
                # 如果槽位太靠右，用 right 防止出界
                if fx >= 0.88:
                    ha = "right"
                else:
                    ha = "left"

                if region == "lower":
                    va = "center"
                elif region == "upper":
                    va = "center"
                else:
                    va = "center"

                rect = _rect(tx, ty, w, h, ha, va)

                # 惩罚：越靠前的候选槽位越优先
                slot_score = 0.40 * j

                # 惩罚：离点太远也不好，但比避免重叠优先级低
                dist_score = (
                    abs(tx - px) / x_span
                    + abs(ty - py) / y_span
                )

                point_overlap = _point_overlap_count(rect, current_roi=roi)

                label_overlap = sum(
                    _overlap(
                        rect,
                        old,
                        pad_x=0.006 * x_span,
                        pad_y=0.010 * y_span,
                    )
                    for old in placed_rects
                )

                outside = _outside_penalty(rect)

                score = (
                    slot_score
                    + 1.0 * dist_score
                    + 16.0 * point_overlap
                    + 40.0 * label_overlap
                    + 100.0 * outside
                )

                if score < best_score:
                    best_score = score
                    best = (tx, ty, ha, va, rect)

            if best is None:
                continue

            tx, ty, ha, va, rect = best
            placed_rects.append(rect)

            ax.annotate(
                label,
                xy=(px, py),
                xytext=(tx, ty),
                textcoords="data",
                ha=ha,
                va=va,
                fontsize=fontsize,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    facecolor="white",
                    edgecolor="lightgray",
                    alpha=0.94,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.8,
                    shrinkA=2,
                    shrinkB=3,
                    alpha=0.85,
                ),
                zorder=30,
            )

    # ==================================================
    # 7. Plot scatter
    # ==================================================
    fig, ax = plt.subplots(figsize=figsize)

    if color_values is None:
        sc = ax.scatter(
            x,
            y,
            s=point_size,
            alpha=alpha,
            edgecolor="k",
            linewidth=0.25,
        )
    else:
        sc = ax.scatter(
            x,
            y,
            c=color_values,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            edgecolor="k",
            linewidth=0.25,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.040, pad=0.03)
        cbar.set_label(color_by)

    # Regression line
    xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    yy = slope * xx + intercept

    if show_reg_line:
        ax.plot(
            xx,
            yy,
            color="red",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
        )

    ax.grid(alpha=0.22, linestyle="--")
    ax.axhline(0, linestyle=":", color="gray", linewidth=1.2)
    ax.margins(x=0.10, y=0.12)

    # 先设置标题轴标签，保证坐标范围稳定
    ax.set_xlabel(f"Stage 1 lesion response ({response_col})")
    ax.set_ylabel(f"Stage 4 ROI-wise recovery ({recovery_col})")
    ax.set_title("Lesion response vs ROI recovery")

    # ==================================================
    # 8. Build callout labels
    # ==================================================
    callout_parts = []

    # lesion ROI
    if lesion_idx is not None:
        lesion_rows = df[df["roi"].astype(int) == int(lesion_idx)]
        if len(lesion_rows) > 0:
            r0 = lesion_rows.iloc[0]

            ax.scatter(
                [float(r0[response_col])],
                [float(r0[recovery_col])],
                s=180,
                facecolors="none",
                edgecolors="blue",
                linewidths=2.0,
                zorder=10,
            )

            lesion_label_df = lesion_rows.copy()
            lesion_label_df["_callout_label"] = [
                f"Lesion ROI\n{str(r0['_label_name'])}"
            ]

            # 按红线判断 lesion 标签区域
            pred_y = slope * float(r0[response_col]) + intercept
            lesion_label_df["_label_region"] = (
                "upper" if float(r0[recovery_col]) >= pred_y else "lower"
            )
            lesion_label_df["_label_priority"] = 0
            callout_parts.append(lesion_label_df)

    # Top-K response nodes
    if label_top_k is not None and label_top_k > 0:
        top_df = df.nlargest(int(label_top_k), response_col).copy()

        if lesion_idx is not None:
            top_df = top_df[top_df["roi"].astype(int) != int(lesion_idx)].copy()

        top_df["_callout_label"] = top_df["_label_name"].astype(str)
        top_df["_label_priority"] = 1

        # 红线以上/以下分区
        pred_line = slope * top_df[response_col].astype(float).values + intercept
        is_above = top_df[recovery_col].astype(float).values >= pred_line
        top_df["_label_region"] = np.where(is_above, "upper", "lower")

        callout_parts.append(top_df)

    # Draw callouts after axis limits are settled
    if len(callout_parts) > 0:
        callout_df = (
            pd.concat(callout_parts, axis=0)
            .drop_duplicates("roi", keep="first")
            .reset_index(drop=True)
        )

        _draw_region_callout_labels(
            ax=ax,
            full_df=df,
            callout_df=callout_df,
            x_col=response_col,
            y_col=recovery_col,
            label_col="_callout_label",
            region_col="_label_region",
            priority_col="_label_priority",
            fontsize=8,
        )

    # ==================================================
    # 9. Correlation text
    # ==================================================
    txt = (
        f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.1e})\n"
        f"Spearman ρ = {spearman_rho:.3f} (p = {spearman_p:.1e})"
    )
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.85,
            edgecolor="gray",
        ),
        zorder=40,
    )

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    summary = {
        "n_roi": int(len(df)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "slope": float(slope),
        "intercept": float(intercept),
    }

    return fig, ax, df, summary


def make_stage4_roi_recovery_topk_table(
    lesion_response_df,
    roi_recovery_df,
    top_k: int = 20,
    roi_names=None,
    sort_by: str = "roi_recovery",
    response_col: str = "response_abs",
    round_digits: int = 6,
    save_csv_path: str = None,
):
    """
    Build Stage 4 ROI recovery Top-K table.

    Columns:
        ROI
        Stage1 response_abs
        D_lesion_roi
        D_policy_roi
        roi_recovery
        improvement

    Parameters
    ----------
    lesion_response_df:
        Usually stage1_res["lesion_response_df"].

    roi_recovery_df:
        Usually stage4_res["roi_recovery_df"].

    sort_by:
        "roi_recovery" or "improvement".
    """
    import os
    import numpy as np
    import pandas as pd

    # -----------------------------
    # 1. basic checks
    # -----------------------------
    if "roi" not in lesion_response_df.columns:
        raise KeyError("lesion_response_df must contain column: roi")

    if response_col not in lesion_response_df.columns:
        raise KeyError(f"lesion_response_df must contain column: {response_col}")

    required_recovery_cols = ["roi", "D_lesion_roi", "D_policy_roi"]
    for c in required_recovery_cols:
        if c not in roi_recovery_df.columns:
            raise KeyError(f"roi_recovery_df must contain column: {c}")

    # -----------------------------
    # 2. prepare Stage 1 response table
    # -----------------------------
    stage1_cols = ["roi", response_col]

    if "roi_name" in lesion_response_df.columns:
        stage1_cols.append("roi_name")

    stage1_df = lesion_response_df[stage1_cols].copy()

    # -----------------------------
    # 3. prepare Stage 4 recovery table
    # -----------------------------
    recovery_df = roi_recovery_df.copy()

    if "roi_recovery" not in recovery_df.columns:
        recovery_df["roi_recovery"] = (
            1.0 - recovery_df["D_policy_roi"] / (recovery_df["D_lesion_roi"] + 1e-12)
        )

    if "improvement" not in recovery_df.columns:
        recovery_df["improvement"] = recovery_df["D_lesion_roi"] - recovery_df["D_policy_roi"]

    recovery_cols = [
        "roi",
        "D_lesion_roi",
        "D_policy_roi",
        "roi_recovery",
        "improvement",
    ]

    recovery_df = recovery_df[recovery_cols].copy()

    # -----------------------------
    # 4. merge Stage 1 and Stage 4
    # -----------------------------
    table_df = recovery_df.merge(
        stage1_df,
        on="roi",
        how="left"
    )

    # -----------------------------
    # 5. add ROI label
    # -----------------------------
    if "roi_name" in table_df.columns:
        table_df["ROI"] = table_df["roi_name"].astype(str)
    elif roi_names is not None:
        table_df["ROI"] = [roi_names[int(r)] for r in table_df["roi"].astype(int).values]
    else:
        table_df["ROI"] = table_df["roi"].astype(str)

    # -----------------------------
    # 6. sort
    # -----------------------------
    if sort_by not in table_df.columns:
        raise KeyError(f"sort_by={sort_by} not found. Available columns: {table_df.columns.tolist()}")

    table_df = (
        table_df
        .sort_values(sort_by, ascending=False)
        .head(int(top_k))
        .reset_index(drop=True)
    )

    table_df.insert(0, "rank", np.arange(1, len(table_df) + 1))

    # -----------------------------
    # 7. select and rename columns
    # -----------------------------
    out = table_df[
        [
            "rank",
            "roi",
            "ROI",
            response_col,
            "D_lesion_roi",
            "D_policy_roi",
            "roi_recovery",
            "improvement",
        ]
    ].copy()

    out = out.rename(columns={
        response_col: "Stage1_response_abs",
    })

    # -----------------------------
    # 8. round numeric columns
    # -----------------------------
    numeric_cols = [
        "Stage1_response_abs",
        "D_lesion_roi",
        "D_policy_roi",
        "roi_recovery",
        "improvement",
    ]

    for c in numeric_cols:
        out[c] = out[c].astype(float).round(round_digits)

    # -----------------------------
    # 9. save
    # -----------------------------
    if save_csv_path is not None:
        save_dir = os.path.dirname(save_csv_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        out.to_csv(save_csv_path, index=False, encoding="utf-8-sig")

    return out


def plot_stage4_roi_recovery_topk_table(
    table_df,
    title: str = "Top-K ROI-wise recovery table",
    figsize=None,
    font_size: int = 8,
    save_path: str = None,
):
    """
    Render Stage 4 ROI recovery Top-K table as a figure.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    df = table_df.copy()

    n_rows, n_cols = df.shape

    if figsize is None:
        figsize = (
            min(17, 1.35 * n_cols),
            max(3.5, 0.42 * n_rows + 1.5)
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1.0, 1.25)

    # -----------------------------
    # style
    # -----------------------------
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eaeaea")
        else:
            if row % 2 == 0:
                cell.set_facecolor("#f8f8f8")

    ax.set_title(title, fontsize=13, pad=12)

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax