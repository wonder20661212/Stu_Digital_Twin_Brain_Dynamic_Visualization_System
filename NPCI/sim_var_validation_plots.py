# sim_var_validation_plots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 0. Global figure style
# ============================================================

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
        "figure.dpi": 120,
    })


# ============================================================
# 1. Utility functions
# ============================================================

def _to_float_series(s):
    """
    Convert possible string columns such as '2.99×' to float.
    """
    return (
        s.astype(str)
        .str.replace("×", "", regex=False)
        .str.replace("x", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )


def load_ratio_comparison_table(path):
    """
    Load and clean sim_var_validation_ratio_comparison.csv
    """
    df = pd.read_csv(path)

    # strip spaces in column names
    df.columns = [c.strip() for c in df.columns]

    numeric_cols = [
        "Compression ratio",
        "Lesion node coverage",
        "Lesion edge coverage",
        "Global edge coverage",
        "Screening node FPR",
        "Recall",
        "Precision",
        "FPR",
        "AUPRC",
        "AUROC",
        "Lesion recall",
        "Lesion Jaccard",
        "Speedup",
        "Screen nodes",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = _to_float_series(df[col])

    # Parse ratio numeric from Config like "ratio=0.55"
    if "Config" in df.columns:
        df["ratio_numeric"] = (
            df["Config"].astype(str).str.extract(r"([0-9.]+)")[0].astype(float)
        )
        df = df.sort_values("ratio_numeric").reset_index(drop=True)
    else:
        raise ValueError("Input CSV must contain column 'Config'.")

    return df


def _save_figure(fig, out_dir, out_stem):
    """
    Save figure as PNG + PDF.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{out_stem}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    return {"png": str(png_path)}


def _beautify_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
    )

    ax.tick_params(
        axis="both",
        direction="out",
        length=4,
        width=1.0,
    )


# ============================================================
# 2. Figure 1: Core recovery metrics across ratios
# ============================================================

def plot_core_metrics_by_ratio(
    df,
    out_dir,
    out_stem="fig1_core_metrics_by_ratio",
    show=False,
):
    """
    Plot core recovery metrics across screening ratios.

    Metrics:
        - Lesion edge coverage
        - Recall
        - Lesion recall
        - Precision
        - AUPRC
        - AUROC
    """
    metrics = [
        "Lesion edge coverage",
        "Recall",
        "Lesion recall",
        "Precision",
        "AUPRC",
        "AUROC",
    ]

    markers = ["o", "s", "^", "D", "v", "P"]
    linestyles = ["-", "-", "-", "--", "--", "--"]

    x = df["ratio_numeric"].values
    xlabels = df["Config"].tolist()

    fig, ax = plt.subplots(figsize=(7.8, 5.0))

    for i, metric in enumerate(metrics):
        ax.plot(
            x,
            df[metric].values,
            marker=markers[i],
            linestyle=linestyles[i],
            linewidth=2.0,
            markersize=6,
            label=metric,
        )

    ax.set_xlabel("Screening ratio setting")
    ax.set_ylabel("Metric value")
    ax.set_title("Core recovery metrics across screening ratios")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    ax.set_ylim(0.0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))

    _beautify_axis(ax)

    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower right",
        handlelength=2.5,
    )

    fig.tight_layout()
    saved_paths = _save_figure(fig, out_dir, out_stem)

    if show:
        plt.show()

    plt.close(fig)
    return saved_paths


# ============================================================
# 3. Figure 2: FPR comparison
# ============================================================

def plot_fpr_comparison(
    df,
    out_dir,
    out_stem="fig2_fpr_comparison",
    show=False,
):
    """
    Plot final edge FPR and screening node FPR across ratios.

    Uses:
    - categorical x positions for even spacing
    - log-scale y-axis because the two FPRs differ by orders of magnitude
    - padded x/y limits so labels stay inside figure
    """
    import matplotlib.ticker as mticker

    x = np.arange(len(df))
    xlabels = df["Config"].tolist()

    y_final = df["FPR"].values
    y_screen = df["Screening node FPR"].values

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        x,
        y_final,
        marker="o",
        linestyle="-",
        linewidth=2.2,
        markersize=6,
        label="Final edge FPR",
    )

    ax.plot(
        x,
        y_screen,
        marker="s",
        linestyle="--",
        linewidth=2.2,
        markersize=6,
        label="Screening node FPR",
    )

    ax.set_xlabel("Screening ratio setting")
    ax.set_ylabel("FPR (log scale)")
    ax.set_title("False positive behavior across screening ratios")

    # ---------- x axis: categorical + side padding ----------
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(-0.10, len(x) - 1 + 0.15)   # 控制左右边缘不要太近

    # ---------- y axis: log scale + better limits ----------
    ax.set_yscale("log")

    y_all = np.concatenate([y_final, y_screen])
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))

    # 在 log scale 下，用倍率留白更自然
    ax.set_ylim(y_min / 1.6, y_max * 1.45)

    # 更规范的 log 刻度
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    _beautify_axis(ax)
    ax.legend(frameon=False, loc="upper left")

    # ---------- 标注：用 offset points，避免挤出图框 ----------
    for xi, yi in zip(x, y_final):
        ax.annotate(
            f"{yi:.3f}",
            xy=(xi, yi),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    for xi, yi in zip(x, y_screen):
        ax.annotate(
            f"{yi:.3f}",
            xy=(xi, yi),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fig.tight_layout()
    saved_paths = _save_figure(fig, out_dir, out_stem)

    if show:
        plt.show()

    plt.close(fig)
    return saved_paths


# ============================================================
# 4. Figure 3: Compression-recall trade-off
# ============================================================

def plot_tradeoff_summary(
    df,
    out_dir,
    out_stem="fig3_compression_recall_tradeoff",
    show=False,
):
    """
    Plot compression–recall trade-off.

    Left y-axis:
        - Compression ratio
        - Lesion edge coverage
        - Lesion recall

    Right y-axis:
        - Speedup

    Uses categorical x positions to avoid uneven spacing between 0.70 and 0.75.
    """
    x = np.arange(len(df))
    xlabels = df["Config"].tolist()

    fig, ax1 = plt.subplots(figsize=(7.6, 5.0))

    # ----------------------------
    # Left axis: 0-1 metrics
    # ----------------------------
    ax1.plot(
        x,
        df["Compression ratio"].values,
        marker="o",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        label="Compression ratio",
    )

    ax1.plot(
        x,
        df["Lesion edge coverage"].values,
        marker="s",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        label="Lesion edge coverage",
    )

    ax1.plot(
        x,
        df["Lesion recall"].values,
        marker="^",
        linestyle="--",
        linewidth=2.0,
        markersize=6,
        label="Lesion recall",
    )

    ax1.set_xlabel("Screening ratio setting")
    ax1.set_ylabel("Coverage / recall / compression")
    ax1.set_title("Compression–recall trade-off across screening ratios")

    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels)

    ax1.set_ylim(0.0, 1.10)
    ax1.set_yticks(np.linspace(0, 1.0, 6))

    _beautify_axis(ax1)

    # ----------------------------
    # Right axis: speedup
    # ----------------------------
    ax2 = ax1.twinx()

    speedup = df["Speedup"].values
    y_min = float(np.min(speedup))
    y_max = float(np.max(speedup))
    y_pad = max(0.15, 0.12 * (y_max - y_min))

    ax2.plot(
        x,
        speedup,
        marker="D",
        linestyle=":",
        linewidth=2.4,
        markersize=6.5,
        label="Speedup",
    )

    ax2.set_ylabel("Speedup over full LPCMCI")
    ax2.set_ylim(y_min - y_pad, y_max + y_pad)

    # 右轴边框处理
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(
        axis="y",
        direction="out",
        length=4,
        width=1.0,
    )

    # Speedup 标注：用 offset points，避免超出图框
    for xi, yi in zip(x, speedup):
        ax2.annotate(
            f"{yi:.2f}×",
            xy=(xi, yi),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )

    fig.tight_layout()

    saved_paths = _save_figure(fig, out_dir, out_stem)

    if show:
        plt.show()

    plt.close(fig)
    return saved_paths


# ============================================================
# 5. Figure 4: Speedup only
# ============================================================

def plot_speedup_by_ratio(
    df,
    out_dir,
    out_stem="fig4_speedup_by_ratio",
    show=False,
):
    """
    Plot speedup only across screening ratios.

    Uses categorical x positions and padded y-limits so labels stay inside figure.
    """
    x = np.arange(len(df))
    xlabels = df["Config"].tolist()
    y = df["Speedup"].values

    fig, ax = plt.subplots(figsize=(6.8, 4.5))

    ax.plot(
        x,
        y,
        marker="o",
        linestyle="-",
        linewidth=2.4,
        markersize=6.5,
        label="Speedup",
    )

    ax.set_xlabel("Screening ratio setting")
    ax.set_ylabel("Speedup over full LPCMCI")
    ax.set_title("Runtime speedup across screening ratios")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    # 给上下都留空间，避免 2.99× 超出图框
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_pad = max(0.15, 0.12 * (y_max - y_min))

    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    _beautify_axis(ax)

    ax.legend(frameon=False, loc="upper right")

    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi:.2f}×",
            xy=(xi, yi),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    saved_paths = _save_figure(fig, out_dir, out_stem)

    if show:
        plt.show()

    plt.close(fig)
    return saved_paths


# ============================================================
# 6. Main entry
# ============================================================

def make_sim_var_validation_figures(
    ratio_compare_csv="./results/tables/sim_var_validation/sim_var_validation_ratio_comparison.csv",
    out_dir="./results/figures/sim_var_validation_figs_paper",
    show=False,
):
    """
    Main entry function.

    Parameters
    ----------
    ratio_compare_csv : str
        Path to sim_var_validation_ratio_comparison.csv
    out_dir : str
        Output directory for figures
    show : bool, default False
        Whether to display figures in notebook.
        Default is False: save only, do not display.

    Returns
    -------
    df : pd.DataFrame
    saved_paths : dict
    """
    _setup_paper_style()

    df = load_ratio_comparison_table(ratio_compare_csv)

    saved_paths = {}

    saved_paths["fig1_core_metrics"] = plot_core_metrics_by_ratio(
        df,
        out_dir=out_dir,
        out_stem="fig1_core_metrics_by_ratio",
        show=show,
    )

    saved_paths["fig2_fpr"] = plot_fpr_comparison(
        df,
        out_dir=out_dir,
        out_stem="fig2_fpr_comparison",
        show=show,
    )

    saved_paths["fig3_tradeoff"] = plot_tradeoff_summary(
        df,
        out_dir=out_dir,
        out_stem="fig3_compression_recall_tradeoff",
        show=show,
    )

    saved_paths["fig4_speedup"] = plot_speedup_by_ratio(
        df,
        out_dir=out_dir,
        out_stem="fig4_speedup_by_ratio",
        show=show,
    )

    return df, saved_paths
