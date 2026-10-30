import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# 0. Global figure style
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
    Convert possible string columns such as '9.41×' to float.
    """
    return (
        s.astype(str)
        .str.replace("×", "", regex=False)
        .str.replace("x", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )


def _standardize_columns(df):
    """
    Standardize possible column names.
    For example:
        Top-50 -> Top-50 Recall
        Top-100 -> Top-100 Recall
        Top-200 -> Top-200 Recall
    """
    rename_map = {}

    for col in df.columns:
        c = col.strip()

        if c == "Top-50":
            rename_map[col] = "Top-50 Recall"
        elif c == "Top-100":
            rename_map[col] = "Top-100 Recall"
        elif c == "Top-200":
            rename_map[col] = "Top-200 Recall"
        else:
            rename_map[col] = c

    return df.rename(columns=rename_map)


def load_lagged_table(path):
    """
    Load and clean paper table.
    """
    df = pd.read_csv(path)
    df = _standardize_columns(df)

    numeric_cols = [
        "ROI size",
        "Screen nodes",
        "Full edges",
        "Local edges",
        "Intersection",
        "Recall",
        "Precision",
        "Jaccard",
        "Speedup",
        "Top-50 Recall",
        "Top-100 Recall",
        "Top-200 Recall",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = _to_float_series(df[col])

    df = df.sort_values("ROI size").reset_index(drop=True)
    return df


def _save_figure(fig, out_dir, out_stem):
    """
    Save each figure as both PNG and PDF.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{out_stem}.png"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")

    return {"png": str(png_path)}


def _beautify_axis(ax):
    """
    Paper-style axis formatting.
    """
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
# 2. Figure 1 / Figure 2
# ============================================================

def plot_metric_lines_by_roi(
    df,
    hop_label,
    out_dir,
    out_stem,
    show=False,
):
    """
    Figure 1 / Figure 2:
    Plot Recall / Precision / Jaccard / Top-K Recall across ROI sizes.

    Parameters
    ----------
    df : pd.DataFrame
    hop_label : str
    out_dir : str or Path
    out_stem : str
    show : bool, default False
        Whether to display the figure in notebook.
        Default is False: save only, do not display.
    """
    metric_cols = [
        "Recall",
        "Precision",
        "Jaccard",
        "Top-50 Recall",
        "Top-100 Recall",
        "Top-200 Recall",
    ]

    line_markers = ["o", "s", "^", "D", "v", "P"]
    line_styles = ["-", "-", "-", "--", "--", "--"]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    x = df["ROI size"].values

    for i, metric in enumerate(metric_cols):
        ax.plot(
            x,
            df[metric].values,
            marker=line_markers[i],
            linestyle=line_styles[i],
            linewidth=2.0,
            markersize=6,
            label=metric,
        )

    ax.set_xlabel("ROI size")
    ax.set_ylabel("Metric value")
    ax.set_title(f"{hop_label}: metric comparison across ROI sizes")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])

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
# 3. Figure 3
# ============================================================

def plot_hop_average_comparison(
    hop1_df,
    hop2_df,
    out_dir,
    out_stem,
    show=False,
):
    """
    Figure 3:
    Grouped bar plot for average metrics between hop=1 and hop=2.

    Parameters
    ----------
    hop1_df : pd.DataFrame
    hop2_df : pd.DataFrame
    out_dir : str or Path
    out_stem : str
    show : bool, default False
        Whether to display the figure in notebook.
        Default is False: save only, do not display.
    """
    metric_cols = [
        "Recall",
        "Precision",
        "Jaccard",
        "Top-50 Recall",
        "Top-100 Recall",
        "Top-200 Recall",
    ]

    hop1_avg = hop1_df[metric_cols].mean()
    hop2_avg = hop2_df[metric_cols].mean()

    x = np.arange(len(metric_cols))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    bars1 = ax.bar(
        x - width / 2,
        hop1_avg.values,
        width=width,
        label="Hop-1",
        edgecolor="black",
        linewidth=0.8,
    )

    bars2 = ax.bar(
        x + width / 2,
        hop2_avg.values,
        width=width,
        label="Hop-2",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_ylabel("Average metric value")
    ax.set_title("Average metric comparison between Hop-1 and Hop-2")

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Recall", "Precision", "Jaccard", "Top-50", "Top-100", "Top-200"],
        rotation=20,
        ha="right",
    )

    ax.set_ylim(0.0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))

    _beautify_axis(ax)
    ax.legend(frameon=False, loc="upper right")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015,
                f"{h:.3f}",
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
# 4. Figure 4
# ============================================================

def plot_speedup_by_roi(
    hop1_df,
    hop2_df=None,
    out_dir="./results/figures/lagged_hypothesis_figs_paper",
    out_stem="fig4_speedup_by_roi",
    show=False,
):
    """
    Figure 4:
    Plot speedup only.

    Speedup is independent of hop because hop=1 and hop=2 are recomputed
    from the same Full LPCMCI and Screening-local LPCMCI runtimes.
    Therefore, only one speedup curve is plotted.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.5))

    x = hop1_df["ROI size"].values
    y = hop1_df["Speedup"].values

    ax.plot(
        x,
        y,
        marker="o",
        linestyle="-",
        linewidth=2.4,
        markersize=6.5,
        label="Screening-local speedup",
    )

    ax.set_xlabel("ROI size")
    ax.set_ylabel("Speedup over full LPCMCI")
    ax.set_title("Runtime speedup across ROI sizes")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])

    # 给 y 轴上下留出空间，避免文字被挤出图框
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_pad = max(0.4, 0.10 * (y_max - y_min))

    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # 左右边距也稍微留一点
    ax.set_xlim(x[0] - 1.5, x[-1] + 1.5)

    _beautify_axis(ax)

    ax.legend(frameon=False, loc="upper left")

    # 用 annotate 而不是直接 ax.text(x, y+常数)
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
# 5. Main entry
# ============================================================

def make_lagged_hypothesis_figures(
    hop1_csv="./results/tables/lagged_hypothesis/lagged_hypothesis_hop1_paper_table.csv",
    hop2_csv="./results/tables/lagged_hypothesis/lagged_hypothesis_hop2_paper_table.csv",
    out_dir="./results/figures/lagged_hypothesis_figs_paper",
    show=False,
):
    """
    Main entry function.

    Parameters
    ----------
    hop1_csv : str
    hop2_csv : str
    out_dir : str
    show : bool, default False
        Whether to display figures in notebook.
        Default is False: save only, do not display.

    Returns
    -------
    hop1_df : pd.DataFrame
    hop2_df : pd.DataFrame
    saved_paths : dict
    """
    _setup_paper_style()

    hop1_df = load_lagged_table(hop1_csv)
    hop2_df = load_lagged_table(hop2_csv)

    saved_paths = {}

    saved_paths["fig1_hop1"] = plot_metric_lines_by_roi(
        hop1_df,
        hop_label="Hop-1 lesion neighborhood",
        out_dir=out_dir,
        out_stem="fig1_hop1_metrics_by_roi",
        show=show,
    )

    saved_paths["fig2_hop2"] = plot_metric_lines_by_roi(
        hop2_df,
        hop_label="Hop-2 lesion neighborhood",
        out_dir=out_dir,
        out_stem="fig2_hop2_metrics_by_roi",
        show=show,
    )

    saved_paths["fig3_hop_compare"] = plot_hop_average_comparison(
        hop1_df,
        hop2_df,
        out_dir=out_dir,
        out_stem="fig3_hop1_vs_hop2_average_metrics",
        show=show,
    )

    saved_paths["fig4_speedup"] = plot_speedup_by_roi(
        hop1_df,
        hop2_df=None,
        out_dir=out_dir,
        out_stem="fig4_speedup_by_roi",
        show=show,
    )

    return hop1_df, hop2_df, saved_paths
