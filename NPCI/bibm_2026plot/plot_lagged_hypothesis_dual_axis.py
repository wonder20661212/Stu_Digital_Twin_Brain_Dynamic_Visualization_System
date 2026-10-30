from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _setup_paper_style():
    plt.rcParams.update(
        {
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
        }
    )
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


def plot_metrics_with_runtime(
    paper_csv: str,
    raw_csv: str,
    output_path: str,
    title: str,
) -> None:
    _setup_paper_style()
    paper_df = pd.read_csv(paper_csv)
    raw_df = pd.read_csv(raw_csv)

    df = paper_df.merge(
        raw_df[["roi_size", "full_time_sec", "screen_local_time_sec"]],
        left_on="ROI size",
        right_on="roi_size",
        how="left",
    )
    df["Relative runtime"] = 100.0 * df["screen_local_time_sec"] / df["full_time_sec"]

    x = df["ROI size"].astype(int).values

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    fig.patch.set_facecolor("white")

    left_series = [
        ("Recall", {"marker": "o", "linestyle": "-", "linewidth": 1.8}),
        ("Precision", {"marker": "s", "linestyle": "-", "linewidth": 1.8}),
        ("Jaccard", {"marker": "^", "linestyle": "-", "linewidth": 1.8}),
        ("Top-50 Recall", {"marker": "d", "linestyle": "--", "linewidth": 1.4}),
        ("Top-100 Recall", {"marker": "P", "linestyle": "--", "linewidth": 1.4}),
        ("Top-200 Recall", {"marker": "*", "linestyle": "--", "linewidth": 1.4}),
    ]

    for col, style in left_series:
        ax.plot(x, df[col].astype(float).values, label=col, **style)

    ax.set_xlabel("ROI size")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.05)
    _beautify_axis(ax)

    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    ax2.patch.set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.plot(
        x,
        df["Relative runtime"].values,
        color="black",
        marker="o",
        linestyle="-.",
        linewidth=1.8,
        label="Relative runtime",
    )
    ax2.set_ylabel("Relative runtime (%)", color="black")
    ax2.tick_params(axis="y", colors="black")
    ax2.set_ylim(0.0, max(df["Relative runtime"].max() * 1.25, 20.0))

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="lower right",
        frameon=False,
        ncol=2,
    )

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    plot_metrics_with_runtime(
        paper_csv="./results/tables/lagged_hypothesis/lagged_hypothesis_hop2_paper_table.csv",
        raw_csv="./results/tables/lagged_hypothesis/lagged_hypothesis_hop2_raw.csv",
        output_path="./results/figures/lagged_hypothesis_figs_paper/fig2_hop2_metrics_by_roi.png",
        title="Hop-2 lesion neighborhood: metric comparison across ROI sizes",
    )
