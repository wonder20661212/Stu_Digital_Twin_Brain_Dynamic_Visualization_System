import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle
from matplotlib import cm, colors
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Patch
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union


def _draw_network_callout_labels(
    ax,
    pos,
    labels,
    roles,
    lesion_idx,
    label_nodes=None,
    fontsize=8,
):
    """
    节点标签：方框 + 引导线。
    左侧节点标签放左边，右侧节点标签放右边，lesion 放上方。
    """
    if label_nodes is None:
        label_nodes = list(labels.keys())

    label_nodes = set(int(x) for x in label_nodes)

    for node, (x, y) in pos.items():
        node = int(node)
        if node not in label_nodes:
            continue

        role = roles.get(node, "other")
        label = labels.get(node, str(node))

        if node == int(lesion_idx):
            xytext = (x, y + 0.70)
            ha, va = "center", "bottom"
        elif x < 0:
            xytext = (x - 0.55, y)
            ha, va = "right", "center"
        elif x > 0:
            xytext = (x + 0.55, y)
            ha, va = "left", "center"
        else:
            xytext = (x, y - 0.65)
            ha, va = "center", "top"

        ax.annotate(
            label,
            xy=(x, y),
            xytext=xytext,
            textcoords="data",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.92,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="gray",
                lw=0.8,
                shrinkA=2,
                shrinkB=3,
                alpha=0.85,
            ),
            zorder=10,
        )


def _draw_direct_core_callout_labels(
    ax,
    pos,
    labels,
    lesion_idx,
    label_nodes=None,
    fontsize=8,
    x_offset=0.72,
    y_offset=0.72,
):
    """
    Direct lesion core graph 标签：
    - 左侧节点：标签放左边
    - 右侧节点：标签放右边
    - lesion：标签放上方
    - 标签为方框 + 引导线
    """
    if label_nodes is None:
        label_nodes = list(labels.keys())

    label_nodes = set(int(x) for x in label_nodes)

    for n, p in pos.items():
        n = int(n)
        if n not in label_nodes:
            continue

        x, y = float(p[0]), float(p[1])
        label = str(labels.get(n, str(n)))

        if n == int(lesion_idx):
            xytext = (x, y + y_offset)
            ha, va = "center", "bottom"
        elif x < 0:
            xytext = (x - x_offset, y)
            ha, va = "right", "center"
        elif x > 0:
            xytext = (x + x_offset, y)
            ha, va = "left", "center"
        else:
            xytext = (x, y - y_offset)
            ha, va = "center", "top"

        ax.annotate(
            label,
            xy=(x, y),
            xytext=xytext,
            textcoords="data",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.94,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="gray",
                lw=0.8,
                shrinkA=2,
                shrinkB=4,
                alpha=0.85,
            ),
            zorder=20,
        )


def _spread_layer_y(n, span=4.8, avoid_zero=True):
    """
    给一列节点分配 y 坐标。
    avoid_zero=True 时尽量避开 y=0，避免节点挡住 lesion 发出的中心线。
    """
    n = int(n)

    if n <= 0:
        return []

    if n == 1:
        return [0.0]

    if avoid_zero and n % 2 == 1:
        # 多生成一个点，然后去掉最接近 0 的点
        ys = np.linspace(span / 2, -span / 2, n + 1)
        drop_idx = int(np.argmin(np.abs(ys)))
        ys = np.delete(ys, drop_idx)
    else:
        ys = np.linspace(span / 2, -span / 2, n)

    return list(ys.astype(float))


def get_stage1_top_response_nodes(
    brainmap_df: pd.DataFrame,
    value_col: str = "response_abs",
    lesion_idx: int = None,
    top_k: int = 20,
):
    """
    从 Stage 1 response 结果中取 Top-K 响应 ROI。
    """
    if "roi" not in brainmap_df.columns:
        raise KeyError("brainmap_df must contain column 'roi'.")

    df = brainmap_df.dropna(subset=["roi", value_col]).copy()
    df["roi"] = df["roi"].astype(int)
    df["_score"] = np.abs(df[value_col].astype(float))

    if lesion_idx is not None:
        df = df[df["roi"] != int(lesion_idx)]

    top_nodes = (
        df.sort_values("_score", ascending=False)
          .head(int(top_k))["roi"]
          .astype(int)
          .tolist()
    )

    return top_nodes

def get_stage2_top_screening_nodes(
    node_score_df: pd.DataFrame,
    score_col: str = "final_screen_score",
    lesion_idx: int = None,
    top_k: int = 20,
):
    """
    从 Stage 2 lagged screening 结果中取 Top-K 节点。
    """
    df = node_score_df.copy()

    if "node" not in df.columns:
        raise KeyError("node_score_df must contain column 'node'.")

    # 兼容不同 score 列名
    if score_col not in df.columns:
        fallback_cols = [
            "candidate_score",
            "screening_score",
            "final_score",
            "score",
            "lagged_score",
        ]

        found = None
        for c in fallback_cols:
            if c in df.columns:
                found = c
                break

        if found is None:
            raise KeyError(
                f"Cannot find score column. Tried {score_col} and {fallback_cols}. "
                f"Current columns are: {df.columns.tolist()}"
            )

        score_col = found

    if "selected" in df.columns:
        df = df[df["selected"].astype(bool)].copy()

    df["node"] = df["node"].astype(int)
    df["_score"] = df[score_col].astype(float)

    if lesion_idx is not None:
        df = df[df["node"] != int(lesion_idx)]

    top_nodes = (
        df.sort_values("_score", ascending=False)
          .head(int(top_k))["node"]
          .astype(int)
          .tolist()
    )

    return top_nodes



def plot_stage2_screening_overlap(
    overlap_info: dict,
    roi_names=None,
    title: str = "Stage 1 response and lagged-screening overlap",
    figsize=(9.8, 4.8),
    max_overlap_show: int = 8,
    save_path=None,
):
    """
    Paper-style overlap visualization.

    Fix:
    - Red Stage 1 title and blue Stage 2 title are separated.
    - Titles are above their own circles and use white bbox.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    stage1_nodes = set(overlap_info["stage1_nodes"])
    screening_nodes = set(overlap_info["screening_nodes"])
    overlap_nodes = set(overlap_info["overlap_nodes"])

    only_stage1 = stage1_nodes - screening_nodes
    only_screening = screening_nodes - stage1_nodes

    n_only_stage1 = len(only_stage1)
    n_only_screening = len(only_screening)
    n_overlap = len(overlap_nodes)

    recall = float(overlap_info["stage1_screen_overlap_topK"])
    precision = float(overlap_info["screening_overlap_precision"])
    jaccard = float(overlap_info["jaccard"])

    # overlap labels
    overlap_list = sorted(list(overlap_nodes))
    if roi_names is not None:
        overlap_labels = [
            roi_names[int(x)] if int(x) < len(roi_names) else str(int(x))
            for x in overlap_list
        ]
    else:
        overlap_labels = [str(int(x)) for x in overlap_list]

    if len(overlap_labels) > max_overlap_show:
        shown_labels = overlap_labels[:max_overlap_show] + ["..."]
    else:
        shown_labels = overlap_labels

    # figure
    fig, (ax_venn, ax_bar) = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    # --------------------------------------------------
    # Left: Venn-like plot
    # --------------------------------------------------
    ax_venn.set_aspect("equal")
    ax_venn.axis("off")

    r = 0.82
    c1 = (-0.50, 0.0)
    c2 = (0.50, 0.0)

    circle1 = Circle(
        c1, r,
        facecolor="#f4a3a3",
        edgecolor="#df6f6f",
        alpha=0.62,
        linewidth=1.2,
    )
    circle2 = Circle(
        c2, r,
        facecolor="#9ecae1",
        edgecolor="#4f91c2",
        alpha=0.62,
        linewidth=1.2,
    )

    ax_venn.add_patch(circle1)
    ax_venn.add_patch(circle2)

    # separated titles
    ax_venn.text(
        c1[0] - 0.33, 1.13,
        "Stage 1 top-30\nresponse nodes",
        ha="center",
        va="center",
        fontsize=10,
        color="#a83232",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.80),
    )

    ax_venn.text(
        c2[0] + 0.33, 1.13,
        "Stage 2 top-30\nlagged screening nodes",
        ha="center",
        va="center",
        fontsize=10,
        color="#1f5f99",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.80),
    )

    # counts
    ax_venn.text(c1[0] - 0.37, 0.0, str(n_only_stage1),
                 ha="center", va="center", fontsize=18, fontweight="bold", color="#a83232")
    ax_venn.text(0.0, 0.0, str(n_overlap),
                 ha="center", va="center", fontsize=20, fontweight="bold", color="black")
    ax_venn.text(c2[0] + 0.37, 0.0, str(n_only_screening),
                 ha="center", va="center", fontsize=18, fontweight="bold", color="#1f5f99")

    # overlap nodes box below
    overlap_text = "Overlap nodes: " + ", ".join(shown_labels)
    ax_venn.text(
        0.0,
        -1.18,
        overlap_text,
        ha="center",
        va="center",
        fontsize=8.5,
        color="dimgray",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="lightgray",
            alpha=0.94,
        ),
    )

    ax_venn.set_xlim(-1.55, 1.55)
    ax_venn.set_ylim(-1.35, 1.38)

    # --------------------------------------------------
    # Right: metrics bar
    # --------------------------------------------------
    metric_names = ["Recall", "Precision", "Jaccard"]
    metric_vals = [recall, precision, jaccard]

    x = np.arange(len(metric_names))
    bars = ax_bar.bar(
        x,
        metric_vals,
        width=0.62,
        color="#4C88B8",
        edgecolor="#2F5F85",
        alpha=0.90,
    )

    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Overlap metrics", fontsize=11)

    for bar, val in zip(bars, metric_vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.035,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="dimgray",
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, rotation=30, ha="right", fontsize=9)

    ax_bar.tick_params(axis="y", labelsize=9)
    ax_bar.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=13, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    return fig, (ax_venn, ax_bar)


def plot_stage1_screening_overlap_metrics(
    overlap_info: dict,
    figsize=(5.2, 3.8),
    save_path=None,
):
    """
    Plot overlap metrics as a compact bar chart.
    """

    metrics = {
        "Recall": overlap_info["stage1_screen_overlap_topK"],
        "Precision": overlap_info["screening_overlap_precision"],
        "Jaccard": overlap_info["jaccard"],
    }

    fig, ax = plt.subplots(figsize=figsize)

    names = list(metrics.keys())
    vals = list(metrics.values())

    ax.bar(names, vals, edgecolor="black", linewidth=0.5, alpha=0.88)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Overlap score")
    ax.set_title("Stage1-response / lagged-screening overlap")

    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax


def plot_screening_node_scores(
    node_score_df: pd.DataFrame,
    top_k: int = 25,
    roi_names: Optional[Sequence[str]] = None,
    lesion_idx: Optional[int] = None,
    stage1_res: Optional[Dict] = None,
    stage1_top_k: int = 30,
    score_col: str = "final_screen_score",
    ax=None,
    figsize=(11.5, 4.8),
    x_rotation: int = 60,
    legend_outside: bool = True,
    show: bool = True,
    return_ax: bool = False,
):
    """
    Plot Top-K whole-brain lagged screening candidates with node-category colors.

    默认修改：
      - top_k = 25，可改成 20
      - x轴标签旋转 60°
      - legend 放到图右侧
      - 标出 lesion node
      - 默认不返回 Axes，避免 Jupyter 输出 <Axes: ...>

    Categories:
        1. lesion node
        2. Stage 1 response overlap node
        3. lagged parent proxy
        4. lagged child proxy
        5. mixed lagged proxy
        6. other selected node
    """

    df = node_score_df.copy()

    # --------------------------------------------------
    # 1. 兼容 score 列
    # --------------------------------------------------
    if score_col not in df.columns:
        fallback_cols = [
            "candidate_score",
            "screening_score",
            "final_score",
            "score",
            "lagged_score",
        ]
        found = None
        for c in fallback_cols:
            if c in df.columns:
                found = c
                break

        if found is None:
            raise KeyError(
                f"Cannot find score column. Tried {score_col} and {fallback_cols}. "
                f"Current columns are: {df.columns.tolist()}"
            )

        score_col = found

    # --------------------------------------------------
    # 2. 只取 selected 节点
    # --------------------------------------------------
    if "selected" in df.columns:
        df = df[df["selected"].astype(bool)].copy()

    df = df.sort_values(score_col, ascending=False).head(int(top_k)).copy()

    if len(df) == 0:
        raise ValueError("No selected screening nodes to plot.")

    if "node" not in df.columns:
        raise KeyError("node_score_df must contain column 'node'.")

    df["node"] = df["node"].astype(int)

    # --------------------------------------------------
    # 3. 构造 Stage 1 overlap node set
    # --------------------------------------------------
    stage1_overlap_nodes = set()

    if stage1_res is not None:
        if isinstance(stage1_res, dict):
            if "lesion_response_df" in stage1_res:
                response_df = stage1_res["lesion_response_df"].copy()
            elif "response_df" in stage1_res:
                response_df = stage1_res["response_df"].copy()
            else:
                response_df = None
        elif isinstance(stage1_res, pd.DataFrame):
            response_df = stage1_res.copy()
        else:
            response_df = None

        if response_df is not None and len(response_df) > 0:
            if "node" not in response_df.columns and "roi" in response_df.columns:
                response_df = response_df.rename(columns={"roi": "node"})

            response_score_col = None
            for c in [
                "response_abs",
                "lesion_response",
                "abnormality_score",
                "spread_score",
                "candidate_score",
                "score",
            ]:
                if c in response_df.columns:
                    response_score_col = c
                    break

            if response_score_col is not None and "node" in response_df.columns:
                stage1_overlap_nodes = set(
                    response_df
                    .sort_values(response_score_col, ascending=False)
                    ["node"]
                    .astype(int)
                    .head(int(stage1_top_k))
                    .tolist()
                )

    # --------------------------------------------------
    # 4. 自动识别 parent / child proxy 分数列
    # --------------------------------------------------
    parent_cols = [
        "parent_proxy",
        "lagged_parent_score",
        "to_lesion_score",
        "lag_to_lesion",
        "score_to_lesion",
    ]

    child_cols = [
        "child_proxy",
        "lagged_child_score",
        "from_lesion_score",
        "lag_from_lesion",
        "score_from_lesion",
    ]

    parent_col = next((c for c in parent_cols if c in df.columns), None)
    child_col = next((c for c in child_cols if c in df.columns), None)

    has_proxy_cols = parent_col is not None and child_col is not None

    # --------------------------------------------------
    # 5. 节点分类
    # --------------------------------------------------
    def classify_node(row):
        node = int(row["node"])

        if lesion_idx is not None and node == int(lesion_idx):
            return "lesion node"

        if node in stage1_overlap_nodes:
            return "Stage 1 overlap"

        if has_proxy_cols:
            p = float(row[parent_col])
            c = float(row[child_col])

            if p > 0 and c > 0:
                ratio = min(p, c) / (max(p, c) + 1e-12)
                if ratio >= 0.65:
                    return "mixed lagged proxy"

            if p >= c:
                return "lagged parent proxy"
            else:
                return "lagged child proxy"

        return "other selected"

    df["node_role"] = df.apply(classify_node, axis=1)

    # --------------------------------------------------
    # 6. 颜色设置
    # --------------------------------------------------
    role_colors = {
        "lesion node": "#1f4eaa",             # 深蓝
        "Stage 1 overlap": "#d62728",        # 红
        "lagged parent proxy": "#ff7f0e",    # 橙
        "lagged child proxy": "#2ca02c",     # 绿
        "mixed lagged proxy": "#9467bd",     # 紫
        "other selected": "#7f7f7f",         # 灰
    }

    colors = [role_colors[x] for x in df["node_role"].values]

    # --------------------------------------------------
    # 7. 标签
    # --------------------------------------------------
    labels = [str(int(x)) for x in df["node"].values]

    if roi_names is not None:
        labels = [roi_names[int(x)] for x in df["node"].values]

    # --------------------------------------------------
    # 8. 画图
    # --------------------------------------------------
    created_fig = False

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    x = np.arange(len(df))
    y = df[score_col].astype(float).values

    bars = ax.bar(
        x,
        y,
        color=colors,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.92,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=x_rotation,
        ha="right",
        fontsize=9
    )

    ax.set_ylabel("Screening score", fontsize=11)
    ax.set_title(
        f"Top {top_k} whole-brain lagged screening candidates",
        fontsize=13
    )

    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    # --------------------------------------------------
    # 9. 标出 lesion node
    # --------------------------------------------------
    if lesion_idx is not None:
        ymax = float(np.max(y))

        for i, node in enumerate(df["node"].astype(int).values):
            if node == int(lesion_idx):
                # 加粗 lesion 柱子的边框
                bars[i].set_edgecolor("#1f4eaa")
                bars[i].set_linewidth(1.8)

                # 在柱子上方标注 lesion
                ax.text(
                    i,
                    y[i] + 0.035 * ymax,
                    "lesion",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#1f4eaa",
                    fontweight="bold",
                )

                # 可选：小箭头，进一步突出
                ax.annotate(
                    "",
                    xy=(i, y[i]),
                    xytext=(i, y[i] + 0.025 * ymax),
                    arrowprops=dict(
                        arrowstyle="-",
                        color="#1f4eaa",
                        lw=1.0
                    )
                )

    # --------------------------------------------------
    # 10. 图例放到右侧
    # --------------------------------------------------
    used_roles = df["node_role"].unique().tolist()

    legend_handles = [
        Patch(
            facecolor=role_colors[r],
            edgecolor="black",
            label=r,
            alpha=0.92,
        )
        for r in role_colors.keys()
        if r in used_roles
    ]

    if legend_outside:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=9,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=(0, 0, 0.84, 1))
    else:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=9,
            loc="upper right",
        )
        fig.tight_layout()
    if show and created_fig:
        plt.show()

    if return_ax:
        return ax


def plot_edge_table_network(
    edge_df: pd.DataFrame,
    lesion_idx: int,
    roi_names: Optional[Sequence[str]] = None,
    weight_col: str = "abs_val",
    ax=None,
):
    """
    用 NetworkX 画最终边表网络。它会把 lesion 节点画得更大
    适合快速检查：
        lesion 是否位于核心？
        边是否太密？
        上下游结构是否清楚？
        是否有明显 bridge node？
    """
    G = nx.DiGraph()
    for _, row in edge_df.iterrows():
        s, d = int(row["src"]), int(row["dst"])
        G.add_edge(s, d, weight=float(row[weight_col]))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    if len(G.nodes) == 0:
        ax.set_title("Empty lesion-centered causal core")
        ax.axis("off")
        return ax

    pos = nx.spring_layout(G, seed=123, weight="weight")
    labels = {n: (roi_names[n] if roi_names is not None else str(n)) for n in G.nodes()}
    node_sizes = [700 if n == int(lesion_idx) else 350 for n in G.nodes()]
    max_w = float(edge_df[weight_col].max()) if len(edge_df) > 0 else 1.0
    widths = [1.0 + 3.0 * G[u][v].get("weight", 1.0) / (max_w + 1e-12) for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, width=widths, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    ax.set_title("Lesion-centered lagged LPCMCI core")
    ax.axis("off")
    return ax


def _get_stage1_response_map(stage1_res):
    """
    Extract response_abs map:
        node -> response_abs
    """
    if stage1_res is None:
        return {}

    if isinstance(stage1_res, dict):
        if "lesion_response_df" in stage1_res:
            df = stage1_res["lesion_response_df"].copy()
        elif "response_df" in stage1_res:
            df = stage1_res["response_df"].copy()
        else:
            return {}
    elif isinstance(stage1_res, pd.DataFrame):
        df = stage1_res.copy()
    else:
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


def _get_candidate_score_map(candidate_df):
    """
    Extract candidate score map:
        node -> candidate_score
    """
    if candidate_df is None or len(candidate_df) == 0:
        return {}

    df = candidate_df.copy()

    if "node" not in df.columns:
        if "roi" in df.columns:
            df = df.rename(columns={"roi": "node"})
        else:
            return {}

    score_col = None
    for c in [
        "candidate_score",
        "objective",
        "single_objective",
        "response_abs",
        "score",
    ]:
        if c in df.columns:
            score_col = c
            break

    if score_col is None:
        return {}

    return dict(zip(df["node"].astype(int), df[score_col].astype(float)))


def _normalize_values(values, eps=1e-12):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values

    lo = np.nanmin(values)
    hi = np.nanmax(values)

    if hi - lo < eps:
        return np.ones_like(values) * 0.5

    return (values - lo) / (hi - lo + eps)


def classify_lpcmci_core_nodes(
    lesion_core_edges: pd.DataFrame,
    lesion_idx: int,
):
    """
    Classify nodes in lesion-centered LPCMCI core graph.

    Roles:
        lesion
        parent      : node -> lesion
        child       : lesion -> node
        spouse      : node -> child, where lesion -> child also exists
        downstream  : child -> node
        bridge      : connected to both parent-side and child-side nodes
        second_order
    """
    lesion_idx = int(lesion_idx)

    if lesion_core_edges is None or len(lesion_core_edges) == 0:
        return {lesion_idx: "lesion"}

    df = lesion_core_edges.copy()
    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)

    nodes = set(df["src"].tolist()) | set(df["dst"].tolist()) | {lesion_idx}

    parents = set(df.loc[df["dst"] == lesion_idx, "src"].astype(int).tolist())
    children = set(df.loc[df["src"] == lesion_idx, "dst"].astype(int).tolist())

    parents.discard(lesion_idx)
    children.discard(lesion_idx)

    first_order = parents | children

    # Skeleton adjacency
    adj = {int(n): set() for n in nodes}
    out_adj = {int(n): set() for n in nodes}
    in_adj = {int(n): set() for n in nodes}

    for _, row in df.iterrows():
        s = int(row["src"])
        d = int(row["dst"])
        adj.setdefault(s, set()).add(d)
        adj.setdefault(d, set()).add(s)
        out_adj.setdefault(s, set()).add(d)
        in_adj.setdefault(d, set()).add(s)

    roles = {}
    roles[lesion_idx] = "lesion"

    for n in parents:
        roles[n] = "parent"

    for n in children:
        if n in roles:
            roles[n] = "parent/child"
        else:
            roles[n] = "child"

    for n in nodes:
        n = int(n)
        if n in roles:
            continue

        # spouse: another parent of lesion child
        # n -> c and lesion -> c
        is_spouse = False
        for c in children:
            if c in out_adj.get(n, set()):
                is_spouse = True
                break

        if is_spouse:
            roles[n] = "spouse"
            continue

        # downstream: child -> n
        is_downstream = False
        for c in children:
            if n in out_adj.get(c, set()):
                is_downstream = True
                break

        if is_downstream:
            roles[n] = "downstream"
            continue

        # bridge: connected with both parent-side and child-side nodes
        neigh = adj.get(n, set())
        if len(neigh & parents) > 0 and len(neigh & children) > 0:
            roles[n] = "bridge"
        else:
            roles[n] = "second_order"

    return roles


def build_lpcmci_core_graph_layout(
    lesion_core_edges: pd.DataFrame,
    lesion_idx: int,
    roles: dict,
):
    """
    Build a deterministic layered layout.

    Center:
        lesion
    First ring:
        parents on left, children on right
    Second ring:
        bridge/spouse/downstream/second_order outside
    """
    lesion_idx = int(lesion_idx)

    nodes = sorted(list(roles.keys()))
    pos = {}

    pos[lesion_idx] = np.array([0.0, 0.0])

    parents = [n for n, r in roles.items() if r in ["parent", "parent/child"] and n != lesion_idx]
    children = [n for n, r in roles.items() if r in ["child", "parent/child"] and n != lesion_idx]

    spouse_nodes = [n for n, r in roles.items() if r == "spouse"]
    downstream_nodes = [n for n, r in roles.items() if r == "downstream"]
    bridge_nodes = [n for n, r in roles.items() if r == "bridge"]
    second_nodes = [n for n, r in roles.items() if r == "second_order"]

    # parent side: left arc
    if len(parents) > 0:
        angles = np.linspace(130, 230, len(parents))
        for n, a in zip(parents, angles):
            rad = np.deg2rad(a)
            pos[n] = np.array([1.45 * np.cos(rad), 1.45 * np.sin(rad)])

    # child side: right arc
    if len(children) > 0:
        angles = np.linspace(-50, 50, len(children))
        for n, a in zip(children, angles):
            rad = np.deg2rad(a)
            pos[n] = np.array([1.45 * np.cos(rad), 1.45 * np.sin(rad)])

    # outer ring
    outer_nodes = bridge_nodes + spouse_nodes + downstream_nodes + second_nodes
    if len(outer_nodes) > 0:
        angles = np.linspace(0, 360, len(outer_nodes), endpoint=False)
        for n, a in zip(outer_nodes, angles):
            rad = np.deg2rad(a)
            pos[n] = np.array([2.55 * np.cos(rad), 2.55 * np.sin(rad)])

    # fallback for any missing node
    missing = [n for n in nodes if n not in pos]
    if len(missing) > 0:
        angles = np.linspace(0, 360, len(missing), endpoint=False)
        for n, a in zip(missing, angles):
            rad = np.deg2rad(a)
            pos[n] = np.array([2.9 * np.cos(rad), 2.9 * np.sin(rad)])

    return pos


def plot_stage2_top_core_edges(
    lesion_core_edges: pd.DataFrame,
    top_k: int = 15,
    roi_names=None,
    lesion_idx: int = None,
    value_col: str = "val",
    abs_col: str = "abs_val",
    plot_type: str = "lollipop",   # "lollipop" or "bar"
    direct_lesion_only: bool = False,
    figsize=(8.4, 5.8),
    title: str = None,
    save_path: str = None,
):
    """
    Plot Top-K strongest LPCMCI core edges as a paper-style supplement figure.

    Purpose
    -------
    This figure replaces dense network visualization for global strongest edges.

    It shows:
      - strongest core edge ranking
      - edge direction: source -> target
      - signed LPCMCI causal value
      - absolute edge strength ranking

    Parameters
    ----------
    lesion_core_edges:
        Stage 2 final core edge table.
        Required:
            src, dst
        Recommended:
            val, abs_val

    top_k:
        Number of strongest edges to show.

    roi_names:
        Optional ROI name list.

    lesion_idx:
        Optional lesion ROI index.
        If provided, lesion-related edge labels can be emphasized.

    value_col:
        Signed LPCMCI value column. Usually "val".

    abs_col:
        Absolute LPCMCI value column. Usually "abs_val".

    plot_type:
        "lollipop" or "bar".

    direct_lesion_only:
        If True, only plot edges directly connected to lesion_idx.

    Returns
    -------
    fig, ax, plot_df
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if lesion_core_edges is None or len(lesion_core_edges) == 0:
        raise ValueError("lesion_core_edges is empty.")

    df = lesion_core_edges.copy()

    required_cols = ["src", "dst"]
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"lesion_core_edges missing columns: {missing}")

    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)

    # --------------------------------------------------
    # 1. 兼容 value / abs column
    # --------------------------------------------------
    if value_col not in df.columns:
        if "weight" in df.columns:
            value_col = "weight"
        elif "val" in df.columns:
            value_col = "val"
        else:
            raise KeyError(
                f"Cannot find signed edge value column. "
                f"Tried '{value_col}', 'weight', 'val'."
            )

    df[value_col] = df[value_col].astype(float)

    if abs_col not in df.columns:
        if "abs_weight" in df.columns:
            abs_col = "abs_weight"
        else:
            df["_abs_val_tmp"] = np.abs(df[value_col].astype(float))
            abs_col = "_abs_val_tmp"

    df[abs_col] = df[abs_col].astype(float)

    # --------------------------------------------------
    # 2. 可选：只看 lesion 直接相关边
    # --------------------------------------------------
    if direct_lesion_only:
        if lesion_idx is None:
            raise ValueError("lesion_idx is required when direct_lesion_only=True.")

        lesion_idx = int(lesion_idx)

        df = df[
            (df["src"] == lesion_idx)
            | (df["dst"] == lesion_idx)
        ].copy()

        if len(df) == 0:
            raise ValueError("No direct lesion-connected edges found.")

    # --------------------------------------------------
    # 3. Top-K strongest edges
    # --------------------------------------------------
    plot_df = (
        df.sort_values(abs_col, ascending=False)
          .head(int(top_k))
          .copy()
          .reset_index(drop=True)
    )

    # 为了水平图从上到下是 strongest -> weaker，画图前反转
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    # --------------------------------------------------
    # 4. Edge label
    # --------------------------------------------------
    def _roi_label(idx):
        idx = int(idx)
        if roi_names is not None and 0 <= idx < len(roi_names):
            return str(roi_names[idx])
        return str(idx)

    labels = []

    for _, r in plot_df.iterrows():
        s = int(r["src"])
        t = int(r["dst"])

        src_label = _roi_label(s)
        dst_label = _roi_label(t)

        labels.append(f"{src_label} → {dst_label}")

    plot_df["edge_label"] = labels

    # --------------------------------------------------
    # 5. Colors by sign
    # --------------------------------------------------
    signed_vals = plot_df[value_col].astype(float).values
    abs_vals = plot_df[abs_col].astype(float).values

    pos_color = "#D55E5E"
    neg_color = "#3B77B6"
    zero_color = "#888888"

    colors = []
    for v in signed_vals:
        if v > 0:
            colors.append(pos_color)
        elif v < 0:
            colors.append(neg_color)
        else:
            colors.append(zero_color)

    # --------------------------------------------------
    # 6. Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    fig.subplots_adjust(
        left=0.34,
        right=0.94,
        bottom=0.12,
        top=0.88,
    )

    y = np.arange(len(plot_df))

    if plot_type == "bar":
        ax.barh(
            y,
            abs_vals,
            color=colors,
            edgecolor="black",
            linewidth=0.45,
            alpha=0.88,
        )

    elif plot_type == "lollipop":
        # horizontal stems
        for yi, xval, col in zip(y, abs_vals, colors):
            ax.hlines(
                yi,
                0,
                xval,
                color=col,
                linewidth=2.0,
                alpha=0.75,
            )

        ax.scatter(
            abs_vals,
            y,
            s=72,
            c=colors,
            edgecolors="black",
            linewidths=0.45,
            zorder=3,
        )

    else:
        raise ValueError("plot_type must be 'lollipop' or 'bar'.")

    # --------------------------------------------------
    # 7. Value annotation
    # --------------------------------------------------
    x_offset = 0.015 * max(np.max(abs_vals), 1e-12)

    for yi, xval, sval in zip(y, abs_vals, signed_vals):
        ax.text(
            xval + x_offset,
            yi,
            f"{sval:+.3f}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="black",
        )

    # --------------------------------------------------
    # 8. Axes style
    # --------------------------------------------------
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["edge_label"].tolist(), fontsize=8.5)

    # --------------------------------------------------
    # 9. Lesion-related edge emphasis
    # --------------------------------------------------
    if lesion_idx is not None:
        lesion_idx = int(lesion_idx)

        for tick_label, (_, r) in zip(ax.get_yticklabels(), plot_df.iterrows()):
            if int(r["src"]) == lesion_idx or int(r["dst"]) == lesion_idx:
                tick_label.set_fontweight("bold")

    ax.set_xlabel("|LPCMCI causal value|", fontsize=10)

    if title is None:
        if direct_lesion_only:
            title = f"Top {top_k} direct lesion-connected core edges"
        else:
            title = f"Top {top_k} strongest LPCMCI core edges"

    ax.set_title(title, fontsize=13, pad=10)

    ax.grid(axis="x", alpha=0.22, linestyle="--", linewidth=0.7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", labelsize=9)

    xmax = max(np.max(abs_vals), 1e-12)
    ax.set_xlim(0, xmax * 1.20)

    # --------------------------------------------------
    # 10. Legend
    # --------------------------------------------------
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color=pos_color,
            label="positive causal value",
            markerfacecolor=pos_color,
            markeredgecolor="black",
            linewidth=2.0,
        ),
        Line2D(
            [0], [0],
            marker="o",
            color=neg_color,
            label="negative causal value",
            markerfacecolor=neg_color,
            markeredgecolor="black",
            linewidth=2.0,
        ),
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=8.5,
        loc="lower right",
    )

    # --------------------------------------------------
    # 11. Save
    # --------------------------------------------------
    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    # 返回原始强度从强到弱的表，更方便检查
    return_df = plot_df.iloc[::-1].reset_index(drop=True)

    return fig, ax, return_df


def plot_lpcmci_lesion_core_chord(
    lesion_core_edges: pd.DataFrame,
    lesion_idx: int,
    candidate_df: pd.DataFrame = None,
    stage1_res: dict = None,
    roi_names=None,
    node_score_source: str = "response_abs",
    weight_col: str = "val",
    abs_col: str = "abs_val",
    max_edges: int = 30,
    top_score_nodes: int = 0,
    top_label_k: int = 8,
    figsize=(13.8, 6.4),
    title: str = "Stage 2 lesion-centered LPCMCI core graph",
    node_cmap: str = "Oranges",
    edge_cmap: str = "RdBu_r",
    ring_width: float = 0.12,          # 保留兼容，不再使用
    edge_width_scale: float = 3.0,
    edge_alpha_min: float = 0.18,
    edge_alpha_max: float = 0.78,
    label_fontsize: int = 8,
    save_path: str = None,
):
    """
    Interface-compatible replacement for the old directed chord plot.

    This version no longer draws a chord diagram. It draws a lesion-centered
    directed layered graph, which is more suitable for LPCMCI causal edges.

    Left panel:
        node color = node_score_source, e.g. response_abs

    Right panel:
        edge color = signed LPCMCI causal value
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D

    lesion_idx = int(lesion_idx)

    # =========================================================
    # 0. Basic checks
    # =========================================================
    if "src" not in lesion_core_edges.columns or "dst" not in lesion_core_edges.columns:
        raise KeyError("lesion_core_edges must contain columns: 'src', 'dst'.")

    raw_edges = lesion_core_edges.copy()
    raw_edges["src"] = raw_edges["src"].astype(int)
    raw_edges["dst"] = raw_edges["dst"].astype(int)

    if weight_col not in raw_edges.columns:
        if "val" in raw_edges.columns:
            weight_col = "val"
        elif "causal_val" in raw_edges.columns:
            weight_col = "causal_val"
        else:
            raw_edges["val"] = 1.0
            weight_col = "val"

    raw_edges[weight_col] = raw_edges[weight_col].astype(float)

    if abs_col not in raw_edges.columns:
        raw_edges[abs_col] = np.abs(raw_edges[weight_col].astype(float))
    else:
        raw_edges[abs_col] = raw_edges[abs_col].astype(float)

    # strongest edges
    # =========================================================
    # Select lesion-centered edges instead of global strongest edges
    # =========================================================

    raw_edges["_abs_for_rank"] = np.abs(raw_edges[abs_col].astype(float))

    # 1) 直接与 lesion 相连的边必须保留
    direct_lesion_edges = raw_edges[
        (raw_edges["src"].astype(int) == int(lesion_idx)) |
        (raw_edges["dst"].astype(int) == int(lesion_idx))
        ].copy()

    direct_lesion_edges = direct_lesion_edges.sort_values(
        "_abs_for_rank",
        ascending=False
    )

    # 2) 构建完整有向图，用于找 lesion 的上游/下游
    import networkx as nx

    G_full = nx.DiGraph()
    for _, r in raw_edges.iterrows():
        G_full.add_edge(int(r["src"]), int(r["dst"]))

    # 3) 找 lesion 的一阶和二阶邻域
    lesion_neighbors = set([int(lesion_idx)])

    if int(lesion_idx) in G_full.nodes:
        # 一阶 parents / children
        lesion_neighbors.update([int(n) for n in G_full.predecessors(int(lesion_idx))])
        lesion_neighbors.update([int(n) for n in G_full.successors(int(lesion_idx))])

        # 二阶 upstream / downstream，避免只剩 direct edges
        for p in list(G_full.predecessors(int(lesion_idx))):
            lesion_neighbors.update([int(n) for n in G_full.predecessors(int(p))])

        for c in list(G_full.successors(int(lesion_idx))):
            lesion_neighbors.update([int(n) for n in G_full.successors(int(c))])

    # 4) lesion-centered subgraph edges
    local_edges = raw_edges[
        raw_edges["src"].astype(int).isin(lesion_neighbors) &
        raw_edges["dst"].astype(int).isin(lesion_neighbors)
        ].copy()

    local_edges = local_edges.sort_values(
        "_abs_for_rank",
        ascending=False
    )

    # 5) 先保留 direct lesion edges，再用 local strongest edges 填充
    edge_df = pd.concat(
        [direct_lesion_edges, local_edges],
        axis=0
    ).drop_duplicates(["src", "dst"], keep="first")

    edge_df = edge_df.head(int(max_edges)).copy().reset_index(drop=True)

    # 6) 如果 direct lesion edges 为空，给出明确提示
    if len(direct_lesion_edges) == 0:
        print(
            "[Warning] No direct lesion-connected edges found. "
            "The plot will use local strongest edges, but lesion may appear isolated."
        )

    # 7) 如果 edge_df 仍然为空，直接报错
    if len(edge_df) == 0:
        raise ValueError(
            "No edges selected for plotting. Check lesion_core_edges, lesion_idx, and max_edges."
        )

    # ensure lesion included
    edge_nodes = set(edge_df["src"].astype(int).tolist()) | set(edge_df["dst"].astype(int).tolist())
    edge_nodes.add(lesion_idx)

    # =========================================================
    # 1. Helpers
    # =========================================================
    def _safe_roi_name(idx: int):
        idx = int(idx)
        if roi_names is not None and 0 <= idx < len(roi_names):
            return str(roi_names[idx])
        return str(idx)

    def _build_node_score_df():
        # collect all possible nodes from edges / stage1 / candidate
        node_set = set(raw_edges["src"].astype(int).tolist()) | set(raw_edges["dst"].astype(int).tolist())
        node_set.add(lesion_idx)

        out = pd.DataFrame({"node": sorted(node_set)})

        # Stage 1 response table
        if stage1_res is not None:
            if isinstance(stage1_res, dict) and "lesion_response_df" in stage1_res:
                tmp = stage1_res["lesion_response_df"].copy()
            elif isinstance(stage1_res, pd.DataFrame):
                tmp = stage1_res.copy()
            else:
                tmp = None

            if tmp is not None:
                if "roi" in tmp.columns and "node" not in tmp.columns:
                    tmp = tmp.rename(columns={"roi": "node"})
                elif "node" not in tmp.columns:
                    tmp["node"] = tmp.index.astype(int)

                keep_cols = ["node"]
                for c in ["response_abs", "response_signed", "roi_name"]:
                    if c in tmp.columns:
                        keep_cols.append(c)

                tmp = tmp[keep_cols].copy()
                tmp["node"] = tmp["node"].astype(int)
                out = out.merge(tmp, on="node", how="left")

        # Candidate score table
        if candidate_df is not None:
            tmp = candidate_df.copy()

            if "roi" in tmp.columns and "node" not in tmp.columns:
                tmp = tmp.rename(columns={"roi": "node"})

            if "node" in tmp.columns:
                tmp["node"] = tmp["node"].astype(int)

                cand_score_col = None
                for c in ["candidate_score", "screening_score", "mixed_score", "score"]:
                    if c in tmp.columns:
                        cand_score_col = c
                        break

                keep_cols = ["node"]
                if cand_score_col is not None:
                    keep_cols.append(cand_score_col)
                if "roi_name" in tmp.columns:
                    keep_cols.append("roi_name")

                tmp = tmp[keep_cols].drop_duplicates("node").copy()

                if cand_score_col is not None and cand_score_col != "candidate_score":
                    tmp = tmp.rename(columns={cand_score_col: "candidate_score"})

                out = out.merge(tmp, on="node", how="left", suffixes=("", "_cand"))

                if "roi_name" in out.columns and "roi_name_cand" in out.columns:
                    out["roi_name"] = out["roi_name"].fillna(out["roi_name_cand"])
                elif "roi_name" not in out.columns and "roi_name_cand" in out.columns:
                    out["roi_name"] = out["roi_name_cand"]

        if "roi_name" not in out.columns:
            out["roi_name"] = [_safe_roi_name(n) for n in out["node"].values]
        else:
            out["roi_name"] = out["roi_name"].fillna(
                pd.Series([_safe_roi_name(n) for n in out["node"].values], index=out.index)
            )

        if node_score_source in out.columns:
            out["_node_score"] = out[node_score_source].astype(float).fillna(0.0)
        elif node_score_source == "candidate_score" and "candidate_score" in out.columns:
            out["_node_score"] = out["candidate_score"].astype(float).fillna(0.0)
        else:
            out["_node_score"] = 0.0

        return out

    def _draw_callout_labels(ax, pos, labels, roles, label_nodes):
        label_nodes = set(int(x) for x in label_nodes)

        for n, p in pos.items():
            n = int(n)
            if n not in label_nodes:
                continue

            x, y = float(p[0]), float(p[1])
            label = str(labels.get(n, str(n)))
            role = roles.get(n, "other")

            if n == lesion_idx:
                xytext = (x, y + 0.72)
                ha, va = "center", "bottom"
            elif x < -0.3:
                xytext = (x - 0.55, y)
                ha, va = "right", "center"
            elif x > 0.3:
                xytext = (x + 0.55, y)
                ha, va = "left", "center"
            else:
                if y >= 0:
                    xytext = (x, y + 0.60)
                    ha, va = "center", "bottom"
                else:
                    xytext = (x, y - 0.60)
                    ha, va = "center", "top"

            ax.annotate(
                label,
                xy=(x, y),
                xytext=xytext,
                textcoords="data",
                ha=ha,
                va=va,
                fontsize=label_fontsize,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="lightgray",
                    alpha=0.93,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.8,
                    shrinkA=2,
                    shrinkB=3,
                    alpha=0.82,
                ),
                zorder=10,
            )

    # =========================================================
    # 2. Node score table and optional extra nodes
    # =========================================================
    score_df = _build_node_score_df()

    if top_score_nodes is not None and int(top_score_nodes) > 0:
        extra_nodes = (
            score_df[~score_df["node"].isin(edge_nodes)]
            .sort_values("_node_score", ascending=False)
            .head(int(top_score_nodes))["node"]
            .astype(int)
            .tolist()
        )
    else:
        extra_nodes = []

    node_set = set(edge_nodes) | set(extra_nodes)
    nodes = sorted(node_set)

    node_df = (
        pd.DataFrame({"node": nodes})
        .merge(score_df[["node", "roi_name", "_node_score"]], on="node", how="left")
    )
    node_df["_node_score"] = node_df["_node_score"].fillna(0.0)
    node_df["roi_name"] = node_df["roi_name"].fillna(
        pd.Series([_safe_roi_name(n) for n in node_df["node"].values], index=node_df.index)
    )

    # lesion score: make lesion visible but do not distort colorbar too much
    if lesion_idx in node_df["node"].values:
        current_max = float(node_df["_node_score"].max())
        if current_max > 0:
            node_df.loc[node_df["node"] == lesion_idx, "_node_score"] = max(
                float(node_df.loc[node_df["node"] == lesion_idx, "_node_score"].iloc[0]),
                current_max * 0.85,
            )

    node_score_map = {
        int(r["node"]): float(r["_node_score"])
        for _, r in node_df.iterrows()
    }

    labels = {
        int(r["node"]): str(r["roi_name"])
        for _, r in node_df.iterrows()
    }

    # =========================================================
    # 3. Directed graph and roles
    # =========================================================
    import networkx as nx

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(int(n))

    for _, r in edge_df.iterrows():
        s = int(r["src"])
        d = int(r["dst"])
        if s in node_set and d in node_set:
            G.add_edge(
                s,
                d,
                value=float(r[weight_col]),
                abs_value=float(r[abs_col]),
            )

    G.add_node(lesion_idx)

    parents = set(G.predecessors(lesion_idx)) if lesion_idx in G.nodes else set()
    children = set(G.successors(lesion_idx)) if lesion_idx in G.nodes else set()
    parent_child = parents & children
    parents_only = parents - children
    children_only = children - parents

    ancestors = nx.ancestors(G, lesion_idx) if lesion_idx in G.nodes else set()
    descendants = nx.descendants(G, lesion_idx) if lesion_idx in G.nodes else set()

    upstream = ancestors - parents - {lesion_idx}
    downstream = descendants - children - {lesion_idx}

    assigned = (
        {lesion_idx}
        | set(parents_only)
        | set(children_only)
        | set(parent_child)
        | set(upstream)
        | set(downstream)
    )
    others = set(nodes) - assigned

    roles = {int(lesion_idx): "lesion"}
    for n in upstream:
        roles[int(n)] = "upstream"
    for n in parents_only:
        roles[int(n)] = "parent"
    for n in children_only:
        roles[int(n)] = "child"
    for n in parent_child:
        roles[int(n)] = "parent_child"
    for n in downstream:
        roles[int(n)] = "downstream"
    for n in others:
        roles[int(n)] = "other"

    # =========================================================
    # 4. Layered layout
    # =========================================================
    pos = {}

    def _sort_by_score(ns):
        return sorted(list(ns), key=lambda x: node_score_map.get(int(x), 0.0), reverse=True)

    def _place_vertical(ns, x, y_span=3.8):
        ns = _sort_by_score(ns)
        if len(ns) == 0:
            return
        if len(ns) == 1:
            ys = [0.0]
        else:
            ys = np.linspace(y_span / 2, -y_span / 2, len(ns))
        for n, y in zip(ns, ys):
            pos[int(n)] = np.array([float(x), float(y)])

    _place_vertical(upstream, x=-5.0, y_span=4.0)
    _place_vertical(parents_only, x=-3.0, y_span=4.8)

    pos[lesion_idx] = np.array([0.0, 0.0])

    _place_vertical(children_only, x=3.0, y_span=4.6)
    _place_vertical(parent_child, x=4.3, y_span=4.2)
    _place_vertical(downstream, x=5.6, y_span=4.0)

    # others / isolated extra nodes placed at bottom
    if len(others) > 0:
        other_sorted = _sort_by_score(others)
        xs = np.linspace(-1.6, 1.6, len(other_sorted))
        for n, x in zip(other_sorted, xs):
            pos[int(n)] = np.array([float(x), -3.1])

    # =========================================================
    # 5. Label nodes
    # =========================================================
    label_nodes = [lesion_idx]

    top_score_label_nodes = (
        node_df[node_df["node"] != lesion_idx]
        .sort_values("_node_score", ascending=False)
        .head(int(top_label_k))["node"]
        .astype(int)
        .tolist()
    )
    label_nodes.extend(top_score_label_nodes)

    # add direct parents/children if still within limit
    direct_nodes = list(parents_only | children_only | parent_child)
    direct_nodes = _sort_by_score(direct_nodes)
    for n in direct_nodes:
        if len(label_nodes) >= int(top_label_k) + 1:
            break
        if int(n) not in label_nodes:
            label_nodes.append(int(n))

    label_nodes = list(dict.fromkeys(label_nodes))

    # =========================================================
    # 6. Normalizers
    # =========================================================
    node_vals = np.array([node_score_map.get(int(n), 0.0) for n in nodes], dtype=float)
    nvmin, nvmax = float(np.nanmin(node_vals)), float(np.nanmax(node_vals))
    if np.isclose(nvmin, nvmax):
        nvmax = nvmin + 1e-6
    node_norm = Normalize(vmin=nvmin, vmax=nvmax)
    node_cmap_obj = plt.get_cmap(node_cmap)

    edge_vals = np.array([float(G.edges[e]["value"]) for e in G.edges()], dtype=float)
    if len(edge_vals) == 0:
        edge_vals = np.array([0.0])
    evmax = float(np.nanmax(np.abs(edge_vals)))
    if evmax <= 1e-12:
        evmax = 1.0
    edge_norm = TwoSlopeNorm(vmin=-evmax, vcenter=0.0, vmax=evmax)
    edge_cmap_obj = plt.get_cmap(edge_cmap)

    max_abs_edge = max(
        [float(G.edges[e]["abs_value"]) for e in G.edges()] + [1e-12]
    )

    # =========================================================
    # 7. Panel drawing
    # =========================================================
    def _role_facecolor(n):
        role = roles.get(int(n), "other")
        if role == "lesion":
            return "#E6F2FF"
        if role == "parent":
            return "#DDEBFF"
        if role == "child":
            return "#DFF3E3"
        if role == "parent_child":
            return "#EFE3FF"
        if role == "upstream":
            return "#EEF4FF"
        if role == "downstream":
            return "#EAF8EE"
        return "#EAEAEA"

    def _node_marker_edge(n):
        role = roles.get(int(n), "other")
        if int(n) == lesion_idx:
            return "*", "#2b6cb0", 2.2
        if role in ["parent", "upstream"]:
            return "^", "#2b6cb0", 1.5
        if role in ["child", "downstream"]:
            return "s", "#2f855a", 1.5
        if role == "parent_child":
            return "D", "#805ad5", 1.6
        return "o", "gray", 1.2

    def _draw_edges(ax, mode="neutral"):
        for i, (s, d) in enumerate(G.edges()):
            if s not in pos or d not in pos:
                continue

            p1 = np.asarray(pos[s], dtype=float)
            p2 = np.asarray(pos[d], dtype=float)

            val = float(G.edges[s, d]["value"])
            abs_val = float(G.edges[s, d]["abs_value"])

            if mode == "causal":
                color = edge_cmap_obj(edge_norm(val))
                alpha = edge_alpha_min + (edge_alpha_max - edge_alpha_min) * abs_val / (max_abs_edge + 1e-12)
            else:
                color = "#8f8f8f"
                alpha = 0.45

            width = 0.9 + edge_width_scale * abs_val / (max_abs_edge + 1e-12)

            # edge curvature to reduce overlap
            rad = 0.08 if i % 2 == 0 else -0.08
            if s == lesion_idx or d == lesion_idx:
                rad = 0.10 if p2[1] >= p1[1] else -0.10

            patch = FancyArrowPatch(
                posA=p1,
                posB=p2,
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}",
                linewidth=width,
                color=color,
                alpha=alpha,
                mutation_scale=14,
                shrinkA=18,
                shrinkB=18,
                zorder=1,
            )
            ax.add_patch(patch)

    def _draw_nodes(ax, mode="score"):
        for n in nodes:
            n = int(n)
            if n not in pos:
                continue

            marker, edgecolor, lw = _node_marker_edge(n)

            if mode == "score":
                facecolor = node_cmap_obj(node_norm(node_score_map.get(n, 0.0)))
            else:
                facecolor = _role_facecolor(n)

            if n == lesion_idx:
                size = 1050
            else:
                size = 250 + 520 * node_score_map.get(n, 0.0) / (nvmax + 1e-12)

            ax.scatter(
                [pos[n][0]],
                [pos[n][1]],
                s=size,
                marker=marker,
                c=[facecolor],
                edgecolors=edgecolor,
                linewidths=lw,
                alpha=0.96,
                zorder=3,
            )

    def _setup_axis(ax, panel_title):
        # layer headers only when layer has nodes
        y_header = max([float(p[1]) for p in pos.values()] + [2.0]) + 0.72

        layer_headers = [
            ("Upstream", -5.0, upstream),
            ("Parents", -3.0, parents_only),
            ("Lesion", 0.0, {lesion_idx}),
            ("Children", 3.0, children_only),
            ("Parent/child", 4.3, parent_child),
            ("Downstream", 5.6, downstream),
        ]
        for lab, x, group_nodes in layer_headers:
            if len(group_nodes) == 0:
                continue
            ax.text(
                x,
                y_header,
                lab,
                ha="center",
                va="center",
                fontsize=9,
                color="dimgray",
                fontweight="bold",
                zorder=12,
            )

        _draw_callout_labels(
            ax=ax,
            pos=pos,
            labels=labels,
            roles=roles,
            label_nodes=label_nodes,
        )

        ax.set_title(panel_title, fontsize=12, pad=12)
        ax.axis("off")

        xs = np.array([p[0] for p in pos.values()], dtype=float)
        ys = np.array([p[1] for p in pos.values()], dtype=float)
        ax.set_xlim(xs.min() - 1.15, xs.max() + 1.25)
        ax.set_ylim(ys.min() - 1.15, ys.max() + 1.65)

    # =========================================================
    # 8. Draw two-panel figure
    # =========================================================
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    ax_node, ax_edge = axes

    # Left: node response
    _draw_edges(ax_node, mode="neutral")
    _draw_nodes(ax_node, mode="score")
    _setup_axis(ax_node, f"Node-level {node_score_source}")

    sm_node = cm.ScalarMappable(norm=node_norm, cmap=node_cmap_obj)
    sm_node.set_array([])
    cbar_node = fig.colorbar(sm_node, ax=ax_node, fraction=0.035, pad=0.025)
    cbar_node.set_label(node_score_source, fontsize=10)

    # Right: edge causal value
    _draw_edges(ax_edge, mode="causal")
    _draw_nodes(ax_edge, mode="role")
    _setup_axis(ax_edge, "Edge-level LPCMCI causal value")

    sm_edge = cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap_obj)
    sm_edge.set_array([])
    cbar_edge = fig.colorbar(sm_edge, ax=ax_edge, fraction=0.035, pad=0.025)
    cbar_edge.set_label("LPCMCI causal value", fontsize=10)

    # figure-level legend
    handles = [
        Line2D(
            [0], [0],
            marker="*",
            color="w",
            label="lesion",
            markerfacecolor="#E6F2FF",
            markeredgecolor="#2b6cb0",
            markersize=13,
            markeredgewidth=1.8,
        ),
        Line2D(
            [0], [0],
            marker="^",
            color="w",
            label="parent / upstream",
            markerfacecolor="#DDEBFF",
            markeredgecolor="#2b6cb0",
            markersize=9,
            markeredgewidth=1.4,
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label="child / downstream",
            markerfacecolor="#DFF3E3",
            markeredgecolor="#2f855a",
            markersize=9,
            markeredgewidth=1.4,
        ),
        Line2D(
            [0], [0],
            marker="D",
            color="w",
            label="parent/child",
            markerfacecolor="#EFE3FF",
            markeredgecolor="#805ad5",
            markersize=9,
            markeredgewidth=1.4,
        ),
        Line2D(
            [0], [0],
            color="#8f8f8f",
            lw=2.0,
            label="directed LPCMCI edge",
        ),
    ]

    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(title.replace("(directed chord)", "(directed layered graph)"), fontsize=13, y=0.99)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    chord_info = {
        "edge_df_plotted": edge_df.reset_index(drop=True),
        "node_df": node_df.reset_index(drop=True),
        "roles": roles,
        "positions": pos,
        "label_nodes": label_nodes,
        "n_nodes": len(nodes),
        "n_edges": len(edge_df),
        "node_score_source": node_score_source,
        "weight_col": weight_col,
        "abs_col": abs_col,
        "plot_type": "directed_layered_graph_replacement_for_chord",
    }

    return fig, axes, chord_info



def plot_lpcmci_lesion_core_graph(
    lesion_core_edges: pd.DataFrame,
    lesion_idx: int,
    candidate_df: pd.DataFrame = None,
    stage1_res=None,
    roi_names=None,
    node_score_source: str = "response_abs",
    max_edges: int = 30,
    top_score_nodes: int = 12,
    top_label_k: int = 8,
    edge_alpha: float = 0.48,
    edge_width_scale: float = 2.4,
    node_size_min: float = 240,
    node_size_scale: float = 480,
    lesion_node_size: float = 1050,
    title: str = "Stage 2 direct lesion-connected LPCMCI core graph",
    save_path: str = None,
):
    """
    Paper-style direct lesion-connected LPCMCI core graph.

    一行两列：
    左图：Node-level response_abs
        - 节点颜色表示 response_abs
        - 边为中性灰色，只表示方向

    右图：Edge-level LPCMCI causal value
        - 节点颜色按角色
        - 边颜色表示 LPCMCI causal value

    布局：
        parent nodes -> lesion -> child / parent-child nodes
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D

    lesion_idx = int(lesion_idx)

    if "src" not in lesion_core_edges.columns or "dst" not in lesion_core_edges.columns:
        raise KeyError("lesion_core_edges must contain columns: src, dst")

    edge_df = lesion_core_edges.copy()
    edge_df["src"] = edge_df["src"].astype(int)
    edge_df["dst"] = edge_df["dst"].astype(int)

    # --------------------------------------------------
    # 1. 只保留直接与 lesion 相连的边
    # --------------------------------------------------
    edge_df = edge_df[
        (edge_df["src"] == lesion_idx) |
        (edge_df["dst"] == lesion_idx)
    ].copy()

    if len(edge_df) == 0:
        raise ValueError("No direct lesion-connected LPCMCI edges found.")

    # edge value column
    if "val" in edge_df.columns:
        value_col = "val"
    elif "causal_val" in edge_df.columns:
        value_col = "causal_val"
    elif "weight" in edge_df.columns:
        value_col = "weight"
    else:
        edge_df["val"] = 1.0
        value_col = "val"

    edge_df[value_col] = edge_df[value_col].astype(float)

    if "abs_val" not in edge_df.columns:
        edge_df["abs_val"] = np.abs(edge_df[value_col].astype(float))
    else:
        edge_df["abs_val"] = edge_df["abs_val"].astype(float)

    edge_df = (
        edge_df.sort_values("abs_val", ascending=False)
        .head(int(max_edges))
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 2. 角色划分
    # --------------------------------------------------
    parents = set(edge_df.loc[edge_df["dst"] == lesion_idx, "src"].astype(int))
    children = set(edge_df.loc[edge_df["src"] == lesion_idx, "dst"].astype(int))

    parent_child = parents & children
    parents_only = parents - children
    children_only = children - parents

    # 右侧统一放 child 和 parent_child，避免方形节点跑到中间挡线
    left_nodes = sorted(list(parents_only))
    right_nodes = sorted(list(children_only | parent_child))

    all_nodes = [lesion_idx] + left_nodes + right_nodes
    all_nodes = list(dict.fromkeys([int(x) for x in all_nodes]))

    roles = {int(lesion_idx): "lesion"}
    for n in parents_only:
        roles[int(n)] = "parent"
    for n in children_only:
        roles[int(n)] = "child"
    for n in parent_child:
        roles[int(n)] = "parent_child"

    # --------------------------------------------------
    # 3. 节点分数：优先 stage1 response_abs
    # --------------------------------------------------
    node_scores = {int(n): 0.0 for n in all_nodes}

    # stage1_res
    s1_df = None
    if stage1_res is not None:
        if isinstance(stage1_res, dict) and "lesion_response_df" in stage1_res:
            s1_df = stage1_res["lesion_response_df"].copy()
        elif isinstance(stage1_res, pd.DataFrame):
            s1_df = stage1_res.copy()

    if s1_df is not None:
        if "node" in s1_df.columns and "roi" not in s1_df.columns:
            s1_df = s1_df.rename(columns={"node": "roi"})
        if "roi" in s1_df.columns and node_score_source in s1_df.columns:
            s1_df["roi"] = s1_df["roi"].astype(int)
            for _, r in s1_df.iterrows():
                roi = int(r["roi"])
                if roi in node_scores:
                    node_scores[roi] = float(r[node_score_source])

    # candidate_df fallback
    if candidate_df is not None:
        cdf = candidate_df.copy()
        if "roi" in cdf.columns:
            id_col = "roi"
        elif "node" in cdf.columns:
            id_col = "node"
        else:
            id_col = None

        if id_col is not None and node_score_source in cdf.columns:
            cdf[id_col] = cdf[id_col].astype(int)
            for _, r in cdf.iterrows():
                roi = int(r[id_col])
                if roi in node_scores and node_scores[roi] == 0:
                    node_scores[roi] = float(r[node_score_source])

    # lesion 保证可见
    max_score = max(max(node_scores.values()), 1e-12)
    node_scores[lesion_idx] = max(node_scores.get(lesion_idx, 0.0), 0.85 * max_score)

    # 如果限制 top_score_nodes，需要保证 lesion 和 direct edges 的重要节点优先保留
    if top_score_nodes is not None and int(top_score_nodes) > 0:
        keep_neighbors = sorted(
            [n for n in all_nodes if n != lesion_idx],
            key=lambda n: node_scores.get(int(n), 0.0),
            reverse=True,
        )[:int(top_score_nodes)]

        keep_nodes = set([lesion_idx] + keep_neighbors)

        edge_df = edge_df[
            edge_df["src"].isin(keep_nodes) &
            edge_df["dst"].isin(keep_nodes)
        ].copy()

        left_nodes = [n for n in left_nodes if n in keep_nodes]
        right_nodes = [n for n in right_nodes if n in keep_nodes]
        all_nodes = [lesion_idx] + left_nodes + right_nodes

    # --------------------------------------------------
    # 4. 标签
    # --------------------------------------------------
    labels = {}
    for n in all_nodes:
        n = int(n)
        if roi_names is not None and 0 <= n < len(roi_names):
            labels[n] = str(roi_names[n])
        else:
            labels[n] = str(n)

    # 只标 lesion + top response 节点
    label_nodes = [lesion_idx]
    label_nodes += sorted(
        [n for n in all_nodes if n != lesion_idx],
        key=lambda n: node_scores.get(int(n), 0.0),
        reverse=True,
    )[:int(top_label_k)]
    label_nodes = list(dict.fromkeys([int(x) for x in label_nodes]))

    # --------------------------------------------------
    # 5. 布局：左、中心、右三层
    # --------------------------------------------------
    pos = {}

    x_left = -3.8
    x_lesion = 0.0
    x_right = 3.8

    left_sorted = sorted(
        left_nodes,
        key=lambda n: node_scores.get(int(n), 0.0),
        reverse=True,
    )
    right_sorted = sorted(
        right_nodes,
        key=lambda n: node_scores.get(int(n), 0.0),
        reverse=True,
    )

    left_ys = _spread_layer_y(len(left_sorted), span=5.2, avoid_zero=True)
    right_ys = _spread_layer_y(len(right_sorted), span=5.2, avoid_zero=True)

    for n, y in zip(left_sorted, left_ys):
        pos[int(n)] = np.array([x_left, y], dtype=float)

    pos[lesion_idx] = np.array([x_lesion, 0.0], dtype=float)

    for n, y in zip(right_sorted, right_ys):
        pos[int(n)] = np.array([x_right, y], dtype=float)

    # --------------------------------------------------
    # 6. normalizer
    # --------------------------------------------------
    node_vals = np.array([node_scores.get(int(n), 0.0) for n in all_nodes], dtype=float)
    node_vmin = float(np.nanmin(node_vals))
    node_vmax = float(np.nanmax(node_vals))
    if np.isclose(node_vmin, node_vmax):
        node_vmax = node_vmin + 1e-6

    node_norm = Normalize(vmin=node_vmin, vmax=node_vmax)
    node_cmap = plt.get_cmap("Oranges")

    edge_vals = edge_df[value_col].astype(float).values
    edge_vmax = float(np.nanmax(np.abs(edge_vals))) if len(edge_vals) > 0 else 1.0
    if edge_vmax <= 1e-12:
        edge_vmax = 1.0

    edge_norm = TwoSlopeNorm(vmin=-edge_vmax, vcenter=0.0, vmax=edge_vmax)
    edge_cmap = plt.get_cmap("RdBu_r")

    max_abs_edge = float(edge_df["abs_val"].max()) if len(edge_df) > 0 else 1.0
    max_abs_edge = max(max_abs_edge, 1e-12)

    max_node_score = max(float(np.nanmax(node_vals)), 1e-12)

    # --------------------------------------------------
    # 7. panel drawing helpers
    # --------------------------------------------------
    def _role_marker_style(n):
        n = int(n)
        role = roles.get(n, "other")

        if n == lesion_idx:
            return "*", "#2b6cb0", 2.2

        if role == "parent":
            return "^", "#2b6cb0", 1.6

        if role == "child":
            return "s", "#2f855a", 1.6

        if role == "parent_child":
            return "D", "#805ad5", 1.7

        return "o", "gray", 1.2

    def _role_facecolor(n):
        n = int(n)
        role = roles.get(n, "other")

        if n == lesion_idx:
            return "#E6F2FF"
        if role == "parent":
            return "#DDEBFF"
        if role == "child":
            return "#DFF3E3"
        if role == "parent_child":
            return "#EFE3FF"
        return "#EAEAEA"

    def _draw_edges(ax, color_mode="neutral"):
        # 用于判断双向边，给双向边不同弧度
        edge_pairs = set((int(r["src"]), int(r["dst"])) for _, r in edge_df.iterrows())

        for i, (_, e) in enumerate(edge_df.iterrows()):
            s = int(e["src"])
            d = int(e["dst"])

            if s not in pos or d not in pos:
                continue

            p1 = np.asarray(pos[s], dtype=float)
            p2 = np.asarray(pos[d], dtype=float)

            val = float(e[value_col])
            abs_val = float(e["abs_val"])

            if color_mode == "causal":
                color = edge_cmap(edge_norm(val))
                alpha = 0.35 + 0.45 * abs_val / max_abs_edge
            else:
                color = "#8F8F8F"
                alpha = edge_alpha

            width = 0.9 + edge_width_scale * abs_val / max_abs_edge

            # 双向边加大弧度，避免重叠
            has_reverse = (d, s) in edge_pairs
            if has_reverse:
                if s == lesion_idx:
                    rad = 0.22
                else:
                    rad = -0.22
            else:
                # 非双向边根据 y 方向轻微弯曲，减少挡住节点
                rad = 0.12 if p2[1] >= p1[1] else -0.12

            patch = FancyArrowPatch(
                posA=p1,
                posB=p2,
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}",
                linewidth=width,
                color=color,
                alpha=alpha,
                mutation_scale=15,
                shrinkA=26,
                shrinkB=26,
                zorder=1,
            )
            ax.add_patch(patch)

    def _draw_nodes(ax, node_mode="response"):
        for n in all_nodes:
            n = int(n)
            if n not in pos:
                continue

            marker, edgecolor, lw = _role_marker_style(n)

            if node_mode == "response":
                facecolor = node_cmap(node_norm(node_scores.get(n, 0.0)))
            else:
                facecolor = _role_facecolor(n)

            if n == lesion_idx:
                size = lesion_node_size
            else:
                size = node_size_min + node_size_scale * node_scores.get(n, 0.0) / max_node_score

            ax.scatter(
                [pos[n][0]],
                [pos[n][1]],
                s=size,
                marker=marker,
                c=[facecolor],
                edgecolors=edgecolor,
                linewidths=lw,
                zorder=4,
                alpha=0.97,
            )

    def _setup_panel(ax, panel_title):
        y_header = 3.35

        if len(left_nodes) > 0:
            ax.text(
                x_left,
                y_header,
                "Parents",
                ha="center",
                va="center",
                fontsize=10,
                color="dimgray",
                fontweight="bold",
            )

        ax.text(
            x_lesion,
            y_header,
            "Lesion",
            ha="center",
            va="center",
            fontsize=10,
            color="dimgray",
            fontweight="bold",
        )

        if len(right_nodes) > 0:
            ax.text(
                x_right,
                y_header,
                "Children / parent-child",
                ha="center",
                va="center",
                fontsize=10,
                color="dimgray",
                fontweight="bold",
            )

        _draw_direct_core_callout_labels(
            ax=ax,
            pos=pos,
            labels=labels,
            lesion_idx=lesion_idx,
            label_nodes=label_nodes,
            fontsize=8,
            x_offset=0.78,
            y_offset=0.78,
        )

        ax.set_title(panel_title, fontsize=12, pad=12)
        ax.axis("off")

        xs = np.array([p[0] for p in pos.values()], dtype=float)
        ys = np.array([p[1] for p in pos.values()], dtype=float)

        ax.set_xlim(xs.min() - 1.60, xs.max() + 1.75)
        ax.set_ylim(ys.min() - 1.25, max(ys.max() + 1.55, 3.75))

    # --------------------------------------------------
    # 8. 一行两列画图
    # --------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.8),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    ax_resp, ax_edge = axes

    # 左图：node response_abs
    _draw_edges(ax_resp, color_mode="neutral")
    _draw_nodes(ax_resp, node_mode="response")
    _setup_panel(ax_resp, f"Node-level {node_score_source}")

    sm_node = cm.ScalarMappable(norm=node_norm, cmap=node_cmap)
    sm_node.set_array([])
    cbar_node = fig.colorbar(sm_node, ax=ax_resp, fraction=0.035, pad=0.025)
    cbar_node.set_label(node_score_source, fontsize=10)

    # 右图：edge causal value
    _draw_edges(ax_edge, color_mode="causal")
    _draw_nodes(ax_edge, node_mode="role")
    _setup_panel(ax_edge, "Edge-level LPCMCI causal value")

    sm_edge = cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
    sm_edge.set_array([])
    cbar_edge = fig.colorbar(sm_edge, ax=ax_edge, fraction=0.035, pad=0.025)
    cbar_edge.set_label("LPCMCI causal value", fontsize=10)

    # 图例：图片下面，两列两行
    handles = [
        Line2D(
            [0], [0],
            marker="*",
            color="w",
            label="lesion",
            markerfacecolor="#E6F2FF",
            markeredgecolor="#2b6cb0",
            markersize=13,
            markeredgewidth=1.8,
        ),
        Line2D(
            [0], [0],
            marker="^",
            color="w",
            label="parent",
            markerfacecolor="#DDEBFF",
            markeredgecolor="#2b6cb0",
            markersize=9,
            markeredgewidth=1.4,
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="w",
            label="child",
            markerfacecolor="#DFF3E3",
            markeredgecolor="#2f855a",
            markersize=9,
            markeredgewidth=1.4,
        ),
        Line2D(
            [0], [0],
            marker="D",
            color="w",
            label="parent/child",
            markerfacecolor="#EFE3FF",
            markeredgecolor="#805ad5",
            markersize=9,
            markeredgewidth=1.4,
        ),
    ]

    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(title, fontsize=13, y=0.99)

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    info = {
        "edge_df_plotted": edge_df.reset_index(drop=True),
        "node_scores": node_scores,
        "roles": roles,
        "positions": pos,
        "label_nodes": label_nodes,
        "left_nodes": left_nodes,
        "right_nodes": right_nodes,
        "n_edges": len(edge_df),
        "n_nodes": len(all_nodes),
        "value_col": value_col,
        "node_score_source": node_score_source,
    }

    return fig, axes, info


def plot_lpcmci_lesion_core_graph_split(
    lesion_core_edges: pd.DataFrame,
    lesion_idx: int,
    candidate_df: pd.DataFrame = None,
    stage1_res=None,
    roi_names=None,
    node_score_source: str = "response_abs",
    max_edges: int = 30,
    top_score_nodes: int = 12,
    top_label_k: int = 8,
    figsize=(10.8, 6.8),
    cmap_nodes="Oranges",
    cmap_edges="RdBu_r",
    title_prefix="Stage 2 direct lesion-connected LPCMCI core graph",
    edge_alpha: float = 0.45,
    edge_width_scale: float = 1.8,
    node_size_min: float = 240.0,
    node_size_scale: float = 480.0,
    lesion_node_size: float = 1050.0,
    save_prefix=None,
):
    """
    Split version:
    1) response_abs node-level graph
    2) LPCMCI causal value edge-level graph

    Compared with the previous version:
    - labels are callout boxes with leader lines;
    - layout is wider and less crowded;
    - legend is placed below the graph;
    - response_abs and LPCMCI causal value are separated into two figures.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D

    lesion_idx = int(lesion_idx)

    if "src" not in lesion_core_edges.columns or "dst" not in lesion_core_edges.columns:
        raise KeyError("lesion_core_edges must contain columns: src, dst")

    edge_df = lesion_core_edges.copy()
    edge_df["src"] = edge_df["src"].astype(int)
    edge_df["dst"] = edge_df["dst"].astype(int)

    # --------------------------------------------------
    # 1. Only direct lesion-connected edges
    # --------------------------------------------------
    edge_df = edge_df[
        (edge_df["src"] == lesion_idx) |
        (edge_df["dst"] == lesion_idx)
    ].copy()

    if len(edge_df) == 0:
        raise ValueError("No direct lesion-connected edges found.")

    # edge value
    if "val" in edge_df.columns:
        value_col = "val"
    elif "causal_val" in edge_df.columns:
        value_col = "causal_val"
    else:
        edge_df["val"] = 1.0
        value_col = "val"

    if "abs_val" not in edge_df.columns:
        edge_df["abs_val"] = np.abs(edge_df[value_col].astype(float))

    edge_df = edge_df.sort_values("abs_val", ascending=False).head(int(max_edges)).copy()

    # --------------------------------------------------
    # 2. Define roles
    # --------------------------------------------------
    parents = sorted(set(edge_df.loc[edge_df["dst"] == lesion_idx, "src"].astype(int)))
    children = sorted(set(edge_df.loc[edge_df["src"] == lesion_idx, "dst"].astype(int)))

    parent_set = set(parents)
    child_set = set(children)
    parent_child = sorted(list(parent_set & child_set))
    parents_only = sorted(list(parent_set - child_set))
    children_only = sorted(list(child_set - parent_set))

    all_neighbor_nodes = sorted(list(parent_set | child_set))
    all_nodes = [lesion_idx] + all_neighbor_nodes

    roles = {lesion_idx: "lesion"}
    for n in parents_only:
        roles[n] = "parent"
    for n in children_only:
        roles[n] = "child"
    for n in parent_child:
        roles[n] = "parent_child"

    # --------------------------------------------------
    # 3. Node scores
    # --------------------------------------------------
    node_scores = {n: 0.0 for n in all_nodes}

    # stage1 response first
    if stage1_res is not None:
        if isinstance(stage1_res, dict) and "lesion_response_df" in stage1_res:
            s1_df = stage1_res["lesion_response_df"].copy()
        elif isinstance(stage1_res, pd.DataFrame):
            s1_df = stage1_res.copy()
        else:
            s1_df = None

        if s1_df is not None:
            if "node" in s1_df.columns and "roi" not in s1_df.columns:
                s1_df = s1_df.rename(columns={"node": "roi"})
            if "roi" in s1_df.columns and node_score_source in s1_df.columns:
                for _, r in s1_df.iterrows():
                    node_scores[int(r["roi"])] = float(r[node_score_source])

    # candidate fallback
    if candidate_df is not None:
        cdf = candidate_df.copy()
        if "roi" in cdf.columns:
            id_col = "roi"
        elif "node" in cdf.columns:
            id_col = "node"
        else:
            id_col = None

        if id_col is not None and node_score_source in cdf.columns:
            for _, r in cdf.iterrows():
                node_scores[int(r[id_col])] = float(r[node_score_source])

    # lesion score
    node_scores[lesion_idx] = max(node_scores.get(lesion_idx, 0.0), max(node_scores.values()) if len(node_scores) else 1.0)

    # optional keep top scored nodes
    if top_score_nodes is not None:
        keep_neighbors = sorted(
            all_neighbor_nodes,
            key=lambda n: node_scores.get(int(n), 0.0),
            reverse=True,
        )[:int(top_score_nodes)]

        keep_nodes = set([lesion_idx] + keep_neighbors)
        edge_df = edge_df[
            edge_df["src"].isin(keep_nodes) &
            edge_df["dst"].isin(keep_nodes)
        ].copy()
        all_nodes = sorted(list(keep_nodes))

        parents_only = [n for n in parents_only if n in keep_nodes]
        children_only = [n for n in children_only if n in keep_nodes]
        parent_child = [n for n in parent_child if n in keep_nodes]

    # --------------------------------------------------
    # 4. Layout: more spread out
    # --------------------------------------------------
    pos = {}

    def place_vertical(nodes, x, y_span=4.4):
        nodes = list(nodes)
        if len(nodes) == 0:
            return
        if len(nodes) == 1:
            ys = [0.0]
        else:
            ys = np.linspace(y_span / 2, -y_span / 2, len(nodes))
        for n, y in zip(nodes, ys):
            pos[int(n)] = np.array([float(x), float(y)])

    place_vertical(parents_only, x=-3.2, y_span=4.8)
    pos[lesion_idx] = np.array([0.0, 0.0])
    place_vertical(children_only, x=3.2, y_span=4.0)
    place_vertical(parent_child, x=4.45, y_span=4.4)

    # if policy/direct unknown nodes exist
    for n in all_nodes:
        if int(n) not in pos:
            pos[int(n)] = np.array([3.9, -2.8])

    # --------------------------------------------------
    # 5. Labels
    # --------------------------------------------------
    labels = {}
    for n in all_nodes:
        n = int(n)
        if roi_names is not None and n < len(roi_names):
            labels[n] = str(roi_names[n])
        else:
            labels[n] = str(n)

    # label nodes: lesion + top score nodes
    label_nodes = [lesion_idx]
    score_order = sorted(
        [n for n in all_nodes if n != lesion_idx],
        key=lambda n: node_scores.get(int(n), 0.0),
        reverse=True,
    )
    label_nodes += score_order[:int(top_label_k)]

    # --------------------------------------------------
    # 6. Common draw function
    # --------------------------------------------------
    def _draw_base_graph(
        fig_title,
        node_mode="response",
        edge_mode="neutral",
        save_path=None,
    ):
        fig, ax = plt.subplots(figsize=figsize)

        # role headers
        ax.text(-3.2, 2.95, "Parents", ha="center", va="center",
                fontsize=10, color="dimgray", fontweight="bold")
        ax.text(0.0, 2.95, "Lesion", ha="center", va="center",
                fontsize=10, color="dimgray", fontweight="bold")
        ax.text(3.2, 2.95, "Children", ha="center", va="center",
                fontsize=10, color="dimgray", fontweight="bold")

        # edge color config
        if edge_mode == "causal":
            edge_vals = edge_df[value_col].astype(float).values
            vmax = float(np.max(np.abs(edge_vals))) if len(edge_vals) else 1.0
            if vmax <= 1e-12:
                vmax = 1.0
            edge_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            edge_cmap = plt.get_cmap(cmap_edges)
        else:
            edge_norm = None
            edge_cmap = None

        # draw edges first
        for i, (_, e) in enumerate(edge_df.iterrows()):
            s = int(e["src"])
            d = int(e["dst"])
            if s not in pos or d not in pos:
                continue

            p1 = np.asarray(pos[s], dtype=float)
            p2 = np.asarray(pos[d], dtype=float)

            if edge_mode == "causal":
                color = edge_cmap(edge_norm(float(e[value_col])))
            else:
                color = "#8f8f8f"

            width = 1.0 + edge_width_scale * float(e["abs_val"]) / (float(edge_df["abs_val"].max()) + 1e-12)

            rad = 0.10 if i % 2 == 0 else -0.10

            patch = FancyArrowPatch(
                posA=p1,
                posB=p2,
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}",
                linewidth=width,
                color=color,
                alpha=edge_alpha,
                mutation_scale=14,
                shrinkA=18,
                shrinkB=18,
                zorder=1,
            )
            ax.add_patch(patch)

        # node color config
        node_vals = np.array([node_scores.get(int(n), 0.0) for n in all_nodes], dtype=float)

        if node_mode == "response":
            vmin = float(np.nanmin(node_vals))
            vmax = float(np.nanmax(node_vals))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6
            node_norm = Normalize(vmin=vmin, vmax=vmax)
            node_cmap = plt.get_cmap(cmap_nodes)
            node_colors = [node_cmap(node_norm(v)) for v in node_vals]
        else:
            node_norm = None
            node_cmap = None
            node_colors = []
            for n in all_nodes:
                role = roles.get(int(n), "other")
                if role == "lesion":
                    node_colors.append("#E6F2FF")
                elif role == "parent":
                    node_colors.append("#DDEBFF")
                elif role == "child":
                    node_colors.append("#DFF3E3")
                elif role == "parent_child":
                    node_colors.append("#EFE3FF")
                else:
                    node_colors.append("#EAEAEA")

        # node sizes
        max_score = max(float(np.nanmax(node_vals)), 1e-12)
        for idx, n in enumerate(all_nodes):
            n = int(n)
            role = roles.get(n, "other")

            if n == lesion_idx:
                marker = "*"
                size = lesion_node_size
                edgecolor = "#2b6cb0"
                lw = 2.2
            elif role == "parent":
                marker = "^"
                size = node_size_min + node_size_scale * node_scores.get(n, 0.0) / max_score
                edgecolor = "#2b6cb0"
                lw = 1.6
            elif role == "child":
                marker = "s"
                size = node_size_min + node_size_scale * node_scores.get(n, 0.0) / max_score
                edgecolor = "#2f855a"
                lw = 1.6
            elif role == "parent_child":
                marker = "D"
                size = node_size_min + node_size_scale * node_scores.get(n, 0.0) / max_score
                edgecolor = "#805ad5"
                lw = 1.7
            else:
                marker = "o"
                size = node_size_min
                edgecolor = "gray"
                lw = 1.2

            ax.scatter(
                [pos[n][0]],
                [pos[n][1]],
                s=size,
                marker=marker,
                c=[node_colors[idx]],
                edgecolors=edgecolor,
                linewidths=lw,
                zorder=3,
                alpha=0.96,
            )

        # callout labels
        _draw_network_callout_labels(
            ax=ax,
            pos=pos,
            labels=labels,
            roles=roles,
            lesion_idx=lesion_idx,
            label_nodes=label_nodes,
            fontsize=8,
        )

        # colorbar
        if node_mode == "response":
            sm = cm.ScalarMappable(norm=node_norm, cmap=node_cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.025)
            cbar.set_label(node_score_source, fontsize=10)

        if edge_mode == "causal":
            sm = cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.025)
            cbar.set_label("LPCMCI causal value", fontsize=10)

        # legend below, four items in two rows
        handles = [
            Line2D([0], [0], marker="*", color="w", label="lesion",
                   markerfacecolor="#E6F2FF", markeredgecolor="#2b6cb0",
                   markersize=13, markeredgewidth=1.8),
            Line2D([0], [0], marker="^", color="w", label="parent",
                   markerfacecolor="#DDEBFF", markeredgecolor="#2b6cb0",
                   markersize=9, markeredgewidth=1.4),
            Line2D([0], [0], marker="s", color="w", label="child",
                   markerfacecolor="#DFF3E3", markeredgecolor="#2f855a",
                   markersize=9, markeredgewidth=1.4),
            Line2D([0], [0], marker="D", color="w", label="parent/child",
                   markerfacecolor="#EFE3FF", markeredgecolor="#805ad5",
                   markersize=9, markeredgewidth=1.4),
        ]

        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=9,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=2,
        )

        ax.set_title(fig_title, fontsize=13, pad=14)
        ax.axis("off")

        xs = np.array([p[0] for p in pos.values()], dtype=float)
        ys = np.array([p[1] for p in pos.values()], dtype=float)

        ax.set_xlim(xs.min() - 1.25, xs.max() + 1.35)
        ax.set_ylim(ys.min() - 1.10, ys.max() + 1.35)

        plt.tight_layout()

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir != "":
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            if save_path.lower().endswith(".png"):
                fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

        return fig, ax

    # --------------------------------------------------
    # 7. Draw two separated figures
    # --------------------------------------------------
    response_save = None
    causal_save = None
    if save_prefix is not None:
        response_save = save_prefix + "_response_abs.png"
        causal_save = save_prefix + "_lpcmci_causal_value.png"

    fig_response, ax_response = _draw_base_graph(
        fig_title=f"{title_prefix}: node-level {node_score_source}",
        node_mode="response",
        edge_mode="neutral",
        save_path=response_save,
    )

    fig_causal, ax_causal = _draw_base_graph(
        fig_title=f"{title_prefix}: edge-level LPCMCI causal value",
        node_mode="role",
        edge_mode="causal",
        save_path=causal_save,
    )

    plt.show()

    graph_info = {
        "n_nodes": len(all_nodes),
        "n_edges_plotted": len(edge_df),
        "roles": roles,
        "node_scores": node_scores,
        "edge_df_plotted": edge_df.reset_index(drop=True),
        "plotted_nodes": sorted(list(all_nodes)),
        "parent_nodes": parents_only,
        "child_nodes": children_only,
        "parent_child_nodes": parent_child,
        "positions": pos,
        "node_score_source": node_score_source,
        "value_col": value_col,
        "abs_col": "abs_val",
    }

    return fig_response, ax_response, fig_causal, ax_causal, graph_info


def _get_roi_label(node, roi_names=None):
    node = int(node)
    if roi_names is not None and node < len(roi_names):
        return str(roi_names[node])
    return str(node)


def order_lpcmci_nodes_by_role(
    lesion_core_edges: pd.DataFrame,
    lesion_idx: int,
    candidate_df: pd.DataFrame = None,
    stage1_res=None,
    node_score_source: str = "candidate_score",
):
    """
    Order nodes for LPCMCI adjacency heatmap.

    Order:
        lesion
        parents
        children
        downstream
        bridge
        spouse
        second_order / others

    Within each group, nodes are sorted by candidate_score or response_abs.
    """
    lesion_idx = int(lesion_idx)

    if lesion_core_edges is None or len(lesion_core_edges) == 0:
        return [lesion_idx], {lesion_idx: "lesion"}, {"lesion": [lesion_idx]}

    edge_df = lesion_core_edges.copy()
    edge_df["src"] = edge_df["src"].astype(int)
    edge_df["dst"] = edge_df["dst"].astype(int)

    # 如果你前面已经在 stage2_visualization.py 里放了 classify_lpcmci_core_nodes，
    # 这里会直接复用它。
    roles = classify_lpcmci_core_nodes(
        lesion_core_edges=edge_df,
        lesion_idx=lesion_idx,
    )

    nodes = sorted(list(roles.keys()))

    candidate_score_map = _get_candidate_score_map(candidate_df)
    response_map = _get_stage1_response_map(stage1_res)

    def node_score(n):
        n = int(n)
        if node_score_source == "response_abs":
            return float(response_map.get(n, candidate_score_map.get(n, 0.0)))
        else:
            return float(candidate_score_map.get(n, response_map.get(n, 0.0)))

    role_order = [
        "lesion",
        "parent",
        "parent/child",
        "child",
        "downstream",
        "bridge",
        "spouse",
        "second_order",
    ]

    grouped_nodes = {}
    node_order = []

    for role in role_order:
        group = [n for n in nodes if roles.get(n, "second_order") == role]
        group = sorted(group, key=lambda x: node_score(x), reverse=True)

        if len(group) > 0:
            grouped_nodes[role] = group
            node_order.extend(group)

    # fallback: 避免有未归类节点
    remaining = [n for n in nodes if n not in set(node_order)]
    remaining = sorted(remaining, key=lambda x: node_score(x), reverse=True)

    if len(remaining) > 0:
        grouped_nodes["others"] = remaining
        node_order.extend(remaining)

    # 保证 lesion 在第一位
    if lesion_idx in node_order:
        node_order = [lesion_idx] + [n for n in node_order if n != lesion_idx]
    else:
        node_order = [lesion_idx] + node_order

    grouped_nodes["lesion"] = [lesion_idx]

    return node_order, roles, grouped_nodes


def build_lpcmci_adjacency_matrix(
    lesion_core_edges: pd.DataFrame,
    node_order,
    value_col: str = "val",
    fill_value: float = 0.0,
):
    """
    Build signed LPCMCI adjacency matrix.

    Matrix convention:
        row    = source ROI
        column = target ROI

    M[src, dst] = causal value
    """
    if lesion_core_edges is None or len(lesion_core_edges) == 0:
        raise ValueError("lesion_core_edges is empty.")

    edge_df = lesion_core_edges.copy()
    edge_df["src"] = edge_df["src"].astype(int)
    edge_df["dst"] = edge_df["dst"].astype(int)

    if value_col not in edge_df.columns:
        if "val" in edge_df.columns:
            value_col = "val"
        elif "abs_val" in edge_df.columns:
            value_col = "abs_val"
        else:
            edge_df["val"] = 1.0
            value_col = "val"

    node_order = [int(x) for x in node_order]
    idx = {node: i for i, node in enumerate(node_order)}

    mat = np.full(
        (len(node_order), len(node_order)),
        fill_value,
        dtype=float,
    )

    for _, row in edge_df.iterrows():
        s = int(row["src"])
        d = int(row["dst"])

        if s not in idx or d not in idx:
            continue

        mat[idx[s], idx[d]] = float(row[value_col])

    adj_df = pd.DataFrame(
        mat,
        index=node_order,
        columns=node_order,
    )

    return adj_df


def plot_lpcmci_adjacency_heatmap(
    lesion_core_edges: pd.DataFrame,
    lesion_idx: int,
    candidate_df: pd.DataFrame = None,
    stage1_res=None,
    roi_names=None,
    node_score_source: str = "candidate_score",
    value_col: str = "val",
    max_nodes: int = 45,
    figsize=(10.5, 8.4),
    cmap="RdBu_r",
    title="Stage 2 LPCMCI core-edge adjacency heatmap",
    show_role_boundaries: bool = True,
    role_order=None,
    grid_linewidth: float = 0.04,
    save_path=None,
):
    """
    Paper-style LPCMCI adjacency heatmap.

    Rows:
        source ROI

    Columns:
        target ROI

    Color:
        signed LPCMCI causal value

    Key improvements:
        1. sort nodes by role: lesion -> parent -> child -> spouse -> second_order
        2. put role labels in the left blank margin
        3. reduce grid lines
        4. reduce x/y tick label fontsize
        5. show core edges only, i.e., lesion_core_edges should be final pruned core edges
    """
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    lesion_idx = int(lesion_idx)

    if lesion_core_edges is None or len(lesion_core_edges) == 0:
        raise ValueError(
            "lesion_core_edges is empty. "
            "Please pass final pruned core edges, not local LPCMCI edges."
        )

    edge_df = lesion_core_edges.copy()
    edge_df["src"] = edge_df["src"].astype(int)
    edge_df["dst"] = edge_df["dst"].astype(int)

    if "abs_val" not in edge_df.columns:
        if "val" in edge_df.columns:
            edge_df["abs_val"] = np.abs(edge_df["val"].astype(float))
        elif "weight" in edge_df.columns:
            edge_df["abs_val"] = np.abs(edge_df["weight"].astype(float))
        else:
            edge_df["abs_val"] = 1.0

    if "val" not in edge_df.columns:
        if "weight" in edge_df.columns:
            edge_df["val"] = edge_df["weight"].astype(float)
        else:
            edge_df["val"] = edge_df["abs_val"].astype(float)

    edge_df["abs_val"] = edge_df["abs_val"].astype(float)
    edge_df["val"] = edge_df["val"].astype(float)

    # --------------------------------------------------
    # 1. 原始 role 识别
    # --------------------------------------------------
    node_order_raw, roles, grouped_nodes = order_lpcmci_nodes_by_role(
        lesion_core_edges=edge_df,
        lesion_idx=lesion_idx,
        candidate_df=candidate_df,
        stage1_res=stage1_res,
        node_score_source=node_score_source,
    )

    # --------------------------------------------------
    # 2. 如果节点太多，只保留 lesion + 高强度相关节点
    # --------------------------------------------------
    if max_nodes is not None and len(node_order_raw) > int(max_nodes):
        candidate_score_map = _get_candidate_score_map(candidate_df)
        response_map = _get_stage1_response_map(stage1_res)

        incident_strength = {int(n): 0.0 for n in node_order_raw}

        for _, row in edge_df.iterrows():
            s = int(row["src"])
            d = int(row["dst"])
            w = float(row["abs_val"])

            incident_strength[s] = incident_strength.get(s, 0.0) + w
            incident_strength[d] = incident_strength.get(d, 0.0) + w

        def rank_score(n):
            n = int(n)
            base = incident_strength.get(n, 0.0)
            cand = float(candidate_score_map.get(n, 0.0))
            resp = float(response_map.get(n, 0.0))
            return base + 0.5 * cand + 0.5 * resp

        keep_nodes = [lesion_idx]
        rest = [int(n) for n in node_order_raw if int(n) != lesion_idx]
        rest = sorted(rest, key=rank_score, reverse=True)
        keep_nodes.extend(rest[: int(max_nodes) - 1])

        keep_set = set(map(int, keep_nodes))

        edge_df = edge_df[
            edge_df["src"].astype(int).isin(keep_set)
            & edge_df["dst"].astype(int).isin(keep_set)
        ].copy()

        node_order_raw = [int(n) for n in node_order_raw if int(n) in keep_set]

    # --------------------------------------------------
    # 3. 按指定 role 顺序重新排序
    # --------------------------------------------------
    if role_order is None:
        role_order = [
            "lesion",
            "parent",
            "parent/child",
            "child",
            "spouse",
            "bridge",
            "downstream",
            "second_order",
            "candidate",
            "other",
        ]

    # lesion 角色强制指定
    roles[int(lesion_idx)] = "lesion"

    # 根据节点强度在每个 role 内部排序
    incident_strength = {int(n): 0.0 for n in node_order_raw}

    for _, row in edge_df.iterrows():
        s = int(row["src"])
        d = int(row["dst"])
        w = float(row["abs_val"])
        incident_strength[s] = incident_strength.get(s, 0.0) + w
        incident_strength[d] = incident_strength.get(d, 0.0) + w

    ordered_nodes = []

    for role in role_order:
        role_nodes = [
            int(n)
            for n in node_order_raw
            if roles.get(int(n), "other") == role
        ]

        role_nodes = sorted(
            role_nodes,
            key=lambda n: incident_strength.get(int(n), 0.0),
            reverse=True
        )

        ordered_nodes.extend(role_nodes)

    # 加上没有被 role_order 覆盖的节点
    remaining_nodes = [
        int(n)
        for n in node_order_raw
        if int(n) not in set(ordered_nodes)
    ]

    remaining_nodes = sorted(
        remaining_nodes,
        key=lambda n: incident_strength.get(int(n), 0.0),
        reverse=True
    )

    node_order = ordered_nodes + remaining_nodes
    node_order = list(dict.fromkeys(node_order))

    # 只保留 node_order 内部的 core edges
    keep_set = set(node_order)

    edge_df = edge_df[
        edge_df["src"].astype(int).isin(keep_set)
        & edge_df["dst"].astype(int).isin(keep_set)
    ].copy()

    # --------------------------------------------------
    # 4. 构造邻接矩阵
    # --------------------------------------------------
    adj_df = build_lpcmci_adjacency_matrix(
        lesion_core_edges=edge_df,
        node_order=node_order,
        value_col=value_col,
        fill_value=0.0,
    )

    mat = adj_df.values.astype(float)

    vmax = float(np.max(np.abs(mat))) if mat.size > 0 else 1.0
    if vmax <= 1e-12:
        vmax = 1.0

    # --------------------------------------------------
    # 5. ROI 标签和字体大小
    # --------------------------------------------------
    labels = [_get_roi_label(n, roi_names=roi_names) for n in node_order]

    n_nodes = len(labels)

    if n_nodes <= 25:
        tick_fontsize = 7
    elif n_nodes <= 45:
        tick_fontsize = 5.5
    else:
        tick_fontsize = 4.8

    # --------------------------------------------------
    # 6. 画 heatmap
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # 给左侧 role label 留白
    fig.subplots_adjust(
        left=0.24,
        right=0.92,
        bottom=0.24,
        top=0.90
    )

    sns.heatmap(
        adj_df,
        ax=ax,
        cmap=cmap,
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        linewidths=grid_linewidth,
        linecolor="#e5e5e5",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={
            "label": "LPCMCI causal value",
            "shrink": 0.78,
            "pad": 0.02,
        },
    )

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Target ROI", fontsize=10)
    ax.set_ylabel("Source ROI", fontsize=10, labelpad=34)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=60,
        ha="right",
        fontsize=tick_fontsize,
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=tick_fontsize,
    )

    ax.tick_params(axis="both", length=0)

    # colorbar 字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("LPCMCI causal value", fontsize=10)

    # --------------------------------------------------
    # 7. 标出 lesion 行/列
    # --------------------------------------------------
    if lesion_idx in node_order:
        lesion_pos = node_order.index(lesion_idx)

        ax.axhline(lesion_pos, color="#1f4eaa", linewidth=1.4)
        ax.axhline(lesion_pos + 1, color="#1f4eaa", linewidth=1.4)
        ax.axvline(lesion_pos, color="#1f4eaa", linewidth=1.4)
        ax.axvline(lesion_pos + 1, color="#1f4eaa", linewidth=1.4)

    # --------------------------------------------------
    # 8. role 边界线 + 左侧 role label
    # --------------------------------------------------
    role_segments = []

    start = 0
    ordered_roles = [roles.get(int(n), "other") for n in node_order]

    for i in range(1, len(ordered_roles) + 1):
        if i == len(ordered_roles) or ordered_roles[i] != ordered_roles[start]:
            role = ordered_roles[start]
            end = i
            role_segments.append((role, start, end))
            start = i

    if show_role_boundaries:
        # role 分界线
        for _, start, end in role_segments:
            if start > 0:
                ax.axhline(start, color="black", linewidth=0.65, alpha=0.75)
                ax.axvline(start, color="black", linewidth=0.65, alpha=0.75)

        # 最外框
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.8)
        ax.axhline(len(node_order), color="black", linewidth=0.8, alpha=0.8)
        ax.axvline(len(node_order), color="black", linewidth=0.8, alpha=0.8)

        # role label 放到左侧空白，不和 ROI label 混在一起
        for role, start, end in role_segments:
            mid = (start + end) / 2

            ax.text(
                -0.18,
                mid,
                role,
                ha="right",
                va="center",
                fontsize=8,
                color="black",
                fontweight="bold" if role == "lesion" else "normal",
                transform=ax.get_yaxis_transform(),
                clip_on=False,
            )

    # --------------------------------------------------
    # 9. 保存
    # --------------------------------------------------
    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    heatmap_info = {
        "node_order": node_order,
        "roles": roles,
        "role_segments": role_segments,
        "adjacency_df": adj_df,
        "edge_df_plotted": edge_df,
        "n_nodes": len(node_order),
        "n_edges": len(edge_df),
    }

    return fig, ax, heatmap_info

def plot_stage2_sparsification_summary(
    summary: dict,
    figsize=(8.4, 4.6),
    title="Stage 2 sparsification process",
    save_path=None,
    ax=None,
):
    """
    Stage 2 sparsification summary panel.

    Show how Stage 2 compresses the graph:
        screened nodes
        local LPCMCI edges
        lesion-subgraph edges
        core edges after FC pruning

    Required keys in summary:
        n_screen_nodes
        n_local_edges
        n_lesion_sub_edges
        n_core_edges
    """
    required_keys = [
        "n_screen_nodes",
        "n_local_edges",
        "n_lesion_sub_edges",
        "n_core_edges",
    ]
    for k in required_keys:
        if k not in summary:
            raise KeyError(f"summary must contain key: {k}")

    labels = [
        "Screened\nnodes",
        "Local LPCMCI\nedges",
        "Lesion-subgraph\nedges",
        "Core edges after\nFC pruning",
    ]
    values = [
        int(summary["n_screen_nodes"]),
        int(summary["n_local_edges"]),
        int(summary["n_lesion_sub_edges"]),
        int(summary["n_core_edges"]),
    ]

    # 计算稀疏化比例（只对 edge→edge 部分更有意义）
    keep_ratio_local_to_sub = values[2] / max(values[1], 1)
    keep_ratio_sub_to_core = values[3] / max(values[2], 1)
    keep_ratio_local_to_core = values[3] / max(values[1], 1)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    x = np.arange(len(labels))
    colors = ["#bcd7f4", "#7fb3d5", "#4f81bd", "#1f4e79"]

    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.8)

    # 数值标注
    y_max = max(values)
    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.02 * y_max,
            f"{v}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # x 轴标签
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # 在图中补充压缩比例说明
    txt = (
        f"Lesion-subgraph keep ratio = {keep_ratio_local_to_sub:.2%}\n"
        f"Core keep ratio (subgraph → core) = {keep_ratio_sub_to_core:.2%}\n"
        f"Overall keep ratio (local → core) = {keep_ratio_local_to_core:.2%}"
    )
    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="gray")
    )

    # 额外脚注说明：第一个是节点数，后三个是边数
    ax.text(
        0.01, -0.18,
        "Note: the first bar is number of screened nodes; the remaining three bars are edge counts.",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
        color="dimgray",
    )

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if created_fig:
        plt.show()

    panel_info = pd.DataFrame({
        "stage": ["screened_nodes", "local_edges", "lesion_subgraph_edges", "core_edges"],
        "count": values,
    })

    return fig, ax, panel_info


def plot_stage2_sparsification_flowchart(
    summary,
    title="Stage 2 sparsification flowchart",
    figsize=(12, 3.6),
    box_fc="#F4F7FB",
    box_ec="#4C78A8",
    arrow_color="#4A4A4A",
    num_color="#1F1F1F",
    label_color="#2F2F2F",
    save_path=None,
):
    """
    summary : dict
        Required keys:
            - n_screen_nodes
            - n_local_edges
            - n_lesion_sub_edges
            - n_core_edges

    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str or None
        If not None, save the figure.
    """
    # -------- 1. 读取统计量 --------
    n_screen_nodes = summary.get("n_screen_nodes", None)
    n_local_edges = summary.get("n_local_edges", None)
    n_lesion_sub_edges = summary.get("n_lesion_sub_edges", None)
    n_core_edges = summary.get("n_core_edges", None)

    # -------- 2. 画布 --------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # -------- 3. 标题 --------
    ax.text(
        0.5, 0.93, title,
        ha="center", va="center",
        fontsize=15, fontweight="bold", color="#1F1F1F"
    )

    # -------- 4. 节点位置 --------
    xs = [0.12, 0.37, 0.62, 0.87]
    y = 0.52
    box_w = 0.18
    box_h = 0.28

    labels = [
        "Screened nodes",
        "Local LPCMCI edges",
        "Lesion-subgraph edges",
        "FC-pruned core edges",
    ]
    numbers = [
        n_screen_nodes,
        n_local_edges,
        n_lesion_sub_edges,
        n_core_edges,
    ]
    subtitles = [
        "Whole-brain lagged screening",
        "LPCMCI on candidate subnetwork",
        "Lesion-centered extracted subgraph",
        "Compressed optimization-ready core",
    ]

    # -------- 5. 画 box --------
    for x, lab, num, sub in zip(xs, labels, numbers, subtitles):
        rect = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            linewidth=1.6,
            edgecolor=box_ec,
            facecolor=box_fc,
        )
        ax.add_patch(rect)

        ax.text(
            x, y + 0.07, lab,
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=label_color
        )

        ax.text(
            x, y - 0.005, f"{num}",
            ha="center", va="center",
            fontsize=22, fontweight="bold", color=num_color
        )

        ax.text(
            x, y - 0.085, sub,
            ha="center", va="center",
            fontsize=9.2, color="#5B5B5B"
        )

    # -------- 6. 箭头 + 压缩比例标注 --------

    # 注意：
    # 第一个箭头 screened nodes -> local LPCMCI edges
    # 是“节点数 -> 边数”，单位不同，不建议写压缩比例。
    # 后两个箭头是 edge -> edge，可以写保留比例 / 压缩比例。

    arrow_labels = [
        "LPCMCI",
        f"{n_local_edges} → {n_lesion_sub_edges}\n"
        f"retained {n_lesion_sub_edges / max(n_local_edges, 1):.1%}",
        f"{n_lesion_sub_edges} → {n_core_edges}\n"
        f"retained {n_core_edges / max(n_lesion_sub_edges, 1):.1%}",
    ]

    for i in range(len(xs) - 1):
        x1 = xs[i] + box_w / 2 + 0.012
        x2 = xs[i + 1] - box_w / 2 - 0.012

        ax.annotate(
            "",
            xy=(x2, y),
            xytext=(x1, y),
            arrowprops=dict(
                arrowstyle="->",
                lw=2.0,
                color=arrow_color,
                shrinkA=0,
                shrinkB=0,
            ),
        )

        # 箭头中点位置
        xm = (x1 + x2) / 2

        ax.text(
            xm,
            y + 0.11,
            arrow_labels[i],
            ha="center",
            va="center",
            fontsize=9.5,
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.95,
            ),
        )

    # -------- 7. 可选：在底部补充压缩比例 --------
    try:
        ratio1 = n_local_edges / max(n_screen_nodes, 1)
        ratio2 = n_lesion_sub_edges / max(n_local_edges, 1)
        ratio3 = n_core_edges / max(n_lesion_sub_edges, 1)

        bottom_text = (
            f"Compression: {n_local_edges} → {n_lesion_sub_edges} → {n_core_edges}"
        )
        ax.text(
            0.5, 0.14, bottom_text,
            ha="center", va="center",
            fontsize=10, color="#444444"
        )
    except Exception:
        pass

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax
