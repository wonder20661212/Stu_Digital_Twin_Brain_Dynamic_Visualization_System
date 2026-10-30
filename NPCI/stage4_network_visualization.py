from __future__ import annotations

from typing import Optional, Sequence, Dict, Tuple, List, Union
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import itertools
from collections import Counter
import matplotlib as mpl


def _extract_best_policy_node(best_policy):
    """
    支持:
    - best_policy = [{"node": 8, ...}]
    - best_policy = {"node": 8, ...}
    """
    if best_policy is None:
        return None

    if isinstance(best_policy, dict):
        return int(best_policy["node"])

    if isinstance(best_policy, (list, tuple)) and len(best_policy) > 0:
        return int(best_policy[0]["node"])

    return None



def analyze_policy_lesion_path(
    core_edge_df: pd.DataFrame,
    lesion_idx: int,
    best_policy,
    score_col: str = "abs_val",
):
    """
    检查 best policy node 和 lesion node 在 Stage 2 有向 core graph 中：
    1. 是否有直接边
    2. 是否存在 policy -> lesion 路径
    3. 是否存在 lesion -> policy 路径
    4. 是否至少存在无向连通路径
    """
    best_policy_node = _extract_best_policy_node(best_policy)

    lesion_idx = int(lesion_idx)
    if best_policy_node is None:
        return {
            "best_policy_node": None,
            "lesion_idx": lesion_idx,
            "has_direct_policy_to_lesion": False,
            "has_direct_lesion_to_policy": False,
            "has_path_policy_to_lesion": False,
            "has_path_lesion_to_policy": False,
            "has_undirected_path": False,
            "shortest_path_policy_to_lesion": None,
            "shortest_path_lesion_to_policy": None,
            "shortest_path_undirected": None,
        }

    df = core_edge_df.copy()
    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)

    G = nx.DiGraph()
    for _, r in df.iterrows():
        src = int(r["src"])
        dst = int(r["dst"])
        w = float(r[score_col]) if score_col in r and pd.notna(r[score_col]) else 1.0
        G.add_edge(src, dst, weight=w)

    has_direct_policy_to_lesion = G.has_edge(best_policy_node, lesion_idx)
    has_direct_lesion_to_policy = G.has_edge(lesion_idx, best_policy_node)

    # policy -> lesion
    if best_policy_node in G and lesion_idx in G and nx.has_path(G, best_policy_node, lesion_idx):
        path_policy_to_lesion = nx.shortest_path(G, best_policy_node, lesion_idx)
    else:
        path_policy_to_lesion = None

    # lesion -> policy
    if best_policy_node in G and lesion_idx in G and nx.has_path(G, lesion_idx, best_policy_node):
        path_lesion_to_policy = nx.shortest_path(G, lesion_idx, best_policy_node)
    else:
        path_lesion_to_policy = None

    # undirected
    G_u = G.to_undirected()
    if best_policy_node in G_u and lesion_idx in G_u and nx.has_path(G_u, best_policy_node, lesion_idx):
        path_undirected = nx.shortest_path(G_u, best_policy_node, lesion_idx)
    else:
        path_undirected = None

    return {
        "best_policy_node": int(best_policy_node),
        "lesion_idx": int(lesion_idx),
        "has_direct_policy_to_lesion": bool(has_direct_policy_to_lesion),
        "has_direct_lesion_to_policy": bool(has_direct_lesion_to_policy),
        "has_path_policy_to_lesion": path_policy_to_lesion is not None,
        "has_path_lesion_to_policy": path_lesion_to_policy is not None,
        "has_undirected_path": path_undirected is not None,
        "shortest_path_policy_to_lesion": path_policy_to_lesion,
        "shortest_path_lesion_to_policy": path_lesion_to_policy,
        "shortest_path_undirected": path_undirected,
    }


def _safe_roi_name(roi, roi_names=None):
    roi = int(roi)
    if roi_names is not None and roi < len(roi_names):
        return str(roi_names[roi])
    return str(roi)


def build_directed_stage4_edge_metrics(
    core_edge_df: pd.DataFrame,
    FC_healthy: np.ndarray,
    FC_lesion: np.ndarray,
    FC_policy: np.ndarray,
) -> pd.DataFrame:
    """
    Attach Stage 4 recovery metrics to directed Stage 2 LPCMCI core edges.

    For each directed edge src -> dst:
        lesion_abn = |FC_lesion[src,dst] - FC_healthy[src,dst]|
        policy_abn = |FC_policy[src,dst] - FC_healthy[src,dst]|
        delta_fc_recovery = lesion_abn - policy_abn

    delta_fc_recovery > 0:
        policy is closer to healthy, edge recovered.

    delta_fc_recovery < 0:
        policy is farther from healthy, edge worsened.
    """
    edge_df = core_edge_df.copy()

    if "src" not in edge_df.columns or "dst" not in edge_df.columns:
        raise KeyError("core_edge_df must contain 'src' and 'dst' columns.")

    edge_df["src"] = edge_df["src"].astype(int)
    edge_df["dst"] = edge_df["dst"].astype(int)

    if "val" not in edge_df.columns:
        if "abs_val" in edge_df.columns:
            edge_df["val"] = edge_df["abs_val"].astype(float)
        else:
            edge_df["val"] = 1.0

    if "abs_val" not in edge_df.columns:
        edge_df["abs_val"] = np.abs(edge_df["val"].astype(float))

    fc_h, fc_l, fc_p = [], [], []
    for _, r in edge_df.iterrows():
        s = int(r["src"])
        d = int(r["dst"])
        fc_h.append(float(FC_healthy[s, d]))
        fc_l.append(float(FC_lesion[s, d]))
        fc_p.append(float(FC_policy[s, d]))

    edge_df["fc_healthy"] = fc_h
    edge_df["fc_lesion"] = fc_l
    edge_df["fc_policy"] = fc_p

    edge_df["lesion_abn"] = np.abs(edge_df["fc_lesion"] - edge_df["fc_healthy"])
    edge_df["policy_abn"] = np.abs(edge_df["fc_policy"] - edge_df["fc_healthy"])
    edge_df["delta_fc_recovery"] = edge_df["lesion_abn"] - edge_df["policy_abn"]
    edge_df["delta_fc_raw"] = edge_df["fc_policy"] - edge_df["fc_lesion"]

    return edge_df


def select_directed_causal_focus_edges(
    edge_df: pd.DataFrame,
    lesion_idx: int,
    best_policy_node: int = None,
    top_k_edges: int = 20,
    mode: str = "lesion_neighborhood",
    score_col: str = "abs_val",
):
    """
    Select directed edges for causal intervention graph.

    This version preserves lesion-related and best-policy-related edges first,
    then fills remaining slots with strongest core edges.
    """

    lesion_idx = int(lesion_idx)
    if best_policy_node is not None:
        best_policy_node = int(best_policy_node)

    df = edge_df.copy()
    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)

    if score_col not in df.columns:
        if "abs_val" in df.columns:
            score_col = "abs_val"
        elif "delta_fc_recovery" in df.columns:
            score_col = "delta_fc_recovery"
        else:
            df["_fallback_score"] = 1.0
            score_col = "_fallback_score"

    df["_rank_score"] = np.abs(df[score_col].astype(float))

    # --------------------------------------------------
    # Helper: append and deduplicate
    # --------------------------------------------------
    def _dedup_edges(x):
        if len(x) == 0:
            return x
        return x.drop_duplicates(["src", "dst"]).reset_index(drop=True)

    # --------------------------------------------------
    # Mode 1: lesion neighborhood
    # --------------------------------------------------
    if mode == "lesion_neighborhood":
        parts = []

        # 1) lesion incident edges must be preserved
        lesion_edges = df[
            (df["src"] == lesion_idx) |
            (df["dst"] == lesion_idx)
        ].copy()
        lesion_edges = lesion_edges.sort_values("_rank_score", ascending=False)
        parts.append(lesion_edges)

        # 2) best policy node incident edges also preserved
        if best_policy_node is not None:
            bp_edges = df[
                (df["src"] == best_policy_node) |
                (df["dst"] == best_policy_node)
            ].copy()
            bp_edges = bp_edges.sort_values("_rank_score", ascending=False)
            parts.append(bp_edges)

        # 3) fill with strongest edges
        strong_edges = df.sort_values("_rank_score", ascending=False)
        parts.append(strong_edges)

        out = pd.concat(parts, axis=0)
        out = _dedup_edges(out)

        # 关键：这里不再先全局排序再 head，
        # 而是按拼接顺序保留 lesion / policy 相关边优先级。
        out = out.head(int(top_k_edges)).reset_index(drop=True)

    # --------------------------------------------------
    # Mode 2: lesion outgoing
    # --------------------------------------------------
    elif mode == "lesion_outgoing":
        G = nx.DiGraph()
        for _, r in df.iterrows():
            G.add_edge(int(r["src"]), int(r["dst"]))

        G.add_node(lesion_idx)

        if lesion_idx in G.nodes:
            reachable = set(nx.descendants(G, lesion_idx))
        else:
            reachable = set()

        reachable.add(lesion_idx)

        out = df[
            df["src"].isin(reachable) &
            df["dst"].isin(reachable)
        ].copy()

        if len(out) == 0:
            # fallback：至少保留 lesion incident edges
            out = df[
                (df["src"] == lesion_idx) |
                (df["dst"] == lesion_idx)
            ].copy()

        out = out.sort_values("_rank_score", ascending=False).head(int(top_k_edges)).reset_index(drop=True)

    # --------------------------------------------------
    # Mode 3: full top-k
    # --------------------------------------------------
    elif mode == "full_topk":
        out = df.sort_values("_rank_score", ascending=False).head(int(top_k_edges)).copy()
        out = out.reset_index(drop=True)

    else:
        raise ValueError(
            "mode must be one of {'lesion_neighborhood', 'lesion_outgoing', 'full_topk'}"
        )

    return out


def build_causal_layer_layout(
    edge_df: pd.DataFrame,
    lesion_idx: int,
    best_policy_node: int = None,
):
    """
    Build deterministic causal-direction layout.

    This version is robust when lesion_idx or best_policy_node
    is not included in selected edges.
    """
    lesion_idx = int(lesion_idx)
    if best_policy_node is not None:
        best_policy_node = int(best_policy_node)

    G = nx.DiGraph()

    # --------------------------------------------------
    # 1. Add directed edges
    # --------------------------------------------------
    if edge_df is not None and len(edge_df) > 0:
        for _, r in edge_df.iterrows():
            G.add_edge(int(r["src"]), int(r["dst"]))

    # --------------------------------------------------
    # 2. Critical fix:
    #    explicitly add lesion and best-policy node
    # --------------------------------------------------
    G.add_node(lesion_idx)

    if best_policy_node is not None:
        G.add_node(best_policy_node)

    nodes = sorted([int(n) for n in G.nodes()])

    # --------------------------------------------------
    # 3. Safe parent / child extraction
    # --------------------------------------------------
    parents = sorted([int(u) for u, v in G.in_edges(lesion_idx)])
    children = sorted([int(v) for u, v in G.out_edges(lesion_idx)])

    # --------------------------------------------------
    # 4. Downstream nodes
    # --------------------------------------------------
    downstream = set()
    for c in children:
        if c in G.nodes:
            downstream |= nx.descendants(G, c)

    downstream = sorted(
        list(downstream - set(children) - {lesion_idx})
    )

    # --------------------------------------------------
    # 5. Upstream second-order nodes
    # --------------------------------------------------
    upstream2 = set()
    for p in parents:
        if p in G.nodes:
            upstream2 |= nx.ancestors(G, p)

    upstream2 = sorted(
        list(upstream2 - set(parents) - {lesion_idx})
    )

    # --------------------------------------------------
    # 6. Others
    # --------------------------------------------------
    assigned = (
        {lesion_idx}
        | set(parents)
        | set(children)
        | set(downstream)
        | set(upstream2)
    )

    others = [n for n in nodes if n not in assigned]

    groups = {
        "upstream2": upstream2,
        "parents": parents,
        "lesion": [lesion_idx],
        "children": children,
        "downstream": downstream,
        "others": others,
    }

    pos = {}

    def place_vertical(group_nodes, x, span=2.6):
        group_nodes = list(group_nodes)
        if len(group_nodes) == 0:
            return

        if len(group_nodes) == 1:
            ys = [0.0]
        else:
            ys = np.linspace(span / 2, -span / 2, len(group_nodes))

        for n, y in zip(group_nodes, ys):
            pos[int(n)] = np.array([float(x), float(y)])

    # --------------------------------------------------
    # 7. Layer layout
    # --------------------------------------------------
    place_vertical(upstream2, -2.8, span=2.2)
    place_vertical(parents, -1.55, span=2.8)

    pos[lesion_idx] = np.array([0.0, 0.0])

    place_vertical(children, 1.55, span=2.8)
    place_vertical(downstream, 2.8, span=2.2)

    # others 放在底部
    if len(others) > 0:
        xs = np.linspace(-1.2, 1.2, len(others))
        for n, x in zip(others, xs):
            pos[int(n)] = np.array([float(x), -2.0])

    # --------------------------------------------------
    # 8. Best policy node: make sure it has position
    # --------------------------------------------------
    if best_policy_node is not None:
        if best_policy_node not in pos:
            # 如果 best policy node 没有边，就放在 lesion 上方
            pos[best_policy_node] = np.array([0.0, 1.25])
        elif best_policy_node != lesion_idx:
            # 如果已有位置，稍微向上抬，突出显示
            pos[best_policy_node] = pos[best_policy_node] + np.array([0.0, 0.35])

    # --------------------------------------------------
    # 9. Role dict
    # --------------------------------------------------
    roles = {}

    # for n in upstream2:
    #     roles[int(n)] = "upstream"

    for n in parents:
        roles[int(n)] = "parent"

    roles[lesion_idx] = "lesion"

    for n in children:
        roles[int(n)] = "child"

    # for n in downstream:
    #     roles[int(n)] = "downstream"

    for n in others:
        roles[int(n)] = "other"

    if best_policy_node is not None:
        base_role = roles.get(best_policy_node, "policy_target")
        if "policy" not in base_role:
            roles[best_policy_node] = base_role + "/policy"

    return pos, roles, groups



def _draw_callout_node_labels_for_causal_graph(
    ax,
    pos,
    labels,
    roles,
    lesion_idx=None,
    best_policy_node=None,
    fontsize=8,
):
    """
    Draw node labels outside nodes with callout lines.
    This avoids labels overflowing triangle/square/circle markers.
    """
    for n, p in pos.items():
        n = int(n)
        x, y = float(p[0]), float(p[1])
        role = roles.get(n, "other")

        label = labels.get(n, str(n))

        if n == int(lesion_idx):
            xytext = (x, y + 0.48)
            ha, va = "center", "bottom"
        elif best_policy_node is not None and n == int(best_policy_node):
            xytext = (x, y - 0.48)
            ha, va = "center", "top"
        elif "parent" in role or "upstream" in role:
            xytext = (x - 0.52, y)
            ha, va = "right", "center"
        elif "child" in role or "downstream" in role:
            xytext = (x + 0.52, y)
            ha, va = "left", "center"
        else:
            xytext = (x, y - 0.42)
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
                fc="white",
                ec="lightgray",
                alpha=0.92,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="gray",
                lw=0.75,
                shrinkA=2,
                shrinkB=3,
                alpha=0.85,
            ),
            zorder=10,
        )


def _draw_policy_to_lesion_relation(
    ax,
    pos,
    lesion_idx,
    best_policy_node,
    text=True,
    direction="policy_to_lesion",
):
    """
    Draw dashed golden intervention relation.

    direction:
        "policy_to_lesion":
            policy node  ---> lesion node

        "lesion_to_policy":
            lesion node  ---> policy node

    This arrow denotes the optimized intervention relation,
    not an LPCMCI causal edge.
    """
    if best_policy_node is None:
        return

    lesion_idx = int(lesion_idx)
    best_policy_node = int(best_policy_node)

    if lesion_idx not in pos or best_policy_node not in pos:
        return

    if lesion_idx == best_policy_node:
        return

    # --------------------------------------------------
    # 1. Choose arrow direction
    # --------------------------------------------------
    if direction == "policy_to_lesion":
        src = best_policy_node
        dst = lesion_idx
        label_text = "policy → lesion\nintervention"
    elif direction == "lesion_to_policy":
        src = lesion_idx
        dst = best_policy_node
        label_text = "lesion → policy\nrelation"
    else:
        raise ValueError("direction must be 'policy_to_lesion' or 'lesion_to_policy'")

    p1 = np.asarray(pos[src], dtype=float)
    p2 = np.asarray(pos[dst], dtype=float)

    # --------------------------------------------------
    # 2. Draw arrow
    # shrinkA / shrinkB 很重要：
    # 防止箭头头部被节点圆圈盖住
    # --------------------------------------------------
    patch = FancyArrowPatch(
        posA=p1,
        posB=p2,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=-0.28",
        linewidth=2.3,
        linestyle="--",
        color="goldenrod",
        alpha=0.95,
        mutation_scale=24,
        shrinkA=22,
        shrinkB=32,
        zorder=8,
        clip_on=False,
    )
    ax.add_patch(patch)

    # --------------------------------------------------
    # 3. Text label
    # --------------------------------------------------
    if text:
        mid = (p1 + p2) / 2.0

        ax.text(
            mid[0],
            mid[1] + 0.28,
            label_text,
            ha="center",
            va="center",
            fontsize=8,
            color="goldenrod",
            fontweight="bold",
            zorder=9,
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="none",
                alpha=0.75,
            ),
        )


def _build_causal_node_labels(nodes, lesion_idx, best_policy_node=None, roi_names=None):
    """
    Build short labels for callout annotation.
    Labels are outside nodes, so no need to force them inside markers.
    """
    labels = {}
    for n in nodes:
        n = int(n)
        base = _safe_roi_name(n, roi_names=roi_names)

        if n == int(lesion_idx):
            labels[n] = f"Lesion\n{base}"
        elif best_policy_node is not None and n == int(best_policy_node):
            labels[n] = f"Policy\n{base}"
        else:
            labels[n] = str(base)

    return labels


def _draw_causal_graph_core_panel(
    ax,
    G,
    pos,
    roles,
    node_df,
    labels,
    lesion_idx,
    best_policy_node,
    node_facecolors,
    node_size_map,
    edge_colors,
    edge_widths,
    show_policy_to_lesion_link=True,
    title="",
):
    """
    Draw one causal graph panel without colorbar.
    """

    # 顶部只保留核心角色
    group_y = 2.08
    group_labels = [
        ("Parents", -1.55),
        ("Lesion", 0.0),
        ("Children", 1.55),
    ]
    for lab, x in group_labels:
        ax.text(
            x,
            group_y,
            lab,
            ha="center",
            va="center",
            fontsize=10,
            color="dimgray",
            fontweight="bold",
        )

    # 边
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=18,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.10",
        alpha=0.86,
        min_source_margin=20,
        min_target_margin=20,
    )

    # 虚线：优化干预关系，不是 LPCMCI 边
    if show_policy_to_lesion_link:
        _draw_policy_to_lesion_relation(
            ax=ax,
            pos=pos,
            lesion_idx=lesion_idx,
            best_policy_node=best_policy_node,
            text=True,
            direction="policy_to_lesion",
        )

    # 节点
    for i, roi in enumerate(node_df["roi"].astype(int).values):
        role = roles.get(int(roi), "other")

        if int(roi) == int(lesion_idx):
            edgecolor = "darkred"
            linewidth = 3.0
            marker = "o"
        elif best_policy_node is not None and int(roi) == int(best_policy_node):
            edgecolor = "gold"
            linewidth = 3.0
            marker = "o"
        elif "parent" in role:
            edgecolor = "#2b6cb0"
            linewidth = 1.8
            marker = "^"
        elif "child" in role:
            edgecolor = "#2f855a"
            linewidth = 1.8
            marker = "s"
        elif "downstream" in role:
            edgecolor = "#805ad5"
            linewidth = 1.5
            marker = "o"
        else:
            edgecolor = "gray"
            linewidth = 1.2
            marker = "o"

        ax.scatter(
            pos[int(roi)][0],
            pos[int(roi)][1],
            s=node_size_map[int(roi)],
            c=[node_facecolors[i]],
            edgecolors=edgecolor,
            linewidths=linewidth,
            marker=marker,
            zorder=3,
            alpha=0.97,
        )

    # 外置 callout 标签，不再把文字放进节点
    _draw_callout_node_labels_for_causal_graph(
        ax=ax,
        pos=pos,
        labels=labels,
        roles=roles,
        lesion_idx=lesion_idx,
        best_policy_node=best_policy_node,
        fontsize=8,
    )

    ax.set_title(title, fontsize=13, pad=14)
    ax.axis("off")

    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    ax.set_xlim(xs.min() - 1.05, xs.max() + 1.05)
    ax.set_ylim(ys.min() - 1.05, ys.max() + 1.10)


def plot_stage4_directed_causal_intervention_graph_split(
    core_edge_df: pd.DataFrame,
    stage4_metrics: dict,
    lesion_idx: int,
    best_policy=None,
    roi_recovery_df: pd.DataFrame = None,
    stage1_response_df: pd.DataFrame = None,
    roi_names=None,
    top_k_edges: int = 20,
    mode: str = "lesion_neighborhood",
    figsize=(11.5, 6.8),
    show_policy_to_lesion_link: bool = True,
    save_prefix: str = None,
):
    """
    Draw two separate Stage 4 directed causal intervention graphs.

    Figure A:
        Node-level ROI recovery graph
        - node color = roi_recovery
        - edges are neutral gray, only showing causal direction/background

    Figure B:
        Edge-level ΔFC recovery graph
        - edge color = delta_fc_recovery
        - nodes are role-colored/neutral

    This separates node recovery and edge recovery to avoid mixing two recovery scales.
    """

    # --------------------------------------------------
    # 1. Basic inputs
    # --------------------------------------------------
    FC_healthy = stage4_metrics["FC_healthy"]
    FC_lesion = stage4_metrics["FC_lesion"]
    FC_policy = stage4_metrics["FC_policy"]

    lesion_idx = int(lesion_idx)
    best_policy_node = _extract_best_policy_node(best_policy)

    # --------------------------------------------------
    # 2. Path info
    # --------------------------------------------------
    path_info = analyze_policy_lesion_path(
        core_edge_df=core_edge_df,
        lesion_idx=lesion_idx,
        best_policy=best_policy,
        score_col="abs_val",
    )

    # --------------------------------------------------
    # 3. Edge metrics
    # --------------------------------------------------
    edge_metric_df = build_directed_stage4_edge_metrics(
        core_edge_df=core_edge_df,
        FC_healthy=FC_healthy,
        FC_lesion=FC_lesion,
        FC_policy=FC_policy,
    )

    focus_edges = select_directed_causal_focus_edges(
        edge_df=edge_metric_df,
        lesion_idx=lesion_idx,
        best_policy_node=best_policy_node,
        top_k_edges=top_k_edges,
        mode=mode,
        score_col="abs_val",
    )

    # --------------------------------------------------
    # 4. Layout
    # --------------------------------------------------
    pos, roles, groups = build_causal_layer_layout(
        edge_df=focus_edges,
        lesion_idx=lesion_idx,
        best_policy_node=best_policy_node,
    )

    nodes = sorted(pos.keys())

    # --------------------------------------------------
    # 5. Node df
    # --------------------------------------------------
    node_df = pd.DataFrame({"roi": nodes})
    node_df["roi"] = node_df["roi"].astype(int)

    if roi_recovery_df is not None:
        rr = roi_recovery_df.copy()
        if "node" in rr.columns and "roi" not in rr.columns:
            rr = rr.rename(columns={"node": "roi"})
        rr["roi"] = rr["roi"].astype(int)

        keep = ["roi"] + [
            c for c in [
                "roi_recovery",
                "improvement",
                "D_lesion_roi",
                "D_policy_roi",
            ]
            if c in rr.columns
        ]
        node_df = node_df.merge(rr[keep], on="roi", how="left")

    if stage1_response_df is not None:
        s1 = stage1_response_df.copy()
        if "node" in s1.columns and "roi" not in s1.columns:
            s1 = s1.rename(columns={"node": "roi"})
        s1["roi"] = s1["roi"].astype(int)

        keep = ["roi"] + [
            c for c in [
                "response_abs",
                "candidate_score",
            ]
            if c in s1.columns
        ]
        node_df = node_df.merge(s1[keep], on="roi", how="left")

    # --------------------------------------------------
    # 6. Labels outside nodes
    # --------------------------------------------------
    labels = _build_causal_node_labels(
        nodes=nodes,
        lesion_idx=lesion_idx,
        best_policy_node=best_policy_node,
        roi_names=roi_names,
    )

    # --------------------------------------------------
    # 7. Graph
    # --------------------------------------------------
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(int(n))

    for _, r in focus_edges.iterrows():
        G.add_edge(
            int(r["src"]),
            int(r["dst"]),
            val=float(r.get("val", 0.0)),
            abs_val=float(r.get("abs_val", 1.0)),
            delta_fc_recovery=float(r.get("delta_fc_recovery", 0.0)),
            lesion_abn=float(r.get("lesion_abn", 0.0)),
            policy_abn=float(r.get("policy_abn", 0.0)),
        )

    # --------------------------------------------------
    # 8. Node size
    # --------------------------------------------------
    if "roi_recovery" in node_df.columns:
        svals = np.abs(node_df["roi_recovery"].astype(float).fillna(0.0).values)
    elif "response_abs" in node_df.columns:
        svals = np.abs(node_df["response_abs"].astype(float).fillna(0.0).values)
    else:
        svals = np.ones(len(node_df), dtype=float)

    if np.max(svals) > 0:
        sizes = 520 + 1050 * svals / (np.max(svals) + 1e-12)
    else:
        sizes = np.ones(len(node_df), dtype=float) * 760

    node_size_map = {
        int(roi): float(size)
        for roi, size in zip(node_df["roi"], sizes)
    }

    # --------------------------------------------------
    # 9. Edge width
    # --------------------------------------------------
    if len(G.edges()) > 0:
        edge_delta_vals = np.array(
            [G.edges[e].get("delta_fc_recovery", 0.0) for e in G.edges()],
            dtype=float,
        )
        edge_abs_vals = np.array(
            [G.edges[e].get("abs_val", 1.0) for e in G.edges()],
            dtype=float,
        )
    else:
        edge_delta_vals = np.array([], dtype=float)
        edge_abs_vals = np.array([], dtype=float)

    if len(edge_abs_vals) > 0 and np.max(edge_abs_vals) > 0:
        edge_widths_background = 1.1 + 3.4 * edge_abs_vals / (np.max(edge_abs_vals) + 1e-12)
    else:
        edge_widths_background = np.ones(len(edge_abs_vals), dtype=float) * 1.6

    if len(edge_delta_vals) > 0 and np.max(np.abs(edge_delta_vals)) > 0:
        edge_widths_delta = 1.2 + 4.2 * np.abs(edge_delta_vals) / (np.max(np.abs(edge_delta_vals)) + 1e-12)
    else:
        edge_widths_delta = np.ones(len(edge_delta_vals), dtype=float) * 1.8

    # ==================================================
    # Figure A: node-level recovery graph
    # ==================================================
    if "roi_recovery" in node_df.columns:
        node_vals = node_df["roi_recovery"].astype(float).fillna(0.0).values
        node_color_label = "ROI recovery"
    elif "response_abs" in node_df.columns:
        node_vals = node_df["response_abs"].astype(float).fillna(0.0).values
        node_color_label = "Stage 1 response_abs"
    else:
        node_vals = np.ones(len(node_df), dtype=float)
        node_color_label = "Node value"

    node_vmin = float(np.min(node_vals))
    node_vmax = float(np.max(node_vals))
    if np.isclose(node_vmin, node_vmax):
        node_vmax = node_vmin + 1e-6

    node_norm = Normalize(vmin=node_vmin, vmax=node_vmax)
    node_cmap = plt.get_cmap("YlGnBu")
    node_colors_recovery = [node_cmap(node_norm(v)) for v in node_vals]

    # edges neutral in node-level figure
    neutral_edge_colors = ["#8f8f8f"] * len(G.edges())

    fig_node, ax_node = plt.subplots(figsize=figsize)

    _draw_causal_graph_core_panel(
        ax=ax_node,
        G=G,
        pos=pos,
        roles=roles,
        node_df=node_df,
        labels=labels,
        lesion_idx=lesion_idx,
        best_policy_node=best_policy_node,
        node_facecolors=node_colors_recovery,
        node_size_map=node_size_map,
        edge_colors=neutral_edge_colors,
        edge_widths=edge_widths_background,
        show_policy_to_lesion_link=show_policy_to_lesion_link,
        title="Stage 4 directed causal graph: ROI-level recovery",
    )

    sm_node = cm.ScalarMappable(norm=node_norm, cmap=node_cmap)
    sm_node.set_array([])
    cbar_node = fig_node.colorbar(sm_node, ax=ax_node, fraction=0.035, pad=0.025)
    cbar_node.set_label(node_color_label, fontsize=10)

    handles_node = [
        Line2D([0], [0], marker="o", color="w", label="Lesion ROI",
               markerfacecolor="lightcoral", markeredgecolor="darkred", markersize=10, markeredgewidth=2.2),
        Line2D([0], [0], marker="o", color="w", label="Best policy node",
               markerfacecolor="khaki", markeredgecolor="gold", markersize=10, markeredgewidth=2.2),
        Line2D([0], [0], marker="^", color="w", label="Parent",
               markerfacecolor="lightgray", markeredgecolor="#2b6cb0", markersize=9),
        Line2D([0], [0], marker="s", color="w", label="Child",
               markerfacecolor="lightgray", markeredgecolor="#2f855a", markersize=9),
        Line2D([0], [0], color="#8f8f8f", lw=2.4, label="Directed causal edge"),
        Line2D([0], [0], color="goldenrod", lw=2.2, linestyle="--",
               label="Optimized intervention relation"),
    ]

    ax_node.legend(
        handles=handles_node,
        frameon=False,
        fontsize=9,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.12),
    )

    plt.tight_layout()

    if save_prefix is not None:
        fig_node.savefig(
            f"{save_prefix}_node_roi_recovery.png",
            dpi=300,
            bbox_inches="tight",
        )

    # ==================================================
    # Figure B: edge-level ΔFC recovery graph
    # ==================================================
    vmax_delta = float(np.max(np.abs(edge_delta_vals))) if len(edge_delta_vals) > 0 else 1.0
    if vmax_delta <= 1e-12:
        vmax_delta = 1.0

    edge_norm = TwoSlopeNorm(vmin=-vmax_delta, vcenter=0.0, vmax=vmax_delta)
    edge_cmap = plt.get_cmap("RdBu")
    edge_colors_delta = [edge_cmap(edge_norm(v)) for v in edge_delta_vals]

    # nodes role-neutral in edge-level figure
    role_node_colors = []
    for roi in node_df["roi"].astype(int).values:
        role = roles.get(int(roi), "other")
        if int(roi) == int(lesion_idx):
            role_node_colors.append("#F4B6B6")
        elif best_policy_node is not None and int(roi) == int(best_policy_node):
            role_node_colors.append("#FFF1A8")
        elif "parent" in role:
            role_node_colors.append("#DDEBFF")
        elif "child" in role:
            role_node_colors.append("#DFF3E3")
        else:
            role_node_colors.append("#EAEAEA")

    fig_edge, ax_edge = plt.subplots(figsize=figsize)

    _draw_causal_graph_core_panel(
        ax=ax_edge,
        G=G,
        pos=pos,
        roles=roles,
        node_df=node_df,
        labels=labels,
        lesion_idx=lesion_idx,
        best_policy_node=best_policy_node,
        node_facecolors=role_node_colors,
        node_size_map=node_size_map,
        edge_colors=edge_colors_delta,
        edge_widths=edge_widths_delta,
        show_policy_to_lesion_link=show_policy_to_lesion_link,
        title=r"Stage 4 directed causal graph: edge-level $\Delta FC$ recovery",
    )

    sm_edge = cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
    sm_edge.set_array([])
    cbar_edge = fig_edge.colorbar(sm_edge, ax=ax_edge, fraction=0.035, pad=0.025)
    cbar_edge.set_label(r"$\Delta FC$ recovery", fontsize=10)

    recovered_color = edge_cmap(edge_norm(vmax_delta))
    worsened_color = edge_cmap(edge_norm(-vmax_delta))

    handles_edge = [
        Line2D([0], [0], marker="o", color="w", label="Lesion ROI",
               markerfacecolor="#F4B6B6", markeredgecolor="darkred", markersize=10, markeredgewidth=2.2),
        Line2D([0], [0], marker="o", color="w", label="Best policy node",
               markerfacecolor="#FFF1A8", markeredgecolor="gold", markersize=10, markeredgewidth=2.2),
        Line2D([0], [0], marker="^", color="w", label="Parent",
               markerfacecolor="#DDEBFF", markeredgecolor="#2b6cb0", markersize=9),
        Line2D([0], [0], marker="s", color="w", label="Child",
               markerfacecolor="#DFF3E3", markeredgecolor="#2f855a", markersize=9),
        Line2D([0], [0], color=recovered_color, lw=3, label="Recovered edge"),
        Line2D([0], [0], color=worsened_color, lw=3, label="Worsened edge"),
        Line2D([0], [0], color="goldenrod", lw=2.2, linestyle="--",
               label="Optimized intervention relation"),
    ]

    ax_edge.legend(
        handles=handles_edge,
        frameon=False,
        fontsize=9,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.12),
    )

    plt.tight_layout()

    if save_prefix is not None:
        fig_edge.savefig(
            f"{save_prefix}_edge_delta_fc_recovery.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    info = {
        "focus_edges": focus_edges,
        "node_df": node_df,
        "roles": roles,
        "groups": groups,
        "n_nodes": len(nodes),
        "n_edges": len(focus_edges),
        "best_policy_node": best_policy_node,
        "lesion_idx": int(lesion_idx),
        "path_info": path_info,
    }

    return fig_node, ax_node, fig_edge, ax_edge, info


def _get_best_policy_node_from_policy(best_policy=None, best_policy_node=None):
    """
    Extract best policy node from:
        best_policy = [{"node": ...}]
        best_policy = {"node": ...}
        best_policy_node = int
    """
    if best_policy_node is not None:
        return int(best_policy_node)

    if best_policy is None:
        raise ValueError("Either best_policy or best_policy_node must be provided.")

    if isinstance(best_policy, dict):
        if "node" not in best_policy:
            raise KeyError("best_policy dict must contain key 'node'.")
        return int(best_policy["node"])

    if isinstance(best_policy, (list, tuple)) and len(best_policy) > 0:
        if not isinstance(best_policy[0], dict):
            raise TypeError("best_policy list must contain dict items.")
        if "node" not in best_policy[0]:
            raise KeyError("best_policy[0] must contain key 'node'.")
        return int(best_policy[0]["node"])

    if isinstance(best_policy, pd.Series):
        if "node" not in best_policy.index:
            raise KeyError("best_policy Series must contain index 'node'.")
        return int(best_policy["node"])

    raise TypeError(f"Unsupported best_policy type: {type(best_policy)}")


def _safe_roi_label(roi, roi_names=None):
    roi = int(roi)
    if roi_names is not None and roi < len(roi_names):
        return str(roi_names[roi])
    return str(roi)


def _prepare_core_edge_df_for_path(core_edge_df: pd.DataFrame, score_col: str = "abs_val") -> pd.DataFrame:
    """
    Ensure core_edge_df has src/dst/score_col.
    """
    df = core_edge_df.copy()

    if "src" not in df.columns or "dst" not in df.columns:
        raise KeyError("core_edge_df must contain columns: 'src', 'dst'.")

    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)

    if score_col not in df.columns:
        if "abs_val" in df.columns:
            score_col = "abs_val"
        elif "val" in df.columns:
            df["abs_val"] = np.abs(df["val"].astype(float))
            score_col = "abs_val"
        elif "fused_score" in df.columns:
            score_col = "fused_score"
        else:
            df["abs_val"] = 1.0
            score_col = "abs_val"

    df[score_col] = df[score_col].astype(float)

    # 去重：同一条 src->dst 如果重复，保留 score 最大的一条
    df["_path_score_for_sort"] = np.abs(df[score_col].astype(float))
    df = (
        df.sort_values("_path_score_for_sort", ascending=False)
        .drop_duplicates(["src", "dst"], keep="first")
        .reset_index(drop=True)
    )

    return df


def _path_score(path, edge_attr, score_col="abs_val", score_mode="mean"):
    """
    Score a path according to edge scores.
    score_mode:
        "mean", "sum", "min"
    """
    vals = []
    for u, v in zip(path[:-1], path[1:]):
        attr = edge_attr.get((int(u), int(v)), None)
        if attr is None:
            attr = edge_attr.get((int(v), int(u)), None)
        if attr is None:
            vals.append(0.0)
        else:
            vals.append(abs(float(attr.get(score_col, 0.0))))

    if len(vals) == 0:
        return 0.0

    vals = np.asarray(vals, dtype=float)

    if score_mode == "mean":
        return float(np.mean(vals))
    elif score_mode == "sum":
        return float(np.sum(vals))
    elif score_mode == "min":
        return float(np.min(vals))
    else:
        raise ValueError("score_mode must be one of {'mean', 'sum', 'min'}.")


# =========================================================
# Function 1: extract_policy_lesion_paths
# =========================================================
def extract_policy_lesion_paths(
    lesion_core_edges: pd.DataFrame,
    lesion_idx: int,
    best_policy=None,
    best_policy_node=None,
    direction: str = "auto",
    max_hops: int = 4,
    top_k_paths: int = 5,
    score_col: str = "abs_val",
    score_mode: str = "mean",
    path_mode: str = "topk_union",
    allow_undirected_fallback: bool = True,
    max_raw_paths: int = 500,
):
    """
    Extract paths between best policy node and lesion node using the full lesion_core_edges.

    Parameters
    ----------
    direction:
        "policy_to_lesion"
        "lesion_to_policy"
        "auto"

    path_mode:
        "shortest":
            select one shortest path. If multiple shortest paths exist,
            choose the one with higher path score.

        "topk_union":
            select top-k scored paths and later draw their union graph.

    score_mode:
        "mean", "sum", "min"

    Returns
    -------
    dict with:
        selected_paths
        selected_path_scores
        all_paths
        all_path_scores
        direction_used
        search_type
    """
    if path_mode not in {"shortest", "topk_union"}:
        raise ValueError("path_mode must be 'shortest' or 'topk_union'.")

    lesion_idx = int(lesion_idx)
    policy_node = _get_best_policy_node_from_policy(
        best_policy=best_policy,
        best_policy_node=best_policy_node,
    )

    df = _prepare_core_edge_df_for_path(lesion_core_edges, score_col=score_col)

    # build directed graph
    G = nx.DiGraph()
    edge_attr = {}

    for _, r in df.iterrows():
        u = int(r["src"])
        v = int(r["dst"])
        attr = r.to_dict()
        G.add_edge(u, v, **attr)
        edge_attr[(u, v)] = attr

    # make sure nodes exist
    G.add_node(policy_node)
    G.add_node(lesion_idx)

    def _limited_all_simple_paths(graph, source, target, cutoff):
        if source not in graph.nodes or target not in graph.nodes:
            return []
        if not nx.has_path(graph, source, target):
            return []
        gen = nx.all_simple_paths(graph, source=source, target=target, cutoff=cutoff)
        return list(itertools.islice(gen, int(max_raw_paths)))

    search_type = "directed"
    direction_used = None
    source, target = None, None
    candidate_paths = []

    # --------------------------------------------------
    # directed search
    # --------------------------------------------------
    if direction == "policy_to_lesion":
        source, target = policy_node, lesion_idx
        direction_used = "policy_to_lesion"
        candidate_paths = _limited_all_simple_paths(G, source, target, max_hops)

    elif direction == "lesion_to_policy":
        source, target = lesion_idx, policy_node
        direction_used = "lesion_to_policy"
        candidate_paths = _limited_all_simple_paths(G, source, target, max_hops)

    elif direction == "auto":
        # 优先 policy -> lesion
        p2l_paths = _limited_all_simple_paths(G, policy_node, lesion_idx, max_hops)
        if len(p2l_paths) > 0:
            source, target = policy_node, lesion_idx
            direction_used = "policy_to_lesion"
            candidate_paths = p2l_paths
        else:
            l2p_paths = _limited_all_simple_paths(G, lesion_idx, policy_node, max_hops)
            if len(l2p_paths) > 0:
                source, target = lesion_idx, policy_node
                direction_used = "lesion_to_policy"
                candidate_paths = l2p_paths
            else:
                direction_used = "none"
                candidate_paths = []
    else:
        raise ValueError("direction must be 'policy_to_lesion', 'lesion_to_policy', or 'auto'.")

    # --------------------------------------------------
    # undirected fallback
    # --------------------------------------------------
    if len(candidate_paths) == 0 and allow_undirected_fallback:
        G_u = G.to_undirected()
        search_type = "undirected_fallback"

        if policy_node in G_u.nodes and lesion_idx in G_u.nodes and nx.has_path(G_u, policy_node, lesion_idx):
            gen = nx.all_simple_paths(
                G_u,
                source=policy_node,
                target=lesion_idx,
                cutoff=max_hops,
            )
            candidate_paths = list(itertools.islice(gen, int(max_raw_paths)))
            source, target = policy_node, lesion_idx
            direction_used = "policy_to_lesion_undirected"
        else:
            candidate_paths = []

    # --------------------------------------------------
    # score and select paths
    # --------------------------------------------------
    if len(candidate_paths) == 0:
        return {
            "best_policy_node": int(policy_node),
            "lesion_idx": int(lesion_idx),
            "direction_used": direction_used,
            "search_type": search_type,
            "has_path": False,
            "source": source,
            "target": target,
            "all_paths": [],
            "all_path_scores": [],
            "selected_paths": [],
            "selected_path_scores": [],
            "path_mode": path_mode,
            "max_hops": int(max_hops),
        }

    path_scores = [
        _path_score(
            p,
            edge_attr=edge_attr,
            score_col=score_col,
            score_mode=score_mode,
        )
        for p in candidate_paths
    ]

    # shortest: first sort by length, then score
    if path_mode == "shortest":
        order = sorted(
            range(len(candidate_paths)),
            key=lambda i: (len(candidate_paths[i]), -path_scores[i])
        )
        selected_idx = order[:1]

    # top-k union: first sort by score, then shorter path
    else:
        order = sorted(
            range(len(candidate_paths)),
            key=lambda i: (-path_scores[i], len(candidate_paths[i]))
        )
        selected_idx = order[:int(top_k_paths)]

    all_order = sorted(
        range(len(candidate_paths)),
        key=lambda i: (-path_scores[i], len(candidate_paths[i]))
    )

    all_paths_sorted = [candidate_paths[i] for i in all_order]
    all_scores_sorted = [path_scores[i] for i in all_order]

    selected_paths = [candidate_paths[i] for i in selected_idx]
    selected_scores = [path_scores[i] for i in selected_idx]

    return {
        "best_policy_node": int(policy_node),
        "lesion_idx": int(lesion_idx),
        "direction_used": direction_used,
        "search_type": search_type,
        "has_path": True,
        "source": int(source) if source is not None else None,
        "target": int(target) if target is not None else None,
        "all_paths": all_paths_sorted,
        "all_path_scores": all_scores_sorted,
        "selected_paths": selected_paths,
        "selected_path_scores": selected_scores,
        "path_mode": path_mode,
        "max_hops": int(max_hops),
    }


# =========================================================
# Function 2: build_policy_lesion_path_subgraph
# =========================================================
def build_policy_lesion_path_subgraph(
    lesion_core_edges: pd.DataFrame,
    path_list,
    score_col: str = "abs_val",
):
    """
    Build a path-union edge table from selected paths.

    The output uses:
        src, dst
    as the DRAWING direction along the selected path.

    It also keeps:
        causal_src, causal_dst
    as the original direction in lesion_core_edges.

    If a path comes from undirected fallback, an edge may be drawn
    opposite to its original causal direction. In that case:
        is_reverse_traversal = True
    """
    if path_list is None or len(path_list) == 0:
        return pd.DataFrame()

    df = _prepare_core_edge_df_for_path(lesion_core_edges, score_col=score_col)

    rows = []
    edge_counter = Counter()

    for path_id, path in enumerate(path_list, start=1):
        for step_id, (u, v) in enumerate(zip(path[:-1], path[1:]), start=1):
            u = int(u)
            v = int(v)
            edge_counter[(u, v)] += 1

            exact = df[(df["src"] == u) & (df["dst"] == v)]
            reverse = df[(df["src"] == v) & (df["dst"] == u)]

            if len(exact) > 0:
                r = exact.iloc[0].copy()
                is_reverse = False
                causal_src = u
                causal_dst = v
            elif len(reverse) > 0:
                r = reverse.iloc[0].copy()
                is_reverse = True
                causal_src = v
                causal_dst = u
            else:
                # This should rarely happen, but keep robust.
                r = pd.Series({"src": u, "dst": v, score_col: 0.0})
                is_reverse = False
                causal_src = u
                causal_dst = v

            r["path_id"] = path_id
            r["path_step"] = step_id

            # drawing direction
            r["src"] = u
            r["dst"] = v

            # original causal direction
            r["causal_src"] = int(causal_src)
            r["causal_dst"] = int(causal_dst)
            r["is_reverse_traversal"] = bool(is_reverse)

            rows.append(r)

    if len(rows) == 0:
        return pd.DataFrame()

    raw_edge_df = pd.DataFrame(rows)

    # union duplicated drawing edges
    agg_dict = {
        "path_id": lambda x: sorted(set(int(v) for v in x)),
        "path_step": "min",
        "causal_src": "first",
        "causal_dst": "first",
        "is_reverse_traversal": "max",
    }

    for c in raw_edge_df.columns:
        if c in ["src", "dst", "path_id", "path_step", "causal_src", "causal_dst", "is_reverse_traversal"]:
            continue
        if pd.api.types.is_numeric_dtype(raw_edge_df[c]):
            agg_dict[c] = "max"
        else:
            agg_dict[c] = "first"

    edge_union_df = (
        raw_edge_df
        .groupby(["src", "dst"], as_index=False)
        .agg(agg_dict)
    )

    edge_union_df["path_count"] = [
        edge_counter[(int(s), int(d))]
        for s, d in zip(edge_union_df["src"], edge_union_df["dst"])
    ]

    return edge_union_df.reset_index(drop=True)


# =========================================================
# Layout helper
# =========================================================
def _build_policy_lesion_path_layer_layout(
    path_list,
    x_gap: float = 1.45,
    y_span: float = 2.15,
):
    """
    Build a clearer layer layout according to path hop index.

    Improvements:
    - larger horizontal gap between layers
    - larger vertical spread within the same layer
    - source on the left, target on the right
    """
    if path_list is None or len(path_list) == 0:
        return {}

    main_path = path_list[0]
    source = int(main_path[0])
    target = int(main_path[-1])

    max_layer = max(len(p) - 1 for p in path_list)

    node_layers = {}
    for path in path_list:
        for i, n in enumerate(path):
            n = int(n)

            if n == source:
                layer = 0
            elif n == target:
                layer = max_layer
            else:
                layer = i

            if n not in node_layers:
                node_layers[n] = layer
            else:
                if n == target:
                    node_layers[n] = max_layer
                else:
                    node_layers[n] = min(node_layers[n], layer)

    layer_to_nodes = {}
    for n, layer in node_layers.items():
        layer_to_nodes.setdefault(layer, []).append(n)

    main_order = {int(n): i for i, n in enumerate(main_path)}

    pos = {}
    for layer, nodes in layer_to_nodes.items():
        nodes = sorted(nodes, key=lambda n: main_order.get(n, 999))

        if len(nodes) == 1:
            ys = [0.0]
        else:
            ys = np.linspace(y_span / 2, -y_span / 2, len(nodes))

        x = float(layer) * float(x_gap)
        for n, y in zip(nodes, ys):
            pos[int(n)] = np.array([x, float(y)])

    return pos


def _draw_path_callout_labels(
    ax,
    pos,
    node_roles,
    roi_names=None,
    fontsize=8,
):
    """
    Draw outside labels with callout lines.

    Improvements:
    - bridge labels are placed in upper/lower label lanes
    - labels avoid path edges as much as possible
    - labels avoid layer title area
    """
    if len(pos) == 0:
        return

    ys_all = np.array([float(p[1]) for p in pos.values()])
    max_y = float(np.max(ys_all))
    min_y = float(np.min(ys_all))

    top_label_y = max_y + 0.48
    bottom_label_y = min_y - 0.48

    # 同一层多个 top/bottom 标签时稍微错开
    top_counter = {}
    bottom_counter = {}

    for n, p in pos.items():
        n = int(n)
        x, y = float(p[0]), float(p[1])
        role = node_roles.get(n, "bridge")

        base = _safe_roi_label(n, roi_names=roi_names)

        if role == "policy":
            label = f"Policy\n{base}"
            xytext = (x, y + 0.60)
            ha, va = "center", "bottom"

        elif role == "lesion":
            label = f"Lesion\n{base}"
            xytext = (x, y + 0.60)
            ha, va = "center", "bottom"

        else:
            label = f"Bridge\n{base}"

            layer_key = round(x, 3)

            if y >= 0:
                k = top_counter.get(layer_key, 0)
                top_counter[layer_key] = k + 1
                xytext = (x, top_label_y + 0.16 * k)
                ha, va = "center", "bottom"
            else:
                k = bottom_counter.get(layer_key, 0)
                bottom_counter[layer_key] = k + 1
                xytext = (x, bottom_label_y - 0.16 * k)
                ha, va = "center", "top"

        ax.annotate(
            label,
            xy=(x, y),
            xytext=xytext,
            textcoords="data",
            ha=ha,
            va=va,
            fontsize=fontsize,
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc="white",
                ec="lightgray",
                alpha=0.95,
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

def _draw_path_summary_box(
    ax,
    selected_paths,
    selected_scores,
    roi_names=None,
    loc: str = "lower right",
    fontsize: float = 8.0,
):
    """
    Draw path summary as a boxed legend-like annotation.

    loc:
        "lower right" or "lower left"
    """
    if selected_paths is None or len(selected_paths) == 0:
        return

    lines = []
    for i, (p, s) in enumerate(zip(selected_paths, selected_scores), start=1):
        pstr = _format_path_string(p, roi_names=roi_names, max_len=90)
        lines.append(f"P{i}: {pstr} | score={s:.4f}")

    summary_text = "\n".join(lines)

    if loc == "lower left":
        x, y = 0.02, 0.03
        ha, va = "left", "bottom"
    else:
        x, y = 0.98, 0.03
        ha, va = "right", "bottom"

    ax.text(
        x,
        y,
        summary_text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color="dimgray",
        bbox=dict(
            boxstyle="round,pad=0.28",
            fc="white",
            ec="lightgray",
            alpha=0.92,
        ),
        zorder=20,
    )

def _format_path_string(path, roi_names=None, max_len=120):
    labels = [_safe_roi_label(n, roi_names=roi_names) for n in path]
    s = " → ".join(labels)
    if len(s) > max_len:
        s = s[:max_len] + " ..."
    return s


# =========================================================
# Function 3: plot_stage4_policy_lesion_path_graph
# =========================================================
def plot_stage4_policy_lesion_path_graph(
    lesion_core_edges: pd.DataFrame,
    lesion_idx: int,
    best_policy=None,
    best_policy_node=None,
    roi_names=None,
    direction: str = "auto",
    max_hops: int = 4,
    top_k_paths: int = 3,
    path_mode: str = "topk_union",   # "shortest" or "topk_union"
    score_col: str = "abs_val",
    score_mode: str = "mean",
    allow_undirected_fallback: bool = True,
    max_raw_paths: int = 500,
    figsize=(10.8, 5.6),
    edge_cmap_name="Blues",
    show_path_summary: bool = True,
    save_path: str = None,
):
    """
    Plot policy-lesion intervention path graph.

    path_mode:
        "shortest":
            draw one representative shortest path.

        "topk_union":
            draw the union of top-k paths. This shows all possible
            intermediate nodes appearing in selected policy-lesion paths.

    The graph is drawn using the full lesion_core_edges, not direct lesion edges.
    """

    # --------------------------------------------------
    # 1. Extract paths
    # --------------------------------------------------
    path_res = extract_policy_lesion_paths(
        lesion_core_edges=lesion_core_edges,
        lesion_idx=lesion_idx,
        best_policy=best_policy,
        best_policy_node=best_policy_node,
        direction=direction,
        max_hops=max_hops,
        top_k_paths=top_k_paths,
        score_col=score_col,
        score_mode=score_mode,
        path_mode=path_mode,
        allow_undirected_fallback=allow_undirected_fallback,
        max_raw_paths=max_raw_paths,
    )

    policy_node = int(path_res["best_policy_node"])
    lesion_idx = int(path_res["lesion_idx"])

    # --------------------------------------------------
    # 2. No path case
    # --------------------------------------------------
    if not path_res["has_path"]:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        ax.text(
            0.5,
            0.60,
            "No policy-lesion path found in the Stage 2 lesion-centered core graph.",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.42,
            f"policy node = {_safe_roi_label(policy_node, roi_names)}, "
            f"lesion node = {_safe_roi_label(lesion_idx, roi_names)}\n"
            f"direction = {direction}, max_hops = {max_hops}",
            ha="center",
            va="center",
            fontsize=10,
            color="dimgray",
        )

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir != "":
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

        return fig, ax, {
            "path_res": path_res,
            "path_edge_df": pd.DataFrame(),
            "node_roles": {},
            "pos": {},
        }

    selected_paths = path_res["selected_paths"]
    selected_scores = path_res["selected_path_scores"]

    # --------------------------------------------------
    # 3. Build path-union edge table
    # --------------------------------------------------
    path_edge_df = build_policy_lesion_path_subgraph(
        lesion_core_edges=lesion_core_edges,
        path_list=selected_paths,
        score_col=score_col,
    )

    # --------------------------------------------------
    # 4. Layout and graph
    # --------------------------------------------------
    pos = _build_policy_lesion_path_layer_layout(selected_paths)

    G = nx.DiGraph()
    for n in pos.keys():
        G.add_node(int(n))

    for _, r in path_edge_df.iterrows():
        u = int(r["src"])
        v = int(r["dst"])
        score = float(r[score_col]) if score_col in r and pd.notna(r[score_col]) else 0.0
        G.add_edge(
            u,
            v,
            score=score,
            path_count=int(r.get("path_count", 1)),
            is_reverse_traversal=bool(r.get("is_reverse_traversal", False)),
        )

    # --------------------------------------------------
    # 5. Node roles
    # --------------------------------------------------
    node_roles = {}
    for n in G.nodes():
        n = int(n)
        if n == policy_node:
            node_roles[n] = "policy"
        elif n == lesion_idx:
            node_roles[n] = "lesion"
        else:
            node_roles[n] = "bridge"

    # --------------------------------------------------
    # 6. Edge colors and widths
    # --------------------------------------------------
    edge_scores = np.array(
        [abs(G.edges[e].get("score", 0.0)) for e in G.edges()],
        dtype=float,
    )

    if len(edge_scores) == 0:
        edge_scores = np.array([0.0])

    vmax = float(np.max(edge_scores))
    if vmax <= 1e-12:
        vmax = 1.0

    edge_norm = Normalize(vmin=0.0, vmax=vmax)
    edge_cmap = plt.get_cmap(edge_cmap_name)

    edge_colors = []
    edge_widths = []
    edge_styles = []

    for e in G.edges():
        score = abs(float(G.edges[e].get("score", 0.0)))
        count = int(G.edges[e].get("path_count", 1))
        is_reverse = bool(G.edges[e].get("is_reverse_traversal", False))

        edge_colors.append(edge_cmap(edge_norm(score)))
        edge_widths.append(1.6 + 3.4 * edge_norm(score) + 0.35 * max(0, count - 1))
        edge_styles.append("--" if is_reverse else "-")

    # --------------------------------------------------
    # 7. Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Draw layer background
    if len(pos) > 0:
        max_x = max(float(p[0]) for p in pos.values())
        for layer in range(int(max_x) + 1):
            ax.axvline(layer, color="lightgray", lw=0.6, alpha=0.30, zorder=0)

    # Draw edges manually for better style control
    for idx, (u, v) in enumerate(G.edges()):
        p1 = np.asarray(pos[int(u)], dtype=float)
        p2 = np.asarray(pos[int(v)], dtype=float)

        # small curvature when there are multiple paths
        rad = 0.05
        if abs(p1[1] - p2[1]) > 1e-6:
            rad = 0.10 if p2[1] >= p1[1] else -0.10

        patch = FancyArrowPatch(
            posA=p1,
            posB=p2,
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            linewidth=edge_widths[idx],
            linestyle=edge_styles[idx],
            color=edge_colors[idx],
            alpha=0.88,
            mutation_scale=18,
            shrinkA=22,
            shrinkB=22,
            zorder=1,
        )
        ax.add_patch(patch)

    # Draw nodes
    for n, p in pos.items():
        n = int(n)
        role = node_roles.get(n, "bridge")

        if role == "policy":
            marker = "o"
            facecolor = "#FFD94A"
            edgecolor = "goldenrod"
            size = 1050
            lw = 2.8
        elif role == "lesion":
            marker = "o"
            facecolor = "#F4A6A6"
            edgecolor = "darkred"
            size = 1050
            lw = 2.8
        else:
            marker = "D"
            facecolor = "#9ECAE1"
            edgecolor = "#2B6CB0"
            size = 780
            lw = 1.8

        ax.scatter(
            [p[0]],
            [p[1]],
            s=size,
            marker=marker,
            c=[facecolor],
            edgecolors=edgecolor,
            linewidths=lw,
            zorder=3,
        )

    # Callout labels
    _draw_path_callout_labels(
        ax=ax,
        pos=pos,
        node_roles=node_roles,
        roi_names=roi_names,
        fontsize=8,
    )

    # Layer title labels
    # Layer title labels: put them above label lanes
    if len(pos) > 0:
        xs_unique = sorted(set(round(float(p[0]), 6) for p in pos.values()))
        ys_all = np.array([float(p[1]) for p in pos.values()])
        layer_label_y = float(np.max(ys_all)) + 1.05

        for layer_idx, x in enumerate(xs_unique):
            if layer_idx == 0:
                txt = "Source"
            elif layer_idx == len(xs_unique) - 1:
                txt = "Target"
            else:
                txt = f"Bridge layer {layer_idx}"

            ax.text(
                x,
                layer_label_y,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color="dimgray",
                fontweight="bold",
                zorder=12,
            )

    # Title
    title = (
        f"Policy-lesion causal path graph "
        f"({path_res['direction_used']}, {path_mode})"
    )
    if path_res["search_type"] == "undirected_fallback":
        title += " [undirected fallback]"

    ax.set_title(title, fontsize=13, pad=14)

    # Summary text
    # Path summary as a legend-like box
    if show_path_summary:
        _draw_path_summary_box(
            ax=ax,
            selected_paths=selected_paths,
            selected_scores=selected_scores,
            roi_names=roi_names,
            loc="lower right",  # 也可以改成 "lower left"
            fontsize=8.0,
        )

    # Legend
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Policy node",
            markerfacecolor="#FFD94A",
            markeredgecolor="goldenrod",
            markersize=10,
            markeredgewidth=2.0,
        ),
        Line2D(
            [0], [0],
            marker="D",
            color="w",
            label="Bridge node",
            markerfacecolor="#9ECAE1",
            markeredgecolor="#2B6CB0",
            markersize=9,
            markeredgewidth=1.6,
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Lesion node",
            markerfacecolor="#F4A6A6",
            markeredgecolor="darkred",
            markersize=10,
            markeredgewidth=2.0,
        ),
        Line2D(
            [0], [0],
            color=edge_cmap(edge_norm(vmax)),
            lw=3,
            label=f"Path edge ({score_col})",
        ),
        Line2D(
            [0], [0],
            color="gray",
            lw=2,
            linestyle="--",
            label="Reverse traversal in undirected fallback",
        ),
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
    )

    # Colorbar
    sm = cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.025)
    cbar.set_label(score_col, fontsize=10)

    # Limits: leave enough room for callout labels and layer titles
    if len(pos) > 0:
        xs = np.array([p[0] for p in pos.values()], dtype=float)
        ys = np.array([p[1] for p in pos.values()], dtype=float)

        ax.set_xlim(xs.min() - 0.95, xs.max() + 0.95)
        ax.set_ylim(ys.min() - 1.35, ys.max() + 1.55)

    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax, {
        "path_res": path_res,
        "path_edge_df": path_edge_df,
        "node_roles": node_roles,
        "pos": pos,
    }

# =========================================================
# 1) 给 directed core edge 表补上 Stage 4 的恢复信息
# =========================================================
def build_policy_lesion_recovery_edge_df(
    core_edge_df: pd.DataFrame,
    stage4_metrics: dict,
    weight_col: str = "val",
    abs_col: str = "abs_val",
) -> pd.DataFrame:
    """
    给 Stage 2 core edge 补充 Stage 4 recovery 信息。

    delta_fc_recovery =
        |FC_lesion - FC_healthy| - |FC_policy - FC_healthy|

    delta_fc_recovery > 0:
        policy 后更接近 healthy，表示恢复。

    delta_fc_recovery < 0:
        policy 后更远离 healthy，表示恶化。
    """
    import numpy as np
    import pandas as pd

    if core_edge_df is None or len(core_edge_df) == 0:
        raise ValueError("core_edge_df is empty.")

    required = ["FC_healthy", "FC_lesion", "FC_policy"]
    for k in required:
        if k not in stage4_metrics:
            raise KeyError(f"stage4_metrics must contain '{k}'.")

    FC_healthy = np.asarray(stage4_metrics["FC_healthy"], dtype=float)
    FC_lesion = np.asarray(stage4_metrics["FC_lesion"], dtype=float)
    FC_policy = np.asarray(stage4_metrics["FC_policy"], dtype=float)

    df = core_edge_df.copy()

    if "src" not in df.columns or "dst" not in df.columns:
        raise KeyError("core_edge_df must contain columns: src, dst")

    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)

    if weight_col not in df.columns:
        if "causal_val" in df.columns:
            weight_col = "causal_val"
        elif "val" in df.columns:
            weight_col = "val"
        else:
            df["val"] = 1.0
            weight_col = "val"

    if abs_col not in df.columns:
        df[abs_col] = np.abs(df[weight_col].astype(float))

    rows = []
    for _, r in df.iterrows():
        src = int(r["src"])
        dst = int(r["dst"])

        fc_h = float(FC_healthy[src, dst])
        fc_l = float(FC_lesion[src, dst])
        fc_p = float(FC_policy[src, dst])

        lesion_abn = abs(fc_l - fc_h)
        policy_abn = abs(fc_p - fc_h)

        row = dict(r)
        row.update({
            "src": src,
            "dst": dst,
            "val": float(r[weight_col]),
            "abs_val": float(r[abs_col]),
            "fc_healthy": fc_h,
            "fc_lesion": fc_l,
            "fc_policy": fc_p,
            "lesion_abn": lesion_abn,
            "policy_abn": policy_abn,
            "delta_fc_recovery": lesion_abn - policy_abn,
            "delta_fc_raw": fc_p - fc_l,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def _extract_policy_node_from_best_policy(best_policy):
    """
    兼容：
        best_policy = [{"node": ...}]
        best_policy = {"node": ...}
        best_policy = pd.Series(...)
    """
    import pandas as pd

    if best_policy is None:
        raise ValueError("best_policy is None.")

    if isinstance(best_policy, dict):
        return int(best_policy["node"])

    if isinstance(best_policy, pd.Series):
        return int(best_policy["node"])

    if isinstance(best_policy, (list, tuple)) and len(best_policy) > 0:
        return int(best_policy[0]["node"])

    raise TypeError(f"Unsupported best_policy type: {type(best_policy)}")


def _safe_node_name(n, roi_names=None):
    n = int(n)
    if roi_names is not None and 0 <= n < len(roi_names):
        return str(roi_names[n])
    return str(n)


def _format_path_text(path, roi_names=None, max_chars=92):
    names = [_safe_node_name(n, roi_names=roi_names) for n in path]
    s = " → ".join(names)
    if len(s) > max_chars:
        s = s[:max_chars] + " ..."
    return s


def _build_recovery_path_layer_layout(
    selected_paths,
    policy_node: int,
    lesion_idx: int,
    x_gap: float = 1.55,
    y_gap: float = 0.92,
):
    """
    根据 selected_paths 自动分层：
        policy -> bridge layer 1 -> bridge layer 2 -> lesion

    bridge layer 内节点自动纵向展开，减少节点/标签/连线重叠。
    """
    import numpy as np
    from collections import defaultdict

    policy_node = int(policy_node)
    lesion_idx = int(lesion_idx)

    if selected_paths is None or len(selected_paths) == 0:
        return {}, {}, {}

    max_bridge_depth = max(max(0, len(p) - 2) for p in selected_paths)
    target_layer = max_bridge_depth + 1

    node_layer = {
        policy_node: 0,
        lesion_idx: target_layer,
    }

    # 每个 bridge 节点放在它第一次出现的最早 bridge layer
    for path in selected_paths:
        path = [int(x) for x in path]
        for i, node in enumerate(path[1:-1], start=1):
            if node not in node_layer:
                node_layer[node] = i
            else:
                node_layer[node] = min(node_layer[node], i)

    layer_to_nodes = defaultdict(list)
    for node, layer in node_layer.items():
        layer_to_nodes[int(layer)].append(int(node))

    # 让主路径节点优先靠近中间
    main_path = [int(x) for x in selected_paths[0]]
    main_order = {n: i for i, n in enumerate(main_path)}

    pos = {}
    for layer in sorted(layer_to_nodes.keys()):
        nodes = sorted(layer_to_nodes[layer], key=lambda n: main_order.get(n, 999))

        x = float(layer) * x_gap

        if len(nodes) == 1:
            ys = [0.0]
        else:
            # 自动展开：节点越多，纵向跨度越大
            span = max(y_gap * (len(nodes) - 1), 1.15)
            ys = np.linspace(span / 2, -span / 2, len(nodes))

        for node, y in zip(nodes, ys):
            pos[int(node)] = np.array([x, float(y)])

    layer_names = {0: "Source"}
    for layer in range(1, target_layer):
        layer_names[layer] = f"Bridge layer {layer}"
    layer_names[target_layer] = "Target"

    return pos, node_layer, layer_names


def _draw_recovery_path_labels(
    ax,
    pos,
    policy_node: int,
    lesion_idx: int,
    roi_names=None,
    fontsize: int = 8,
):
    """
    标签使用方框 + 连线。
    标签放在上/下 label lane，不压在路径线上。
    """
    import numpy as np

    if len(pos) == 0:
        return

    ys = np.array([float(p[1]) for p in pos.values()])
    top_lane = float(np.max(ys)) + 0.58
    bottom_lane = float(np.min(ys)) - 0.58

    top_count = {}
    bottom_count = {}

    for node, p in pos.items():
        node = int(node)
        x, y = float(p[0]), float(p[1])

        if node == int(policy_node):
            label = f"Policy\n{_safe_node_name(node, roi_names)}"
            xytext = (x, top_lane)
            ha, va = "center", "bottom"
            lane_key = ("top", round(x, 3))
        elif node == int(lesion_idx):
            label = f"Lesion\n{_safe_node_name(node, roi_names)}"
            xytext = (x, top_lane)
            ha, va = "center", "bottom"
            lane_key = ("top", round(x, 3))
        else:
            label = f"Bridge\n{_safe_node_name(node, roi_names)}"

            if y >= 0:
                k = top_count.get(round(x, 3), 0)
                top_count[round(x, 3)] = k + 1
                xytext = (x, top_lane + 0.16 * k)
                ha, va = "center", "bottom"
            else:
                k = bottom_count.get(round(x, 3), 0)
                bottom_count[round(x, 3)] = k + 1
                xytext = (x, bottom_lane - 0.16 * k)
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
                fc="white",
                ec="lightgray",
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
            zorder=20,
        )


def _draw_compact_path_summary_box(
    ax,
    selected_paths,
    selected_path_scores,
    roi_names=None,
    loc: str = "lower right",
    fontsize: float = 8.0,
):
    """
    路径说明框，紧凑放在左下角或右下角。
    """
    if selected_paths is None or len(selected_paths) == 0:
        return

    lines = []
    for i, (path, score) in enumerate(zip(selected_paths, selected_path_scores), start=1):
        path_text = _format_path_text(path, roi_names=roi_names, max_chars=80)
        lines.append(f"P{i}: {path_text} | score={score:.4f}")

    txt = "\n".join(lines)

    if loc == "lower left":
        x, y = 0.02, 0.04
        ha, va = "left", "bottom"
    else:
        x, y = 0.98, 0.04
        ha, va = "right", "bottom"

    ax.text(
        x,
        y,
        txt,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color="dimgray",
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec="lightgray",
            alpha=0.92,
        ),
        zorder=30,
    )


def plot_stage4_policy_lesion_recovery_path_graph(
    core_edge_df: pd.DataFrame,
    stage4_metrics: dict,
    best_policy,
    lesion_idx: int,
    roi_names=None,
    direction: str = "policy_to_lesion",
    path_mode: str = "topk_union",   # "shortest" or "topk_union"
    top_k_paths: int = 4,
    max_hops: int = 5,
    score_col: str = "abs_val",
    score_mode: str = "mean",
    allow_undirected_fallback: bool = True,
    figsize=(11.8, 5.8),
    summary_loc: str = "lower right",
    save_path: str = None,
):
    """
    Policy-lesion recovery path graph.

    路径结构：
        仍然由 policy-lesion causal path 决定。

    边颜色：
        delta_fc_recovery

        蓝色：恢复
        红色：恶化
        白色附近：变化弱
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D

    policy_node = _extract_policy_node_from_best_policy(best_policy)
    lesion_idx = int(lesion_idx)

    # --------------------------------------------------
    # 1. 补 Stage 4 recovery edge metric
    # --------------------------------------------------
    edge_metric_df = build_policy_lesion_recovery_edge_df(
        core_edge_df=core_edge_df,
        stage4_metrics=stage4_metrics,
        weight_col="val",
        abs_col="abs_val",
    )

    # --------------------------------------------------
    # 2. 路径仍然按 causal abs_val 提取
    # --------------------------------------------------
    path_res = extract_policy_lesion_paths(
        lesion_core_edges=edge_metric_df,
        lesion_idx=lesion_idx,
        best_policy=best_policy,
        direction=direction,
        max_hops=max_hops,
        top_k_paths=top_k_paths,
        score_col=score_col,
        score_mode=score_mode,
        path_mode=path_mode,
        allow_undirected_fallback=allow_undirected_fallback,
    )

    selected_paths = path_res["selected_paths"]
    selected_scores = path_res["selected_path_scores"]

    if not path_res["has_path"] or len(selected_paths) == 0:
        raise ValueError("No policy-lesion path found. Try larger max_hops or allow_undirected_fallback=True.")

    # --------------------------------------------------
    # 3. 路径并集边表
    # --------------------------------------------------
    path_edge_df = build_policy_lesion_path_subgraph(
        lesion_core_edges=edge_metric_df,
        path_list=selected_paths,
        score_col=score_col,
    )

    if len(path_edge_df) == 0:
        raise ValueError("No path edge found after building path subgraph.")

    # --------------------------------------------------
    # 4. 分层布局
    # --------------------------------------------------
    pos, node_layer, layer_names = _build_recovery_path_layer_layout(
        selected_paths=selected_paths,
        policy_node=policy_node,
        lesion_idx=lesion_idx,
    )

    # --------------------------------------------------
    # 5. 颜色映射：蓝色恢复，红色恶化
    # --------------------------------------------------
    delta_vals = path_edge_df["delta_fc_recovery"].astype(float).values
    vmax = float(np.max(np.abs(delta_vals))) if len(delta_vals) > 0 else 1.0
    vmax = max(vmax, 1e-12)

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # 关键：RdBu 是负值红，正值蓝
    cmap = plt.get_cmap("RdBu")

    max_abs = float(path_edge_df["abs_val"].max()) if "abs_val" in path_edge_df.columns else 1.0
    max_abs = max(max_abs, 1e-12)

    # --------------------------------------------------
    # 6. 画图
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # layer guide lines and titles
    max_layer = max(node_layer.values())
    y_top = max(float(p[1]) for p in pos.values()) + 1.05

    for layer in range(max_layer + 1):
        xs = [float(p[0]) for n, p in pos.items() if node_layer[int(n)] == layer]
        if len(xs) == 0:
            continue
        x_layer = xs[0]

        ax.axvline(
            x_layer,
            color="lightgray",
            lw=0.7,
            alpha=0.32,
            zorder=0,
        )

        ax.text(
            x_layer,
            y_top,
            layer_names.get(layer, f"Layer {layer}"),
            ha="center",
            va="center",
            fontsize=9,
            color="dimgray",
            fontweight="bold",
        )

    # draw edges
    for i, (_, r) in enumerate(path_edge_df.iterrows()):
        src = int(r["src"])
        dst = int(r["dst"])

        if src not in pos or dst not in pos:
            continue

        p1 = np.asarray(pos[src], dtype=float)
        p2 = np.asarray(pos[dst], dtype=float)

        delta = float(r["delta_fc_recovery"])
        abs_val = float(r.get("abs_val", 0.0))

        color = cmap(norm(delta))
        width = 1.6 + 4.2 * abs_val / max_abs

        # 同层或相近 y 的边微弯，避免重叠
        rad = 0.05
        if abs(p2[1] - p1[1]) < 1e-6:
            rad = 0.12 if i % 2 == 0 else -0.12
        elif p2[1] > p1[1]:
            rad = 0.08
        else:
            rad = -0.08

        patch = FancyArrowPatch(
            posA=p1,
            posB=p2,
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            linewidth=width,
            color=color,
            alpha=0.92,
            mutation_scale=18,
            shrinkA=24,
            shrinkB=24,
            zorder=2,
        )
        ax.add_patch(patch)

    # draw nodes
    nodes = sorted(pos.keys())
    bridge_nodes = set()
    for path in selected_paths:
        bridge_nodes.update([int(x) for x in path[1:-1]])

    for n in nodes:
        x, y = pos[int(n)]

        if int(n) == int(policy_node):
            marker = "o"
            size = 980
            face = "#FFD94A"
            edge = "goldenrod"
            lw = 2.6
        elif int(n) == int(lesion_idx):
            marker = "o"
            size = 980
            face = "#F5B5B5"
            edge = "darkred"
            lw = 2.6
        else:
            marker = "D"
            size = 740
            face = "#A6CEE3"
            edge = "#2B6CB0"
            lw = 1.9

        ax.scatter(
            [x],
            [y],
            s=size,
            marker=marker,
            c=[face],
            edgecolors=edge,
            linewidths=lw,
            zorder=5,
        )

    # labels
    _draw_recovery_path_labels(
        ax=ax,
        pos=pos,
        policy_node=policy_node,
        lesion_idx=lesion_idx,
        roi_names=roi_names,
        fontsize=8,
    )

    # path summary box
    _draw_compact_path_summary_box(
        ax=ax,
        selected_paths=selected_paths,
        selected_path_scores=selected_scores,
        roi_names=roi_names,
        loc=summary_loc,
        fontsize=8.0,
    )

    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.025)
    cbar.set_label(r"$\Delta FC$ recovery", fontsize=10)

    # legend
    recovered_color = cmap(norm(vmax))
    worsened_color = cmap(norm(-vmax))

    handles = [
        Line2D([0], [0], marker="o", color="w", label="Policy node",
               markerfacecolor="#FFD94A", markeredgecolor="goldenrod",
               markeredgewidth=2, markersize=10),
        Line2D([0], [0], marker="D", color="w", label="Bridge node",
               markerfacecolor="#A6CEE3", markeredgecolor="#2B6CB0",
               markeredgewidth=1.8, markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Lesion node",
               markerfacecolor="#F5B5B5", markeredgecolor="darkred",
               markeredgewidth=2, markersize=10),
        Line2D([0], [0], color=recovered_color, lw=3.0, label="Recovered edge"),
        Line2D([0], [0], color=worsened_color, lw=3.0, label="Worsened edge"),
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
    )

    # title
    if path_mode == "shortest":
        mode_title = "shortest path"
    else:
        mode_title = f"top-{top_k_paths} path union"

    ax.set_title(
        f"Policy-lesion recovery path graph ({mode_title})",
        fontsize=13,
        pad=14,
    )

    ax.axis("off")

    xs = np.array([float(p[0]) for p in pos.values()])
    ys = np.array([float(p[1]) for p in pos.values()])

    ax.set_xlim(xs.min() - 0.85, xs.max() + 0.85)
    ax.set_ylim(ys.min() - 1.10, ys.max() + 1.65)

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    info = {
        "path_res": path_res,
        "selected_paths": selected_paths,
        "selected_path_scores": selected_scores,
        "selected_edge_df": path_edge_df.reset_index(drop=True),
        "edge_metric_df": edge_metric_df.reset_index(drop=True),
        "pos": pos,
        "node_layer": node_layer,
        "policy_node": int(policy_node),
        "lesion_idx": int(lesion_idx),
        "path_mode": path_mode,
        "delta_fc_recovery_min": float(np.min(delta_vals)),
        "delta_fc_recovery_max": float(np.max(delta_vals)),
        "n_paths": len(selected_paths),
        "n_edges": len(path_edge_df),
    }

    return fig, ax, info


def plot_stage4_policy_lesion_recovery_path_graph_pair(
    core_edge_df: pd.DataFrame,
    stage4_metrics: dict,
    best_policy,
    lesion_idx: int,
    roi_names=None,
    direction: str = "auto",
    top_k_paths: int = 4,
    max_hops: int = 5,
    save_prefix: str = "./results/stage4_policy_lesion_recovery_path",
):
    """
    一次性画两张图：
        1. shortest path recovery graph
        2. top-k union recovery graph
    """
    import os

    if save_prefix is not None:
        save_dir = os.path.dirname(save_prefix)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

    fig_short, ax_short, info_short = plot_stage4_policy_lesion_recovery_path_graph(
        core_edge_df=core_edge_df,
        stage4_metrics=stage4_metrics,
        best_policy=best_policy,
        lesion_idx=lesion_idx,
        roi_names=roi_names,
        direction=direction,
        path_mode="shortest",
        top_k_paths=1,
        max_hops=max_hops,
        summary_loc="lower right",
        save_path=None if save_prefix is None else save_prefix + "_shortest.png",
    )

    fig_union, ax_union, info_union = plot_stage4_policy_lesion_recovery_path_graph(
        core_edge_df=core_edge_df,
        stage4_metrics=stage4_metrics,
        best_policy=best_policy,
        lesion_idx=lesion_idx,
        roi_names=roi_names,
        direction=direction,
        path_mode="topk_union",
        top_k_paths=top_k_paths,
        max_hops=max_hops,
        summary_loc="lower right",
        save_path=None if save_prefix is None else save_prefix + "_topk_union.png",
    )

    return {
        "shortest": {
            "fig": fig_short,
            "ax": ax_short,
            "info": info_short,
        },
        "topk_union": {
            "fig": fig_union,
            "ax": ax_union,
            "info": info_union,
        },
    }

