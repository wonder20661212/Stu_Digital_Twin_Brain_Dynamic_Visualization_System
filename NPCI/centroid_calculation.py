import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

dlabel_path = r"./data/derived/MNI-MMP_centroid/HCP_MMP1.0_Glasser.32k_fs_LR.dlabel.nii"
left_surf_path = r"./data/derived/MNI-MMP_centroid/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
right_surf_path = r"./data/derived/MNI-MMP_centroid/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii"

out_csv_path = r"./data/derived/MNI-MMP_centroid/mmp360_centroids.csv"



def standardize_mmp_name(name: str) -> str:
    """
    Standardize HCP-MMP ROI names.
    L_V1_ROI  -> L_V1
    R_3b_ROI  -> R_3b
    """
    name = str(name).strip()
    # 去掉 dlabel 里的后缀
    if name.endswith("_ROI"):
        name = name[:-4]
    # 有些文件可能带空格
    name = name.replace(" ", "")
    return name


def load_surface_vertices(surf_gii_path):
    """
    Read GIFTI surface file and return vertex coordinates.
    Expected shape: (n_vertices, 3)
    """
    gii = nib.load(surf_gii_path)
    coords = gii.darrays[0].data.astype(float)
    return coords


# 读取 CIFTI dlabel，并提取左右半球每个 vertex 的 label
def load_cifti_surface_labels(dlabel_path):
    """
    Load CIFTI dlabel file and extract:
      - left hemisphere vertex labels
      - right hemisphere vertex labels
      - label table (id -> name)

    Returns
    -------
    left_labels : np.ndarray, shape (n_left_vertices_used,)
    right_labels: np.ndarray, shape (n_right_vertices_used,)
    left_vertex_indices : np.ndarray
        indices in 32k surface for left cortex
    right_vertex_indices : np.ndarray
        indices in 32k surface for right cortex
    label_dict : dict
        label_id -> label_name
    """
    img = nib.load(dlabel_path)
    data = np.asarray(img.dataobj).squeeze()  # shape ~ (59412,)
    hdr = img.header

    # CIFTI axes
    # axis 0: label axis
    # axis 1: brain models axis
    label_axis = hdr.get_axis(0)
    brain_axis = hdr.get_axis(1)

    # label table: id -> name
    label_dict = {}
    # label_axis.label is typically a list/dict structure
    if hasattr(label_axis, "label"):
        # Most often label_axis.label[0] is a dict {label_id: (name, rgba...)}
        raw_label_table = label_axis.label[0]
        for k, v in raw_label_table.items():
            try:
                label_name = v[0]
            except Exception:
                label_name = str(v)
            label_dict[int(k)] = str(label_name)

    left_labels = None
    right_labels = None
    left_vertex_indices = None
    right_vertex_indices = None

    offset = 0
    for bm in brain_axis.iter_structures():
        structure_name, slc, subaxis = bm
        subdata = data[slc]

        # vertex indices in corresponding 32k mesh
        vertex_indices = subaxis.vertex

        if structure_name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            left_labels = subdata.astype(int)
            left_vertex_indices = np.asarray(vertex_indices, dtype=int)

        elif structure_name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            right_labels = subdata.astype(int)
            right_vertex_indices = np.asarray(vertex_indices, dtype=int)

    if left_labels is None or right_labels is None:
        raise ValueError("Failed to extract left/right cortex labels from dlabel file.")

    return left_labels, right_labels, left_vertex_indices, right_vertex_indices, label_dict


# 计算 360 个 ROI centroid
def compute_mmp360_centroids(dlabel_path, left_surf_path, right_surf_path):
    """
    Compute centroid (mean xyz) for each labeled ROI in HCP-MMP1.0.

    Returns
    -------
    centroid_df : pd.DataFrame
        columns:
            label_id, roi_name, hemi, x, y, z, n_vertices
    """
    # load data
    left_coords = load_surface_vertices(left_surf_path)   # (32492, 3)
    right_coords = load_surface_vertices(right_surf_path) # (32492, 3)

    left_labels, right_labels, left_vertex_indices, right_vertex_indices, label_dict = \
        load_cifti_surface_labels(dlabel_path)

    rows = []

    # ----- left hemisphere -----
    unique_left = np.unique(left_labels)
    for lab in unique_left:
        if lab == 0:
            continue  # background / medial wall
        roi_name = label_dict.get(int(lab), f"Label_{lab}")

        mask = (left_labels == lab)
        used_vertices = left_vertex_indices[mask]
        xyz = left_coords[used_vertices]

        if xyz.shape[0] == 0:
            continue

        centroid = xyz.mean(axis=0)
        rows.append({
            "label_id": int(lab),
            "roi_name": roi_name,
            "hemi": "L",
            "x": float(centroid[0]),
            "y": float(centroid[1]),
            "z": float(centroid[2]),
            "n_vertices": int(xyz.shape[0]),
        })

    # ----- right hemisphere -----
    unique_right = np.unique(right_labels)
    for lab in unique_right:
        if lab == 0:
            continue
        roi_name = label_dict.get(int(lab), f"Label_{lab}")

        mask = (right_labels == lab)
        used_vertices = right_vertex_indices[mask]
        xyz = right_coords[used_vertices]

        if xyz.shape[0] == 0:
            continue

        centroid = xyz.mean(axis=0)
        rows.append({
            "label_id": int(lab),
            "roi_name": roi_name,
            "hemi": "R",
            "x": float(centroid[0]),
            "y": float(centroid[1]),
            "z": float(centroid[2]),
            "n_vertices": int(xyz.shape[0]),
        })

    centroid_df = pd.DataFrame(rows)

    # 过滤掉明显非 ROI 的项
    # 常见背景名可能是 ??? 或 UNKNOWN，可按需加规则
    centroid_df = centroid_df[centroid_df["roi_name"].notna()].copy()

    return centroid_df


def _get_roi_name_col(df: pd.DataFrame):
    """
    Pick ROI name column automatically.
    """
    for c in ["roi_name", "roi_name_stage1", "roi_name_std", "node_label"]:
        if c in df.columns:
            return c
    return None


def _build_numbered_label_df(
    df: pd.DataFrame,
    value_col: str,
    lesion_idx: int = None,
    top_k_label: int = 5,
):
    """
    Build a dataframe for numbered ROI annotations.

    Returns
    -------
    label_df:
        columns include:
            roi, roi_name, value_col, label_id, is_lesion
    legend_lines:
        list[str], for side text legend
    """
    name_col = _get_roi_name_col(df)

    if name_col is None:
        df = df.copy()
        df["roi_name_fallback"] = df["roi"].astype(int).astype(str)
        name_col = "roi_name_fallback"

    df = df.copy()
    df["roi"] = df["roi"].astype(int)

    selected_rows = []

    # 1) lesion ROI first
    if lesion_idx is not None and "roi" in df.columns:
        lesion_rows = df[df["roi"] == int(lesion_idx)]
        if len(lesion_rows) > 0:
            r = lesion_rows.iloc[0].copy()
            r["is_lesion"] = True
            selected_rows.append(r)

    # 2) top-k response ROI (exclude lesion if duplicated)
    top_df = df.nlargest(int(top_k_label), value_col).copy()

    used_roi = set()
    for r in selected_rows:
        used_roi.add(int(r["roi"]))

    for _, r in top_df.iterrows():
        roi = int(r["roi"])
        if roi in used_roi:
            continue
        rr = r.copy()
        rr["is_lesion"] = False
        selected_rows.append(rr)
        used_roi.add(roi)

    if len(selected_rows) == 0:
        return pd.DataFrame(), []

    label_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    label_df["label_id"] = np.arange(1, len(label_df) + 1)

    legend_lines = []
    for _, r in label_df.iterrows():
        idx = int(r["label_id"])
        roi_name = str(r[name_col])
        if bool(r.get("is_lesion", False)):
            legend_lines.append(f"{idx} = lesion: {roi_name}")
        else:
            legend_lines.append(f"{idx} = {roi_name}")

    return label_df, legend_lines


def _annotate_numbered_nodes_to_nearest_edge(
    ax,
    label_df: pd.DataFrame,
    xcol: str = "x",
    ycol: str = "z",
    label_col: str = None,
    edge_inset: float = 0.035,      # 编号距离坐标轴边缘的内部距离
    min_sep: float = 0.055,
    number_fontsize: int = 8,
    line_color: str = "gray",
    circle_ec: str = "gray",
    circle_fc: str = "white",
    lesion_ec: str = "blue",
):
    """
    Put numbered labels inside the plotting box, close to the nearest border,
    and connect them to the corresponding ROI node.

    编号位置：
        - 不放到坐标轴外面
        - 放在图内最近边缘附近
        - 用连线指向对应 ROI 节点

    label_df should contain:
        xcol, ycol, and either label_id or label_num
    """
    import numpy as np

    if label_df is None or len(label_df) == 0:
        return

    label_df = label_df.copy().reset_index(drop=True)

    if label_col is None:
        if "label_id" in label_df.columns:
            label_col = "label_id"
        elif "label_num" in label_df.columns:
            label_col = "label_num"
        else:
            raise KeyError("label_df must contain 'label_id' or 'label_num'.")

    # --------------------------------------------------
    # 1. data 坐标 -> axes fraction 坐标
    # --------------------------------------------------
    points = []

    for _, r in label_df.iterrows():
        x_data = float(r[xcol])
        y_data = float(r[ycol])

        # data -> display -> axes fraction
        x_disp, y_disp = ax.transData.transform((x_data, y_data))
        x_frac, y_frac = ax.transAxes.inverted().transform((x_disp, y_disp))

        # 限制到图内，避免极端点导致判断异常
        x_frac = float(np.clip(x_frac, 0.0, 1.0))
        y_frac = float(np.clip(y_frac, 0.0, 1.0))

        # 距离四条边的距离
        distances = {
            "left": x_frac,
            "right": 1.0 - x_frac,
            "bottom": y_frac,
            "top": 1.0 - y_frac,
        }

        side = min(distances, key=distances.get)

        # --------------------------------------------------
        # 关键修改：
        # 编号放在图内边缘，而不是图外
        # --------------------------------------------------
        if side == "left":
            tx = edge_inset
            ty = np.clip(y_frac, edge_inset, 1.0 - edge_inset)
            edge_coord = ty

        elif side == "right":
            tx = 1.0 - edge_inset
            ty = np.clip(y_frac, edge_inset, 1.0 - edge_inset)
            edge_coord = ty

        elif side == "top":
            tx = np.clip(x_frac, edge_inset, 1.0 - edge_inset)
            ty = 1.0 - edge_inset
            edge_coord = tx

        else:  # bottom
            tx = np.clip(x_frac, edge_inset, 1.0 - edge_inset)
            ty = edge_inset
            edge_coord = tx

        points.append({
            "row": r,
            "side": side,
            "x_data": x_data,
            "y_data": y_data,
            "tx": tx,
            "ty": ty,
            "edge_coord": edge_coord,
        })

    # --------------------------------------------------
    # 2. 同一条边上的编号做轻微防重叠
    # --------------------------------------------------
    def _spread(vals, lo, hi, min_sep):
        vals = np.asarray(vals, dtype=float)

        if len(vals) <= 1:
            return vals

        order = np.argsort(vals)
        sorted_vals = vals[order].copy()

        # forward pass
        for i in range(1, len(sorted_vals)):
            if sorted_vals[i] - sorted_vals[i - 1] < min_sep:
                sorted_vals[i] = sorted_vals[i - 1] + min_sep

        # 如果超过上界，整体往回移
        if sorted_vals[-1] > hi:
            sorted_vals -= sorted_vals[-1] - hi

        # backward pass
        for i in range(len(sorted_vals) - 2, -1, -1):
            if sorted_vals[i + 1] - sorted_vals[i] < min_sep:
                sorted_vals[i] = sorted_vals[i + 1] - min_sep

        # 如果低于下界，整体往上移
        if sorted_vals[0] < lo:
            sorted_vals += lo - sorted_vals[0]

        sorted_vals = np.clip(sorted_vals, lo, hi)

        out = vals.copy()
        out[order] = sorted_vals
        return out

    lo = edge_inset
    hi = 1.0 - edge_inset

    for side in ["left", "right", "top", "bottom"]:
        idxs = [i for i, p in enumerate(points) if p["side"] == side]

        if len(idxs) <= 1:
            continue

        coords = [points[i]["edge_coord"] for i in idxs]
        coords_new = _spread(coords, lo=lo, hi=hi, min_sep=min_sep)

        for i, cnew in zip(idxs, coords_new):
            if side in ["left", "right"]:
                points[i]["ty"] = float(cnew)
            else:
                points[i]["tx"] = float(cnew)

    # --------------------------------------------------
    # 3. 画编号和连接线
    # --------------------------------------------------
    for p in points:
        r = p["row"]

        is_lesion = bool(r.get("is_lesion", False))

        ec = lesion_ec if is_lesion else circle_ec
        text_color = lesion_ec if is_lesion else "black"

        ax.annotate(
            str(int(r[label_col])),
            xy=(p["x_data"], p["y_data"]),
            xycoords="data",
            xytext=(p["tx"], p["ty"]),
            textcoords=ax.transAxes,
            ha="center",
            va="center",
            fontsize=number_fontsize,
            color=text_color,
            bbox=dict(
                boxstyle="circle,pad=0.20",
                fc=circle_fc,
                ec=ec,
                lw=1.0,
                alpha=0.96,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=ec if is_lesion else line_color,
                lw=0.8,
                shrinkA=4,
                shrinkB=4,
                alpha=0.90,
                connectionstyle="arc3,rad=0.0",
            ),
            annotation_clip=False,
            zorder=8,
        )

        # lesion 节点额外加蓝圈
        if is_lesion:
            ax.scatter(
                [p["x_data"]],
                [p["y_data"]],
                s=260,
                facecolors="none",
                edgecolors=lesion_ec,
                linewidths=2.0,
                zorder=7,
            )


def _add_side_legend_text(
    fig,
    legend_lines,
    x: float = 0.84,
    y: float = 0.82,
    fontsize: int = 8,
    title: str = "ROI labels",
):
    """
    Add number-to-ROI mapping text on the right side of the figure.
    """
    if legend_lines is None or len(legend_lines) == 0:
        return

    text = title + "\n" + "\n".join(legend_lines)

    fig.text(
        x,
        y,
        text,
        ha="left",
        va="top",
        fontsize=fontsize,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="lightgray",
            alpha=0.95,
        ),
    )

def plot_stage1_lesion_response_brainmap(
    brainmap_df: pd.DataFrame,
    value_col: str = "response_abs",
    lesion_idx: int = None,
    top_k_label: int = 5,
    figsize=(15.5, 4.8),
    cmap: str = "Reds",
    size_scale: float = 260.0,
    size_min: float = 18.0,
    annotate_only_xz: bool = True,
    label_fontsize: int = 8,
    side_legend_fontsize: int = 8,
    save_path: str = None,
):
    """
    Three-view lesion response brain map with:
      - numbered ROI labels
      - offset annotation + connector lines
      - side legend mapping number -> roi_name
      - default: annotate only x-z main view
    """
    df = brainmap_df.dropna(subset=["x", "y", "z", value_col]).copy()
    df["roi"] = df["roi"].astype(int)

    values = df[value_col].values.astype(float)
    if len(values) == 0:
        raise ValueError("No valid rows to plot.")

    # node sizes
    absv = np.abs(values)
    if absv.max() < 1e-12:
        sizes = np.full_like(absv, size_min, dtype=float)
    else:
        sizes = size_min + size_scale * absv / absv.max()

    # norm
    if value_col == "response_abs":
        vmin, vmax = 0.0, float(np.max(values))
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        vmax = float(np.max(np.abs(values)))
        norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # label set
    label_df, legend_lines = _build_numbered_label_df(
        df=df,
        value_col=value_col,
        lesion_idx=lesion_idx,
        top_k_label=top_k_label,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # leave room for side legend + colorbar
    fig.subplots_adjust(left=0.05, right=0.82, bottom=0.12, top=0.85, wspace=0.18)

    pairs = [
        ("x", "y", "x-y"),
        ("y", "z", "y-z"),
        ("x", "z", "x-z"),
    ]

    sc_ref = None
    for ax, (a, b, title_suffix) in zip(axes, pairs):
        sc = ax.scatter(
            df[a],
            df[b],
            c=values,
            s=sizes,
            cmap=cmap,
            norm=norm,
            alpha=0.85,
            edgecolor="k",
            linewidth=0.25,
            zorder=2,
        )
        sc_ref = sc

        ax.set_title(f"Stage 1 lesion response map ({title_suffix})", fontsize=12)
        ax.set_xlabel(a, fontsize=10)
        ax.set_ylabel(b, fontsize=10)
        ax.grid(alpha=0.18, linestyle="--")

        # only annotate x-z by default
        if annotate_only_xz:
            if (a, b) == ("x", "z"):
                tmp = label_df.copy()
                _annotate_numbered_nodes_to_nearest_edge(
                    ax,
                    label_df,
                    xcol="x",
                    ycol="z",
                    number_fontsize=label_fontsize,
                    edge_inset=0.035,
                    min_sep=0.055,
                )
        else:
            tmp = label_df.copy()
            _annotate_numbered_nodes_to_nearest_edge(
                ax,
                label_df,
                xcol="x",
                ycol="z",
                number_fontsize=label_fontsize,
                edge_inset=0.035,
                min_sep=0.055,
            )

    # outer colorbar
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    cbar = fig.colorbar(sc_ref, cax=cax)
    cbar.set_label(value_col, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # side legend
    _add_side_legend_text(
        fig,
        legend_lines,
        x=0.83,
        y=0.82,
        fontsize=side_legend_fontsize,
        title="ROI labels",
    )

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()
    return fig, axes



def plot_stage1_centroid_brainmap_paper(
    brainmap_df: pd.DataFrame,
    value_col: str = "response_abs",
    lesion_idx: int = None,
    top_k_label: int = 5,
    figsize=(8.2, 5.8),
    cmap: str = None,
    size_scale: float = 260.0,
    size_min: float = 20.0,
    size_max: float = 160.0,
    label_fontsize: int = 8,
    side_legend_fontsize: int = 8,
    title: str = "Stage 1 lesion-response map",
    save_path: str = None,
):
    """
    Single x-z main-view plot with:
      - numbered ROI labels
      - offset annotation + connector lines
      - side legend mapping number -> roi_name
      - correct colorbar normalization
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.colors import Normalize

    df = brainmap_df.dropna(subset=["x", "z", value_col]).copy()
    df["roi"] = df["roi"].astype(int)

    values = df[value_col].astype(float).values

    if len(values) == 0:
        raise ValueError("No valid rows to plot.")

    # --------------------------------------------------
    # 1. node size
    # --------------------------------------------------
    absv = np.abs(values)

    if np.nanmax(absv) < 1e-12:
        sizes = np.full_like(absv, size_min, dtype=float)
    else:
        sizes = size_min + size_scale * absv / np.nanmax(absv)
        sizes = np.clip(sizes, size_min, size_max)

    # --------------------------------------------------
    # 2. color norm：这里是关键修改
    # --------------------------------------------------
    if value_col == "response_abs":
        # 绝对响应：必须从 0 开始
        vmax = float(np.nanmax(values))

        if vmax <= 1e-12:
            vmax = 1.0

        norm = Normalize(vmin=0.0, vmax=vmax)

        if cmap is None:
            cmap = "Reds"

    else:
        # signed response：必须以 0 为中心
        vmax = float(np.nanmax(np.abs(values)))

        if vmax <= 1e-12:
            vmax = 1.0

        norm = colors.TwoSlopeNorm(
            vmin=-vmax,
            vcenter=0.0,
            vmax=vmax
        )

        if cmap is None:
            cmap = "RdBu_r"

    # --------------------------------------------------
    # 3. labels
    # --------------------------------------------------
    label_df, legend_lines = _build_numbered_label_df(
        df=df,
        value_col=value_col,
        lesion_idx=lesion_idx,
        top_k_label=top_k_label,
    )

    # --------------------------------------------------
    # 4. figure layout
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # 左边给主图，右边给 colorbar + ROI label table
    fig.subplots_adjust(
        left=0.10,
        right=0.78,
        bottom=0.12,
        top=0.88
    )

    # --------------------------------------------------
    # 5. scatter：这里必须用同一个 values + norm
    # --------------------------------------------------
    sc = ax.scatter(
        df["x"].values,
        df["z"].values,
        c=values,
        s=sizes,
        cmap=cmap,
        norm=norm,
        alpha=0.90,
        edgecolor="k",
        linewidth=0.25,
        zorder=2,
    )

    # --------------------------------------------------
    # 6. numbered annotation
    # --------------------------------------------------
    _annotate_numbered_nodes_to_nearest_edge(
        ax,
        label_df,
        xcol="x",
        ycol="z",
        number_fontsize=label_fontsize,
        edge_inset=0.035,
        min_sep=0.055,
    )

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("z", fontsize=10)
    ax.grid(alpha=0.18, linestyle="--")

    # --------------------------------------------------
    # 7. outer colorbar：必须绑定 scatter 返回的 sc
    # --------------------------------------------------
    cax = fig.add_axes([0.80, 0.20, 0.020, 0.60])
    cbar = fig.colorbar(sc, cax=cax)

    if value_col == "response_abs":
        cbar.set_label("response_abs", fontsize=10)
    else:
        cbar.set_label("response_signed", fontsize=10)

    cbar.ax.tick_params(labelsize=8)

    # --------------------------------------------------
    # 8. side legend
    # --------------------------------------------------
    _add_side_legend_text(
        fig,
        legend_lines,
        x=0.83,
        y=0.82,
        fontsize=side_legend_fontsize,
        title="ROI labels",
    )

    # --------------------------------------------------
    # 9. save
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
