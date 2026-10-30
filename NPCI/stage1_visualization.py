import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import os
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



def standardize_mmp_name(name: str) -> str:
    """
    Standardize HCP-MMP ROI name.
    Examples
    L_V1_ROI  -> L_V1
    """
    name = str(name).strip()
    if name.endswith("_ROI"):
        name = name[:-4]
    name = name.replace(" ", "")
    return name


def get_stage1_roi_name_col(df: pd.DataFrame) -> str:
    """
    Find the most appropriate ROI name column from a merged dataframe.
    """
    for col in ["roi_name", "roi_name_stage1", "roi_name_x", "roi_name_std"]:
        if col in df.columns:
            return col
    raise KeyError(f"Cannot find ROI name column in dataframe. Columns: {df.columns.tolist()}")




def plot_stage1_rollout_stability_diagnostics(
    lesion_result,
    figsize=(7.2, 4.6),
    title="Stage 1 rollout stability diagnostics"
):
    """
    判断 Stage 1 中 lesion response over time 的上升，
    到底是病灶效应扩散，还是模型长时 rollout 本身发散。

    Required keys in lesion_result:
        - healthy_traj
        - lesion_traj

    healthy_traj / lesion_traj expected shape:
        (n_seed_windows, rollout_horizon, n_roi)
    or:
        (rollout_horizon, n_roi)
    """

    healthy_traj = np.asarray(lesion_result["healthy_traj"], dtype=float)
    lesion_traj = np.asarray(lesion_result["lesion_traj"], dtype=float)

    # 如果只有单条轨迹，shape=(T, ROI)，补成 shape=(1, T, ROI)
    if healthy_traj.ndim == 2:
        healthy_traj = healthy_traj[None, :, :]

    if lesion_traj.ndim == 2:
        lesion_traj = lesion_traj[None, :, :]

    if healthy_traj.shape != lesion_traj.shape:
        raise ValueError(
            f"healthy_traj and lesion_traj must have the same shape, "
            f"but got {healthy_traj.shape} and {lesion_traj.shape}"
        )

    # shape = (n_seed, T)
    healthy_norm = np.linalg.norm(healthy_traj, axis=-1)
    lesion_norm = np.linalg.norm(lesion_traj, axis=-1)

    # 对 seed 求平均，得到每个 rollout step 的平均轨迹范数
    healthy_norm_mean = healthy_norm.mean(axis=0)
    lesion_norm_mean = lesion_norm.mean(axis=0)

    # 每个时间步的 Mean |lesion - healthy|
    # shape = (T,)
    diff_abs_mean = np.mean(np.abs(lesion_traj - healthy_traj), axis=-1).mean(axis=0)

    T = healthy_traj.shape[1]
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, healthy_norm_mean, label="Healthy trajectory norm", linewidth=2.2)
    ax.plot(x, lesion_norm_mean, label="Lesion trajectory norm", linewidth=2.2)
    ax.plot(x, diff_abs_mean, label="Mean |lesion - healthy|", linewidth=2.2)

    ax.set_title(title)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Magnitude")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.show()

    return {
        "healthy_norm_mean": healthy_norm_mean,
        "lesion_norm_mean": lesion_norm_mean,
        "diff_abs_mean": diff_abs_mean,
    }


def plot_stage1_response_and_stability(
    lesion_result,
    figsize=(13.5, 8.0),
    save_path=None,
):
    """
    Stage 1 lesion response and rollout stability diagnostics.

    组合图：
    A. Lesion response over rollout time
    B. Healthy / lesion trajectory norm + response on secondary y-axis
    C. Phase-wise response bar: early / middle / late
    D. Summary response bar: initial / mean / peak / final

    Parameters
    ----------
    lesion_result : dict
        必须包含：
            healthy_traj
            lesion_traj
        可选包含：
            temporal_response

    Returns
    -------
    dict
        包含 temporal_response、norm diagnostics、phase summary、response summary。
    """
    # --------------------------------------------------
    # 1. Read trajectories
    # --------------------------------------------------
    healthy_traj = np.asarray(lesion_result["healthy_traj"], dtype=float)
    lesion_traj = np.asarray(lesion_result["lesion_traj"], dtype=float)

    if healthy_traj.ndim == 2:
        healthy_traj = healthy_traj[None, :, :]

    if lesion_traj.ndim == 2:
        lesion_traj = lesion_traj[None, :, :]

    if healthy_traj.shape != lesion_traj.shape:
        raise ValueError(
            f"healthy_traj and lesion_traj must have the same shape, "
            f"but got {healthy_traj.shape} and {lesion_traj.shape}"
        )

    # shape = (n_seed, rollout_horizon, roi_num)
    n_seed, rollout_horizon, roi_num = healthy_traj.shape

    # --------------------------------------------------
    # 2. Basic temporal response
    # --------------------------------------------------
    if "temporal_response" in lesion_result:
        temporal_response = np.asarray(
            lesion_result["temporal_response"], dtype=float
        )
    else:
        temporal_response = np.mean(
            np.abs(lesion_traj - healthy_traj),
            axis=-1
        ).mean(axis=0)

    healthy_norm_mean = np.linalg.norm(healthy_traj, axis=-1).mean(axis=0)
    lesion_norm_mean = np.linalg.norm(lesion_traj, axis=-1).mean(axis=0)
    diff_abs_mean = np.mean(
        np.abs(lesion_traj - healthy_traj),
        axis=-1
    ).mean(axis=0)

    x = np.arange(len(temporal_response))

    # --------------------------------------------------
    # 3. Phase-wise response: early / middle / late
    # --------------------------------------------------
    phase_names = ["Early", "Middle", "Late"]
    phase_indices = np.array_split(np.arange(len(temporal_response)), 3)

    phase_values = []
    for idx in phase_indices:
        if len(idx) == 0:
            phase_values.append(np.nan)
        else:
            phase_values.append(float(np.mean(temporal_response[idx])))

    phase_df = pd.DataFrame({
        "phase": phase_names,
        "mean_response": phase_values,
    })

    # --------------------------------------------------
    # 4. Summary response bars
    # --------------------------------------------------
    initial_response = float(temporal_response[0])
    mean_response = float(np.mean(temporal_response))
    peak_response = float(np.max(temporal_response))
    final_response = float(temporal_response[-1])
    peak_step = int(np.argmax(temporal_response))

    summary_df = pd.DataFrame({
        "metric": ["Initial", "Mean", "Peak", "Final"],
        "value": [
            initial_response,
            mean_response,
            peak_response,
            final_response,
        ],
    })

    growth_abs = final_response - initial_response
    growth_ratio = final_response / (initial_response + 1e-12)

    norm_gap_mean = float(np.mean(np.abs(lesion_norm_mean - healthy_norm_mean)))
    norm_gap_final = float(abs(lesion_norm_mean[-1] - healthy_norm_mean[-1]))

    diagnostics = {
        "n_seed": int(n_seed),
        "rollout_horizon": int(rollout_horizon),
        "roi_num": int(roi_num),
        "initial_response": initial_response,
        "mean_response": mean_response,
        "peak_response": peak_response,
        "peak_step": peak_step,
        "final_response": final_response,
        "growth_abs": float(growth_abs),
        "growth_ratio": float(growth_ratio),
        "norm_gap_mean": norm_gap_mean,
        "norm_gap_final": norm_gap_final,
    }

    # --------------------------------------------------
    # 5. Plot 2 x 2 panel
    # --------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.25, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.35,
        wspace=0.28,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # --------------------------------------------------
    # A. Lesion response over time
    # --------------------------------------------------
    ax_a.plot(
        x,
        temporal_response,
        linewidth=2.3,
        label="Mean |lesion - healthy|",
    )
    ax_a.scatter(
        [peak_step],
        [peak_response],
        s=45,
        zorder=5,
        label=f"Peak step = {peak_step}",
    )
    ax_a.axvline(
        peak_step,
        linestyle="--",
        linewidth=1.1,
        alpha=0.65,
    )

    ax_a.set_title("Lesion response over time")
    ax_a.set_xlabel("Rollout step")
    ax_a.set_ylabel("Mean |lesion - healthy|")
    ax_a.grid(alpha=0.25, linestyle="--")
    ax_a.legend(frameon=False, fontsize=9)

    # --------------------------------------------------
    # B. Rollout stability diagnostics
    #    用双 y 轴，避免 response 被 norm 压扁
    # --------------------------------------------------
    ax_b.plot(
        x,
        healthy_norm_mean,
        label="Healthy trajectory norm",
        linewidth=2.2,
    )
    ax_b.plot(
        x,
        lesion_norm_mean,
        label="Lesion trajectory norm",
        linewidth=2.2,
    )

    ax_b.set_title("Rollout stability diagnostics")
    ax_b.set_xlabel("Rollout step")
    ax_b.set_ylabel("Trajectory norm")
    ax_b.grid(alpha=0.25, linestyle="--")

    ax_b2 = ax_b.twinx()
    ax_b2.plot(
        x,
        diff_abs_mean,
        linestyle="--",
        linewidth=2.0,
        label="Mean |lesion - healthy|",
    )
    ax_b2.set_ylabel("Response magnitude")

    lines1, labels1 = ax_b.get_legend_handles_labels()
    lines2, labels2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        fontsize=9,
        loc="center",
    )

    # --------------------------------------------------
    # C. Phase-wise response bar
    # --------------------------------------------------
    bars_c = ax_c.bar(
        phase_df["phase"],
        phase_df["mean_response"],
        width=0.62,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    for bar, val in zip(bars_c, phase_df["mean_response"]):
        if np.isfinite(val):
            ax_c.text(
                bar.get_x() + bar.get_width() / 2,
                val,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax_c.set_title("Phase-wise lesion response")
    ax_c.set_ylabel("Mean response")
    ax_c.grid(axis="y", alpha=0.25, linestyle="--")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # --------------------------------------------------
    # D. Summary response bar
    # --------------------------------------------------
    bars_d = ax_d.bar(
        summary_df["metric"],
        summary_df["value"],
        width=0.62,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    for bar, val in zip(bars_d, summary_df["value"]):
        ax_d.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    summary_text = (
        f"Growth = {growth_abs:.4f}\n"
        f"Final / initial = {growth_ratio:.2f}×\n"
        f"Mean norm gap = {norm_gap_mean:.4f}"
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

    ax_d.set_title("Response summary")
    ax_d.set_ylabel("Response value")
    ax_d.grid(axis="y", alpha=0.25, linestyle="--")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    # --------------------------------------------------
    # Panel labels
    # --------------------------------------------------
    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], ["A", "B", "C", "D"]):
        ax.text(
            -0.10,
            1.08,
            lab,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.suptitle(
        "Stage 1 lesion response and rollout stability",
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

    plt.show()

    return {
        "fig": fig,
        "axes": {
            "response_time": ax_a,
            "stability_norm": ax_b,
            "stability_response_axis": ax_b2,
            "phase_bar": ax_c,
            "summary_bar": ax_d,
        },
        "temporal_response": temporal_response,
        "healthy_norm_mean": healthy_norm_mean,
        "lesion_norm_mean": lesion_norm_mean,
        "diff_abs_mean": diff_abs_mean,
        "phase_df": phase_df,
        "summary_df": summary_df,
        "diagnostics": diagnostics,
    }
# 1. 读取 surface 与 dlabel

def load_gifti_surface_mesh(surf_path: str):
    """
    Load GIFTI surface mesh.

    Returns
    -------
    coords : (n_vertices, 3)
    faces  : (n_faces, 3)
    """
    img = nib.load(surf_path)
    coords = img.darrays[0].data.astype(float)
    faces = img.darrays[1].data.astype(int)
    return coords, faces


def load_cifti_surface_labels_sparse(dlabel_path: str):
    """
    Load CIFTI dlabel surface labels (sparse version, only valid cortex vertices).

    Returns
    -------
    left_labels_sparse
    right_labels_sparse
    left_vertex_indices
    right_vertex_indices
    label_dict : dict(label_id -> label_name)
    """
    img = nib.load(dlabel_path)
    data = np.asarray(img.dataobj).squeeze()
    hdr = img.header

    label_axis = hdr.get_axis(0)
    brain_axis = hdr.get_axis(1)

    label_dict = {}
    if hasattr(label_axis, "label"):
        raw_label_table = label_axis.label[0]
        for k, v in raw_label_table.items():
            try:
                label_name = v[0]
            except Exception:
                label_name = str(v)
            label_dict[int(k)] = str(label_name)

    left_labels_sparse = None
    right_labels_sparse = None
    left_vertex_indices = None
    right_vertex_indices = None

    for structure_name, slc, subaxis in brain_axis.iter_structures():
        subdata = data[slc]
        vertex_indices = np.asarray(subaxis.vertex, dtype=int)

        if structure_name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            left_labels_sparse = subdata.astype(int)
            left_vertex_indices = vertex_indices

        elif structure_name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            right_labels_sparse = subdata.astype(int)
            right_vertex_indices = vertex_indices

    if left_labels_sparse is None or right_labels_sparse is None:
        raise ValueError("Failed to extract left/right cortex labels from dlabel file.")

    return (
        left_labels_sparse,
        right_labels_sparse,
        left_vertex_indices,
        right_vertex_indices,
        label_dict,
    )


def expand_sparse_labels_to_full_mesh(labels_sparse, vertex_indices, n_vertices):
    """
    Expand sparse dlabel labels to full 32k mesh.

    Output length == n_vertices of the surface mesh.
    Missing vertices are filled with 0.
    """
    full_labels = np.zeros(n_vertices, dtype=int)
    full_labels[vertex_indices] = labels_sparse
    return full_labels


# =========================================================
# 2. Stage1 值映射到 surface vertex
# =========================================================

def build_stage1_value_map(lesion_df: pd.DataFrame, value_col: str = "response_abs"):
    """
    Build dict:
        standardized roi_name -> response value
    """
    roi_col = "roi_name"
    if roi_col not in lesion_df.columns:
        raise KeyError(f"'{roi_col}' not found in lesion_df columns: {lesion_df.columns.tolist()}")

    tmp = lesion_df.copy()
    tmp["roi_name_std"] = tmp["roi_name"].apply(standardize_mmp_name)

    value_map = dict(zip(tmp["roi_name_std"], tmp[value_col].astype(float)))
    return value_map


def map_parcel_values_to_vertices(full_labels, label_dict, value_map, default=np.nan):
    """
    Map parcel-level values to each vertex by label id.

    Parameters
    ----------
    full_labels : (n_vertices,)
        full mesh label ids
    label_dict : dict
        label_id -> label_name
    value_map : dict
        standardized roi_name -> value
    """
    vertex_values = np.full(full_labels.shape[0], default, dtype=float)

    unique_labels = np.unique(full_labels)
    for lab in unique_labels:
        if lab == 0:
            continue
        label_name = label_dict.get(int(lab), None)
        if label_name is None:
            continue

        roi_name_std = standardize_mmp_name(label_name)
        if roi_name_std in value_map:
            vertex_values[full_labels == lab] = float(value_map[roi_name_std])

    return vertex_values


# =========================================================
# 3. 画左右半球 surface 图
# =========================================================

def _make_surface_collection(coords, faces, vertex_values, cmap="Reds", vmin=None, vmax=None,
                             bg_color=(0.85, 0.85, 0.85, 1.0)):
    """
    Create Poly3DCollection for surface rendering.
    """
    triangles = coords[faces]  # (n_faces, 3, 3)

    with np.errstate(invalid="ignore"):
        face_values = np.nanmean(vertex_values[faces], axis=1)

    if vmin is None:
        valid_vals = face_values[~np.isnan(face_values)]
        vmin = np.min(valid_vals) if len(valid_vals) > 0 else 0.0
    if vmax is None:
        valid_vals = face_values[~np.isnan(face_values)]
        vmax = np.max(valid_vals) if len(valid_vals) > 0 else 1.0

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    facecolors = np.tile(np.array(bg_color)[None, :], (len(face_values), 1))

    valid = ~np.isnan(face_values)
    if np.any(valid):
        facecolors[valid] = cmap_obj(norm(face_values[valid]))

    collection = Poly3DCollection(triangles, linewidths=0.0, antialiased=False)
    collection.set_facecolor(facecolors)
    collection.set_edgecolor("none")

    return collection, norm, cmap_obj


def _set_3d_axis_style(ax, coords):
    ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    ax.set_zlim(coords[:, 2].min(), coords[:, 2].max())

    xr = coords[:, 0].max() - coords[:, 0].min()
    yr = coords[:, 1].max() - coords[:, 1].min()
    zr = coords[:, 2].max() - coords[:, 2].min()
    ax.set_box_aspect((xr, yr, zr))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()


def _draw_one_surface(ax, coords, faces, vertex_values, title,
                      cmap="Reds", vmin=None, vmax=None,
                      elev=10, azim=180,
                      lesion_xyz=None):
    collection, norm, cmap_obj = _make_surface_collection(
        coords, faces, vertex_values, cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.add_collection3d(collection)
    _set_3d_axis_style(ax, coords)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11)

    if lesion_xyz is not None:
        ax.scatter(
            [lesion_xyz[0]], [lesion_xyz[1]], [lesion_xyz[2]],
            s=55, c="deepskyblue", edgecolors="k", linewidths=0.7
        )

    return norm, cmap_obj


def plot_stage1_surface_brainmap(
    dlabel_path: str,
    left_surf_path: str,
    right_surf_path: str,
    lesion_df: pd.DataFrame,
    aligned_centroid_df: pd.DataFrame = None,
    lesion_idx: int = None,
    value_col: str = "response_abs",
    cmap: str = "Reds",
    save_path: str = None,
    suptitle: str = "Stage 1 lesion response surface map",
):
    """
    Plot a paper-style surface map:
        Left lateral / Left medial / Right lateral / Right medial
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors, cm
    from matplotlib.colors import Normalize

    # ----------------------------
    # 1. load surfaces
    # ----------------------------
    left_coords, left_faces = load_gifti_surface_mesh(left_surf_path)
    right_coords, right_faces = load_gifti_surface_mesh(right_surf_path)

    # ----------------------------
    # 2. load dlabel sparse + expand to full mesh
    # ----------------------------
    (
        left_labels_sparse,
        right_labels_sparse,
        left_vertex_indices,
        right_vertex_indices,
        label_dict,
    ) = load_cifti_surface_labels_sparse(dlabel_path)

    left_labels_full = expand_sparse_labels_to_full_mesh(
        left_labels_sparse, left_vertex_indices, left_coords.shape[0]
    )
    right_labels_full = expand_sparse_labels_to_full_mesh(
        right_labels_sparse, right_vertex_indices, right_coords.shape[0]
    )

    # ----------------------------
    # 3. stage1 values
    # ----------------------------
    value_map = build_stage1_value_map(lesion_df, value_col=value_col)

    left_vertex_values = map_parcel_values_to_vertices(
        left_labels_full, label_dict, value_map
    )
    right_vertex_values = map_parcel_values_to_vertices(
        right_labels_full, label_dict, value_map
    )

    # ----------------------------
    # 4. color range
    # ----------------------------
    left_valid = left_vertex_values[~np.isnan(left_vertex_values)]
    right_valid = right_vertex_values[~np.isnan(right_vertex_values)]

    if len(left_valid) + len(right_valid) > 0:
        all_vals = np.concatenate([left_valid, right_valid])
    else:
        all_vals = np.array([0.0], dtype=float)

    if value_col == "response_abs":
        vmin = 0.0
        vmax = float(np.nanmax(all_vals))
        if vmax <= 1e-12:
            vmax = 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        vmax = float(np.nanmax(np.abs(all_vals)))
        if vmax <= 1e-12:
            vmax = 1.0
        vmin = -vmax
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    cmap_obj = plt.get_cmap(cmap)

    # ----------------------------
    # 5. optional lesion centroid highlight
    # ----------------------------
    lesion_xyz_left = None
    lesion_xyz_right = None
    if lesion_idx is not None and aligned_centroid_df is not None:
        lesion_row = aligned_centroid_df[
            aligned_centroid_df["roi"].astype(int) == int(lesion_idx)
        ]
        if len(lesion_row) > 0:
            lesion_row = lesion_row.iloc[0]
            xyz = np.array(
                [lesion_row["x"], lesion_row["y"], lesion_row["z"]],
                dtype=float
            )
            hemi = str(lesion_row["hemi"])
            if hemi == "L":
                lesion_xyz_left = xyz
            elif hemi == "R":
                lesion_xyz_right = xyz

    # ----------------------------
    # 6. figure + axes
    # ----------------------------
    fig = plt.figure(figsize=(14, 8))

    axes = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]

    # 手动调整整体布局：右边给 colorbar 留空间
    fig.subplots_adjust(
        left=0.04,
        right=0.88,
        bottom=0.04,
        top=0.90,
        wspace=0.02,
        hspace=0.12,
    )

    # ----------------------------
    # 7. draw surfaces
    # ----------------------------
    _draw_one_surface(
        axes[0], left_coords, left_faces, left_vertex_values,
        title="Left lateral",
        cmap=cmap, vmin=vmin, vmax=vmax,
        elev=10, azim=180,
        lesion_xyz=lesion_xyz_left
    )
    _draw_one_surface(
        axes[1], left_coords, left_faces, left_vertex_values,
        title="Left medial",
        cmap=cmap, vmin=vmin, vmax=vmax,
        elev=10, azim=0,
        lesion_xyz=lesion_xyz_left
    )
    _draw_one_surface(
        axes[2], right_coords, right_faces, right_vertex_values,
        title="Right lateral",
        cmap=cmap, vmin=vmin, vmax=vmax,
        elev=10, azim=0,
        lesion_xyz=lesion_xyz_right
    )
    _draw_one_surface(
        axes[3], right_coords, right_faces, right_vertex_values,
        title="Right medial",
        cmap=cmap, vmin=vmin, vmax=vmax,
        elev=10, azim=180,
        lesion_xyz=lesion_xyz_right
    )

    # ----------------------------
    # 8. outer colorbar
    # ----------------------------
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    # [left, bottom, width, height]
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(sm, cax=cax)

    cbar.set_label(value_col, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # ----------------------------
    # 9. title
    # ----------------------------
    fig.suptitle(
        suptitle,
        x=0.5,
        y=0.96,
        ha="center",
        fontsize=16
    )

    # 不要再用 tight_layout
    # plt.tight_layout(rect=(0, 0, 0.93, 0.95))

    # ----------------------------
    # 10. save
    # ----------------------------
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    return fig, axes

# =========================================================
# 4. Network map: high FC edge 构造
# =========================================================
def get_top_response_node_subset(
    brainmap_df: pd.DataFrame,
    value_col: str = "response_abs",
    lesion_idx: int = None,
    top_n: int = 30,
):
    """
    Select lesion ROI + Top-N lesion-response ROIs.

    Returns
    -------
    node_subset : list[int]
    """
    if "roi" not in brainmap_df.columns:
        raise KeyError("brainmap_df must contain column 'roi'.")

    df = brainmap_df.dropna(subset=["roi", value_col]).copy()
    df["roi"] = df["roi"].astype(int)

    df["_score"] = np.abs(df[value_col].astype(float))

    if lesion_idx is not None:
        top_df = df[df["roi"] != int(lesion_idx)].nlargest(top_n, "_score")
        node_subset = [int(lesion_idx)] + top_df["roi"].astype(int).tolist()
    else:
        top_df = df.nlargest(top_n, "_score")
        node_subset = top_df["roi"].astype(int).tolist()

    # 去重但保持顺序
    node_subset = list(dict.fromkeys(node_subset))

    return node_subset


def build_high_fc_edge_df(
    fc_matrix: np.ndarray,
    abs_percentile: float = 99.5,
    top_k_per_node: int = None,
    node_subset=None,
    lesion_idx: int = None,
    edge_mode: str = "global_percentile",
):
    """
    Build high-FC edge dataframe for Stage 1 network plotting.

    Parameters
    ----------
    fc_matrix:
        FC matrix, shape = (n_roi, n_roi)

    abs_percentile:
        Percentile threshold for |FC| when using global_percentile
        or lesion_centered without top_k_per_node.

    top_k_per_node:
        If provided, keep top-k strongest FC edges per node.

    node_subset:
        Restrict edge construction to this set of nodes.

    lesion_idx:
        Required when edge_mode="lesion_centered".

    edge_mode:
        - "global_percentile":
            keep globally high-|FC| edges within node_subset.
        - "lesion_centered":
            only keep edges connected with lesion_idx.
        - "top_k_per_node":
            for each node, keep its top-k FC edges.

    Returns
    -------
    edge_df:
        columns = src, dst, weight, abs_weight, edge_type
    """

    fc = np.asarray(fc_matrix, dtype=float)

    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError(f"fc_matrix must be square, got shape {fc.shape}")

    n = fc.shape[0]

    if node_subset is None:
        node_subset = list(range(n))
    else:
        node_subset = list(map(int, node_subset))

    node_subset = [i for i in node_subset if 0 <= i < n]
    node_set = set(node_subset)

    keep_pairs = set()

    # -------------------------------------------------
    # Mode A: only lesion-centered edges
    # -------------------------------------------------
    if edge_mode == "lesion_centered":
        if lesion_idx is None:
            raise ValueError("lesion_idx is required when edge_mode='lesion_centered'.")

        lesion_idx = int(lesion_idx)

        if lesion_idx not in node_set:
            node_subset = [lesion_idx] + node_subset
            node_subset = list(dict.fromkeys(node_subset))
            node_set = set(node_subset)

        cand = []

        for j in node_subset:
            if j == lesion_idx:
                continue

            a, b = min(lesion_idx, j), max(lesion_idx, j)
            w = fc[a, b]
            cand.append((a, b, w, abs(w)))

        if len(cand) == 0:
            return pd.DataFrame(columns=["src", "dst", "weight", "abs_weight", "edge_type"])

        # 如果指定 top_k_per_node，则只保留 lesion ROI 最强的 top-k FC 边
        if top_k_per_node is not None and top_k_per_node > 0:
            cand = sorted(cand, key=lambda x: x[3], reverse=True)[:top_k_per_node]
            for a, b, _, _ in cand:
                keep_pairs.add((a, b))
        else:
            abs_vals = np.asarray([x[3] for x in cand], dtype=float)
            thr = np.percentile(abs_vals, abs_percentile)
            for a, b, _, aw in cand:
                if aw >= thr:
                    keep_pairs.add((a, b))

    # -------------------------------------------------
    # Mode B: each node keeps top-k FC edges
    # -------------------------------------------------
    elif edge_mode == "top_k_per_node":
        if top_k_per_node is None or top_k_per_node <= 0:
            raise ValueError("top_k_per_node must be positive when edge_mode='top_k_per_node'.")

        for i in node_subset:
            cand = []

            for j in node_subset:
                if i == j:
                    continue

                a, b = min(i, j), max(i, j)
                cand.append((j, abs(fc[i, j]), a, b))

            cand = sorted(cand, key=lambda x: x[1], reverse=True)[:top_k_per_node]

            for _, _, a, b in cand:
                keep_pairs.add((a, b))

    # -------------------------------------------------
    # Mode C: global percentile threshold
    # -------------------------------------------------
    elif edge_mode == "global_percentile":
        pairs = []
        abs_vals = []

        for i in node_subset:
            for j in node_subset:
                if i >= j:
                    continue

                w = fc[i, j]
                pairs.append((i, j, w))
                abs_vals.append(abs(w))

        abs_vals = np.asarray(abs_vals, dtype=float)

        if len(abs_vals) == 0:
            return pd.DataFrame(columns=["src", "dst", "weight", "abs_weight", "edge_type"])

        thr = np.percentile(abs_vals, abs_percentile)

        for i, j, w in pairs:
            if abs(w) >= thr:
                keep_pairs.add((i, j))

        # 可选：额外保证每个节点有 top-k 边
        if top_k_per_node is not None and top_k_per_node > 0:
            for i in node_subset:
                cand = []

                for j in node_subset:
                    if i == j:
                        continue

                    a, b = min(i, j), max(i, j)
                    cand.append((j, abs(fc[i, j]), a, b))

                cand = sorted(cand, key=lambda x: x[1], reverse=True)[:top_k_per_node]

                for _, _, a, b in cand:
                    keep_pairs.add((a, b))

    else:
        raise ValueError(
            "edge_mode must be one of: "
            "'global_percentile', 'lesion_centered', 'top_k_per_node'."
        )

    rows = []

    for i, j in sorted(list(keep_pairs)):
        rows.append({
            "src": int(i),
            "dst": int(j),
            "weight": float(fc[i, j]),
            "abs_weight": float(abs(fc[i, j])),
            "edge_type": edge_mode,
        })

    edge_df = pd.DataFrame(rows)

    if len(edge_df) > 0:
        edge_df = edge_df.sort_values("abs_weight", ascending=False).reset_index(drop=True)

    return edge_df


# =========================================================
# 5. Network map: LPCMCI core edge 构造
# =========================================================

def build_lpcmci_core_edge_plot_df(
    core_edge_df: pd.DataFrame,
    node_subset=None,
    top_n: int = 40,
    abs_col: str = "abs_val",
    weight_col: str = "val",
    collapse_undirected: bool = True,
):
    """
    Convert LPCMCI core edge table to plotting edge table.

    Expected input columns:
        src, dst, abs_val, val

    Output columns:
        src, dst, weight, abs_weight, edge_type

    用于 Stage 1 / Stage 2 bridge map:
        - 节点来自 Stage1 top response + Stage2 top screening + lesion
        - 边只来自 LPCMCI core edges
    """

    if core_edge_df is None or len(core_edge_df) == 0:
        return pd.DataFrame(columns=["src", "dst", "weight", "abs_weight", "edge_type"])

    df = core_edge_df.copy()

    # -------------------------------------------------
    # 1. 检查必要列
    # -------------------------------------------------
    required_cols = ["src", "dst"]
    missing = [c for c in required_cols if c not in df.columns]

    if len(missing) > 0:
        raise KeyError(f"core_edge_df missing columns: {missing}")

    if abs_col not in df.columns:
        if "abs_weight" in df.columns:
            abs_col = "abs_weight"
        else:
            raise KeyError(
                f"Cannot find abs_col='{abs_col}'. "
                f"Current columns are: {df.columns.tolist()}"
            )

    if weight_col not in df.columns:
        if "weight" in df.columns:
            weight_col = "weight"
        else:
            raise KeyError(
                f"Cannot find weight_col='{weight_col}'. "
                f"Current columns are: {df.columns.tolist()}"
            )

    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)
    df[abs_col] = df[abs_col].astype(float)
    df[weight_col] = df[weight_col].astype(float)

    # -------------------------------------------------
    # 2. 只保留 bridge plot 节点内部的 core edges
    # -------------------------------------------------
    if node_subset is not None:
        node_subset = set(map(int, node_subset))

        df = df[
            df["src"].isin(node_subset)
            & df["dst"].isin(node_subset)
        ].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=["src", "dst", "weight", "abs_weight", "edge_type"])

    # -------------------------------------------------
    # 3. 可选：折叠成无向边，避免同一对节点画两次
    # -------------------------------------------------
    if collapse_undirected:
        df["u"] = df[["src", "dst"]].min(axis=1).astype(int)
        df["v"] = df[["src", "dst"]].max(axis=1).astype(int)

        df = (
            df.sort_values(abs_col, ascending=False)
              .drop_duplicates(["u", "v"])
              .copy()
        )

        df["src"] = df["u"]
        df["dst"] = df["v"]

    # -------------------------------------------------
    # 4. 只画 Top-N strongest core edges
    # -------------------------------------------------
    df = df.sort_values(abs_col, ascending=False)

    if top_n is not None:
        df = df.head(int(top_n)).copy()

    # -------------------------------------------------
    # 5. 转成 plot_stage1_network_map 需要的格式
    # -------------------------------------------------
    out = df[["src", "dst", weight_col, abs_col]].copy()
    out = out.rename(
        columns={
            weight_col: "weight",
            abs_col: "abs_weight",
        }
    )

    out["edge_type"] = "lpcmci_core"

    return out.reset_index(drop=True)


# =========================================================
# 6. 画 network map
# =========================================================

def plot_stage1_fc_subnetwork_matrix(
    brainmap_df: pd.DataFrame,
    fc_matrix,
    lesion_idx: int = None,
    value_col: str = "response_abs",
    roi_names=None,
    node_subset=None,
    top_k: int = 20,
    lesion_first: bool = True,
    figsize=(9.8, 6.4),
    fc_cmap: str = "RdBu_r",
    response_color: str = "#B22222",
    vmax_percentile: float = 98.0,
    title: str = "Stage 1 lesion-responsive FC subnetwork",
    save_path: str = None,
):
    """
    Paper-style Stage 1 FC subnetwork figure.

    Instead of drawing a node-link graph, this function visualizes:
      1. FC matrix among top lesion-responsive ROIs
      2. ROI-wise lesion response bar aligned with the matrix order

    This is more suitable for Stage 1 because FC is undirected and the
    Stage 1 emphasis is response-ranked local subnetwork structure.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    from matplotlib.patches import Rectangle

    # --------------------------------------------------
    # 1. Basic checks
    # --------------------------------------------------
    if "roi" not in brainmap_df.columns:
        raise KeyError("brainmap_df must contain column: 'roi'.")

    if value_col not in brainmap_df.columns:
        raise KeyError(f"brainmap_df must contain column: {value_col}")

    df = brainmap_df.copy()
    df["roi"] = df["roi"].astype(int)

    fc = np.asarray(fc_matrix, dtype=float)

    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError("fc_matrix must be a square 2D matrix.")

    n_roi = fc.shape[0]

    # --------------------------------------------------
    # 2. Select nodes
    # --------------------------------------------------
    if node_subset is not None:
        selected_nodes = [int(x) for x in node_subset]
        selected_nodes = [x for x in selected_nodes if 0 <= x < n_roi]
    else:
        selected_nodes = (
            df.sort_values(value_col, ascending=False)
              .head(int(top_k))["roi"]
              .astype(int)
              .tolist()
        )

    if lesion_idx is not None:
        lesion_idx = int(lesion_idx)
        if lesion_idx not in selected_nodes and 0 <= lesion_idx < n_roi:
            selected_nodes = [lesion_idx] + selected_nodes

    selected_nodes = list(dict.fromkeys(selected_nodes))

    if len(selected_nodes) == 0:
        raise ValueError("No selected nodes to plot.")

    # 只保留 brainmap_df 里存在的节点
    valid_df_nodes = set(df["roi"].astype(int).tolist())
    selected_nodes = [n for n in selected_nodes if n in valid_df_nodes]

    if len(selected_nodes) == 0:
        raise ValueError("No selected nodes overlap with brainmap_df['roi'].")

    # --------------------------------------------------
    # 3. Sort nodes by response_abs, optionally put lesion first
    # --------------------------------------------------
    score_map = {
        int(r["roi"]): float(r[value_col])
        for _, r in df.iterrows()
    }

    if lesion_first and lesion_idx is not None and lesion_idx in selected_nodes:
        rest = [n for n in selected_nodes if n != lesion_idx]
        rest = sorted(rest, key=lambda n: score_map.get(int(n), 0.0), reverse=True)
        ordered_nodes = [lesion_idx] + rest
    else:
        ordered_nodes = sorted(
            selected_nodes,
            key=lambda n: score_map.get(int(n), 0.0),
            reverse=True,
        )

    # 如果节点太多，限制到 top_k，但保留 lesion
    if len(ordered_nodes) > int(top_k):
        if lesion_idx is not None and lesion_idx in ordered_nodes:
            ordered_nodes = [lesion_idx] + [
                n for n in ordered_nodes if n != lesion_idx
            ][: int(top_k) - 1]
        else:
            ordered_nodes = ordered_nodes[: int(top_k)]

    # --------------------------------------------------
    # 4. Labels
    # --------------------------------------------------
    name_col = None
    for c in ["roi_name", "roi_name_stage1", "roi_name_std", "node_label"]:
        if c in df.columns:
            name_col = c
            break

    if roi_names is not None:
        labels = [
            roi_names[int(n)] if int(n) < len(roi_names) else str(int(n))
            for n in ordered_nodes
        ]
    elif name_col is not None:
        name_map = {
            int(r["roi"]): str(r[name_col])
            for _, r in df.iterrows()
        }
        labels = [name_map.get(int(n), str(int(n))) for n in ordered_nodes]
    else:
        labels = [str(int(n)) for n in ordered_nodes]

    # --------------------------------------------------
    # 5. FC submatrix
    # --------------------------------------------------
    sub_fc = fc[np.ix_(ordered_nodes, ordered_nodes)].copy()

    # diagonal does not carry between-ROI FC information
    np.fill_diagonal(sub_fc, np.nan)

    finite_vals = sub_fc[np.isfinite(sub_fc)]

    if len(finite_vals) == 0:
        raise ValueError("FC submatrix has no finite off-diagonal values.")

    vmax = float(np.nanpercentile(np.abs(finite_vals), vmax_percentile))
    if vmax <= 1e-12:
        vmax = float(np.nanmax(np.abs(finite_vals)))
    if vmax <= 1e-12:
        vmax = 1.0

    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    cmap_obj = plt.get_cmap(fc_cmap).copy()
    cmap_obj.set_bad("#f2f2f2")

    response_vals = np.array(
        [score_map.get(int(n), 0.0) for n in ordered_nodes],
        dtype=float,
    )

    # --------------------------------------------------
    # 6. Figure layout
    # --------------------------------------------------
    fig, (ax_mat, ax_bar) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [4.6, 1.35], "wspace": 0.08},
    )

    fig.subplots_adjust(
        left=0.16,
        right=0.86,
        bottom=0.20,
        top=0.86,
    )

    # --------------------------------------------------
    # 7. FC heatmap
    # --------------------------------------------------
    im = ax_mat.imshow(
        sub_fc,
        cmap=cmap_obj,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )

    n = len(ordered_nodes)

    ax_mat.set_xticks(np.arange(n))
    ax_mat.set_yticks(np.arange(n))

    label_fontsize = 7.5 if n <= 20 else 6.2

    ax_mat.set_xticklabels(
        labels,
        rotation=60,
        ha="right",
        fontsize=label_fontsize,
    )

    ax_mat.set_yticklabels(
        labels,
        fontsize=label_fontsize,
    )

    ax_mat.set_title("FC subnetwork among top response ROIs", fontsize=12, pad=8)

    ax_mat.tick_params(length=0)

    # light grid
    ax_mat.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_mat.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_mat.grid(which="minor", color="white", linewidth=0.5)
    ax_mat.tick_params(which="minor", bottom=False, left=False)

    # --------------------------------------------------
    # 8. Highlight lesion row/column
    # --------------------------------------------------
    if lesion_idx is not None and lesion_idx in ordered_nodes:
        li = ordered_nodes.index(lesion_idx)

        ax_mat.axhline(li - 0.5, color="deepskyblue", linewidth=1.4)
        ax_mat.axhline(li + 0.5, color="deepskyblue", linewidth=1.4)
        ax_mat.axvline(li - 0.5, color="deepskyblue", linewidth=1.4)
        ax_mat.axvline(li + 0.5, color="deepskyblue", linewidth=1.4)

        ax_mat.add_patch(
            Rectangle(
                (li - 0.5, li - 0.5),
                1,
                1,
                fill=False,
                edgecolor="deepskyblue",
                linewidth=2.0,
                zorder=5,
            )
        )

    # --------------------------------------------------
    # 9. Response bar aligned with matrix rows
    # --------------------------------------------------
    y = np.arange(n)

    bars = ax_bar.barh(
        y,
        response_vals,
        color=response_color,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.35,
    )

    if lesion_idx is not None and lesion_idx in ordered_nodes:
        li = ordered_nodes.index(lesion_idx)
        bars[li].set_edgecolor("deepskyblue")
        bars[li].set_linewidth(1.8)

    ax_bar.set_ylim(n - 0.5, -0.5)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel(value_col, fontsize=9)
    ax_bar.set_title("Response", fontsize=12, pad=8)
    ax_bar.tick_params(axis="x", labelsize=8)

    ax_bar.grid(axis="x", alpha=0.22, linestyle="--")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_visible(False)

    # --------------------------------------------------
    # 10. Colorbar outside
    # --------------------------------------------------
    cax = fig.add_axes([0.885, 0.28, 0.018, 0.46])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("FC weight", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # --------------------------------------------------
    # 11. Title
    # --------------------------------------------------
    fig.suptitle(title, fontsize=13, y=0.96)

    # --------------------------------------------------
    # 12. Save
    # --------------------------------------------------
    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_path.lower().endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")

    plt.show()

    info = {
        "ordered_nodes": ordered_nodes,
        "labels": labels,
        "sub_fc": sub_fc,
        "response_vals": response_vals,
    }

    return fig, (ax_mat, ax_bar), info
