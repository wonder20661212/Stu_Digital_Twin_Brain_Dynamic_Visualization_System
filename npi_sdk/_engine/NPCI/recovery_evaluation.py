from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def ensure_3d_traj(traj: np.ndarray, name: str = "traj") -> np.ndarray:
    """
    Ensure trajectory has shape (B, H, ROI).

    Supports:
        (H, ROI) -> (1, H, ROI)
        (B, H, ROI) -> unchanged
    """
    arr = np.asarray(traj, dtype=float)

    if arr.ndim == 2:
        arr = arr[None, :, :]

    if arr.ndim != 3:
        raise ValueError(
            f"{name} must have shape (B, H, ROI) or (H, ROI), "
            f"but got {arr.shape}"
        )

    return arr


def validate_three_trajectories(
    healthy_traj: np.ndarray,
    lesion_traj: np.ndarray,
    policy_traj: np.ndarray,
):
    """
    Validate and return three trajectories with the same shape.
    """
    healthy_traj = ensure_3d_traj(healthy_traj, "healthy_traj")
    lesion_traj = ensure_3d_traj(lesion_traj, "lesion_traj")
    policy_traj = ensure_3d_traj(policy_traj, "policy_traj")

    if healthy_traj.shape != lesion_traj.shape:
        raise ValueError(
            f"healthy_traj and lesion_traj shape mismatch: "
            f"{healthy_traj.shape} vs {lesion_traj.shape}"
        )

    if healthy_traj.shape != policy_traj.shape:
        raise ValueError(
            f"healthy_traj and policy_traj shape mismatch: "
            f"{healthy_traj.shape} vs {policy_traj.shape}"
        )

    return healthy_traj, lesion_traj, policy_traj


def normalize_policy(policy):
    """
    Make policy always a list of dicts.

    Supports:
        dict
        list[dict]
        tuple[dict]
    """
    if policy is None:
        return []

    if isinstance(policy, dict):
        return [policy]

    if isinstance(policy, (list, tuple)):
        return list(policy)

    raise TypeError(f"Unsupported policy type: {type(policy)}")


def build_eval_schedule_vector(name: str, horizon: int) -> np.ndarray:
    """
    Local fallback schedule vector builder for Stage 4 evaluation.

    This avoids temporal_cost = 0 when policy only contains schedule name
    but no schedule_vector.
    """
    name = str(name)
    horizon = int(horizon)

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    if name == "constant":
        return np.ones(horizon, dtype=float)

    if name == "ramp_up":
        return np.linspace(0.0, 1.0, horizon)

    if name == "ramp_decay":
        return np.linspace(1.0, 0.0, horizon)

    if name == "exponential_decay":
        x = np.linspace(0.0, 4.0, horizon)
        v = np.exp(-x)
        return v / (v.max() + 1e-12)

    # pulse schedules
    v = np.zeros(horizon, dtype=float)

    if name == "early_pulse":
        a, b = 0, max(1, horizon // 3)
        v[a:b] = 1.0
        return v

    if name == "middle_pulse":
        a, b = horizon // 3, max(horizon // 3 + 1, 2 * horizon // 3)
        v[a:b] = 1.0
        return v

    if name == "late_pulse":
        a, b = 2 * horizon // 3, horizon
        v[a:b] = 1.0
        return v

    # unknown schedule: treat as constant, but still safe
    return np.ones(horizon, dtype=float)


def fc_from_traj(traj: np.ndarray) -> np.ndarray:
    """
    Compute FC from rollout trajectory.
    traj:
        shape (B, horizon, roi_num)
    """
    # 把三维轨迹压成二维,也就是把所有 seed、所有时间步拼起来，当作一段长时间序列。
    X = np.asarray(traj, dtype=float).reshape(-1, traj.shape[-1])
    FC = np.corrcoef(X, rowvar=False)
    return np.nan_to_num(FC, nan=0.0, posinf=0.0, neginf=0.0)


def trajectory_recovery_score(
    healthy_traj: np.ndarray,
    lesion_traj: np.ndarray,
    policy_traj: np.ndarray,
    eps: float = 1e-12,
) -> Dict:
    """
    计算轨迹层面的恢复分数:回答干预后的动态轨迹是否比病灶轨迹更接近健康轨迹？
    R_traj = 1 - D(policy, healthy) / D(lesion, healthy).
    """
    healthy_traj, lesion_traj, policy_traj = validate_three_trajectories(
        healthy_traj,
        lesion_traj,
        policy_traj,
    )

    d_lesion = float(np.mean(np.linalg.norm(lesion_traj - healthy_traj, axis=2)))
    d_policy = float(np.mean(np.linalg.norm(policy_traj - healthy_traj, axis=2)))
    recovery = float(1.0 - d_policy / (d_lesion + eps))
    return {
        "traj_recovery": recovery,
        "D_lesion_traj": d_lesion,
        "D_policy_traj": d_policy,
    }


def time_resolved_recovery(
    healthy_traj: np.ndarray,
    lesion_traj: np.ndarray,
    policy_traj: np.ndarray,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    计算每个 rollout step 上的轨迹恢复程度。回答：干预效果是在早期明显、中期明显，还是后期才出现？
    """
    healthy_traj, lesion_traj, policy_traj = validate_three_trajectories(
        healthy_traj,
        lesion_traj,
        policy_traj,
    )
    dh = np.mean(np.linalg.norm(lesion_traj - healthy_traj, axis=2), axis=0)
    dp = np.mean(np.linalg.norm(policy_traj - healthy_traj, axis=2), axis=0)
    rec = 1.0 - dp / (dh + eps)
    return pd.DataFrame({
        "step": np.arange(len(rec), dtype=int),
        "D_lesion_traj": dh,
        "D_policy_traj": dp,
        "traj_recovery": rec,
    })


def fc_recovery_score(
    healthy_traj: np.ndarray,
    lesion_traj: np.ndarray,
    policy_traj: np.ndarray,
    eps: float = 1e-12,
) -> Dict:
    """
    计算 FC 网络层面的恢复分数。回答：干预是否让全脑功能连接结构恢复到健康状态？
    R_FC = 1 - ||FC_policy - FC_healthy||_F / ||FC_lesion - FC_healthy||_F.
    """
    healthy_traj, lesion_traj, policy_traj = validate_three_trajectories(
        healthy_traj,
        lesion_traj,
        policy_traj,
    )
    FC_h = fc_from_traj(healthy_traj)
    FC_l = fc_from_traj(lesion_traj)
    FC_p = fc_from_traj(policy_traj)
    # 计算 FC 差异，这里用的是 Frobenius norm，也就是矩阵整体差异。
    d_lesion = float(np.linalg.norm(FC_l - FC_h, ord="fro"))
    d_policy = float(np.linalg.norm(FC_p - FC_h, ord="fro"))
    recovery = float(1.0 - d_policy / (d_lesion + eps))
    return {
        "fc_recovery": recovery,
        "D_lesion_fc": d_lesion,
        "D_policy_fc": d_policy,
        "FC_healthy": FC_h,
        "FC_lesion": FC_l,
        "FC_policy": FC_p,
    }


def roiwise_recovery_score(
    healthy_traj: np.ndarray,
    lesion_traj: np.ndarray,
    policy_traj: np.ndarray,
    roi_names: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """ROI-wise recovery values."""
    healthy_traj, lesion_traj, policy_traj = validate_three_trajectories(
        healthy_traj,
        lesion_traj,
        policy_traj,
    )
    # 每个 ROI 的病灶偏离程度对：seed 维度、时间维度取平均，保留 ROI 维度。
    d_lesion = np.mean(np.abs(lesion_traj - healthy_traj), axis=(0, 1))
    d_policy = np.mean(np.abs(policy_traj - healthy_traj), axis=(0, 1))
    recovery = 1.0 - d_policy / (d_lesion + eps)
    out = pd.DataFrame({
        "roi": np.arange(healthy_traj.shape[-1], dtype=int),
        "D_lesion_roi": d_lesion,
        "D_policy_roi": d_policy,
        "roi_recovery": recovery,
        "improvement": d_lesion - d_policy,
    }).sort_values("roi_recovery", ascending=False).reset_index(drop=True)
    if roi_names is not None:
        out["roi_name"] = out["roi"].map(lambda i: roi_names[int(i)])
    return out


def temporal_complexity_from_vector(schedule_vector: np.ndarray) -> float:
    """Temporal complexity penalty of an intervention schedule."""
    v = np.asarray(schedule_vector, dtype=float)
    if len(v) <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(v))))



def evaluate_recovery_objective(
    healthy_traj: np.ndarray,
    lesion_traj: np.ndarray,
    policy_traj: np.ndarray,
    policy: Sequence[Dict],
    alpha_traj: float = 1.0,
    beta_fc: float = 0.8,
    lambda_node: float = 0.0,
    lambda_strength: float = 0.05,
    lambda_temporal: float = 0.03,
) -> Dict:
    """
    Final recovery objective:

        J = alpha_traj * R_traj
            + beta_fc * R_FC
            - lambda_node * number_of_nodes
            - lambda_strength * sum(|strength|)
            - lambda_temporal * temporal_complexity

    Notes
    -----
    This function supports multi-node policy:

        [
            {"node": 51, "sign": 1, "strength": 0.5, "schedule": "constant"},
            {"node": 100, "sign": -1, "strength": 0.3, "schedule": "ramp_decay"},
        ]
    """
    healthy_traj, lesion_traj, policy_traj = validate_three_trajectories(
        healthy_traj,
        lesion_traj,
        policy_traj,
    )

    policy = normalize_policy(policy)
    horizon = int(healthy_traj.shape[1])

    # Recovery terms
    traj_metrics = trajectory_recovery_score(
        healthy_traj,
        lesion_traj,
        policy_traj,
    )

    fc_metrics = fc_recovery_score(
        healthy_traj,
        lesion_traj,
        policy_traj,
    )

    # Cost terms
    node_cost = float(len(policy))

    strength_cost = float(
        sum(abs(float(item.get("strength", 0.0))) for item in policy)
    )

    temporal_cost = 0.0
    for item in policy:
        if "schedule_vector" in item:
            schedule_vector = np.asarray(item["schedule_vector"], dtype=float)
        elif "schedule" in item:
            schedule_vector = build_eval_schedule_vector(
                item["schedule"],
                horizon=horizon,
            )
        else:
            schedule_vector = None

        if schedule_vector is not None:
            temporal_cost += temporal_complexity_from_vector(schedule_vector)
        else:
            temporal_cost += float(item.get("temporal_complexity", 0.0))

    objective = (
        float(alpha_traj) * traj_metrics["traj_recovery"]
        + float(beta_fc) * fc_metrics["fc_recovery"]
        - float(lambda_node) * node_cost
        - float(lambda_strength) * strength_cost
        - float(lambda_temporal) * temporal_cost
    )

    out = {
        "objective": float(objective),

        **traj_metrics,
        **fc_metrics,

        "node_cost": float(node_cost),
        "strength_cost": float(strength_cost),
        "temporal_cost": float(temporal_cost),

        "policy_size": int(len(policy)),
        "policy_nodes": [int(item["node"]) for item in policy if "node" in item],
    }

    return out


def summarize_recovery(
    healthy_traj: np.ndarray,
    lesion_traj: np.ndarray,
    policy_traj: np.ndarray,
    policy: Sequence[Dict],
    roi_names: Optional[Sequence[str]] = None,
    distance_metric: str = "rmse",
    **objective_kwargs,
) -> Dict:
    """
    Return all Stage 4 outputs for inspection.

    Outputs:
        metrics
        roi_recovery_df
        time_recovery_df
        distance_df
    """
    healthy_traj, lesion_traj, policy_traj = validate_three_trajectories(
        healthy_traj,
        lesion_traj,
        policy_traj,
    )

    policy = normalize_policy(policy)

    metrics = evaluate_recovery_objective(
        healthy_traj=healthy_traj,
        lesion_traj=lesion_traj,
        policy_traj=policy_traj,
        policy=policy,
        **objective_kwargs,
    )

    roi_df = roiwise_recovery_score(
        healthy_traj=healthy_traj,
        lesion_traj=lesion_traj,
        policy_traj=policy_traj,
        roi_names=roi_names,
    )

    time_df = time_resolved_recovery(
        healthy_traj=healthy_traj,
        lesion_traj=lesion_traj,
        policy_traj=policy_traj,
    )

    distance_df = compute_three_trajectory_distance_df(
        healthy_traj=healthy_traj,
        lesion_traj=lesion_traj,
        policy_traj=policy_traj,
        metric=distance_metric,
    )

    return {
        "metrics": metrics,
        "roi_recovery_df": roi_df,
        "time_recovery_df": time_df,
        "distance_df": distance_df,
        "policy": policy,
        "policy_size": len(policy),
        "policy_nodes": [int(item["node"]) for item in policy if "node" in item],
    }


def compute_three_trajectory_distance_df(
    healthy_traj,
    lesion_traj,
    policy_traj,
    metric: str = "rmse",
    eps: float = 1e-12,
):
    """
    Compute time-resolved distances among three trajectories:

        D(healthy, lesion)
        D(healthy, policy)

    Parameters
    ----------
    healthy_traj, lesion_traj, policy_traj:
        np.ndarray, shape = (B, H, ROI)
        B   : seed windows
        H   : rollout horizon
        ROI : number of brain regions

    metric:
        "rmse" : sqrt(mean squared difference over ROI), then average over B
        "mae"  : mean absolute difference over ROI, then average over B
        "l2"   : L2 norm over ROI, then average over B

    Returns
    -------
    dist_df:
        pd.DataFrame with columns:
            step
            D_healthy_lesion
            D_healthy_policy
            distance_reduction
            relative_recovery
            policy_closer_than_lesion
    """
    healthy_traj, lesion_traj, policy_traj = validate_three_trajectories(
        healthy_traj,
        lesion_traj,
        policy_traj,
    )

    if healthy_traj.shape != lesion_traj.shape:
        raise ValueError(
            f"healthy_traj and lesion_traj shape mismatch: "
            f"{healthy_traj.shape} vs {lesion_traj.shape}"
        )

    if healthy_traj.shape != policy_traj.shape:
        raise ValueError(
            f"healthy_traj and policy_traj shape mismatch: "
            f"{healthy_traj.shape} vs {policy_traj.shape}"
        )

    if healthy_traj.ndim != 3:
        raise ValueError(
            "trajectories must have shape (B, H, ROI)"
        )

    diff_lesion = lesion_traj - healthy_traj
    diff_policy = policy_traj - healthy_traj

    if metric == "rmse":
        # shape: (B, H)
        D_lh = np.sqrt(np.mean(diff_lesion ** 2, axis=2))
        D_ph = np.sqrt(np.mean(diff_policy ** 2, axis=2))

    elif metric == "mae":
        D_lh = np.mean(np.abs(diff_lesion), axis=2)
        D_ph = np.mean(np.abs(diff_policy), axis=2)

    elif metric == "l2":
        D_lh = np.linalg.norm(diff_lesion, axis=2)
        D_ph = np.linalg.norm(diff_policy, axis=2)

    else:
        raise ValueError("metric must be one of {'rmse', 'mae', 'l2'}")

    # average over seed windows
    D_lh_mean = D_lh.mean(axis=0)
    D_ph_mean = D_ph.mean(axis=0)

    distance_reduction = D_lh_mean - D_ph_mean

    relative_recovery = 1.0 - D_ph_mean / (D_lh_mean + eps)

    dist_df = pd.DataFrame({
        "step": np.arange(healthy_traj.shape[1]),
        "D_healthy_lesion": D_lh_mean,
        "D_healthy_policy": D_ph_mean,
        "distance_reduction": distance_reduction,
        "relative_recovery": relative_recovery,
        "policy_closer_than_lesion": D_ph_mean < D_lh_mean,
    })

    return dist_df



