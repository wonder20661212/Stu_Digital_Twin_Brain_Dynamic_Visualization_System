"""
Stage 3: single-node intervention optimization.

This file searches the best intervention direction, strength and temporal weight
for each candidate node. It does not run LPCMCI and does not choose the lesion
subnetwork by itself.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from lesion_response import rollout_npi, select_seed_windows,build_schedule_vector
from recovery_evaluation import evaluate_recovery_objective



def make_single_node_policy(node: int, sign: int, strength: float, schedule: str, horizon: int) -> List[Dict]:
    """Create one-node policy object."""
    return [{
        "node": int(node),
        "sign": int(sign),
        "strength": float(strength),
        "schedule": str(schedule),
        "schedule_vector": build_schedule_vector(schedule, horizon),
    }]


def simulate_policy_trajectory(
    model: torch.nn.Module,
    seed_inputs: np.ndarray,
    roi_num: int,
    using_steps: int,
    lesion_idx: int,
    lesion_strength: float,
    policy: Sequence[Dict],
    horizon: int = 20,
    lesion_mode: str = "subtract",
    batch_size: int = 256,
) -> np.ndarray:
    """Simulate trajectory under lesion plus intervention policy."""
    return rollout_npi(
        model=model,
        seed_inputs=seed_inputs,
        roi_num=roi_num,
        using_steps=using_steps,
        horizon=horizon,
        lesion_node=lesion_idx,
        lesion_strength=lesion_strength,
        lesion_mode=lesion_mode,
        policy=policy,
        batch_size=batch_size,
    )


def prepare_baseline_rollouts(
    model: torch.nn.Module,
    inputs: np.ndarray,
    roi_num: int,
    using_steps: int,
    lesion_idx: int,
    horizon: int = 20,
    n_seed_windows: int = 64,
    lesion_strength: float = 0.5,
    lesion_mode: str = "subtract",
    random_state: int = 123,
    batch_size: int = 256,
) -> Dict:
    """
    Prepare healthy and lesion rollouts for Stage 3.

    Use this when Stage 1 outputs are not passed in directly.
    """
    seed_inputs, seed_indices = select_seed_windows(inputs, n_seed_windows, random_state)
    healthy_traj = rollout_npi(
        model, seed_inputs, roi_num, using_steps,
        horizon=horizon, lesion_node=None, lesion_strength=0.0, batch_size=batch_size,
    )
    lesion_traj = rollout_npi(
        model, seed_inputs, roi_num, using_steps,
        horizon=horizon, lesion_node=lesion_idx, lesion_strength=lesion_strength,
        lesion_mode=lesion_mode, batch_size=batch_size,
    )
    return {
        "seed_inputs": seed_inputs,
        "seed_indices": seed_indices,
        "healthy_traj": healthy_traj,
        "lesion_traj": lesion_traj,
    }


def evaluate_single_node_policy_grid(
    model: torch.nn.Module,
    seed_inputs: np.ndarray,
    healthy_traj: np.ndarray,
    lesion_traj: np.ndarray,
    roi_num: int,
    using_steps: int,
    lesion_idx: int,
    node: int,
    horizon: int = 20,
    lesion_strength: float = 0.5,
    lesion_mode: str = "subtract",
    signs: Sequence[int] = (-1, +1),
    strengths: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
    schedules=("constant","early_pulse","middle_pulse","late_pulse","ramp_up","ramp_decay","exponential_decay"),
    alpha_traj: float = 1.0,
    beta_fc: float = 0.8,
    lambda_node: float = 0.0,
    lambda_strength: float = 0.05,
    lambda_temporal: float = 0.03,
    batch_size: int = 256,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Evaluate all sign-strength-schedule combinations for one node."""
    rows: List[Dict] = []
    details: Dict[int, Dict] = {}
    total = max(1, len(signs) * len(strengths) * len(schedules))
    done = 0

    for sign in signs:
        for strength in strengths:
            for schedule in schedules:
                if progress_callback is not None:
                    progress_callback({
                        "phase": "single_grid",
                        "node": int(node),
                        "sign": int(sign),
                        "strength": float(strength),
                        "schedule": str(schedule),
                        "node_grid_done": int(done),
                        "node_grid_total": int(total),
                    })
                policy = make_single_node_policy(node, sign, strength, schedule, horizon)
                policy_traj = simulate_policy_trajectory(
                    model=model,
                    seed_inputs=seed_inputs,
                    roi_num=roi_num,
                    using_steps=using_steps,
                    lesion_idx=lesion_idx,
                    lesion_strength=lesion_strength,
                    policy=policy,
                    horizon=horizon,
                    lesion_mode=lesion_mode,
                    batch_size=batch_size,
                )
                metrics = evaluate_recovery_objective(
                    healthy_traj=healthy_traj,
                    lesion_traj=lesion_traj,
                    policy_traj=policy_traj,
                    policy=policy,
                    alpha_traj=alpha_traj,
                    beta_fc=beta_fc,
                    lambda_node=lambda_node,
                    lambda_strength=lambda_strength,
                    lambda_temporal=lambda_temporal,
                )
                row = {
                    "node": int(node),
                    "sign": int(sign),
                    "strength": float(strength),
                    "schedule": schedule,
                    "objective": metrics["objective"],
                    "traj_recovery": metrics["traj_recovery"],
                    "fc_recovery": metrics["fc_recovery"],
                    "D_lesion_traj": metrics["D_lesion_traj"],
                    "D_policy_traj": metrics["D_policy_traj"],
                    "D_lesion_fc": metrics["D_lesion_fc"],
                    "D_policy_fc": metrics["D_policy_fc"],
                    "node_cost": metrics.get("node_cost", 1.0),
                    "strength_cost": metrics["strength_cost"],
                    "temporal_cost": metrics["temporal_cost"],
                }
                rows.append(row)
                details[len(rows) - 1] = {
                    "policy": policy,
                    "policy_traj": policy_traj,
                    "metrics": metrics,
                }
                done += 1
                if progress_callback is not None:
                    progress_callback({
                        "phase": "single_grid",
                        "node": int(node),
                        "sign": int(sign),
                        "strength": float(strength),
                        "schedule": str(schedule),
                        "node_grid_done": int(done),
                        "node_grid_total": int(total),
                    })

    # IMPORTANT:
    # Keep the original grid-row id before sorting, otherwise the trajectory stored in
    # `details` will no longer match the sorted best row after reset_index(drop=True).
    # The old version used df.index after reset_index(), which always became 0,1,2,...
    # and could attach the wrong policy_traj to the best row.
    raw_df = pd.DataFrame(rows)
    raw_df["_detail_id"] = np.arange(len(raw_df), dtype=int)

    df = raw_df.sort_values("objective", ascending=False).reset_index(drop=True)

    # Rebuild details by sorted row order using the preserved original detail id.
    sorted_details: Dict[int, Dict] = {}
    for new_i, old_i in enumerate(df["_detail_id"].astype(int).tolist()):
        sorted_details[new_i] = details[int(old_i)]

    # Hide internal id from user-facing result table.
    df = df.drop(columns=["_detail_id"])
    return df, sorted_details


def evaluate_policy_once(
    model: torch.nn.Module,
    seed_inputs: np.ndarray,
    healthy_traj: np.ndarray,
    lesion_traj: np.ndarray,
    roi_num: int,
    using_steps: int,
    lesion_idx: int,
    policy: Sequence[Dict],
    horizon: int = 20,
    lesion_strength: float = 0.5,
    lesion_mode: str = "subtract",
    alpha_traj: float = 1.0,
    beta_fc: float = 0.8,
    lambda_node: float = 0.0,
    lambda_strength: float = 0.05,
    lambda_temporal: float = 0.03,
    batch_size: int = 256,
) -> Dict:
    """
    Evaluate a complete intervention policy.
    policy 可以包含一个或多个干预节点。
    """
    policy_traj = simulate_policy_trajectory(
        model=model,
        seed_inputs=seed_inputs,
        roi_num=roi_num,
        using_steps=using_steps,
        lesion_idx=lesion_idx,
        lesion_strength=lesion_strength,
        policy=policy,
        horizon=horizon,
        lesion_mode=lesion_mode,
        batch_size=batch_size,
    )

    metrics = evaluate_recovery_objective(
        healthy_traj=healthy_traj,
        lesion_traj=lesion_traj,
        policy_traj=policy_traj,
        policy=policy,
        alpha_traj=alpha_traj,
        beta_fc=beta_fc,
        lambda_node=lambda_node,
        lambda_strength=lambda_strength,
        lambda_temporal=lambda_temporal,
    )

    out = dict(metrics)
    out["policy_traj"] = policy_traj
    return out


def run_greedy_policy_search_with_history(
    single_df: pd.DataFrame,
    model: torch.nn.Module,
    seed_inputs: np.ndarray,
    healthy_traj: np.ndarray,
    lesion_traj: np.ndarray,
    roi_num: int,
    using_steps: int,
    lesion_idx: int,
    horizon: int = 20,
    lesion_strength: float = 0.5,
    lesion_mode: str = "subtract",
    alpha_traj: float = 1.0,
    beta_fc: float = 0.8,
    lambda_node: float = 0.0,
    lambda_strength: float = 0.05,
    lambda_temporal: float = 0.03,
    max_interventions: int = 3,
    top_m_per_step: int = 20,
    min_marginal_gain: float = 1e-4,
    force_accept_first: bool = True,
    batch_size: int = 256,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """
    Forward greedy construction of a multi-node intervention policy.

    Step 1:
        选择 single_df 中 objective 最高的单节点策略。

    Step 2, 3, ...:
        在已有 policy 的基础上，逐个尝试加入一个新节点；
        重新 rollout；
        重新计算 objective；
        接受 marginal_gain 最大且超过阈值的节点。

    返回：
        best_policy
        greedy_history
        greedy_trial_df
        best_policy_traj
        best_metrics
    """
    import copy

    if single_df is None or len(single_df) == 0:
        raise ValueError("single_df is empty.")

    required_cols = ["node", "objective", "sign", "strength", "schedule"]
    for c in required_cols:
        if c not in single_df.columns:
            raise KeyError(f"single_df must contain column: {c}")

    df = single_df.copy()
    df["node"] = df["node"].astype(int)

    # 每个 node 只保留它自己的最佳单节点策略
    candidate_df = (
        df.sort_values("objective", ascending=False)
          .drop_duplicates("node", keep="first")
          .reset_index(drop=True)
    )

    if top_m_per_step is not None:
        candidate_df = candidate_df.head(int(top_m_per_step)).copy()

    def _row_to_policy_item(row):
        return {
            "node": int(row["node"]),
            "sign": int(row["sign"]),
            "strength": float(row["strength"]),
            "schedule": str(row["schedule"]),
            "schedule_vector": build_schedule_vector(str(row["schedule"]), horizon),
        }

    selected_policy: List[Dict] = []
    selected_nodes = set()

    current_objective = 0.0

    greedy_history_rows = []
    greedy_trial_rows = []

    best_policy_traj = None
    best_metrics = None

    for step in range(1, int(max_interventions) + 1):

        best_trial_policy = None
        best_trial_record = None
        best_trial_metrics = None
        step_candidates = [
            int(row["node"])
            for _, row in candidate_df.iterrows()
            if int(row["node"]) not in selected_nodes
        ]
        step_total = max(1, len(step_candidates))
        step_done = 0

        for _, row in candidate_df.iterrows():
            node = int(row["node"])

            if node in selected_nodes:
                continue
            if progress_callback is not None:
                progress_callback({
                    "phase": "greedy",
                    "greedy_step": int(step),
                    "greedy_done": int(step_done),
                    "greedy_total": int(step_total),
                    "candidate_node": int(node),
                    "selected_nodes": [int(x) for x in selected_nodes],
                })

            new_item = _row_to_policy_item(row)
            trial_policy = copy.deepcopy(selected_policy) + [new_item]

            metrics = evaluate_policy_once(
                model=model,
                seed_inputs=seed_inputs,
                healthy_traj=healthy_traj,
                lesion_traj=lesion_traj,
                roi_num=roi_num,
                using_steps=using_steps,
                lesion_idx=lesion_idx,
                policy=trial_policy,
                horizon=horizon,
                lesion_strength=lesion_strength,
                lesion_mode=lesion_mode,
                alpha_traj=alpha_traj,
                beta_fc=beta_fc,
                lambda_node=lambda_node,
                lambda_strength=lambda_strength,
                lambda_temporal=lambda_temporal,
                batch_size=batch_size,
            )

            objective = float(metrics["objective"])
            marginal_gain = objective - float(current_objective)

            trial_record = {
                "step": int(step),
                "candidate_node": int(node),
                "selected_node": int(node),
                "node": int(node),
                "sign": int(new_item["sign"]),
                "strength": float(new_item["strength"]),
                "schedule": str(new_item["schedule"]),
                "policy_size": len(trial_policy),
                "policy_nodes": [int(p["node"]) for p in trial_policy],
                "objective": objective,
                "marginal_gain": float(marginal_gain),
                "traj_recovery": float(metrics.get("traj_recovery", np.nan)),
                "fc_recovery": float(metrics.get("fc_recovery", np.nan)),
                "D_lesion_traj": float(metrics.get("D_lesion_traj", np.nan)),
                "D_policy_traj": float(metrics.get("D_policy_traj", np.nan)),
                "D_lesion_fc": float(metrics.get("D_lesion_fc", np.nan)),
                "D_policy_fc": float(metrics.get("D_policy_fc", np.nan)),
                "node_cost": float(metrics.get("node_cost", np.nan)),
                "strength_cost": float(metrics.get("strength_cost", np.nan)),
                "temporal_cost": float(metrics.get("temporal_cost", np.nan)),
                "policy": copy.deepcopy(trial_policy),
            }

            greedy_trial_rows.append(trial_record)

            if best_trial_record is None or objective > best_trial_record["objective"]:
                best_trial_policy = copy.deepcopy(trial_policy)
                best_trial_record = trial_record
                best_trial_metrics = metrics
            step_done += 1
            if progress_callback is not None:
                progress_callback({
                    "phase": "greedy",
                    "greedy_step": int(step),
                    "greedy_done": int(step_done),
                    "greedy_total": int(step_total),
                    "candidate_node": int(node),
                    "selected_nodes": [int(x) for x in selected_nodes],
                })

        if best_trial_record is None:
            break

        if step == 1 and force_accept_first:
            accept = True
        else:
            accept = best_trial_record["marginal_gain"] > float(min_marginal_gain)

        if not accept:
            break

        selected_policy = copy.deepcopy(best_trial_policy)
        selected_node = int(best_trial_record["selected_node"])
        selected_nodes.add(selected_node)

        current_objective = float(best_trial_record["objective"])
        best_policy_traj = best_trial_metrics["policy_traj"]
        best_metrics = {
            k: v for k, v in best_trial_metrics.items()
            if k != "policy_traj"
        }

        greedy_history_rows.append(best_trial_record)

    greedy_history = pd.DataFrame(greedy_history_rows)
    greedy_trial_df = pd.DataFrame(greedy_trial_rows)

    return {
        "best_policy": selected_policy,
        "greedy_history": greedy_history,
        "greedy_trial_df": greedy_trial_df,
        "best_policy_traj": best_policy_traj,
        "best_metrics": best_metrics,
        "best_objective": float(current_objective),
    }


def run_single_node_search(
    model: torch.nn.Module,
    candidate_nodes: Sequence[int],
    roi_num: int,
    using_steps: int,
    lesion_idx: int,
    horizon: int = 20,
    lesion_strength: float = 0.5,
    lesion_mode: str = "subtract",
    seed_inputs: Optional[np.ndarray] = None,
    healthy_traj: Optional[np.ndarray] = None,
    lesion_traj: Optional[np.ndarray] = None,
    inputs: Optional[np.ndarray] = None,
    n_seed_windows: int = 64,
    random_state: int = 123,
    signs: Sequence[int] = (-1, +1),
    strengths: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    schedules: Sequence[str] = ("constant","early_pulse","middle_pulse","late_pulse","ramp_up","ramp_decay","exponential_decay"),
    alpha_traj: float = 1.0,
    beta_fc: float = 0.8,
    lambda_node: float = 0.0,
    lambda_strength: float = 0.05,
    lambda_temporal: float = 0.03,
    batch_size: int = 256,
    run_greedy: bool = True,
    max_interventions: int = 3,
    top_m_per_step: int = 20,
    min_marginal_gain: float = 1e-4,
    force_accept_first: bool = True,
    progress_callback: Optional[Callable[[Dict], None]] = None,
    cached_node_results: Optional[Dict[int, Dict]] = None,
    node_result_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """
    它对所有候选节点逐个运行：evaluate_single_node_policy_grid(...)然后选出每个节点的最佳单节点干预方案。
    回答的问题是：每个候选节点最优的单节点干预方案是什么？哪个节点的单节点干预效果最好？
    给每个节点找一个最佳方案
    You can pass Stage 1 outputs directly:
        seed_inputs=stage1_res["seed_inputs"]
        healthy_traj=stage1_res["healthy_traj"]
        lesion_traj=stage1_res["lesion_traj"]

    Or pass inputs and let this function prepare baseline rollouts.
    """
    if seed_inputs is None or healthy_traj is None or lesion_traj is None:
        if inputs is None:
            raise ValueError("Either pass Stage 1 outputs or pass inputs to prepare baseline rollouts.")
        base = prepare_baseline_rollouts(
            model=model,
            inputs=inputs,
            roi_num=roi_num,
            using_steps=using_steps,
            lesion_idx=lesion_idx,
            horizon=horizon,
            n_seed_windows=n_seed_windows,
            lesion_strength=lesion_strength,
            lesion_mode=lesion_mode,
            random_state=random_state,
            batch_size=batch_size,
        )
        seed_inputs = base["seed_inputs"]
        healthy_traj = base["healthy_traj"]
        lesion_traj = base["lesion_traj"]

    best_rows: List[pd.Series] = []
    all_grid: Dict[int, pd.DataFrame] = {}
    best_details: Dict[int, Dict] = {}
    candidate_nodes = [int(node) for node in candidate_nodes]
    grid_per_node = max(1, len(signs) * len(strengths) * len(schedules))
    grid_total = max(1, len(candidate_nodes) * grid_per_node)
    grid_done = 0
    cached_node_results = cached_node_results or {}

    for node_pos, node in enumerate(candidate_nodes, start=1):
        if progress_callback is not None:
            progress_callback({
                "phase": "single_node",
                "current_node": int(node),
                "candidate_index": int(node_pos),
                "candidate_total": int(len(candidate_nodes)),
                "grid_done": int(grid_done),
                "grid_total": int(grid_total),
                "grid_per_node": int(grid_per_node),
            })

        def _node_progress(update: Dict) -> None:
            if progress_callback is None:
                return
            node_done = int(update.get("node_grid_done", 0))
            progress_callback({
                **update,
                "current_node": int(node),
                "candidate_index": int(node_pos),
                "candidate_total": int(len(candidate_nodes)),
                "grid_done": int(grid_done + node_done),
                "grid_total": int(grid_total),
                "grid_per_node": int(grid_per_node),
            })

        cached_node = cached_node_results.get(int(node))
        if cached_node is not None and cached_node.get("grid_records"):
            # 已完成候选靶点可以直接读取逐组合网格表；最终最佳轨迹仍会在下面重新模拟，
            # 保持 Stage 4 的输入和当前模型/参数一致。
            grid_df = pd.DataFrame(cached_node["grid_records"])
            grid_df = grid_df.sort_values("objective", ascending=False).reset_index(drop=True)
            if progress_callback is not None:
                progress_callback({
                    "phase": "single_cached",
                    "current_node": int(node),
                    "candidate_index": int(node_pos),
                    "candidate_total": int(len(candidate_nodes)),
                    "grid_done": int(grid_done + grid_per_node),
                    "grid_total": int(grid_total),
                    "grid_per_node": int(grid_per_node),
                    "node_grid_done": int(grid_per_node),
                    "node_grid_total": int(grid_per_node),
                })
        else:
            # 搜索该节点的全部干预组合
            grid_df, detail = evaluate_single_node_policy_grid(
                model=model,
                seed_inputs=seed_inputs,
                healthy_traj=healthy_traj,
                lesion_traj=lesion_traj,
                roi_num=roi_num,
                using_steps=using_steps,
                lesion_idx=lesion_idx,
                node=int(node),
                horizon=horizon,
                lesion_strength=lesion_strength,
                lesion_mode=lesion_mode,
                signs=signs,
                strengths=strengths,
                schedules=schedules,
                alpha_traj=alpha_traj,
                beta_fc=beta_fc,
                lambda_node=lambda_node,
                lambda_strength=lambda_strength,
                lambda_temporal=lambda_temporal,
                batch_size=batch_size,
                progress_callback=_node_progress,
            )
        grid_done += grid_per_node
        # 保存该节点完整网格结果
        all_grid[int(node)] = grid_df
        # 取该节点最优方案
        best_row = grid_df.iloc[0].copy()
        best_rows.append(best_row)
        # 重新构造最佳 policy
        best_policy = [{
            "node": int(best_row["node"]),
            "sign": int(best_row["sign"]),
            "strength": float(best_row["strength"]),
            "schedule": str(best_row["schedule"]),
            "schedule_vector": build_schedule_vector(str(best_row["schedule"]), horizon),
        }]
        # 重新模拟最佳轨迹并重新计算指标
        best_policy_traj = simulate_policy_trajectory(
            model=model,
            seed_inputs=seed_inputs,
            roi_num=roi_num,
            using_steps=using_steps,
            lesion_idx=lesion_idx,
            lesion_strength=lesion_strength,
            lesion_mode=lesion_mode,
            policy=best_policy,
            horizon=horizon,
            batch_size=batch_size,
        )

        best_metrics = evaluate_recovery_objective(
            healthy_traj=healthy_traj,
            lesion_traj=lesion_traj,
            policy_traj=best_policy_traj,
            policy=best_policy,
            alpha_traj=alpha_traj,
            beta_fc=beta_fc,
            lambda_node=lambda_node,
            lambda_strength=lambda_strength,
            lambda_temporal=lambda_temporal,
        )

        best_details[int(node)] = {
            "policy": best_policy,
            "policy_traj": best_policy_traj,
            "metrics": best_metrics,
            "best_row": best_row,
        }
        if node_result_callback is not None and cached_node is None:
            node_result_callback({
                "node": int(node),
                "candidate_index": int(node_pos),
                "candidate_total": int(len(candidate_nodes)),
                "grid_records": grid_df.to_dict(orient="records"),
                "best_row": best_row.to_dict(),
                "best_metrics": best_metrics,
            })

    single_df = (
        pd.DataFrame(best_rows)
        .sort_values("objective", ascending=False)
        .reset_index(drop=True)
    )

    # 单节点最优
    best_single_node = int(single_df.iloc[0]["node"])
    best_single_policy = best_details[best_single_node]["policy"]
    best_single_metrics = best_details[best_single_node]["metrics"]
    best_single_policy_traj = best_details[best_single_node]["policy_traj"]

    result = {
        "single_df": single_df,
        "all_grid": all_grid,
        "best_details": best_details,

        "seed_inputs": seed_inputs,
        "healthy_traj": healthy_traj,
        "lesion_traj": lesion_traj,
        "lambda_node": lambda_node,
        "lambda_strength": lambda_strength,
        "lambda_temporal": lambda_temporal,

        # 保留单节点结果
        "best_single_node": best_single_node,
        "best_single_policy": best_single_policy,
        "best_single_metrics": best_single_metrics,
        "best_single_policy_traj": best_single_policy_traj,

        # 默认先让 best_policy 指向单节点，若 greedy 成功再覆盖
        "best_policy": best_single_policy,
        "best_policy_traj": best_single_policy_traj,
        "best_metrics": best_single_metrics,
        "best_objective": float(single_df.iloc[0]["objective"]),
    }

    # --------------------------------------------------
    # Greedy multi-node policy construction
    # --------------------------------------------------
    if run_greedy:
        greedy_res = run_greedy_policy_search_with_history(
            single_df=single_df,
            model=model,
            seed_inputs=seed_inputs,
            healthy_traj=healthy_traj,
            lesion_traj=lesion_traj,
            roi_num=roi_num,
            using_steps=using_steps,
            lesion_idx=lesion_idx,
            horizon=horizon,
            lesion_strength=lesion_strength,
            lesion_mode=lesion_mode,
            alpha_traj=alpha_traj,
            beta_fc=beta_fc,
            lambda_node=lambda_node,
            lambda_strength=lambda_strength,
            lambda_temporal=lambda_temporal,
            max_interventions=max_interventions,
            top_m_per_step=top_m_per_step,
            min_marginal_gain=min_marginal_gain,
            force_accept_first=force_accept_first,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

        result["greedy_history"] = greedy_res["greedy_history"]
        result["greedy_trial_df"] = greedy_res["greedy_trial_df"]
        result["best_multi_policy"] = greedy_res["best_policy"]
        result["best_multi_policy_traj"] = greedy_res["best_policy_traj"]
        result["best_multi_metrics"] = greedy_res["best_metrics"]
        result["best_multi_objective"] = greedy_res["best_objective"]

        # 如果 greedy 至少选了一个节点，就让最终 best_policy 指向 greedy 结果
        if len(greedy_res["greedy_history"]) > 0:
            result["best_policy"] = greedy_res["best_policy"]
            result["best_policy_traj"] = greedy_res["best_policy_traj"]
            result["best_metrics"] = greedy_res["best_metrics"]
            result["best_objective"] = greedy_res["best_objective"]

    return result


def get_best_single_policy(stage3_res: Dict) -> Dict:
    """
    从 Stage 3 的结果中取出全局最优单节点干预方案。进一步找所有候选节点里面谁最好
    """
    # 读取总表
    single_df = stage3_res["single_df"]
    # 按objective找最高分
    best_row = single_df.sort_values("objective", ascending=False).iloc[0]
    # 根据node找对应详细结果
    node = int(best_row["node"])
    detail = stage3_res["best_details"][node]

    return {
        "best_row": best_row,
        "policy": detail["policy"],
        "policy_traj": detail["policy_traj"],
        "metrics": detail["metrics"],
    }


def attach_candidate_scores(
    single_df: pd.DataFrame,
    candidate_df: pd.DataFrame = None,
    roi_names=None,
) -> pd.DataFrame:
    """
    Attach candidate scores and ROI names to Stage 3 single-node search results.
    把 Stage 2 的候选节点信息和 ROI 名称合并到 Stage 3 的结果表里（信息补充）。增强可解释性
    """
    out = single_df.copy()

    # --------------------------------------------------
    # 1. 合并 candidate_df 里的候选信息
    # --------------------------------------------------
    if candidate_df is not None and len(candidate_df) > 0:
        cand = candidate_df.copy()

        if "node" not in cand.columns and "roi" in cand.columns:
            cand = cand.rename(columns={"roi": "node"})

        keep_cols = ["node"]

        for c in [
            "candidate_score",
            "is_upstream_of_lesion",
            "is_downstream_of_lesion",
            "roi_name",
        ]:
            if c in cand.columns:
                keep_cols.append(c)

        cand = cand[keep_cols].drop_duplicates("node")

        out = out.merge(
            cand,
            on="node",
            how="left",
            suffixes=("", "_candidate")
        )

    # --------------------------------------------------
    # 2. 如果没有 roi_name，则用 roi_names 补
    # --------------------------------------------------
    if "roi_name" not in out.columns and roi_names is not None:
        roi_name_list = [
            roi_names[int(n)] if int(n) < len(roi_names) else str(int(n))
            for n in out["node"].values
        ]

        # 插到 node 后面
        node_pos = out.columns.get_loc("node")
        out.insert(node_pos + 1, "roi_name", roi_name_list)

    # --------------------------------------------------
    # 3. 如果已有 roi_name 但有缺失，也补全
    # --------------------------------------------------
    elif "roi_name" in out.columns and roi_names is not None:
        out["roi_name"] = [
            out.loc[i, "roi_name"]
            if pd.notna(out.loc[i, "roi_name"])
            else (
                roi_names[int(out.loc[i, "node"])]
                if int(out.loc[i, "node"]) < len(roi_names)
                else str(int(out.loc[i, "node"]))
            )
            for i in out.index
        ]

        # 确保 roi_name 在 node 后面
        cols = list(out.columns)
        cols.remove("roi_name")
        node_pos = cols.index("node")
        cols.insert(node_pos + 1, "roi_name")
        out = out[cols]

    return out

