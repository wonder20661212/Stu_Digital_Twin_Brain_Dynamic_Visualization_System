"""
    inputs:  shape (n_samples, using_steps * roi_num)
    signals: shape (T, roi_num)
    ANN:     torch model, 3-step-to-1-step predictor
Main outputs:
    healthy_traj
    lesion_traj
    lesion_response_df
    response_vector
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def predict_next_np(model: torch.nn.Module, x_flat: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    把已经训练好的 NPI 模型拿来做推理，输入一批历史窗口 x_flat，输出每个窗口对应的下一时刻脑区状态预测。
    相当于“单步脑动力学预测器”
    x_flat:
        Input windows, shape (B, using_steps * roi_num).
    Returns
        Predicted next state, shape (B, roi_num).每个 seed window 对应一个下一时刻预测脑状态
    """
    model.eval()
    device = next(model.parameters()).device
    x_flat = np.asarray(x_flat, dtype=np.float32)

    outs: List[np.ndarray] = []
    with torch.no_grad():
        for st in range(0, x_flat.shape[0], batch_size):
            xb = torch.tensor(x_flat[st: st + batch_size], dtype=torch.float32, device=device)
            yb = model(xb)
            if yb.ndim == 3:
                yb = yb[:, 0, :]
            outs.append(yb.detach().cpu().numpy())
    return np.vstack(outs)


def select_seed_windows(
    inputs: np.ndarray,
    n_seed_windows: int = 64,
    random_state: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从所有监督学习输入窗口 inputs 中，随机抽取一部分作为 rollout 的初始窗口。
    因为如果只用一个初始窗口，结果容易受偶然状态影响。现在做法是：
    从很多历史窗口中抽取 B 个 seed
    每个 seed 都 rollout 一条未来轨迹
    最后对 B 条轨迹取平均
    这样得到的 lesion response 更稳定。
    Returns
    seed_inputs, seed_indices：被抽中的输入窗口和这些窗口在原始 inputs 中的下标
    """
    inputs = np.asarray(inputs, dtype=float)
    # 看一共有多少个窗口
    n_total = inputs.shape[0]
    # 实际抽取数量不能超过总数量
    n_seed = min(int(n_seed_windows), n_total)
    # 不放回随机抽样
    rng = np.random.default_rng(random_state)
    # 用 random_state 固定结果
    idx = rng.choice(n_total, size=n_seed, replace=False)
    return inputs[idx], idx


def build_schedule_vector(name: str, horizon: int) -> np.ndarray:
    """
    Build intervention schedule vector.
    Returns
    -------
    v : np.ndarray
        shape = (horizon,)
        each value in [0, 1]
    """
    name = str(name)
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if name == "constant":
        v = np.ones(horizon, dtype=float)

    elif name == "early_pulse":
        v = np.zeros(horizon, dtype=float)
        end = max(1, horizon // 3)
        v[:end] = 1.0

    elif name == "middle_pulse":
        v = np.zeros(horizon, dtype=float)
        st = horizon // 3
        ed = max(st + 1, 2 * horizon // 3)
        v[st:ed] = 1.0

    elif name == "late_pulse":
        v = np.zeros(horizon, dtype=float)
        st = 2 * horizon // 3
        v[st:] = 1.0

    elif name == "ramp_up":
        v = np.linspace(0.0, 1.0, horizon)

    elif name == "ramp_decay":
        v = np.linspace(1.0, 0.0, horizon)

    elif name == "exponential_decay":
        t = np.arange(horizon, dtype=float)
        v = np.exp(-3.0 * t / max(1, horizon - 1))
        v = v / (v.max() + 1e-12)

    else:
        raise ValueError(f"Unknown schedule: {name}")

    return v.astype(float)


def rollout_npi(
    model: torch.nn.Module,
    seed_inputs: np.ndarray,
    roi_num: int,
    using_steps: int,
    horizon: int = 20,
    lesion_node: Optional[int] = None,
    lesion_strength: float = 0.0,
    lesion_mode: str = "subtract",
    policy: Optional[Sequence[Dict]] = None,
    batch_size: int = 256,
) -> np.ndarray:
    """
    用已经训练好的 3-step-to-1-step NPI 模型，递归预测未来多步轨迹
    虽然模型本身只会预测下一步，但是这个函数通过：
    预测 t+1，把 t+1 塞回历史窗口，再预测 t+2，再塞回去……直到 horizon 步，实现多步滚动预测。
    lesion 是在每一个 rollout step 都会重新作用到当前输入窗口的最后状态上，所以它更像：持续性病灶状态扰动
    而不是只在初始时刻打一下，然后让系统自由演化。
    lesion_node:
        If not None, a perturbation is applied to the last state of the input history.
    lesion_strength:
        Perturbation strength.
    lesion_mode:
        "subtract": x[node] -= lesion_strength.
        "zero":     x[node] = 0.
        "scale":    x[node] *= (1 - lesion_strength).
    policy:
        Optional list of interventions. Each item may contain:
        {
            "node": int,
            "sign": -1 or 1,
            "strength": float,
            "schedule": str,
            "schedule_weight": float,
            "schedule_vector": np.ndarray
        }
    Returns
    traj:shape = (B, horizon, roi_num)，例如：(32, 20, 360)
    含义是：32 条 seed 轨迹，每条往未来 rollout 20 步，每一步有 360 个 ROI 状态
    """
    if policy is None:
        policy = []

    seed_inputs = np.asarray(seed_inputs, dtype=float)
    B = seed_inputs.shape[0]
    # 例如：(32, 3, 360)，含义是：32 条初始轨迹，每条轨迹有 3 个历史时间点，每个时间点有 360 个 ROI
    hist = seed_inputs.reshape(B, using_steps, roi_num).copy()
    traj: List[np.ndarray] = []

    # 每个 rollout step 都会先复制当前历史窗口,每一次循环都会:
    # 1. 取当前历史窗口 hist
    # 2. 加病灶扰动 lesion
    # 3. 加干预 policy
    # 4. 用 ANN 预测下一步 y_next
    # 5. 把 y_next 加入历史窗口
    # 6. 继续下一步
    for t in range(int(horizon)):
        hist_in = hist.copy()

        # 1. Apply lesion perturbation
        if lesion_node is not None and lesion_strength != 0.0:
            node = int(lesion_node)
            # 在最后一个历史状态上加病灶扰动。subtract：对该脑区状态减去一个固定强度
            if lesion_mode == "subtract":
                hist_in[:, -1, node] -= float(lesion_strength)

            elif lesion_mode == "zero":
                hist_in[:, -1, node] = 0.0
            # scale：按比例削弱该脑区活动
            elif lesion_mode == "scale":
                hist_in[:, -1, node] *= (1.0 - float(lesion_strength))

            else:
                raise ValueError(
                    "lesion_mode must be one of {'subtract', 'zero', 'scale'}"
                )

        # 2. Apply intervention policy
        for item in policy:
            node = int(item["node"])
            sign = float(item.get("sign", 1.0))
            strength = float(item.get("strength", 0.0))
            # 优先级 1：直接使用 schedule_vector。直接给完整时间权重向量
            if "schedule_vector" in item:
                schedule_vec = np.asarray(item["schedule_vector"], dtype=float)

                if len(schedule_vec) != horizon:
                    raise ValueError(
                        f"schedule_vector length {len(schedule_vec)} "
                        f"does not match horizon {horizon}"
                    )

                w = float(schedule_vec[t])

            else:
                schedule_name = item.get("schedule", "constant")
                schedule_vec = build_schedule_vector(schedule_name, horizon)
                w = float(schedule_vec[t])
            # 实际干预公式：实际干预幅度 = sign × strength × schedule_weight
            hist_in[:, -1, node] += sign * strength * w

        # 3. Predict next state
        x_flat = hist_in.reshape(B, using_steps * roi_num)
        y_next = predict_next_np(model, x_flat, batch_size=batch_size)

        traj.append(y_next)

        # 4. Update rolling history window
        hist = np.concatenate(
            [hist[:, 1:, :], y_next[:, None, :]],
            axis=1
        )

    return np.stack(traj, axis=1)


def compute_lesion_response(
    model: torch.nn.Module,
    inputs: np.ndarray,
    roi_num: int,
    using_steps: int,
    lesion_idx: int,
    roi_names: Optional[Sequence[str]] = None,
    horizon: int = 20,
    n_seed_windows: int = 64,
    lesion_strength: float = 0.5,
    lesion_mode: str = "subtract",
    random_state: int = 123,
    batch_size: int = 256,
) -> Dict:
    """
    同一批 seed windows，分别跑健康 rollout 和病灶 rollout，然后比较二者差异。
    选取一批初始 seed windows，分别做健康 rollout 和病灶 rollout，然后比较两条轨迹的差异，得到每个 ROI 对病灶扰动的响应强度。
    Returns a dict that can be inspected immediately in the Notebook before
    entering Stage 2.
    """
    seed_inputs, seed_indices = select_seed_windows(inputs, n_seed_windows, random_state)

    healthy_traj = rollout_npi(
        model=model,
        seed_inputs=seed_inputs,
        roi_num=roi_num,
        using_steps=using_steps,
        horizon=horizon,
        lesion_node=None,
        lesion_strength=0.0,
        batch_size=batch_size,
    )

    lesion_traj = rollout_npi(
        model=model,
        seed_inputs=seed_inputs,
        roi_num=roi_num,
        using_steps=using_steps,
        horizon=horizon,
        lesion_node=lesion_idx,
        lesion_strength=lesion_strength,
        lesion_mode=lesion_mode,
        batch_size=batch_size,
    )
    # 形状diff.shape = (B, horizon, roi_num)，含义是：
    # 每个 seed，每个未来时间步，每个 ROI，在病灶状态下相对于健康状态的偏移
    diff = lesion_traj - healthy_traj
    # 每个 ROI 在整个 rollout 过程中，平均受到该 lesion 影响的强度。后面 lesion_response_df 就是主要根据它排序
    response_vector = np.mean(np.abs(diff), axis=(0, 1))
    signed_response_vector = np.mean(diff, axis=(0, 1))
    # 每个 rollout step 上，全脑平均病灶响应强度。这个指标适合画时间曲线
    temporal_response = np.mean(np.abs(diff), axis=(0, 2))
    # 每个时间步、每个 ROI 的病灶响应强度
    roi_temporal_response = np.mean(np.abs(diff), axis=0)
    # 把每个 ROI 的响应强度整理成表格,按响应强度排序
    lesion_response_df = pd.DataFrame({
        "roi": np.arange(roi_num, dtype=int),
        "response_abs": response_vector,
        "response_signed": signed_response_vector,
    })
    #生成表格时插入
    if roi_names is not None:
        lesion_response_df.insert(
            lesion_response_df.columns.get_loc("roi") + 1,
            "roi_name",
            lesion_response_df["roi"].astype(int).map(lambda i: roi_names[int(i)])
        )
    # 然后在画图函数里直接用roi_names=roi_names
    lesion_response_df = lesion_response_df.sort_values(
        "response_abs",
        ascending=False
    ).reset_index(drop=True)

    return {
        "lesion_idx": int(lesion_idx),
        "seed_indices": seed_indices,
        "seed_inputs": seed_inputs,
        "healthy_traj": healthy_traj,
        "lesion_traj": lesion_traj,
        "response_vector": response_vector,
        "signed_response_vector": signed_response_vector,
        "temporal_response": temporal_response,
        "roi_temporal_response": roi_temporal_response,
        "lesion_response_df": lesion_response_df,
    }


def load_mmp_labels_cortical_only(label_file):
    roi_names = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            if name.startswith('L_') or name.startswith('R_'):
                roi_names.append(name)
    return roi_names


def top_response_nodes(
    lesion_result: Dict,
    top_k: int = 20,
    exclude_lesion: bool = False,
) -> pd.DataFrame:
    """
    取前 top_k 个响应最强的脑区
    如果：exclude_lesion=True,就会把 lesion 自身脑区排除掉。这点很有用。
    因为有时 lesion 自己响应最大，但你后面做干预候选节点时，不一定想把病灶本身作为候选，而是更关心受病灶影响最大的下游脑区
    Return top lesion-response nodes.
    """
    df = lesion_result["lesion_response_df"].copy()
    if exclude_lesion:
        df = df[df["roi"] != int(lesion_result["lesion_idx"])]
    return df.head(int(top_k)).reset_index(drop=True)



def plot_top_response_bar(
    lesion_result: Dict,
    top_k: int = 20,
    roi_names: Optional[Sequence[str]] = None,
    ax=None,
    show_roi_id: bool = True,
):
    """
    画的是 top-k ROI 的response_abs,也就是病灶响应强度排名柱状图。
    它回答的问题是哪些脑区最受当前 lesion 影响？这个图适合放在：病灶响应分析结果或者作为后续局部子网络构建的依据图。
    """
    df = top_response_nodes(
        lesion_result,
        top_k=top_k,
        exclude_lesion=False
    ).copy()

    if roi_names is not None:
        if show_roi_id:
            labels = [
                f"{int(x)} | {roi_names[int(x)]}"
                for x in df["roi"].values
            ]
        else:
            labels = [
                roi_names[int(x)]
                for x in df["roi"].values
            ]
    else:
        labels = [
            str(int(x))
            for x in df["roi"].values
        ]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.bar(np.arange(len(df)), df["response_abs"].values)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Mean |lesion - healthy|")
    ax.set_title(f"Top {top_k} lesion-response ROIs")
    return ax
