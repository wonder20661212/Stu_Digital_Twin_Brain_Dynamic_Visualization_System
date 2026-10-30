import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# =========================================================
# 0. Reproducibility
# =========================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# =========================================================
# 1. Models
# =========================================================
class ANN_MLP(nn.Module):
    """Use MLP as a surrogate brain"""

    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, pred_steps=2, dropout=0.2):
        super().__init__()
        self.pred_steps = pred_steps
        self.output_dim = output_dim

        self.func = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(latent_dim, output_dim * pred_steps),
        )
        self.to(device)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        out = self.func(x)
        return out.view(batch_size, self.pred_steps, self.output_dim)


class ANN_CNN(nn.Module):
    """Use CNN as a surrogate brain"""

    def __init__(self, in_channels, hidden_channels, out_channels, data_length, pred_steps=2):
        super().__init__()
        self.in_channels = in_channels
        self.data_length = data_length
        self.pred_steps = pred_steps

        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=2),
        )
        self.Linear = nn.Linear(out_channels * (data_length - 2), in_channels * pred_steps)
        self.to(device)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)

        x = x.view(batch_size, self.data_length, self.in_channels)
        x = x.permute(0, 2, 1)  # (batch, channel, time)

        pred = self.CNN(x)
        pred = pred.reshape(batch_size, -1)
        pred = self.Linear(pred)
        return pred.view(batch_size, self.pred_steps, self.in_channels)


class ANN_RNN(nn.Module):
    """Use RNN as a surrogate brain"""

    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, data_length, pred_steps=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.data_length = data_length
        self.pred_steps = pred_steps

        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.rnn = nn.RNN(input_size=1, hidden_size=latent_dim, batch_first=True)
        self.output = nn.Linear(latent_dim, output_dim * pred_steps)
        self.to(device)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)

        x = x.view(batch_size, self.data_length, self.input_dim)
        encodes = self.enc(x)  # (batch, time, latent_dim)

        init_hidden = encodes[:, -1, :].unsqueeze(0).contiguous()  # (1, batch, latent_dim)
        zero_input = torch.zeros((batch_size, 1, 1), dtype=x.dtype, device=x.device)

        ht, _ = self.rnn(zero_input, init_hidden)   # ht: (batch, 1, latent_dim)
        out = self.output(ht[:, 0, :])              # (batch, output_dim * pred_steps)
        return out.view(batch_size, self.pred_steps, self.output_dim)


class ANN_VAR(nn.Module):
    """Use linear VAR-like surrogate"""

    def __init__(self, input_dim, output_dim, pred_steps=2):
        super().__init__()
        self.pred_steps = pred_steps
        self.output_dim = output_dim
        self.func = nn.Linear(input_dim, output_dim * pred_steps)
        self.to(device)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        out = self.func(x)
        return out.view(batch_size, self.pred_steps, self.output_dim)


# =========================================================
# 2. Data utilities
# =========================================================
def multi2one(time_series, input_steps, pred_steps):
    """
    Split the time series into (input_X, target_Y)
    input_X shape : (n_samples, input_steps * n_area)
    target_Y shape: (n_samples, pred_steps  * n_area)
    """
    n_step, n_area = time_series.shape
    n_samples = n_step - input_steps - pred_steps + 1

    input_X = np.zeros((n_samples, n_area * input_steps), dtype=np.float32)
    target_Y = np.zeros((n_samples, n_area * pred_steps), dtype=np.float32)

    for i in range(n_samples):
        input_X[i] = time_series[i:i + input_steps].reshape(-1)
        target_Y[i] = time_series[i + input_steps:i + input_steps + pred_steps].reshape(-1)

    return input_X, target_Y


def split_train_test(input_X, target_Y, train_set_proportion=0.8):
    """
    Chronological split
    """
    n_samples = input_X.shape[0]
    n_train = int(train_set_proportion * n_samples)

    train_X = input_X[:n_train]
    train_Y = target_Y[:n_train]
    test_X = input_X[n_train:]
    test_Y = target_Y[n_train:]

    return train_X, train_Y, test_X, test_Y


class NoisyDataset(data.Dataset):
    """
    Online Gaussian noise augmentation for inputs only.
    """
    def __init__(self, X, Y, noise_std=0.0):
        self.X = X
        self.Y = Y
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        y = self.Y[idx]
        return x, y


# =========================================================
# 3. Training / evaluation
# =========================================================
def evaluate_loader(model, loader, loss_fn, pred_steps):
    model.eval()
    total_loss = 0.0
    total_num = 0

    with torch.no_grad():
        for X, y in loader:
            y_hat = model(X)
            y = y.view(y.size(0), pred_steps, -1)
            loss = loss_fn(y_hat, y)

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_num += bs

    return total_loss / max(total_num, 1)


def train_NN(
    model,
    input_X,
    target_Y,
    pred_steps,
    batch_size=50,
    train_set_proportion=0.8,
    num_epochs=100,
    lr=1e-3,
    l2=0.0,
    patience=10,
    min_delta=0.0,
    noise_std=0.0,
    seed=42,
    verbose=True
):
    """
    Stable training version:
    - chronological split
    - online noise augmentation only for optimization
    - clean/noisy/test losses separated
    - true early stopping with deepcopy(best_state)
    """
    set_seed(seed)

    train_X_np, train_Y_np, test_X_np, test_Y_np = split_train_test(
        input_X, target_Y, train_set_proportion=train_set_proportion
    )

    train_inputs = torch.tensor(train_X_np, dtype=torch.float32).to(device)
    train_targets = torch.tensor(train_Y_np, dtype=torch.float32).to(device)
    test_inputs = torch.tensor(test_X_np, dtype=torch.float32).to(device)
    test_targets = torch.tensor(test_Y_np, dtype=torch.float32).to(device)

    # optimization loader: noisy inputs
    train_dataset_noisy = NoisyDataset(train_inputs, train_targets, noise_std=noise_std)
    train_loader_noisy = data.DataLoader(
        train_dataset_noisy, batch_size=batch_size, shuffle=True, drop_last=False
    )

    # evaluation loaders: clean train / clean test
    train_dataset_clean = data.TensorDataset(train_inputs, train_targets)
    test_dataset_clean = data.TensorDataset(test_inputs, test_targets)

    train_loader_clean = data.DataLoader(
        train_dataset_clean, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader_clean = data.DataLoader(
        test_dataset_clean, batch_size=batch_size, shuffle=False, drop_last=False
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    history = {
        "epoch": [],
        "train_noisy_loss": [],
        "train_clean_loss": [],
        "test_loss": [],
        "best_epoch": None,
        "best_test_loss": None,
    }

    best_test_loss = float("inf")
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(1, num_epochs + 1):
        # -------------------------
        # train on noisy inputs
        # -------------------------
        model.train()
        running_noisy_loss = 0.0
        running_num = 0

        for X, y in train_loader_noisy:
            y_hat = model(X)
            y = y.view(y.size(0), pred_steps, -1)
            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            running_noisy_loss += loss.item() * bs
            running_num += bs

        train_noisy_loss = running_noisy_loss / max(running_num, 1)

        # -------------------------
        # evaluate on clean train / clean test
        # -------------------------
        train_clean_loss = evaluate_loader(model, train_loader_clean, loss_fn, pred_steps)
        test_loss = evaluate_loader(model, test_loader_clean, loss_fn, pred_steps)

        history["epoch"].append(epoch)
        history["train_noisy_loss"].append(float(train_noisy_loss))
        history["train_clean_loss"].append(float(train_clean_loss))
        history["test_loss"].append(float(test_loss))

        if verbose:
            print(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] | "
                f"train_noisy: {train_noisy_loss:.6f} | "
                f"train_clean: {train_clean_loss:.6f} | "
                f"test: {test_loss:.6f}"
            )

        # -------------------------
        # early stopping
        # -------------------------
        if test_loss < best_test_loss - min_delta:
            best_test_loss = test_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}, best epoch = {best_epoch}, best test loss = {best_test_loss:.6f}")
                break

    model.load_state_dict(best_model_state)

    history["best_epoch"] = best_epoch
    history["best_test_loss"] = float(best_test_loss)

    return model, history


# =========================================================
# 4. Prediction helpers
# =========================================================
def predict_array(model, input_X, batch_size=256):
    """
    input_X: np.ndarray, shape (n_samples, input_steps*n_roi)
    return : np.ndarray, shape (n_samples, pred_steps, n_roi)
    """
    model.eval()

    X_tensor = torch.tensor(input_X, dtype=torch.float32).to(device)
    dataset = data.TensorDataset(X_tensor)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for (X,) in loader:
            y_hat = model(X)
            preds.append(y_hat.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    return preds


def evaluate_test(model, input_X, target_Y, pred_steps, train_set_proportion=0.8, batch_size=256):
    """
    Evaluate only on held-out test set
    """
    _, _, test_X_np, test_Y_np = split_train_test(
        input_X, target_Y, train_set_proportion=train_set_proportion
    )

    pred_all = predict_array(model, test_X_np, batch_size=batch_size)
    targets_all = test_Y_np.reshape(-1, pred_steps, pred_all.shape[-1])

    mse_by_step = []
    mae_by_step = []

    for k in range(pred_steps):
        mse_k = np.mean((pred_all[:, k, :] - targets_all[:, k, :]) ** 2)
        mae_k = np.mean(np.abs(pred_all[:, k, :] - targets_all[:, k, :]))
        mse_by_step.append(float(mse_k))
        mae_by_step.append(float(mae_k))

    return {
        "test_inputs": test_X_np,
        "test_targets": targets_all,
        "pred_all": pred_all,
        "mse_by_step": mse_by_step,
        "mae_by_step": mae_by_step,
    }


# =========================================================
# 5. FC / EC / Jacobian
# =========================================================
def corrcoef(signals, eps=1e-8):
    """
    Stable correlation matrix computation
    signals shape: (T, ROI)
    """
    signals = np.asarray(signals, dtype=np.float32)

    # 先把 nan/inf 清掉
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

    # 去均值
    signals = signals - signals.mean(axis=0, keepdims=True)

    # 防止某些ROI方差接近0
    std = signals.std(axis=0, keepdims=True)
    good = std[0] > eps

    z = np.zeros_like(signals, dtype=np.float32)
    z[:, good] = signals[:, good] / std[:, good]

    fc = np.corrcoef(z.T)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)

    # 对角线设为1
    np.fill_diagonal(fc, 1.0)
    return fc.astype(np.float32)


def model_FC(
    model,
    node_num,
    steps,
    pred_steps,
    sim_len=1200,
    burn_in=100,
    noise_std=0.02,
    init_data=None,
    clip_value=3.0
):
    """
    Stable rollout for model FC.
    - 用真实数据片段做初始种子，比全零更稳
    - 每一步做 nan/inf 清理 + clip
    - 给极低方差状态加微小噪声，防止 corrcoef 全NaN
    """
    model.eval()

    # ---------- 初始化 ----------
    if init_data is not None and len(init_data) >= steps:
        start = np.random.randint(0, len(init_data) - steps + 1)
        rollout = [init_data[start + i].astype(np.float32).copy() for i in range(steps)]
    else:
        rollout = [0.01 * np.random.randn(node_num).astype(np.float32) for _ in range(steps)]

    total_steps = sim_len + burn_in

    # ---------- 自回归滚动 ----------
    with torch.no_grad():
        for _ in range(total_steps):
            model_input = np.array(rollout[-steps:], dtype=np.float32).reshape(-1)

            pred = model(torch.tensor(model_input, dtype=torch.float32).to(device))
            next_step = pred[0, 0].detach().cpu().numpy().astype(np.float32)

            # 数值清理
            next_step = np.nan_to_num(
                next_step,
                nan=0.0,
                posinf=clip_value,
                neginf=-clip_value
            )
            next_step = np.clip(next_step, -clip_value, clip_value)

            # 加一点创新噪声，避免塌成常数轨道
            next_step = next_step + noise_std * np.random.randn(node_num).astype(np.float32)

            # 如果方差过小，再补一点极小扰动
            if np.std(next_step) < 1e-6:
                next_step = next_step + 1e-3 * np.random.randn(node_num).astype(np.float32)

            rollout.append(next_step)

    # 丢掉最开始的窗口和 burn-in
    sim_signals = np.asarray(rollout[steps + burn_in:], dtype=np.float32)

    # 再做一次保险清理
    sim_signals = np.nan_to_num(sim_signals, nan=0.0, posinf=0.0, neginf=0.0)

    return corrcoef(sim_signals)



def model_EC(model,input_X,target_Y,pert_strength,pred_steps,step_idx=None,batch_size=256):
    """
    Infer EC by perturbing the last input time step.
    Parameters
    ----------
    model : torch.nn.Module
        Trained surrogate model.
    input_X : np.ndarray
        Shape: (n_samples, input_steps * node_num)
    target_Y : np.ndarray
        Shape: (n_samples, pred_steps * node_num)
    pert_strength : float
        Perturbation strength added to one node at the last input time step.
    pred_steps : int
        Number of predicted future steps.
    step_idx : int or None, default=None
        - None: return EC for all predicted steps, shape = (pred_steps, node_num, node_num)
        - int : return EC for one specific step, shape = (node_num, node_num)
    batch_size : int, default=256
        Batch size used in predict_array.
    Returns
    -------
    np.ndarray
        If step_idx is None:
            shape = (pred_steps, node_num, node_num)
        else:
            shape = (node_num, node_num)
    Notes
    -----
    - Perturbation is applied only to the last input time step.
    - Diagonal is NOT zeroed inside this function.
    """
    input_X = np.asarray(input_X, dtype=np.float32)
    target_Y = np.asarray(target_Y, dtype=np.float32)

    if input_X.ndim != 2 or target_Y.ndim != 2:
        raise ValueError("input_X and target_Y must both be 2D arrays.")

    node_num = target_Y.shape[1] // pred_steps
    if node_num * pred_steps != target_Y.shape[1]:
        raise ValueError("target_Y.shape[1] is not divisible by pred_steps.")

    steps = input_X.shape[1] // node_num
    if steps * node_num != input_X.shape[1]:
        raise ValueError("input_X.shape[1] is not divisible by node_num.")

    if step_idx is not None:
        if not isinstance(step_idx, int):
            raise ValueError("step_idx must be None or an integer.")
        if step_idx < 0 or step_idx >= pred_steps:
            raise ValueError(f"step_idx must be in [0, {pred_steps - 1}]")

    # 基线预测
    base_pred_all = predict_array(model, input_X, batch_size=batch_size)  # (n_samples, pred_steps, node_num)

    # 分配输出
    if step_idx is None:
        NPI_EC = np.zeros((pred_steps, node_num, node_num), dtype=np.float32)
    else:
        NPI_EC = np.zeros((node_num, node_num), dtype=np.float32)

    # 逐个扰动 source node
    for node in range(node_num):
        perturbation = np.zeros((steps, node_num), dtype=np.float32)
        perturbation[-1, node] = pert_strength

        perturbed_X = input_X + perturbation.reshape(1, -1)
        perturbed_pred_all = predict_array(model, perturbed_X, batch_size=batch_size)

        diff = perturbed_pred_all - base_pred_all   # (n_samples, pred_steps, node_num)

        if step_idx is None:
            # 对所有 horizon 都保留
            # mean_diff shape: (pred_steps, node_num)
            mean_diff = np.mean(diff, axis=0)
            NPI_EC[:, node, :] = mean_diff
        else:
            # 只取某一个 horizon
            # mean_diff shape: (node_num,)
            mean_diff = np.mean(diff[:, step_idx, :], axis=0)
            NPI_EC[node, :] = mean_diff

    return NPI_EC

def model_Jacobian(model, input_X, steps, pred_step_idx=0):
    """
    Average Jacobian wrt the last input step.
    """
    model.eval()
    node_num = input_X.shape[1] // steps
    jacobian = np.zeros((node_num, node_num), dtype=np.float32)

    for i in range(input_X.shape[0]):
        x = torch.tensor(input_X[i], dtype=torch.float32, device=device, requires_grad=True)

        J = torch.autograd.functional.jacobian(
            lambda z: model(z.unsqueeze(0))[0, pred_step_idx], x
        )  # (node_num, input_dim)

        jacobian += J[:, -node_num:].detach().cpu().numpy()

    return (jacobian / input_X.shape[0]).T


# # =========================================================
# # 6. Small utilities
# # =========================================================
# def flat_without_diagonal(matrix):
#     matrix = np.asarray(matrix)
#     n = matrix.shape[0]
#     mask = ~np.eye(n, dtype=bool)
#     return matrix[mask]