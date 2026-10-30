# 数字孪生脑扰动可视化 SDK 操作指南

## 1. 准备文件

使用者需要拿到以下文件：

```text
npi_brain_sdk-0.1.7-py3-none-any.whl
SDK_具体操作说明.md
```

将两个文件放在同一个目录，例如：

```text
D:\npi_sdk_package
```

## 2. Python 版本要求

SDK 支持：

```text
Python >= 3.9
```

推荐使用：

```text
Python 3.10.x 或 Python 3.11.x
```

检查 Python 版本：

```powershell
python --version
```

如果显示 `Python 3.9`、`Python 3.10`、`Python 3.11`、`Python 3.12` 或更高版本，即满足 SDK 的 Python 版本要求。实际安装时还需要当前 Python 版本有可用的 PyTorch wheel。

## 3. 安装 SDK

进入 wheel 所在目录：

```powershell
cd D:\npi_sdk_package
```

安装 SDK：

```powershell
python -m pip install npi_brain_sdk-0.1.7-py3-none-any.whl
```

在线环境下，`pip` 会自动安装 SDK 的完整依赖，包括重新训练 NPI 需要的 PyTorch。

离线环境下，仅有 SDK wheel 可能不够，还需要同时提供依赖 wheel 包，尤其是与当前 Python 版本匹配的 `torch` wheel，或提前安装好依赖。

## 4. 快速验证

安装完成后运行：

```powershell
npi-demo
```

也可以使用 Python 方式运行：

```powershell
python -m npi_sdk.demo
```

正常情况下会看到类似结果：

```text
NPI Brain SDK demo check
Available strengths: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EC shape at 0.2: (360, 360)
pcmci mask: {'shape': [360, 360], 'edges': 3762, 'density': 0.029109}
Result: OK
```

需要 JSON 输出时运行：

```powershell
npi-demo --json
```

查看 SDK 内置资源：

```powershell
npi-demo info
```

## 5. 打开可视化网页

启动 Web 服务：

```powershell
npi-demo web --host 127.0.0.1 --port 5002
```

然后在浏览器打开：

```text
http://127.0.0.1:5002/
```

停止服务：

```text
在运行服务的 PowerShell 窗口按 Ctrl + C
```

如果 `5002` 端口被占用，可以换成其他端口：

```powershell
npi-demo web --host 127.0.0.1 --port 5010
```

然后打开：

```text
http://127.0.0.1:5010/
```

## 6. Python 代码调用

最小调用示例：

```python
from npi_sdk import NPIClient

client = NPIClient()

print(client.available_strengths())
print(client.get_ec_matrix(0.2).shape)
print(client.mask_summary(client.compute_mask("pcmci")))
```

读取指定扰动强度的 EC 矩阵：

```python
from npi_sdk import NPIClient

client = NPIClient()
ec = client.get_ec_matrix(0.2)
print(ec.shape)
```

计算或读取因果算法 mask：

```python
mask = client.compute_mask("pcmci")
summary = client.mask_summary(mask)
print(summary)
```

按当前 Web 系统的三种运行模式读取 FASK / Two-Step：

```python
# 三种模式：none、fask、twostep

# 纯 NPI 模式下，FASK 作为普通因果算法使用
fask_normal = client.compute_mask("fask", pipeline_mode="none")

# FASK 前筛 + NPI 模式下，FASK 是前筛算法
fask_prescreen = client.compute_mask("fask", pipeline_mode="fask")

# FASK 前筛 + NPI 模式下，Two-Step 作为普通因果算法使用
twostep_normal_in_fask_mode = client.compute_mask("twostep", pipeline_mode="fask")
```

预计算普通算法：

```python
standard = client.precompute_default_masks(group="standard")
print(standard.keys())
```

注意：`group="standard"` 只运行 Python 内部普通算法，不会默认调用 FASK / Two-Step。

查看某个模式下的全脑因果算法缓存状态：

```python
status = client.precompute_causal_status(pipeline_mode="fask")
print(status["computed_keys"])
print(status["missing_keys"])
```

局部 FASK / Two-Step 筛选：

```python
local = client.local_fmri_mask(
    "fask",
    strength=0.2,
    node_idx=0,
    pipeline_mode="twostep",
    max_nodes=40,
)
print(local["message"])
```

构建扰动节点因果骨架：

```python
backbone = client.causal_backbone(
    0.2,
    node_idx=0,
    top_n=10,
    mask_key="pcmci",
)
print(backbone)
```

把“最强节点筛选”结果接到 NPCI C/D：

```python
# 先按当前 B 阶段骨架取 Top-N 节点，ROI 编号是 0 基索引
top_targets = client.top_backbone_target_nodes(
    0.2,
    node_idx=0,
    top_n=10,
    mask_key="pcmci",
)

# 再把这些节点作为 NPCI Stage C/D 的候选靶点
result = client.npci_intervention(
    0.2,
    node_idx=0,
    target_nodes=top_targets,
    stage3_params={"profile": "fast"},
)
print(result)
```

构建指定 ROI 节点网络：

```python
network = client.build_network(
    0.2,
    node_indices=[0, 5, 12, 18],
    direction_mode="dominant_pairs",
    mask_key="consensus",
)
print(network)
```

读取新版 NPCI 阶段默认配置：

```python
defaults = client.npci_stage3_defaults()
print(defaults)
```

读取已缓存的 NPCI 干预靶点和恢复结果：

```python
result = client.npci_intervention(0.3, node_idx=6, cached_only=True)
print(result)
```

如果要预计算并保存这组 Top-N 候选的 NPCI C/D 结果：

```python
top_targets = client.top_backbone_target_nodes(0.3, node_idx=6, top_n=12)
client.precompute_npci_intervention(
    0.3,
    node_idx=6,
    target_nodes=top_targets,
)
```

## 7. 可用因果算法

基础可用算法：

```text
granger
vargranger
pcmci
sparsevar
consensus
ensemble
```

fMRI 原生算法：

```text
fask
twostep
```

说明：

- `granger`、`vargranger`、`pcmci`、`sparsevar` 是 Python 内部算法，可以作为普通默认预计算使用。
- `consensus`、`ensemble` 是组合/派生结果，依赖基础算法缓存。
- `fask` 当前依赖 Java 运行环境和 SDK 内置的 Tetrad jar。
- `twostep` 当前仍依赖 MATLAB 环境；SDK 不会默认运行 Two-Step。
- `compute_mask("twostep")` 或 `precompute_default_masks(keys=["twostep"])` 只有在用户电脑 MATLAB 环境可用时才适合调用。

## 8. 依赖说明

SDK、预存 EC、因果 mask、demo、Web 可视化和 NPI 训练所需主要依赖：

```text
Flask>=2.3,<4
Flask-Cors>=4,<7
numpy>=1.24,<3
scipy>=1.10,<2
nibabel>=5.1,<6
scikit-image>=0.21,<1
pandas>=1.5,<3
torch>=2.2,<3
matplotlib>=3.7,<4
seaborn>=0.12,<1
tigramite>=5.2,<6
```

原始 notebook 或分析脚本可能还需要：

```text
nilearn>=0.10,<1
jupyter>=1.0,<2
```

## 9. NPI 训练接口

如果是源码运行、离线安装，或安装时使用了 `--no-deps`，需要手动补装完整依赖：

```powershell
python -m pip install -r requirements.txt
```

完整依赖中包含：

```text
torch>=2.2,<3
```

如果当前 Python 版本没有可用 torch wheel，需要安装与该 Python 版本兼容的 PyTorch。

训练调用示例：

```python
from npi_sdk import NPIClient

client = NPIClient()

result = client.train_and_compute_ec(
    mode="none",
    strengths=[0.2, 0.5],
    num_epochs=100,
    save=True,
)

print(result["mode"])
print(result["ec"][0.2].shape)
```

## 10. 常见问题

如果提示 `npi-demo` 不是内部或外部命令：

```powershell
python -m npi_sdk.demo
```

如果网页打不开，先确认运行 Web 服务的 PowerShell 窗口没有关闭，并检查端口是否被占用。可以换端口：

```powershell
npi-demo web --host 127.0.0.1 --port 5010
```

如果安装时依赖下载失败，说明运行环境无法联网或网络源不可用，需要提前准备依赖 wheel 包。

如果调用训练时报 PyTorch 相关错误，说明还没有安装训练依赖，或当前 Python 版本没有匹配的 PyTorch。

如果 FASK 报 Java 相关错误，说明当前电脑没有可用 Java 运行环境。普通 Python 因果算法和 Web 可视化仍然可以使用。

如果 Two-Step 报 MATLAB 相关错误，说明当前电脑没有可用 MATLAB 环境。SDK 默认不会运行 Two-Step；需要 Two-Step 时再单独配置 MATLAB。

如果想避免外部环境问题，建议普通用户优先使用：

```text
granger
vargranger
pcmci
sparsevar
consensus
ensemble
```

## 11. 最小验收标准

完成安装后，至少应通过以下两项。

第一项，运行：

```powershell
npi-demo
```

结果中应看到：

```text
Result: OK
```

第二项，启动网页：

```powershell
npi-demo web --host 127.0.0.1 --port 5002
```

浏览器打开：

```text
http://127.0.0.1:5002/
```

能看到可视化页面，即说明 SDK 基础功能和网页功能可用。

