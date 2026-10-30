# 数字孪生 SDK 交付说明

## 基本信息

- SDK 名称：数字孪生脑扰动可视化 SDK
- Python 包名：`npi_sdk`
- 版本：`0.1.7`
- 推荐 Python：`3.10.x`、`3.11.x` 或 `3.12.x`
- 支持 Python：`>=3.9`
- 提供日期：`2026-08-13`
- 项目位置：`E:\脑扰动可视化2.0`
- 交付 wheel：`dist\npi_brain_sdk-0.1.7-py3-none-any.whl`

SDK、预存数据、因果 mask、demo、Web 可视化和 NPI 训练接口支持 Python 3.9 及以上版本。建议普通交付优先使用 Python 3.10.x、3.11.x 或 3.12.x；如果使用更新 Python，需确认目标版本有可用的 PyTorch wheel。

## 交付内容

- Python SDK：`npi_sdk`
- 标准客户端：`NPIClient`
- 新版前端：运行模式切换、NPCI A/B/C/D 阶段面板、FASK / Two-Step 模式隔离入口
- 内置运行资源：`npi_sdk\_engine`
- 安装配置：`pyproject.toml`
- 使用说明：`SDK_README.md`
- 可分发安装包：`dist\npi_brain_sdk-0.1.7-py3-none-any.whl`

wheel 包已包含 SDK 引擎代码、新版 Web 模板、预计算 EC 数据、普通因果算法缓存、NPCI 运行模块、已生成的 NPCI 缓存、示例 fMRI 文本数据，以及 FASK 所需的 Tetrad jar 资源。

## 依赖安装

完整 SDK 需要以下主要依赖：

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

源码运行时，建议使用 Python 3.10.x、3.11.x 或 3.12.x，并安装完整依赖：

```powershell
python -m pip install -r requirements.txt
```

该文件会安装完整依赖，包括 `torch>=2.2,<3`。如果当前 Python 版本没有匹配的 torch wheel，需要先安装与该 Python 版本兼容的 PyTorch。

在线环境中，直接安装 wheel 时，`pip` 会根据 wheel 元数据自动安装完整运行依赖，包括 PyTorch：

```powershell
python -m pip install dist\npi_brain_sdk-0.1.7-py3-none-any.whl
```

离线环境中，只发送 SDK wheel 不足以保证可用；需要同时提供依赖 wheel 包，尤其是匹配当前 Python 版本的 torch wheel，或先在目标机器安装好上述依赖。

## 后端和前端启动命令

本机可使用当前环境中的 Python：

```powershell
python --version
```

应显示：

```text
Python 3.9 或更高版本
```

安装 SDK wheel：

```powershell
python -m pip install dist\npi_brain_sdk-0.1.7-py3-none-any.whl
```

启动后端 Web 服务：

```powershell
python -c "from npi_sdk import NPIClient; NPIClient().run_web_app(host='127.0.0.1', port=5002)"
```

打开前端页面：

```text
http://127.0.0.1:5002/
```

如果 `5002` 端口被占用，换成 `5010`：

```powershell
python -c "from npi_sdk import NPIClient; NPIClient().run_web_app(host='127.0.0.1', port=5010)"
```

然后打开：

```text
http://127.0.0.1:5010/
```

停止后端服务：在运行服务的 PowerShell 窗口按 `Ctrl + C`。

## wheel 内置 Demo

安装 wheel 后，可以直接运行内置 demo：

```powershell
npi-demo
```

等价的 Python 调用方式：

```powershell
python -m npi_sdk.demo
```

查看 SDK 内置资源信息：

```powershell
npi-demo info
```

启动可视化网页 demo：

```powershell
npi-demo web --host 127.0.0.1 --port 5002
```

然后在浏览器打开：

```text
http://127.0.0.1:5002/
```

## 已封装能力

- 读取 NPI 有效连接矩阵
- 计算或读取因果算法 mask
- 按 `none / fask / twostep` 三种运行模式读取 FASK / Two-Step 缓存
- 启动普通算法、FASK、Two-Step 的全脑预计算状态接口
- 计算局部 FASK / Two-Step 因果筛选结果
- 生成扰动节点因果骨架
- 构建指定 ROI 节点网络
- 启动 Web 可视化服务
- 使用新版 NPCI A/B/C/D 干预靶点与恢复效果接口
- 调用标准 NPI 训练、EC 计算和结果保存流程

## SDK 调用方式

安装后可在任意目录调用：

```python
from npi_sdk import NPIClient

client = NPIClient()
print(client.available_strengths())
```

SDK 0.1.7 与 Web 前端一致，支持三种运行模式：

```text
none      纯 NPI
fask      FASK 前筛 + NPI
twostep   Two-Step 前筛 + NPI
```

FASK / Two-Step 的全脑和局部缓存会按运行模式分开保存。例如：

```python
fask_prescreen = client.compute_mask("fask", pipeline_mode="fask")
twostep_normal = client.compute_mask("twostep", pipeline_mode="fask")
status = client.precompute_causal_status(pipeline_mode="fask")
```

最强节点筛选已经和 NPCI C/D 打通。SDK 中对应流程是先在 B 阶段取 Top-N 节点，再把这些节点作为 C/D 候选靶点：

```python
top_targets = client.top_backbone_target_nodes(
    0.2,
    node_idx=0,
    top_n=10,
    mask_key="pcmci",
)

stage_cd = client.npci_intervention(
    0.2,
    node_idx=0,
    target_nodes=top_targets,
    stage3_params={"profile": "fast"},
)
```

普通用户默认预计算建议使用：

```python
client.precompute_default_masks(group="standard")
```

该命令只运行 Python 内部普通算法，不会默认触发 MATLAB Two-Step。

如需指定外部数据目录：

```python
client = NPIClient(project_root=r"D:\path\to\npi_project")
```

## 最小验收

```python
from npi_sdk import NPIClient

client = NPIClient()
assert 0.2 in client.available_strengths()
assert client.get_ec_matrix(0.2).shape == (360, 360)
print(client.mask_summary(client.compute_mask("pcmci")))
```

通过标准：

- SDK 可正常 `import`
- `NPIClient()` 无需传入 `project_root`
- 可读取扰动强度
- 可读取 `(360, 360)` EC 矩阵
- 可返回因果算法 mask summary

## 注意事项

- 普通读取、因果 mask、骨架网络和 Web 可视化可使用 wheel 内置资源。
- NPI 训练接口需要 PyTorch，NPCI 快速筛查和局部 LPCMCI 需要 matplotlib/seaborn/tigramite；`0.1.7` wheel 已把这些包声明为默认依赖。离线安装或 `--no-deps` 安装时仍需手动提供对应 wheel。
- 默认普通算法包括 `granger`、`vargranger`、`pcmci`、`sparsevar`，它们不依赖 MATLAB。
- `consensus`、`ensemble` 是组合/派生结果，依赖基础算法缓存。
- FASK 依赖 Java 运行环境；wheel 内置 Tetrad jar，但不内置 Java 运行时。
- Two-Step 仍依赖 MATLAB 环境；SDK 不会默认运行 Two-Step。

