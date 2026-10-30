# FASK / Two-Step 接入说明

本项目已经接入论文中的两个因果发现算法：

- FASK：通过 Java + Tetrad 后端运行。
- Two-Step：通过 MATLAB 原版 `.m` 代码运行。

## 页面入口

启动可视化系统后，左侧先选择“运行模式”：

- 纯 NPI
- FASK 前筛 + NPI
- Two-Step 前筛 + NPI

这个模式会作为 B/C/D 后续流程的主模式。选择 FASK 或 Two-Step 前筛后，NPCI 靶点筛查和恢复评价都会使用对应前筛后的 NPI EC；未被选为前筛的另一个 fMRI 算法会保留为普通因果算法，可继续用于后面的骨架查看和局部筛选。选择纯 NPI 时，FASK / Two-Step 都作为普通因果算法使用，不进入主流程前筛。

页面现在保留两条使用路线：

1. 先算法再 NPI：在“运行模式”选择 FASK 前筛或 Two-Step 前筛，然后点击“计算当前运行模式EC”，按下面流程生成可供 NPCI 继续使用的结果：

```text
原始 fMRI 时间序列 → FASK / Two-Step 因果骨架 → NPI EC → NPCI 靶点与恢复分析
```

2. 普通骨架查看 / NPI 后筛对照：在 B 区选择 FASK / Two-Step 或其他因果算法，再点击“临时显示预计算掩码”，只改变当前可视化中的骨架显示，不改变 C/D 主流程。

此外，FASK / Two-Step 支持局部筛选：选择强度和病灶 ROI 后，C 区按当前运行模式显示普通局部筛选按钮。纯 NPI 模式显示 FASK 和 Two-Step；FASK 前筛模式显示 Two-Step 普通筛选；Two-Step 前筛模式显示 FASK 普通筛选。

全脑预计算结果会按按钮分组保存：

- Granger
- VAR-Granger
- PCMCI
- Sparse VAR
- FASK
- Two-Step

同时，显式点击“计算全部EC结果”会保存纯 NPI，以及上述算法前筛后的 NPI EC 文件。后续重新打开系统时，可以直接从已保存文件读取，不需要每次重新计算。

NPCI 的 C/D 预计算结果按运行模式分开保存，避免不同模式之间互相读取：

- 纯 NPI：`data/npci_intervention_results/none/`
- FASK 前筛 + NPI：`data/npci_intervention_results/fask/`
- Two-Step 前筛 + NPI：`data/npci_intervention_results/twostep/`

每个模式下还会单独保存 backbone、stage3 节点缓存和最终 C/D 结果。旧版没有模式分目录的 NPCI 缓存不会再被新版直接复用。

全脑 FASK / Two-Step 结果也按“当前运行模式 + 算法角色”分开保存，避免前筛结果和普通算法结果串用：

- 纯 NPI 下的普通全脑算法：`data/causal_mask_results/full_fmri/none/ordinary/`
- FASK 前筛下的 FASK 全脑前筛：`data/causal_mask_results/full_fmri/fask/prefilter/`
- FASK 前筛下的普通 Two-Step：`data/causal_mask_results/full_fmri/fask/ordinary/`
- Two-Step 前筛下的 Two-Step 全脑前筛：`data/causal_mask_results/full_fmri/twostep/prefilter/`
- Two-Step 前筛下的普通 FASK：`data/causal_mask_results/full_fmri/twostep/ordinary/`

因此，旧版直接保存在 `data/causal_mask_results/fask.npz` 或 `twostep.npz` 的全脑缓存不会再作为新版 FASK / Two-Step 模式结果复用。

局部 FASK / Two-Step 筛选结果也按“当前运行模式 + 算法角色”分开保存，避免 FASK 前筛模式误读 Two-Step 前筛模式的局部结果：

- 纯 NPI 下的普通局部算法：`data/causal_mask_results/local_fmri/none/ordinary/`
- FASK 前筛下的普通 Two-Step：`data/causal_mask_results/local_fmri/fask/ordinary/`
- Two-Step 前筛下的普通 FASK：`data/causal_mask_results/local_fmri/twostep/ordinary/`
- 被选为前筛算法时的专用缓存：`data/causal_mask_results/local_fmri/<mode>/prefilter/`

注意：Two-Step 只会在前端显式点击 Two-Step 或“全部”相关按钮、或在代码中明确指定 `twostep` 时运行。原版 MATLAB Two-Step 在 360 ROI 全脑数据上可能非常慢，系统默认超时时间已提高到 7200 秒。SDK 的安全默认预计算不会自动触发 MATLAB Two-Step。

页面中的预计算按钮已分开：

- 预计算普通算法：Granger、VAR-Granger、PCMCI、Sparse VAR。
- 预计算 FASK 全脑：只运行 FASK 全脑骨架。
- 预计算 Two-Step 全脑：只运行 Two-Step 全脑骨架，可能耗时数小时。
- 预计算全部算法：运行普通算法 + FASK + Two-Step。

## Python 代码调用

在项目源码目录中可以这样调用：

```python
from npi_sdk import NPIClient

client = NPIClient(project_root=r"E:\脑扰动可视化2.0")

# 纯 NPI 模式：FASK / Two-Step 作为普通因果算法使用
fask_normal = client.compute_mask("fask", pipeline_mode="none")

# FASK 前筛 + NPI 模式：FASK 是前筛算法
fask_prescreen = client.compute_mask("fask", pipeline_mode="fask")

# FASK 前筛 + NPI 模式：Two-Step 作为普通因果算法使用
two_step_normal = client.compute_mask("twostep", pipeline_mode="fask")

# 默认只预计算 Python 内部普通算法，不会自动触发 MATLAB Two-Step
standard_status = client.precompute_default_masks(group="standard")

# 查看当前模式下 FASK / Two-Step 全脑预计算状态
fask_mode_status = client.precompute_causal_status(pipeline_mode="fask")
```

注意：`pipeline_mode` 必须和页面左侧选择的运行模式一致，避免读取到其他模式的旧缓存。`twostep` 建议传入较小 ROI 子网络或在研究复现实验中单独调用；如果没有 MATLAB 环境，不建议在用户电脑上直接运行 Two-Step。

如果只想拿到算法函数表：

```python
from npi_sdk.algorithms import get_cached_mask_functions

algorithms = get_cached_mask_functions(
    project_root=r"E:\脑扰动可视化2.0",
    pipeline_mode="fask",
)
fask_func = algorithms["fask"]
two_step_func = algorithms["twostep"]
```

## 项目内依赖路径

FASK 运行包和源码已经放到项目内：

```text
E:\脑扰动可视化2.0\external_algorithms\fask\tetrad-lib-7.6.11-SNAPSHOT-shaded.jar
E:\脑扰动可视化2.0\external_algorithms\fask\tetrad-lib-7.6.11-SNAPSHOT-sources.jar
E:\脑扰动可视化2.0\external_algorithms\fask\source
```

如果以后想临时换成别的 Tetrad jar，可以设置环境变量：

```powershell
$env:NPI_FASK_TETRAD_JAR = "D:\path\to\tetrad-lib-7.6.11-SNAPSHOT-shaded.jar"
```

Two-Step 原版 MATLAB 源码也已经放到项目内：

```text
E:\脑扰动可视化2.0\external_algorithms\two_step\Two-Step_Algorithm
```

如果以后想临时换成别的 Two-Step 源码目录，可以设置：

```powershell
$env:NPI_TWO_STEP_ALGO_DIR = "D:\path\to\Two-Step_Algorithm"
```

MATLAB 路径可以设置：

```powershell
$env:MATLAB_EXE = "D:\path\to\matlab.exe"
```

## 数据格式

系统内部会把 fMRI 信号整理成：

```text
时间点 × ROI
```

FASK 直接使用这个格式。Two-Step 原始代码要求：

```text
变量数 × 样本数
```

这个转置已经在桥接脚本里自动完成，不需要手动处理。
