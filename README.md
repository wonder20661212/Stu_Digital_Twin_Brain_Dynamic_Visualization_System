# 数字孪生脑因果分析与动态干预可视化系统

> 脑影像驱动的因果分析与虚拟干预平台

本项目面向功能磁共振脑网络分析与数字孪生脑干预研究，提供从脑影像数据、因果网络构建、虚拟扰动、关键节点筛选，到候选干预靶点搜索和恢复效果评价的一体化可视化流程。

项目以神经扰动推断（NPI）为基础能力之一，在此之上完成了多类因果筛选算法、面向功能磁共振数据的 FASK 与 Two-Step 前筛流程、NPCI 动态核心与靶点搜索、恢复评价、三维脑网络展示以及 Python SDK 的系统集成。因此，本仓库不是单独的 NPI 算法复现仓库，而是一个围绕“因果脑网络—虚拟扰动—干预评估”构建的完整研究系统。

> **使用边界：** 本系统属于科研与教学软件，输出结果用于方法研究、假设生成和方案比较，不能替代临床诊断、治疗决策或真实神经刺激实验。

## 核心功能

- **全脑因果网络分析**：当前内置数据覆盖 360 个脑区，可分析脑区间的作用方向、连接强度以及兴奋性或抑制性影响。
- **三种主运行模式**：支持纯 NPI、FASK 前筛后运行 NPI、Two-Step 前筛后运行 NPI。
- **多算法因果骨架**：集成 Granger、VAR-Granger、PCMCI-Lite、Sparse VAR-Lite、FASK、Two-Step、Consensus、NPI-Ensemble 和 NPCI 等分析入口。
- **虚拟脑区扰动**：支持多档扰动强度，观察指定脑区扰动后的全脑传播与网络响应。
- **关键节点筛选**：基于当前因果骨架和扰动响应筛选最强节点，并将结果直接传递给后续靶点分析。
- **C阶段靶点搜索**：围绕病灶脑区生成动态局部核心，比较候选节点、刺激方向、强度和时序策略。
- **D阶段恢复评价**：展示轨迹恢复、功能连接恢复、脑区级恢复以及干预前后的网络差异。
- **三维交互可视化**：展示脑区节点、有向连接、正负作用、传播路径、动态核心和候选靶点。
- **预计算与缓存隔离**：不同运行模式、算法角色和局部/全脑结果分别保存，避免跨模式误读。
- **Python SDK**：通过 `NPIClient` 读取预计算结果、调用因果算法、构建网络数据并启动完整网页系统。

## 系统流程

```text
功能磁共振时间序列
        │
        ├─ 纯 NPI
        ├─ FASK 前筛 → NPI
        └─ Two-Step 前筛 → NPI
                    │
                    ▼
          B阶段：因果骨架与关键节点
                    │
                    ▼
          NPCI：病灶响应与动态局部核心
                    │
                    ▼
          C阶段：候选靶点与刺激策略搜索
                    │
                    ▼
          D阶段：恢复轨迹与网络恢复评价
                    │
                    ▼
             三维可视化与结果保存
```

### 三种主运行模式

| 模式 | 主流程 | 未选中的功能磁共振算法 |
|---|---|---|
| 纯 NPI | 直接使用 NPI 有效连接 | FASK 和 Two-Step 均可作为普通比较算法 |
| FASK 前筛 | 原始数据 → FASK 骨架 → NPI | Two-Step 可继续作为普通算法使用 |
| Two-Step 前筛 | 原始数据 → Two-Step 骨架 → NPI | FASK 可继续作为普通算法使用 |

运行模式会同时控制后续 B、C、D 阶段。不同模式的有效连接、因果骨架、局部筛选和 NPCI 结果分别缓存，不会自动复用其他模式的结果。

## 分析方法

| 方法 | 在系统中的作用 | 环境说明 |
|---|---|---|
| NPI | 通过代理脑模型和虚拟扰动估计有效连接 | 重新训练需要 PyTorch |
| Granger | 两两滞后预测筛选 | Python |
| VAR-Granger | 多变量滞后方向筛选 | Python |
| PCMCI-Lite | 候选受限的时间条件独立筛选 | Python / Tigramite |
| Sparse VAR-Lite | 稀疏多变量滞后支持 | Python |
| FASK | 非高斯方向判别并支持反馈边 | 需要 Java 运行环境 |
| Two-Step | 自适应稀疏估计与独立成分方向恢复 | 需要单独安装并授权的 MATLAB |
| Consensus | 汇总多种时间序列筛选证据 | Python |
| NPI-Ensemble | 融合虚拟扰动证据与时间序列证据 | Python |
| NPCI | 病灶响应、动态核心、靶点搜索与恢复评价 | Python |

这些方法提供的是统计或模型层面的因果证据，不等同于真实干预实验中的因果证明。

## 快速开始

### 1. 获取代码和大文件

仓库中的 Java 算法包使用 Git LFS 管理。首次克隆前请安装 Git LFS：

```powershell
git lfs install
git clone https://github.com/wonder20661212/Stu_Digital_Twin_Brain_Dynamic_Visualization_System.git
cd Stu_Digital_Twin_Brain_Dynamic_Visualization_System
git lfs pull
```

### 2. 创建 Python 环境

建议使用 Python 3.10、3.11 或 3.12。以下示例使用 Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. 启动系统

```powershell
.\.venv\Scripts\python.exe app.py
```

浏览器打开：

```text
http://127.0.0.1:5002/
```

可通过以下环境变量覆盖默认路径或端口：

- `NPI_DATA_DIR`
- `NPI_BRAIN_MODEL_FILE`
- `NPI_FMRI_DIR`
- `NPI_HOST`
- `NPI_PORT`

## 外部运行环境

### FASK

项目已包含 FASK 调用代码与 Tetrad 运行包，但用户电脑仍需安装 Java。缺少 Java 时，其他 Python 算法、NPI、NPCI 和已保存结果仍可使用。

### Two-Step

项目包含 Two-Step 算法源码和系统适配入口，但不包含 MATLAB。需要重新计算 Two-Step 时，用户必须自行安装并正确授权 MATLAB。没有 MATLAB 时，可以读取已经保存的 Two-Step 结果，但不能重新运行原版算法。

## Python SDK

从项目根目录安装开发版 SDK：

```powershell
python -m pip install -e .
```

基础示例：

```python
from npi_sdk import NPIClient

client = NPIClient()

# 查看内置扰动强度
print(client.available_strengths())

# 获取有效连接矩阵
ec = client.get_ec_matrix(0.2)
print(ec.shape)

# 在指定运行模式下读取或计算因果掩码
fask_mask = client.compute_mask("fask", pipeline_mode="fask")

# 将B阶段最强节点用于C/D阶段候选靶点
targets = client.top_backbone_target_nodes(
    strength=0.2,
    node_idx=0,
    top_n=10,
)
result = client.npci_intervention(
    strength=0.2,
    node_idx=0,
    target_nodes=targets,
)
```

启动 SDK 内置网页：

```powershell
npi-demo web --host 127.0.0.1 --port 5002
```

更多接口与安装说明见：

- [SDK 具体操作说明](./SDK_具体操作说明.md)
- [SDK API 说明](./SDK_README.md)
- [FASK / Two-Step 接入说明](./CAUSAL_ALGORITHM_INTEGRATION.md)
- [SDK 交付说明](./DIGITAL_TWIN_SDK_DELIVERY.md)

## 项目结构

```text
├─ app.py                         # Flask 后端与系统接口
├─ templates/index.html           # 主前端与三维可视化界面
├─ data/
│  ├─ brain9.nii                  # 脑表面模型
│  ├─ ec_data.npz                 # 当前有效连接数据
│  ├─ ec_results/                 # 不同运行模式的预计算有效连接
│  └─ causal_mask_results/        # 因果算法预计算结果
├─ external_algorithms/
│  ├─ fask/                       # FASK Java 适配与运行包
│  └─ two_step/                   # Two-Step MATLAB 源码与适配入口
├─ NPCI/                          # 动态核心、靶点搜索与恢复评价算法
├─ npi_sdk/
│  ├─ client.py                   # SDK 主客户端
│  ├─ algorithms.py               # 算法函数入口
│  └─ _engine/                    # SDK 独立运行所需的内置引擎副本
├─ requirements.txt               # 完整运行依赖
├─ pyproject.toml                 # SDK 打包配置
└─ README.md
```

### 关于 `npi_sdk/_engine`

`npi_sdk/_engine` 中的 `app.py`、模板、算法和必要数据与项目根目录存在部分重复，这是有意保留的 SDK 运行副本。它保证安装 wheel 后，即使脱离原始项目目录，SDK 仍能启动网页和读取内置资源。删除这些文件会导致独立安装的 SDK 功能不完整，因此它们不属于可直接清理的普通冗余文件。

## 数据与缓存

- `data/ec_results/` 保存三种主运行模式及普通算法对应的有效连接结果。
- `data/causal_mask_results/` 保存全脑或局部因果骨架。
- `data/npci_intervention_results/` 保存 C/D 阶段结果，默认不提交到 Git，由本地运行自动生成。
- 原始和派生的大型研究数据默认通过 `.gitignore` 排除。
- 真实功能磁共振数据在上传或共享前必须确认匿名化和数据授权要求。

## 当前限制

- 当前内置展示数据采用 360 脑区配置；替换脑图谱时需要同步提供坐标、脑表面和有效连接数据。
- 全脑 Two-Step 计算可能耗时较长，更适合局部筛选或预计算。
- FASK 与 Two-Step 的结果受外部运行环境、算法参数和输入数据质量影响。
- NPCI 恢复评价是模型空间中的虚拟评估，不代表临床治疗效果。
- 项目优先在 Windows 环境验证；其他系统可能需要调整 Java、MATLAB 和数据路径。

## 相关论文与致谢

本项目使用并扩展了以下研究方法。使用对应算法进行科研发表时，请引用原始论文：

1. Luo, Z., Peng, K., Liang, Z. *et al.* Mapping effective connectivity by virtually perturbing a surrogate brain. *Nature Methods* (2025). https://doi.org/10.1038/s41592-025-02654-x
2. Sanchez-Romero, R., Ramsey, J. D., Zhang, K. *et al.* Estimating feedforward and feedback effective connections from fMRI time series: Assessments of statistical methods. *Network Neuroscience*. https://doi.org/10.1162/netn_a_00061

NPI、Tetrad/FASK、Two-Step 及其他第三方组件保留各自的作者署名、引用要求和许可条件。本仓库根目录许可证不应被解释为覆盖所有第三方代码和数据。

## 反馈与协作

如发现问题或希望改进算法接口、可视化及文档，请通过本仓库的 GitHub Issues 提交可复现的问题描述、运行环境和必要日志。请勿在 Issue 中上传患者数据、账户信息、密钥或其他敏感内容。

