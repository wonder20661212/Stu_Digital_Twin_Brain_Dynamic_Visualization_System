# NPI

[Mapping effective connectivity by virtually perturbing a surrogate brain](https://www.nature.com/articles/s41592-025-02654-x)

## **Abstract**

Effective connectivity (EC), which reflects the causal interactions between brain regions, is fundamental to understanding information processing in the brain. However, traditional methods for obtaining EC, which rely on neural responses to stimulation, are often invasive or limited in spatial coverage, making them unsuitable for whole-brain EC mapping in humans. To address this gap, we introduce Neural Perturbational Inference (NPI), a data-driven framework for mapping whole-brain EC. NPI employs an artificial neural network trained to model large-scale neural dynamics, serving as a computational surrogate of the brain. By systematically perturbing all regions in the surrogate brain and analyzing the resulting responses in other regions, NPI maps the directionality, strength, and excitatory/inhibitory properties of brain-wide EC. Validation of NPI on generative models with known ground-truth EC demonstrates its superiority over existing methods such as Granger causality and dynamic causal modeling. When applied to resting-state fMRI data across diverse datasets, NPI reveals consistent, structurally supported EC patterns. Furthermore, comparisons with cortico-cortical evoked potentials data show a strong resemblance between NPI-inferred EC and real stimulation propagation patterns. By transitioning from correlational to causal understandings of brain functionality, NPI marks a stride in decoding the brain's functional architecture and facilitating both neuroscience studies and clinical applications.

## **Introduction**

<img src=".\img\NPI_framework.jpg" alt="NPI_framework" style="zoom:80%;" />

This repository contains the code and documentation of the NPI framework, a tool designed for mapping whole-brain EC in the brain. NPI operates by first training an ANN to mimic the complex dynamics of the brain based on the observed neural data. Once trained, virtual perturbations are applied to specific regions of the ANN to simulate the effect of neural stimulation. By monitoring the responses in other brain regions, NPI constructs a comprehensive map of the causal relationships, detailing how each area influences others across the brain. This process allows for the assessment of not only the presence but also the direction and excitatory/inhibitory properties of neural connections, contributing to a deeper comprehension of the brain's functional architecture.

## **Methodology**

<img src=".\img\dynamics.gif" alt="dynamics" style="zoom:100%;" />

1. **Utilize an ANN to serve as a surrogate brain.**

   NPI utilizes ANN to learn the brain’s complex, nonlinear dynamics directly from data. This approach allows NPI to adapt to a wide range of data types and dynamics. The use of advanced AI techniques, such as pre-training (to train a group-level surrogate model) and fine-tuning (to obtain individual-level surrogate models), further enhances our model’s applicability to both group-level and individual-level analyses.

2. **Apply virtual perturbation to ANN for inferring EC.**

   NPI provides flexibility in the pattern of perturbations once the surrogate model is well-trained. It is not constrained to a fixed-size perturbation and can accommodate various forms and scales of perturbations, tailored to specific research needs. This adaptability enhances NPI’s applicability across diverse experimental settings and research questions.

## **Requirements**

**Operating Systems**: Windows 10+, Ubuntu 20.04+

**Python Version Plan**

- **Python 3.10.x / 3.11.x / 3.12.x**: recommended for the SDK, including PyTorch-based NPI retraining and regenerating EC from raw fMRI text files.
- **Python 3.13+**: use only when the target machine has compatible PyTorch wheels available.

**Dependency Files**

- `requirements.txt`: full source-runtime dependencies, including `torch>=2.2,<3`.

**Non-standard Hardware Requirements**: No non-standard hardware is required to start the visualizer. A GPU-capable PyTorch install is still recommended for long NPI training jobs.

## **Installation**

### Install And Start

Use this path for the SDK, the web UI, saved results, and recomputation workflows.

1. Install Python 3.10.x, 3.11.x, or 3.12.x.
2. Create a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   ```

3. Install the full dependency set:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```

## **How to run**

### Web visualizer

Start the Flask server with one of the commands above, then open `http://127.0.0.1:5002/`.

Notes:

- The app now resolves `data/brain9.nii`, `data/ec_data.npz`, `templates/index.html`, and `real_fMRI_data/*.txt` relative to the project directory.
- You can override data locations with environment variables:
  - `NPI_DATA_DIR`
  - `NPI_BRAIN_MODEL_FILE`
  - `NPI_FMRI_DIR`
- If you press the "start computation" action without PyTorch installed, the app will return a clear error telling you to install `torch>=2.2,<3` or `requirements.txt`.

### Notebook demos

The notebook `.ipynb` files still contain the step-by-step workflow for loading data, training a surrogate brain, simulating long time series under noise-driven conditions, calculating model FC, and perturbing the surrogate brain to obtain EC.

### Causal-discovery algorithm notes

The web app provides several causal screening masks for comparing or filtering NPI-inferred EC. Interpret them as statistical screening tools rather than ground-truth causal proof.

- Default precompute order is Granger, VAR-Granger, PCMCI-Lite, then Sparse VAR-Lite. These are all directed lagged time-series screens, which best match the NPI-first fMRI causal-skeleton workflow.
- Pairwise Granger and VAR-Granger combine evidence across fMRI sessions with Fisher's method, apply Benjamini-Hochberg FDR correction across directed ROI pairs, and require non-trivial effect size. The web defaults cap Granger-style mask density to avoid treating tiny but statistically significant fMRI effects as useful causal skeleton edges.
- VAR-Granger caps the lag order to preserve degrees of freedom for 360-ROI data, then uses block coefficient F/Wald tests.
- PCMCI-Lite builds lagged matrices separately inside each fMRI session, limits parent candidates by lagged correlation, caps conditioning sets, and combines session-level p-values. This preserves the main temporal-conditioning idea while keeping full-brain visualization tractable.
- Sparse VAR-Lite uses candidate-limited ridge VAR coefficients to add a sparse directed lagged screen. It is available as an individual front-end screening option and also contributes to the consensus evidence.
- The front-end exposes a `Consensus` option that requires all four default evidence sources. Consensus uses weighted agreement across Granger, VAR-Granger, PCMCI-Lite, and Sparse VAR-Lite, then shows compact evidence summaries: average consensus score, average algorithm support, high-confidence edge count, and source algorithms.
- The `Ensemble` option combines NPI virtual-intervention evidence with the temporal causal screens. NPI evidence is computed from row-normalized directed EC magnitude across perturbation strengths and sign consistency; temporal evidence comes from the weighted causal-screen consensus. The default filter requires NPI support plus at least one causal-screen source and caps density for a conservative directed skeleton.
- PC, FCI, NOTEARS, LiNGAM, and DirectLiNGAM are kept as code-level exploratory routines only. They are disabled in the main web workflow because they are slower, weaker for the current lagged fMRI causal-skeleton task, partially directed rather than cleanly directed, or redundant for the current NPI-first design.
- FASK and Two-Step are treated separately as fMRI-native algorithms. In the web UI and SDK they can be used as pre-screen modes before NPI, or as ordinary comparison algorithms after NPI, with caches separated by `pipeline_mode`.

To run the lightweight algorithm validation suite:

```powershell
py -3.13 validate_causal_algorithms.py
```

### Python SDK

The project also exposes the distributable `npi-brain-sdk` package in
`npi_sdk/`. It follows the earlier user-facing SDK shape: a single
`NPIClient` can read bundled EC data, use cached causal masks, build network
payloads, and start the bundled web visualizer.

Install it from the project root:

```powershell
python -m pip install -e .
```

Or install the built wheel:

```powershell
python -m pip install dist\npi_brain_sdk-0.1.7-py3-none-any.whl
```

Minimal usage:

```python
from npi_sdk import NPIClient

client = NPIClient()
print(client.available_strengths())
print(client.get_ec_matrix(0.2).shape)
print(client.mask_summary(client.compute_mask("pcmci")))

# SDK 0.1.7 follows the web UI pipeline modes for native fMRI algorithms.
# FASK / Two-Step caches are separated by pipeline_mode:
# "none", "fask", and "twostep".
fask_mask = client.compute_mask("fask", pipeline_mode="fask")
status = client.precompute_causal_status(pipeline_mode="fask")

# Reuse B-stage strongest-node screening as NPCI C/D candidates.
top_targets = client.top_backbone_target_nodes(0.2, node_idx=0, top_n=10)
stage_cd = client.npci_intervention(0.2, node_idx=0, target_nodes=top_targets)

# User-safe default precompute only runs Python-native standard algorithms.
client.precompute_default_masks(group="standard")
print(client.npci_stage3_defaults())
```

FASK requires a Java runtime on the target machine. Two-Step still requires MATLAB, so it is not run by the SDK's default precompute path.

Start the SDK web visualizer:

```powershell
npi-demo web --host 127.0.0.1 --port 5002
```

or:

```python
from npi_sdk import NPIClient

NPIClient().run_web_app(host="127.0.0.1", port=5002)
```

The SDK supports Python 3.9 and above. Python 3.10.x, 3.11.x, or 3.12.x is
recommended for the full PyTorch-based retraining workflow and ordinary web use.

## **Demo examples**

Demos `NPI_demo(RNN).ipynb` and `NPI_demo(WBM).ipynb` show the application of NPI to simulated data derived from recurrent neural network (RNN) and whole-brain model (WBM) under given structural connectivity (SC). These demos serve to illustrate the process of applying NPI to infer EC, and comparing the NPI-inferred EC with ground-truth EC and SC. Demo `NPI_demo(fMRI).ipynb` shows applying of NPI on real individual fMRI data from HCP dataset to infer individual EC.

**Demo 1: RNN synthetic data**

- **Overview**: Demo `NPI_demo(RNN).ipynb` applies NPI to ground-truth RNN simulated data. It encompasses the generation of synthetic neural signals, obtaining ground-truth EC by direct perturbation of the RNN, applying NPI to infer EC from the RNN simulated data, and a comparison of the NPI-inferred EC with the ground-truth EC and SC, assessing the accuracy and reliability of the NPI framework in capturing directed causal interactions.
- **Simulation data resources**: Located in the `RNN_simulation data` folder, we provide the code necessary for using RNN to generate synthetic neural data and ground-truth EC. The folder contains detailed descriptions of the model parameters and simulation process.

**Demo 2: WBM synthetic data**

- **Overview**: Demo `NPI_demo(WBM).ipynb` applies NPI to WBM simulated data. It encompasses the generation of synthetic neural signals, obtaining ground-truth EC by direct perturbation of the WBM, applying NPI to infer EC from the WBM simulated data, and a comparison of the NPI-inferred EC with the ground-truth EC and SC, assessing the accuracy and reliability of the NPI framework in capturing direct causal interactions.
- **Simulation data resources**: Located in the `WBM_simulation data` folder, we provide the code necessary for using RNN to generate synthetic neural data and ground-truth EC. The folder contains detailed descriptions of the model parameters and simulation process.

**Demo 3: fMRI data from HCP dataset**

- **Overview**: Demo `NPI_demo(fMRI).ipynb` applies NPI to real fMRI data from HCP dataset of an individual. It encompasses applying NPI to infer EC from BOLD signals of an individual and a comparison of model FC and empirical FC.

**How to run the demos**

1. **Launch Jupyter Notebook**: Start from the project directory by launching the Jupyter Notebook.
2. **Open the demo notebook**: Select the demo notebook corresponding to either RNN or WBM synthetic data.
3. **Follow the instructions**: Each notebook provides step-by-step instructions for loading data, training the model, applying NPI, and analyzing results.
4. **Expected output**: The notebook will display output cells after execution. (Model FC closely resembles empirical FC, while NPI-inferred EC shows a strong correspondence with both ground-truth EC and SC.)
5. **Expected run time**: Expected total run time for the three demos is approximately 30 minutes on a "normal" desktop computer.

## **Whole-brain EC resources**

Within the `Whole-brain_EC` directory, we provide EC data across various brain atlases, offering insights into brain connectivity research. The EC matrices represent the causal interactions between different brain regions, derived from resting-state fMRI data of subjects within the Human Connectome Project (HCP) dataset. A row of the EC matrix represents the output EC from a source region to all other regions. A column of the EC matrix represents the input EC from all other regions to a target region.

Specifically, we calculated the EC matrices based on the MMP (360 regions), AAL (116 regions), MSDL (39 regions) atlases across 800 subjects and the Schaefer2018 atlases (with 100, 200, ..., 1000 regions) across 100 subjects. For each atlas, we calculated the individual EC matrices for all subjects and then averaged these matrices at the group level to obtain a representative EC matrix for each atlas. All EC matrices are normalized so that the strongest connection in each matrix has a value of 1.

We provide a notebook file `Whole-brain_EC/Whole-brain_EC.ipynb` that allow for the visualization of these EC matrices. Offering these detailed EC resources, we aim to facilitate research into the structural and functional interplay of the brain and support the development of novel diagnostic and therapeutic strategies in neuroscience.

<img src=".\img\ebc.jpg" alt="NPI_framework" style="zoom:80%;" />

## **How to cite**

If you use NPI in your research, please cite our paper:

- Luo, Z., Peng, K., Liang, Z. *et al.* Mapping effective connectivity by virtually perturbing a surrogate brain. *Nat Methods* (2025). [https://doi.org/10.1038/s41592-025-02654-x](https://doi.org/10.1038/s41592-025-02654-x)

## **Contact**

For any questions/comments, please contact [NCC Lab](https://www.sustech.edu.cn/en/faculties/liuquanying.html).

## **Copyright**

Copyright © 2024 NCC Lab, Southern University of Science and Technology, Shenzhen, China.

