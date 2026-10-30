# NPI Brain SDK

`npi_sdk` is the Python SDK for the digital-twin brain perturbation system. It can load bundled NPI effective-connectivity data, run causal masks, build causal backbones, start the web visualizer, and run the standard NPI training pipeline.

## Python Version

Recommended Python: `3.10.x`, `3.11.x`, or `3.12.x`.

Supported Python range: `>=3.9`.

The SDK, bundled data, causal masks, demos, web visualizer, and NPI training APIs are intended to run on Python 3.9 and above. Python 3.10/3.11/3.12 remain the safest delivery targets on Windows because PyTorch wheels are widely available for them.

## Install

For delivery to another machine, install the wheel:

```powershell
python -m pip install npi_brain_sdk-0.1.7-py3-none-any.whl
```

If the machine can access a Python package index, `pip` will install the required runtime dependencies declared by the wheel.

For offline delivery, provide the SDK wheel together with dependency wheels. PyTorch is part of the SDK dependency set, so offline delivery should include a matching `torch` wheel for the target Python version.

```powershell
python -m pip install -r requirements.txt
```

For local development from this repository:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation --no-deps
```

## Start Backend And Frontend

Generic install command:

```powershell
python -m pip install dist\npi_brain_sdk-0.1.7-py3-none-any.whl
```

Start the backend web service:

```powershell
python -c "from npi_sdk import NPIClient; NPIClient().run_web_app(host='127.0.0.1', port=5002)"
```

Open the frontend page in a browser:

```text
http://127.0.0.1:5002/
```

If port `5002` is already in use, start another port:

```powershell
python -c "from npi_sdk import NPIClient; NPIClient().run_web_app(host='127.0.0.1', port=5010)"
```

Then open:

```text
http://127.0.0.1:5010/
```

Stop the backend service with `Ctrl + C` in the PowerShell window.

## Demo Commands

After installing the wheel, run the built-in quick demo:

```powershell
npi-demo
```

The same demo can be run through Python:

```powershell
python -m npi_sdk.demo
```

Print SDK resource information:

```powershell
npi-demo info
```

Start the web visualizer demo:

```powershell
npi-demo web --host 127.0.0.1 --port 5002
```

Then open:

```text
http://127.0.0.1:5002/
```

## Dependencies

Required runtime dependencies:

- `Flask>=2.3,<4`
- `Flask-Cors>=4,<7`
- `numpy>=1.24,<3`
- `pandas>=1.5,<3`
- `scipy>=1.10,<2`
- `nibabel>=5.1,<6`
- `scikit-image>=0.21,<1`
- `torch>=2.2,<3`
- `matplotlib>=3.7,<4`
- `seaborn>=0.12,<1`
- `tigramite>=5.2,<6`

Optional notebook/demo dependencies:

- `nilearn>=0.10,<1`
- `jupyter>=1.0,<2`

When installing from the wheel with dependencies enabled, PyTorch is installed automatically. For source installs, use:

```powershell
python -m pip install -r requirements.txt
```

That file includes the SDK dependencies and `torch>=2.2,<3`.

If a machine has no internet access, install a PyTorch wheel that matches the local Python version before calling training APIs.

Optional external runtimes:

- Standard algorithms (`granger`, `vargranger`, `pcmci`, `sparsevar`) run inside Python and do not require Java or MATLAB.
- FASK needs a Java runtime. The SDK wheel includes the Tetrad jar used by the current system, but it does not install Java itself.
- Two-Step still needs a MATLAB runtime/executable. The SDK does not run Two-Step by default; call it only when the target machine has MATLAB configured.

## Quick Check

```python
from npi_sdk import NPIClient

client = NPIClient()
print(client.available_strengths())
print(client.get_ec_matrix(0.2).shape)
print(client.mask_summary(client.compute_mask("pcmci")))
```

Expected basics:

- perturbation strengths include `0.05` through `1.0`
- EC matrix shape is `(360, 360)`
- `pcmci` returns a mask summary

## Main API

```python
from npi_sdk import NPIClient

client = NPIClient()

# cached NPI EC
ec = client.get_ec_matrix(0.2)

# causal masks
pcmci = client.compute_mask("pcmci")
consensus = client.compute_mask("consensus")

# FASK / Two-Step follow the same three pipeline modes as the web UI:
# "none", "fask", and "twostep". This keeps pre-screen caches separate
# from ordinary algorithm caches.
fask_prescreen = client.compute_mask("fask", pipeline_mode="fask")
twostep_as_ordinary_in_fask_mode = client.compute_mask("twostep", pipeline_mode="fask")

# Default precompute is user-safe: it runs Python-native standard algorithms.
standard = client.precompute_default_masks(group="standard")

# Full-brain FASK can be requested explicitly. Two-Step still requires the
# external MATLAB environment in the current system, so it is not run by default.
fask_status = client.precompute_causal_status(pipeline_mode="fask")

# local fMRI screening for a selected perturbation node
local_fask = client.local_fmri_mask(
    "fask",
    strength=0.2,
    node_idx=0,
    pipeline_mode="twostep",
    max_nodes=40,
)

# causal backbone for one perturbation node
backbone = client.causal_backbone(0.2, node_idx=0, top_n=10, mask_key="pcmci")

# B-stage strongest nodes can be reused as NPCI C/D candidate targets
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

# selected-node network
network = client.build_network(
    0.2,
    node_indices=[0, 5, 12, 18],
    direction_mode="dominant_pairs",
    mask_key="consensus",
)

# web visualizer
client.run_web_app(port=5002)
```

NPCI intervention dashboard APIs:

```python
defaults = client.npci_stage3_defaults()
stage_cd = client.npci_intervention(0.3, node_idx=6, cached_only=True)
top_targets = client.top_backbone_target_nodes(0.3, node_idx=6, top_n=12)
client.precompute_npci_intervention(0.3, node_idx=6, target_nodes=top_targets)
status = client.precompute_npci_status()
```

## Training API

One-call training and EC computation:

```python
result = client.train_and_compute_ec(
    mode="none",
    strengths=[0.2, 0.5],
    num_epochs=100,
    save=True,
)
```

Step-by-step training:

```python
training_data = client.load_training_data()
trained = client.train_surrogate_model(training_data, num_epochs=100)
ec_results = client.compute_npi_ec(trained["model"], training_data, strengths=[0.2])
client.save_ec_results(ec_results, mode="none")
```

## Resource Behavior

After wheel installation, `NPIClient()` works from any directory because the wheel includes the SDK engine, cached EC data, causal-mask cache, templates, and sample fMRI text data.

To use another project/data folder, pass:

```python
client = NPIClient(project_root=r"D:\path\to\npi_project")
```

or set `NPI_PROJECT_ROOT`.

