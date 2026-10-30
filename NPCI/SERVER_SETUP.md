# Server Environment Setup

This project is easiest to move to a server when the environment is rebuilt from
scratch instead of copying a local `.venv`.

## Recommended baseline

- Python: `3.12` or `3.13`
- Environment isolation: standard `venv` without `--system-site-packages`
- PyTorch:
  - CPU-only server: install the default wheel from PyPI
  - GPU server: install the wheel that matches the server CUDA runtime

## 1. Create a clean virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

## 2. Upgrade packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install PyTorch

### CPU-only

```bash
python -m pip install "torch>=2.2,<2.13"
```

### GPU server

Use the PyTorch wheel that matches the server driver and CUDA stack. Examples:

```bash
# Example only: choose the index URL that matches the server CUDA runtime
python -m pip install "torch>=2.2,<2.13" --index-url https://download.pytorch.org/whl/cu124
```

If you are unsure which CUDA build to use, check the server with:

```bash
nvidia-smi
```

## 4. Install the remaining project dependencies

```bash
python -m pip install -r requirements.txt
```

## 5. Smoke test

```bash
python -c "import torch, numpy, pandas, scipy, nilearn, nibabel, networkx, sklearn, tigramite; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

## Notes

- Do not copy a local `.venv` from Windows to Linux or from one machine to another.
- This project uses notebooks and analysis scripts, so `jupyter`, `nbformat`, and
  `nbclient` are kept in the base dependency list.
- Some workflows depend on `tigramite`; it is included in `requirements.txt`
  so the full project can be moved and run consistently.
