from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import numpy as np
import nibabel as nib
import pandas as pd
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
from scipy.stats import f as f_dist, norm as norm_dist, chi2 as chi2_dist
from scipy.linalg import expm, lstsq, solve
from scipy.optimize import minimize
import os
import json
import copy
import hashlib
from pathlib import Path
import threading
import time
import warnings
import sys
import importlib.util
import subprocess
import tempfile
import shutil

warnings.filterwarnings('ignore')

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream is not None and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

app = Flask(__name__)
cors_origins = os.getenv("NPI_CORS_ORIGINS", "http://127.0.0.1:5002,http://localhost:5002")
cors_origin_list = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
CORS(app, origins="*" if cors_origins.strip() == "*" else cors_origin_list)

# ====================== 全局路径配置 ======================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("NPI_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR = BASE_DIR / "templates"
REAL_FMRI_DIR = Path(os.getenv("NPI_FMRI_DIR", BASE_DIR / "real_fMRI_data"))
DEFAULT_FMRI_FILENAMES = [
    "102008_REST1_LR.txt",
    "102008_REST1_RL.txt",
    "102008_REST2_LR.txt",
    "102008_REST2_RL.txt",
]
BRAIN_MODEL_FILE = Path(os.getenv("NPI_BRAIN_MODEL_FILE", DATA_DIR / "brain9.nii"))
NPCI_DIR = BASE_DIR / "NPCI"
EXTERNAL_CAUSAL_DIR = BASE_DIR / "external_algorithms"
FASK_RUNNER_DIR = EXTERNAL_CAUSAL_DIR / "fask"
TWO_STEP_RUNNER_DIR = EXTERNAL_CAUSAL_DIR / "two_step"
EXTERNAL_CAUSAL_TMP_DIR = DATA_DIR / "external_causal_tmp"
EXTERNAL_CAUSAL_TMP_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# 项目核心配置
# ==========================
batch_size = 50
train_set_proportion = 0.8
ROI_num = 360
using_steps = 3
strength_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
data_file = DATA_DIR / "ec_data.npz"
ec_meta_file = DATA_DIR / "ec_data_meta.json"
EC_RESULTS_DIR = DATA_DIR / "ec_results"
EC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
mask_cache_file = DATA_DIR / "causal_masks_default.npz"
mask_meta_file = DATA_DIR / "causal_masks_default_meta.json"
MASK_RESULTS_DIR = DATA_DIR / "causal_mask_results"
MASK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_FMRI_MASK_RESULTS_DIR = MASK_RESULTS_DIR / "full_fmri"
FULL_FMRI_MASK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_FMRI_MASK_CACHE_VERSION = 2
LOCAL_FMRI_MASK_RESULTS_DIR = MASK_RESULTS_DIR / "local_fmri"
LOCAL_FMRI_MASK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_FMRI_MASK_CACHE_VERSION = 2
MASK_CACHE_VERSION = 3
NPCI_INTERVENTION_RESULTS_DIR = DATA_DIR / "npci_intervention_results"
NPCI_INTERVENTION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NPCI_BACKBONE_RESULTS_DIR = NPCI_INTERVENTION_RESULTS_DIR / "backbone"
NPCI_BACKBONE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NPCI_BACKBONE_STAGE_CACHE_DIR = NPCI_INTERVENTION_RESULTS_DIR / "backbone_stages"
NPCI_BACKBONE_STAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
NPCI_STAGE3_NODE_CACHE_DIR = NPCI_INTERVENTION_RESULTS_DIR / "stage3_nodes"
NPCI_STAGE3_NODE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
NPCI_INTERVENTION_CACHE_VERSION = 2

# 全局变量
ec_dict = {}
current_ec_meta = None
npi_intervention_score_cache = None
npi_intervention_cache_signature = None
ANN = None
NPI = None
inputs_data = None
targets_data = None
roi_vox = None
brain_verts = None
brain_faces = None
is_initializing = True   # True while startup data load is in progress

computation_status = {
    "is_computing": False,
    "progress": 0,
    "total": 0,
    "message": "",
    "last_error": "",
    "started_at": None,
    "mode": "none",
    "mode_label": "NPI only",
    "screen_first": False,
    "all_modes": False,
}
computation_lock = threading.Lock()

# Global causality masks (ROI_num x ROI_num); True = significant causal link
granger_mask      = None
var_granger_mask  = None   # Multivariate VAR-Granger (BIC lag selection)
pcmci_mask        = None   # PCMCI causal time-series screen
sparse_var_mask   = None   # Candidate-limited sparse VAR screen
fask_mask         = None   # FASK non-Gaussian feedback-oriented screen
two_step_mask     = None   # Two-Step ICA-based cyclic causal screen
fask_local_mask   = None   # Node-centered local FASK screen embedded in full ROI space
two_step_local_mask = None # Node-centered local Two-Step screen embedded in full ROI space
full_fmri_mask_meta = {}
local_fmri_mask_meta = {}
npci_backbone_cache = {}
npci_intervention_cache = {}
npci_runtime_modules = {}

BASE_TEMPORAL_SCREEN_KEYS = [
    "granger",
    "vargranger",
    "pcmci",
    "sparsevar",
]
FMRI_NATIVE_SCREEN_KEYS = [
    "fask",
    "twostep",
]
SCREENING_KEYS = BASE_TEMPORAL_SCREEN_KEYS + FMRI_NATIVE_SCREEN_KEYS
EXPERIMENTAL_MASK_KEYS = []
DERIVED_MASK_KEYS = ["consensus", "ensemble"]
LOCAL_FMRI_MASK_KEYS = ["fask_local", "twostep_local"]
# Full precompute stores every raw-fMRI causal screen so later NPI/NPCI stages
# can read cached skeletons directly.
PRECOMPUTE_MASK_KEYS = SCREENING_KEYS
STANDARD_PRECOMPUTE_MASK_KEYS = BASE_TEMPORAL_SCREEN_KEYS
FMRI_PRECOMPUTE_MASK_KEYS = FMRI_NATIVE_SCREEN_KEYS
LOCAL_ONLY_PRECOMPUTE_KEYS = []
CONSENSUS_SOURCE_KEYS = BASE_TEMPORAL_SCREEN_KEYS
CONSENSUS_WEIGHTS = {
    "granger": 0.20,
    "vargranger": 0.30,
    "pcmci": 0.35,
    "sparsevar": 0.15,
}
ENSEMBLE_WEIGHTS = {
    "npi_intervention": 0.55,
    "temporal_consensus": 0.45,
}
ENSEMBLE_DEFAULTS = {
    "min_score": 0.45,
    "min_npi_score": 0.20,
    "min_support": 1,
    "max_density": 0.10,
}

DEFAULT_MASK_PARAMS = {
    "granger": {"max_lag": 3, "alpha": 0.05, "min_effect": 0.001, "max_density": 0.12, "max_candidates_per_target": None},
    "vargranger": {"max_lag": 8, "alpha": 0.05, "min_effect": 0.001, "max_density": 0.12},
    "pcmci": {"max_lag": 2, "alpha": 0.05, "min_effect": 0.02, "max_density": 0.10, "full_tigramite": False},
    "sparsevar": {"max_lag": 2, "max_candidates_per_target": 32, "min_effect": 0.01, "max_density": 0.10, "ridge": 1.0, "min_session_support_ratio": 0.25},
    "fask": {"two_cycle_alpha": 0.01, "penalty_discount": 2.0, "max_density": 0.10, "timeout_seconds": 900},
    "twostep": {"lambda": 2.0, "min_effect": 0.01, "max_density": 0.10, "timeout_seconds": 7200},
}

DEFAULT_MASK_LABELS = {
    "granger": "Granger",
    "vargranger": "VAR-Granger",
    "pcmci": "PCMCI Screen",
    "sparsevar": "Sparse VAR-Lite",
    "fask": "FASK",
    "twostep": "Two-Step",
    "fask_local": "Local FASK",
    "twostep_local": "Local Two-Step",
    "consensus": "Consensus",
    "ensemble": "NPI-Ensemble",
}

COMPUTE_MODE_LABELS = {
    "none": "NPI only",
    "granger": "Granger-screened NPI",
    "vargranger": "VAR-Granger-screened NPI",
    "pcmci": "PCMCI-screened NPI",
    "sparsevar": "Sparse VAR-Lite-screened NPI",
    "fask": "FASK-screened NPI",
    "twostep": "Two-Step-screened NPI",
    "consensus": "Consensus-screened NPI",
    # NPI-Ensemble itself needs NPI scores, so its pre-NPI stage uses the
    # temporal consensus mask. The post-hoc Ensemble score can still be used
    # as evidence, but not as a true pre-NPI screen.
    "ensemble": "Temporal-consensus-screened NPI",
}

EC_RESULT_MODES = [
    "none",
    "granger",
    "vargranger",
    "pcmci",
    "sparsevar",
    "fask",
    "twostep",
    "consensus",
    "ensemble",
]
for _npci_mode_name in EC_RESULT_MODES:
    _npci_mode_dir = NPCI_INTERVENTION_RESULTS_DIR / _npci_mode_name
    _npci_mode_dir.mkdir(parents=True, exist_ok=True)
    (_npci_mode_dir / "backbone").mkdir(parents=True, exist_ok=True)
    (_npci_mode_dir / "backbone_stages").mkdir(parents=True, exist_ok=True)
    (_npci_mode_dir / "stage3_nodes").mkdir(parents=True, exist_ok=True)
BATCH_EC_RESULT_MODES = [
    "none",
    "granger",
    "vargranger",
    "pcmci",
    "sparsevar",
    "fask",
    "twostep",
    "consensus",
    "ensemble",
]
BATCH_PRE_NPI_SCREEN_KEYS = [
    key for key in BATCH_EC_RESULT_MODES
    if key in SCREENING_KEYS
]
NPCI_DEFAULT_PARAMS = {
    "horizon": 12,
    "n_seed_windows": 32,
    "tau": 1,
    "max_candidate_nodes": 32,
    "min_candidate_nodes": 16,
    "top_k_direct_lagged": 20,
    "top_k_response": 16,
    "top_k_fc": 0,
    "w_lagged": 1.0,
    "w_response": 0.6,
    "w_fc": 0.0,
    "pc_alpha": 0.05,
    "max_core_edges": 24,
    "max_direct_lesion_edges": 8,
    "strong_quantile": 0.85,
    "causal_weight": 0.7,
    "fc_weight": 0.3,
}

PC_MAX_CONDITIONING_NEIGHBORS = int(os.getenv("NPI_PC_MAX_CONDITIONING_NEIGHBORS", "12"))
PCMCI_MAX_PARENTS_PER_TARGET = int(os.getenv("NPI_PCMCI_MAX_PARENTS_PER_TARGET", "16"))
PCMCI_MAX_CONDITIONING_SET = int(os.getenv("NPI_PCMCI_MAX_CONDITIONING_SET", "4"))
PCMCI_PC_MAX_ORDER = int(os.getenv("NPI_PCMCI_PC_MAX_ORDER", "1"))

screening_status = {
    key: {
        "is_running": False,
        "message": "",
        "last_error": "",
        "params": None,
        "started_at": None,
        "completed_at": None,
    }
    for key in (SCREENING_KEYS + EXPERIMENTAL_MASK_KEYS)
}
screening_lock = threading.Lock()
causal_precompute_status = {
    "is_running": False,
    "progress": 0,
    "total": len(PRECOMPUTE_MASK_KEYS),
    "expected_keys": PRECOMPUTE_MASK_KEYS,
    "current_key": "",
    "current_label": "",
    "current_started_at": None,
    "message": "",
    "last_error": "",
    "started_at": None,
    "completed_at": None,
    "last_heartbeat_at": None,
    "computed_keys": [],
    "missing_keys": PRECOMPUTE_MASK_KEYS,
    "failed_results": [],
    "pipeline_mode": "none",
}
causal_precompute_lock = threading.Lock()
causal_precompute_cancel_event = threading.Event()
external_causal_proc_lock = threading.Lock()
external_causal_current_proc = None
npci_precompute_status = {
    "is_running": False,
    "progress": 0,
    "total": 1,
    "strength": None,
    "node_idx": None,
    "node_label": "",
    "profile": "fast",
    "profile_label": "快速筛查",
    "stage": "",
    "candidate_count": 0,
    "candidate_nodes": [],
    "grid_done": 0,
    "grid_total": 0,
    "grid_per_node": 0,
    "current_node": None,
    "current_node_label": "",
    "candidate_index": 0,
    "node_grid_done": 0,
    "node_grid_total": 0,
    "current_sign": None,
    "current_policy_strength": None,
    "current_schedule": "",
    "cached_node_count": 0,
    "saved_node_count": 0,
    "last_saved_node": None,
    "last_saved_node_label": "",
    "last_node_cache_file": None,
    "stage3_node_cache_dir": None,
    "backbone_cache_dir": None,
    "stage1_cache_file": None,
    "screening_cache_file": None,
    "local_edges_cache_file": None,
    "local_edge_count": 0,
    "core_edge_count": 0,
    "greedy_step": 0,
    "greedy_done": 0,
    "greedy_total": 0,
    "estimated_policy_count": 0,
    "message": "",
    "last_error": "",
    "started_at": None,
    "completed_at": None,
    "last_heartbeat_at": None,
    "cache_file": None,
}
npci_precompute_lock = threading.Lock()


def get_npi_module():
    """Import NPI lazily and report a clear dependency error if PyTorch is missing."""
    global NPI
    if NPI is None:
        try:
            import NPI as npi_module
        except ModuleNotFoundError as exc:
            missing = exc.name or "训练依赖"
            raise RuntimeError(
                f"无法导入训练模块（缺少 {missing}）。"
                "完整 SDK 安装应包含 PyTorch；如果是离线安装或使用了 --no-deps，"
                "请补装 torch>=2.2,<3，或在源码目录运行 python -m pip install -r requirements.txt。"
            ) from exc
        NPI = npi_module
    return NPI


def _load_python_module(module_name: str, file_path: Path):
    if module_name in npci_runtime_modules:
        return npci_runtime_modules[module_name]
    if str(file_path.parent) not in sys.path:
        sys.path.insert(0, str(file_path.parent))
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    npci_runtime_modules[module_name] = module
    return module


def get_npci_modules():
    if not NPCI_DIR.exists():
        raise RuntimeError(f"未找到NPCI目录: {NPCI_DIR}")
    modules = {
        "lesion_response": _load_python_module("lesion_response", NPCI_DIR / "lesion_response.py"),
        "local_lpcmci": _load_python_module("local_lpcmci", NPCI_DIR / "local_lpcmci.py"),
        "recovery_evaluation": _load_python_module("recovery_evaluation", NPCI_DIR / "recovery_evaluation.py"),
        "intervention_optimization": _load_python_module("intervention_optimization", NPCI_DIR / "intervention_optimization.py"),
    }
    return modules


def _fc_from_signals(signals: np.ndarray) -> np.ndarray:
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("signals必须是二维矩阵")
    fc = np.corrcoef(arr, rowvar=False)
    return np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)


def _hashable_npci_params(params: dict) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in params.items()))


def _relative_cache_path(path) -> str:
    """Return a project-relative path for API/UI display."""
    try:
        p = Path(path).resolve()
        try:
            rel = p.relative_to(BASE_DIR.resolve())
        except ValueError:
            rel = Path(os.path.relpath(p, BASE_DIR.resolve()))
        return rel.as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def _to_plain_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_plain_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_python(v) for v in obj]
    return obj


def _stable_json_hash(payload: dict, n_chars: int = 16) -> str:
    text = json.dumps(_to_plain_python(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n_chars]


def _npci_ec_signature() -> dict:
    meta = current_ec_meta or {}
    return {
        "mode": meta.get("mode", "none"),
        "mode_label": meta.get("mode_label", ""),
        "updated_at": meta.get("updated_at"),
        "file": meta.get("file"),
    }


def _npci_signature_mode(signature: dict | None = None) -> str:
    signature = signature or {}
    ec = signature.get("ec") if isinstance(signature.get("ec"), dict) else {}
    mode = str(ec.get("mode") or "none").lower()
    if mode not in EC_RESULT_MODES:
        mode = "none"
    return mode


def _npci_mode_root(signature: dict | None = None) -> Path:
    path = NPCI_INTERVENTION_RESULTS_DIR / _npci_signature_mode(signature)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _npci_mode_backbone_dir(signature: dict | None = None) -> Path:
    path = _npci_mode_root(signature) / "backbone"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _npci_mode_backbone_stage_dir(signature: dict | None = None) -> Path:
    path = _npci_mode_root(signature) / "backbone_stages"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _npci_mode_stage3_node_dir(signature: dict | None = None) -> Path:
    path = _npci_mode_root(signature) / "stage3_nodes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _npci_intervention_signature(
    strength: float,
    lesion_idx: int,
    cleaned_backbone: dict,
    cleaned_stage3: dict,
) -> dict:
    payload = {
        "cache_version": NPCI_INTERVENTION_CACHE_VERSION,
        "roi_num": ROI_num,
        "using_steps": using_steps,
        "strength": float(strength),
        "lesion_idx": int(lesion_idx),
        "ec": _npci_ec_signature(),
        "backbone": _to_plain_python(cleaned_backbone),
        "stage3": _to_plain_python(cleaned_stage3),
    }
    payload["digest"] = _stable_json_hash(payload)
    return payload


def _npci_backbone_signature(strength: float, lesion_idx: int, cleaned_backbone: dict) -> dict:
    payload = {
        "cache_version": NPCI_INTERVENTION_CACHE_VERSION,
        "roi_num": ROI_num,
        "using_steps": using_steps,
        "strength": float(strength),
        "lesion_idx": int(lesion_idx),
        "ec": _npci_ec_signature(),
        "backbone_params": _to_plain_python(cleaned_backbone),
    }
    payload["digest"] = _stable_json_hash(payload)
    return payload


def _npci_intervention_result_file(signature: dict) -> Path:
    strength_tag = str(signature["strength"]).replace("-", "m").replace(".", "p")
    node_tag = int(signature["lesion_idx"]) + 1
    mode_tag = _npci_signature_mode(signature)
    return _npci_mode_root(signature) / f"npci_cd_{mode_tag}_s{strength_tag}_roi{node_tag}_{signature['digest']}.json"


def _npci_backbone_result_file(signature: dict) -> Path:
    strength_tag = str(signature["strength"]).replace("-", "m").replace(".", "p")
    node_tag = int(signature["lesion_idx"]) + 1
    mode_tag = _npci_signature_mode(signature)
    return _npci_mode_backbone_dir(signature) / f"npci_backbone_{mode_tag}_s{strength_tag}_roi{node_tag}_{signature['digest']}.json"


def _npci_backbone_stage_dir(signature: dict) -> Path:
    return _npci_mode_backbone_stage_dir(signature) / str(signature["digest"])


def _npci_backbone_stage_file(signature: dict, stage_name: str) -> Path:
    return _npci_backbone_stage_dir(signature) / f"{stage_name}.json"


def _load_npci_intervention_file(signature: dict) -> dict | None:
    path = _npci_intervention_result_file(signature)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if payload.get("cache_version") != NPCI_INTERVENTION_CACHE_VERSION:
        return None
    if payload.get("signature") != signature:
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    result = _to_plain_python(result)
    result["cached"] = True
    result["cache_file"] = _relative_cache_path(path)
    result["saved_at"] = payload.get("saved_at")
    return result


def _load_npci_backbone_file(signature: dict) -> dict | None:
    path = _npci_backbone_result_file(signature)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if payload.get("cache_version") != NPCI_INTERVENTION_CACHE_VERSION:
        return None
    if payload.get("signature") != signature:
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    result = _to_plain_python(result)
    result["cached"] = True
    result["cache_source"] = "backbone_file"
    result["cache_file"] = _relative_cache_path(path)
    result["saved_at"] = payload.get("saved_at")
    return result


def _save_npci_backbone_file(signature: dict, result: dict) -> Path:
    path = _npci_backbone_result_file(signature)
    payload = {
        "cache_version": NPCI_INTERVENTION_CACHE_VERSION,
        "saved_at": time.time(),
        "signature": signature,
        "result": _to_plain_python(result),
    }
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def _save_npci_intervention_file(signature: dict, result: dict) -> Path:
    path = _npci_intervention_result_file(signature)
    payload = {
        "cache_version": NPCI_INTERVENTION_CACHE_VERSION,
        "saved_at": time.time(),
        "signature": signature,
        "result": _to_plain_python(result),
    }
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def _npci_stage3_node_cache_dir(signature: dict) -> Path:
    digest = str(signature.get("digest", "unknown"))
    return _npci_mode_stage3_node_dir(signature) / digest


def _npci_stage3_node_result_file(signature: dict, node_idx: int) -> Path:
    return _npci_stage3_node_cache_dir(signature) / f"roi{int(node_idx) + 1:03d}.json"


def _load_npci_stage3_node_cache(signature: dict, candidate_nodes: list[int]) -> dict[int, dict]:
    cached: dict[int, dict] = {}
    for node in candidate_nodes:
        path = _npci_stage3_node_result_file(signature, int(node))
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if payload.get("cache_version") != NPCI_INTERVENTION_CACHE_VERSION:
            continue
        if payload.get("signature") != signature:
            continue
        grid_records = payload.get("grid_records")
        if not isinstance(grid_records, list) or not grid_records:
            continue
        cached[int(node)] = {
            "grid_records": grid_records,
            "best_row": payload.get("best_row", {}),
            "saved_at": payload.get("saved_at"),
            "cache_file": _relative_cache_path(path),
        }
    return cached


def _save_npci_stage3_node_cache(signature: dict, node_payload: dict) -> Path:
    node_idx = int(node_payload["node"])
    cache_dir = _npci_stage3_node_cache_dir(signature)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _npci_stage3_node_result_file(signature, node_idx)
    payload = {
        "cache_version": NPCI_INTERVENTION_CACHE_VERSION,
        "saved_at": time.time(),
        "signature": signature,
        "node": node_idx,
        "node_label": f"ROI {node_idx + 1}",
        "candidate_index": int(node_payload.get("candidate_index", 0) or 0),
        "candidate_total": int(node_payload.get("candidate_total", 0) or 0),
        "grid_records": _to_plain_python(node_payload.get("grid_records", [])),
        "best_row": _to_plain_python(node_payload.get("best_row", {})),
        "best_metrics": _to_plain_python(_public_scalar_metrics(node_payload.get("best_metrics"))),
    }
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def _save_npci_backbone_stage(signature: dict, stage_name: str, data: dict) -> Path:
    stage_dir = _npci_backbone_stage_dir(signature)
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = _npci_backbone_stage_file(signature, stage_name)
    payload = {
        "cache_version": NPCI_INTERVENTION_CACHE_VERSION,
        "saved_at": time.time(),
        "signature": signature,
        "stage": str(stage_name),
        "data": _to_plain_python(data),
    }
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def _load_npci_backbone_stage(signature: dict, stage_name: str) -> dict | None:
    path = _npci_backbone_stage_file(signature, stage_name)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if payload.get("cache_version") != NPCI_INTERVENTION_CACHE_VERSION:
        return None
    if payload.get("signature") != signature:
        return None
    data = payload.get("data")
    return data if isinstance(data, dict) else None


def _stage1_backbone_cache_payload(stage1_res: dict) -> dict:
    return {
        "lesion_idx": int(stage1_res.get("lesion_idx", -1)),
        "response_vector": np.asarray(stage1_res.get("response_vector", [])).tolist(),
        "signed_response_vector": np.asarray(stage1_res.get("signed_response_vector", [])).tolist(),
        "temporal_response": np.asarray(stage1_res.get("temporal_response", [])).tolist(),
        "roi_temporal_response": np.asarray(stage1_res.get("roi_temporal_response", [])).tolist(),
        "lesion_response_df": _df_records(stage1_res.get("lesion_response_df"), limit=None),
    }


def _restore_stage1_backbone_cache(payload: dict) -> dict:
    out = dict(payload)
    out["response_vector"] = np.asarray(payload.get("response_vector", []), dtype=np.float64)
    out["signed_response_vector"] = np.asarray(payload.get("signed_response_vector", []), dtype=np.float64)
    out["temporal_response"] = np.asarray(payload.get("temporal_response", []), dtype=np.float64)
    out["roi_temporal_response"] = np.asarray(payload.get("roi_temporal_response", []), dtype=np.float64)
    out["lesion_response_df"] = pd.DataFrame(payload.get("lesion_response_df", []))
    return out


def _screening_cache_payload(screen_res: dict) -> dict:
    return {
        "candidate_nodes": [int(x) for x in screen_res.get("candidate_nodes", [])],
        "info": _to_plain_python(screen_res.get("info", {})),
        "screening_edge_df": _df_records(screen_res.get("screening_edge_df"), limit=None),
        "node_score_df": _df_records(screen_res.get("node_score_df"), limit=None),
    }


def _restore_screening_cache(payload: dict) -> dict:
    return {
        "candidate_nodes": [int(x) for x in payload.get("candidate_nodes", [])],
        "info": payload.get("info", {}),
        "screening_edge_df": pd.DataFrame(payload.get("screening_edge_df", [])),
        "node_score_df": pd.DataFrame(payload.get("node_score_df", [])),
    }


def _normalize01(values, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def get_default_fmri_files() -> list[Path]:
    return [REAL_FMRI_DIR / filename for filename in DEFAULT_FMRI_FILENAMES]


npci_runtime_lock = threading.Lock()


def _sanitize_npci_params(params: dict | None = None) -> dict:
    cleaned = dict(NPCI_DEFAULT_PARAMS)
    if isinstance(params, dict):
        for key, value in params.items():
            if key in cleaned and value is not None:
                cleaned[key] = value

    def _clamp_int(name: str, low: int, high: int):
        cleaned[name] = max(low, min(high, int(cleaned[name])))

    def _clamp_float(name: str, low: float, high: float):
        cleaned[name] = max(low, min(high, float(cleaned[name])))

    _clamp_int("horizon", 5, 60)
    _clamp_int("n_seed_windows", 8, 256)
    _clamp_int("tau", 1, 3)
    _clamp_int("max_candidate_nodes", 8, ROI_num)
    _clamp_int("min_candidate_nodes", 4, ROI_num)
    _clamp_int("top_k_direct_lagged", 4, ROI_num - 1)
    _clamp_int("top_k_response", 4, ROI_num - 1)
    _clamp_int("top_k_fc", 0, ROI_num - 1)
    _clamp_float("w_lagged", 0.0, 5.0)
    _clamp_float("w_response", 0.0, 5.0)
    _clamp_float("w_fc", 0.0, 5.0)
    _clamp_float("pc_alpha", 1e-4, 0.2)
    _clamp_int("max_core_edges", 8, ROI_num * 2)
    _clamp_int("max_direct_lesion_edges", 1, ROI_num)
    _clamp_float("strong_quantile", 0.5, 0.99)
    _clamp_float("causal_weight", 0.0, 1.0)
    _clamp_float("fc_weight", 0.0, 1.0)
    if cleaned["min_candidate_nodes"] > cleaned["max_candidate_nodes"]:
        cleaned["min_candidate_nodes"] = cleaned["max_candidate_nodes"]
    return cleaned




def _ensure_npci_runtime_ready():
    global ANN, inputs_data, targets_data
    if ANN is not None and inputs_data is not None and targets_data is not None:
        return
    with npci_runtime_lock:
        if ANN is not None and inputs_data is not None and targets_data is not None:
            return
        npi = get_npi_module()
        signals = []
        inputs = []
        targets = []
        for f in get_default_fmri_files():
            if os.path.exists(f):
                sig = np.loadtxt(f)[30:, :ROI_num]
                inp, tar = npi.multi2one(sig, using_steps)
                signals.append(sig)
                inputs.append(inp)
                targets.append(tar)
        if not signals:
            raise FileNotFoundError(f"未找到NPCI运行所需的fMRI文本数据，请检查目录: {REAL_FMRI_DIR}")
        inputs_data = np.vstack(inputs)
        targets_data = np.vstack(targets)
        ANN = npi.ANN_MLP(
            input_dim=using_steps * ROI_num,
            hidden_dim=2 * ROI_num,
            latent_dim=int(0.8 * ROI_num),
            output_dim=ROI_num,
        )
        ANN, _, _ = npi.train_NN(
            ANN,
            inputs_data,
            targets_data,
            batch_size=batch_size,
            train_set_proportion=train_set_proportion,
            num_epochs=100,
            lr=1e-3,
            l2=5e-5,
        )


def _pick_npci_screen_signal(signals: list[np.ndarray]) -> np.ndarray:
    if not signals:
        raise ValueError("未找到可用于NPCI筛查的原始时序")
    # LPCMCI expects one contiguous run; choose the longest session to avoid
    # artificial lag links across session boundaries.
    return max((np.asarray(sig, dtype=np.float64) for sig in signals), key=lambda arr: arr.shape[0])


def _signed_lagged_corr_matrix(signals: np.ndarray, tau: int = 1) -> np.ndarray:
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= tau + 2:
        raise ValueError("signals长度不足，无法计算滞后相关")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    past = arr[:-tau, :]
    future = arr[tau:, :]
    past = (past - past.mean(axis=0, keepdims=True)) / np.maximum(past.std(axis=0, keepdims=True), 1e-12)
    future = (future - future.mean(axis=0, keepdims=True)) / np.maximum(future.std(axis=0, keepdims=True), 1e-12)
    corr = (past.T @ future) / max(past.shape[0], 1)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    return corr


def _screening_edges_to_signed_df(
    screening_edge_df: pd.DataFrame,
    candidate_nodes: list[int],
    signed_score_matrix: np.ndarray,
) -> pd.DataFrame:
    if screening_edge_df is None or len(screening_edge_df) == 0:
        return pd.DataFrame(columns=["src", "dst", "tau", "link", "val", "abs_val", "p"])
    candidate_set = set(int(x) for x in candidate_nodes)
    df = screening_edge_df[
        screening_edge_df["src"].astype(int).isin(candidate_set)
        & screening_edge_df["dst"].astype(int).isin(candidate_set)
    ].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["src", "dst", "tau", "link", "val", "abs_val", "p"])
    vals = []
    for _, row in df.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])
        vals.append(float(signed_score_matrix[src, dst]))
    df["val"] = vals
    df["abs_val"] = np.abs(df["val"].to_numpy(dtype=np.float64))
    df["p"] = np.clip(0.05 - 0.04 * _normalize01(df["abs_val"].to_numpy(dtype=np.float64)), 0.001, 0.05)
    df["link"] = "screen"
    return df.sort_values(["abs_val", "p"], ascending=[False, True]).reset_index(drop=True)


def _build_npci_nodes_and_edges(core_edges: pd.DataFrame, lesion_idx: int) -> tuple[list[dict], list[dict], list[dict]]:
    if core_edges is None or len(core_edges) == 0:
        return [], [], []
    lesion_idx = int(lesion_idx)
    involved_nodes = set(core_edges["src"].astype(int).tolist()) | set(core_edges["dst"].astype(int).tolist())
    involved_nodes.add(lesion_idx)
    score_map = {}
    in_degree = {int(n): 0 for n in involved_nodes}
    out_degree = {int(n): 0 for n in involved_nodes}
    for _, row in core_edges.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])
        val = float(row["val"])
        abs_val = abs(val)
        out_degree[src] = out_degree.get(src, 0) + 1
        in_degree[dst] = in_degree.get(dst, 0) + 1
        score_map[src] = score_map.get(src, 0.0) + val
        score_map[dst] = score_map.get(dst, 0.0) + val
        if src == lesion_idx:
            score_map[dst] = score_map.get(dst, 0.0) + abs_val
        elif dst == lesion_idx:
            score_map[src] = score_map.get(src, 0.0) - abs_val

    nodes = []
    for idx in sorted(involved_nodes):
        if idx == lesion_idx:
            continue
        score = float(score_map.get(idx, 0.0))
        nodes.append({
            "label": f"ROI {idx + 1}",
            "index": int(idx),
            "ec": round(score, 4),
            "type": "positive" if score >= 0 else "negative",
            "coord": roi_vox[idx].tolist(),
            "in_degree": int(in_degree.get(idx, 0)),
            "out_degree": int(out_degree.get(idx, 0)),
            "evidence": _edge_evidence(lesion_idx, idx),
        })
    nodes.sort(key=lambda item: abs(item["ec"]), reverse=True)

    edges_positive = []
    edges_negative = []
    for _, row in core_edges.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])
        val = float(row["val"])
        edge = {
            "source": int(src),
            "target": int(dst),
            "from": roi_vox[src].tolist(),
            "to": roi_vox[dst].tolist(),
            "strength": round(abs(val), 4) if val < 0 else round(val, 4),
            "evidence": _edge_evidence(src, dst),
        }
        if val >= 0:
            edge["strength"] = round(val, 4)
            edges_positive.append(edge)
        else:
            edge["strength"] = round(abs(val), 4)
            edges_negative.append(edge)
    return nodes, edges_positive, edges_negative


def _filter_npci_result_view(result: dict, top_n: int | None = None, filter_type: str = "all") -> dict:
    view = copy.deepcopy(_to_plain_python(result))
    filter_type = str(filter_type or "all").lower()
    root_idx = int(result.get("pert_node_index", -1))
    nodes = list(view.get("nodes", []))
    pos_edges = list(view.get("edges_positive", []))
    neg_edges = list(view.get("edges_negative", []))

    if filter_type == "positive":
        nodes = [n for n in nodes if n.get("ec", 0) >= 0]
        neg_edges = []
    elif filter_type == "negative":
        nodes = [n for n in nodes if n.get("ec", 0) < 0]
        pos_edges = []

    if top_n is not None:
        top_n = max(1, int(top_n))
        nodes = sorted(nodes, key=lambda item: abs(float(item.get("ec", 0.0))), reverse=True)[:top_n]
        keep_indices = {int(n["index"]) for n in nodes}
        if root_idx >= 0:
            keep_indices.add(root_idx)

        def _edge_keep(edge: dict) -> bool:
            src = int(edge.get("source", -1))
            dst = int(edge.get("target", -1))
            return src in keep_indices and dst in keep_indices

        pos_edges = [e for e in pos_edges if _edge_keep(e)]
        neg_edges = [e for e in neg_edges if _edge_keep(e)]

    view["nodes"] = nodes
    view["edges_positive"] = pos_edges
    view["edges_negative"] = neg_edges
    stats = view.get("stats", {})
    stats["num_nodes"] = len(nodes) + 1
    stats["num_positive"] = len(pos_edges)
    stats["num_negative"] = len(neg_edges)
    stats["total_edges"] = len(pos_edges) + len(neg_edges)
    if top_n is not None:
        stats["top_n"] = len(nodes)
        stats["max_abs_ec"] = max([abs(float(n.get("ec", 0.0))) for n in nodes], default=0.0)
        stats["min_abs_ec"] = min([abs(float(n.get("ec", 0.0))) for n in nodes], default=0.0)
    view["stats"] = stats
    return view


def _run_npci_backbone(strength: float, lesion_idx: int, params: dict | None = None) -> dict:
    cleaned = _sanitize_npci_params(params)
    signature = _npci_backbone_signature(strength, lesion_idx, cleaned)
    cache_key = signature["digest"]
    cached = npci_backbone_cache.get(cache_key)
    if cached is not None:
        return cached
    cached = _load_npci_backbone_file(signature)
    if cached is not None:
        npci_backbone_cache[cache_key] = copy.deepcopy(cached)
        return cached

    def _backbone_status(stage: str, message: str, progress: int, **extra):
        with npci_precompute_lock:
            active = (
                bool(npci_precompute_status.get("is_running"))
                and int(npci_precompute_status.get("node_idx", -1)) == int(lesion_idx)
                and float(npci_precompute_status.get("strength", strength)) == float(strength)
            )
        if not active:
            return
        payload = {
            "stage": stage,
            "progress": int(progress),
            "total": max(int(npci_precompute_status.get("total") or 1), 5),
            "message": message,
            "last_heartbeat_at": time.time(),
            "backbone_cache_dir": _relative_cache_path(_npci_backbone_stage_dir(signature)),
        }
        payload.update(extra)
        _update_npci_precompute_status(**payload)

    _ensure_npci_runtime_ready()
    _backbone_status("stage0_data", "NPCI Stage0 数据与先验矩阵准备中...", 0)
    raw_signals = _load_signals()
    if not raw_signals:
        raise FileNotFoundError("未找到NPCI所需原始fMRI数据")
    screen_signal = _pick_npci_screen_signal(raw_signals)
    fc_matrix = _fc_from_signals(screen_signal)
    signed_score = _signed_lagged_corr_matrix(screen_signal, tau=int(cleaned["tau"]))

    modules = get_npci_modules()
    lesion_response = modules["lesion_response"]
    local_lpcmci = modules["local_lpcmci"]

    stage1_payload = _load_npci_backbone_stage(signature, "stage1_response")
    if stage1_payload is not None:
        stage1_res = _restore_stage1_backbone_cache(stage1_payload)
        _backbone_status("stage1", "NPCI Stage1 病灶响应已从缓存读取", 1)
    else:
        _backbone_status("stage1", "NPCI Stage1 病灶响应计算中：healthy/lesion rollout...", 1)
        stage1_res = lesion_response.compute_lesion_response(
            model=ANN,
            inputs=inputs_data,
            roi_num=ROI_num,
            using_steps=using_steps,
            lesion_idx=int(lesion_idx),
            horizon=int(cleaned["horizon"]),
            n_seed_windows=int(cleaned["n_seed_windows"]),
            lesion_strength=float(strength),
            lesion_mode="subtract",
            batch_size=256,
        )
        path = _save_npci_backbone_stage(signature, "stage1_response", _stage1_backbone_cache_payload(stage1_res))
        _backbone_status("stage1", "NPCI Stage1 病灶响应已保存", 1, stage1_cache_file=_relative_cache_path(path))

    screening_payload = _load_npci_backbone_stage(signature, "stage2_screening")
    if screening_payload is not None:
        screen_res = _restore_screening_cache(screening_payload)
        _backbone_status(
            "stage2_screening",
            f"NPCI Stage2 候选筛选已从缓存读取：{len(screen_res['candidate_nodes'])} 个候选节点",
            2,
            candidate_count=len(screen_res["candidate_nodes"]),
            candidate_nodes=[int(x) for x in screen_res["candidate_nodes"]],
        )
    else:
        _backbone_status("stage2_screening", "NPCI Stage2 候选筛选中：lagged response + FC priors...", 2)
        screen_res = local_lpcmci.run_lagged_candidate_screening(
            signals=screen_signal,
            lesion_idx=int(lesion_idx),
            lesion_response_prior=stage1_res,
            empirical_FC=fc_matrix,
            tau=int(cleaned["tau"]),
            max_candidate_nodes=int(cleaned["max_candidate_nodes"]),
            min_candidate_nodes=int(cleaned["min_candidate_nodes"]),
            top_k_direct_lagged=int(cleaned["top_k_direct_lagged"]),
            top_k_response=int(cleaned["top_k_response"]),
            top_k_fc=int(cleaned["top_k_fc"]),
            w_lagged=float(cleaned["w_lagged"]),
            w_response=float(cleaned["w_response"]),
            w_fc=float(cleaned["w_fc"]),
            standardize="auto",
        )
        path = _save_npci_backbone_stage(signature, "stage2_screening", _screening_cache_payload(screen_res))
        _backbone_status(
            "stage2_screening",
            f"NPCI Stage2 候选筛选已保存：{len(screen_res['candidate_nodes'])} 个候选节点",
            2,
            candidate_count=len(screen_res["candidate_nodes"]),
            candidate_nodes=[int(x) for x in screen_res["candidate_nodes"]],
            screening_cache_file=_relative_cache_path(path),
        )

    runtime_note = ""
    local_edges_payload = _load_npci_backbone_stage(signature, "stage2_local_edges")
    if local_edges_payload is not None:
        local_edge_df = pd.DataFrame(local_edges_payload.get("local_edges", []))
        runtime_note = str(local_edges_payload.get("runtime_note", ""))
        _backbone_status(
            "stage2_lpcmci",
            f"NPCI Stage2 局部因果边已从缓存读取：{len(local_edge_df)} 条边",
            3,
            local_edge_count=len(local_edge_df),
        )
    else:
        _backbone_status("stage2_lpcmci", "NPCI Stage2 局部 LPCMCI 运行中：这是最慢的一步...", 3)
        try:
            lpcmci_res, node_map = local_lpcmci.run_candidate_subnetwork_lpcmci(
                signals=screen_signal,
                candidate_nodes=screen_res["candidate_nodes"],
                tau_min=1,
                tau_max=int(cleaned["tau"]),
                pc_alpha=float(cleaned["pc_alpha"]),
                verbosity=0,
                standardize="auto",
                max_cond_px=1,
                max_p_global=2,
                max_p_non_ancestral=2,
                max_q_global=5,
                max_pds_set=8,
            )
            local_edge_df = local_lpcmci.lpcmci_to_edge_df(
                lpcmci_res,
                node_map=node_map,
                include_contemp=False,
                lag_only=True,
                alpha_level=float(cleaned["pc_alpha"]),
            )
            if len(local_edge_df) == 0:
                raise RuntimeError("LPCMCI未返回有效滞后边")
        except Exception as exc:
            runtime_note = f"未启用完整LPCMCI，当前使用Stage1+lagged-screening近似核心：{exc}"
            local_edge_df = _screening_edges_to_signed_df(
                screen_res["screening_edge_df"],
                screen_res["candidate_nodes"],
                signed_score,
            )
            if len(local_edge_df) == 0:
                raise RuntimeError("NPCI筛查阶段未产生有效候选边")
        path = _save_npci_backbone_stage(signature, "stage2_local_edges", {
            "local_edges": _df_records(local_edge_df, limit=None),
            "runtime_note": runtime_note,
        })
        _backbone_status(
            "stage2_lpcmci",
            f"NPCI Stage2 局部因果边已保存：{len(local_edge_df)} 条边",
            3,
            local_edge_count=len(local_edge_df),
            local_edges_cache_file=_relative_cache_path(path),
        )

    _backbone_status("stage2_core", "NPCI Stage2 病灶核心子图提取与 FC 约束剪枝中...", 4)
    lesion_nodes, lesion_sub_edges, lesion_info = local_lpcmci.extract_lesion_neighborhood(
        local_edge_df,
        lesion_idx=int(lesion_idx),
        min_nodes=max(6, int(cleaned["min_candidate_nodes"]) // 3),
        max_nodes=max(12, int(cleaned["max_candidate_nodes"]) // 2),
        direction="both",
        expand_to_second_order=True,
    )
    lesion_core_edges, prune_info = local_lpcmci.prune_edges_by_lpcmci_fc_constraints(
        lesion_sub_edges,
        empirical_FC=fc_matrix,
        lesion_idx=int(lesion_idx),
        max_edges=int(cleaned["max_core_edges"]),
        alpha_level=float(cleaned["pc_alpha"]),
        strong_quantile=float(cleaned["strong_quantile"]),
        causal_weight=float(cleaned["causal_weight"]),
        fc_weight=float(cleaned["fc_weight"]),
        max_direct_lesion_edges=int(cleaned["max_direct_lesion_edges"]),
    )

    nodes, edges_positive, edges_negative = _build_npci_nodes_and_edges(lesion_core_edges, lesion_idx)
    abs_vals = lesion_core_edges["abs_val"].to_numpy(dtype=np.float64) if len(lesion_core_edges) else np.array([])
    response_vector = np.asarray(stage1_res.get("response_vector", np.zeros(ROI_num)), dtype=np.float64)

    result = {
        "available": True,
        "strength": float(strength),
        "pert_node_index": int(lesion_idx),
        "mask": {
            "key": "npci",
            "label": "NPCI Core",
            "active": True,
            "available": True,
            "dynamic": True,
        },
        "pert_node_label": f"ROI {int(lesion_idx) + 1}",
        "pert_coord": roi_vox[int(lesion_idx)].tolist(),
        "nodes": nodes,
        "edges_positive": edges_positive,
        "edges_negative": edges_negative,
        "stats": {
            "num_nodes": len(nodes) + 1,
            "num_positive": len(edges_positive),
            "num_negative": len(edges_negative),
            "total_edges": len(edges_positive) + len(edges_negative),
            "max_ec": round(float(abs_vals.max()), 4) if abs_vals.size else 0.0,
            "min_ec": round(float(abs_vals.min()), 4) if abs_vals.size else 0.0,
            "avg_abs_ec": round(float(abs_vals.mean()), 4) if abs_vals.size else 0.0,
            "density": round(
                (len(edges_positive) + len(edges_negative)) / max((len(nodes) + 1) * max(len(nodes), 1), 1),
                4,
            ),
            "global_response": round(float(np.mean(response_vector)), 4),
            "evidence_summary": _evidence_summary_for_edges([
                (int(row["src"]), int(row["dst"])) for _, row in lesion_core_edges.iterrows()
            ]),
            "mode_reduction_vs_npi": _mode_reduction_against_npi(),
            "original_edges": int(len(local_edge_df)),
            "filtered_edges": int(len(lesion_core_edges)),
            "removed_edges": max(int(len(local_edge_df)) - int(len(lesion_core_edges)), 0),
            "removed_pct": round(
                max(int(len(local_edge_df)) - int(len(lesion_core_edges)), 0) / int(len(local_edge_df)),
                4,
            ) if len(local_edge_df) else 0.0,
            "runtime_note": runtime_note,
            "stage1_top_response_mean": round(float(np.mean(np.sort(response_vector)[-10:])), 4) if response_vector.size else 0.0,
            "n_candidate_nodes": int(len(screen_res["candidate_nodes"])),
            "n_local_edges": int(len(local_edge_df)),
            "n_core_edges": int(len(lesion_core_edges)),
            "used_hop": int(lesion_info.get("used_hop", 1)),
            "prune_reason": prune_info.get("reason", ""),
        },
        "npci": {
            "params": cleaned,
            "stage1": {
                "top_response_nodes": stage1_res["lesion_response_df"].head(12).to_dict(orient="records"),
            },
            "screening": {
                "candidate_nodes": [int(x) for x in screen_res["candidate_nodes"]],
                "info": screen_res.get("info", {}),
            },
            "lesion_subgraph": {
                "nodes": [int(x) for x in lesion_nodes],
                "info": lesion_info,
            },
            "core_edges": lesion_core_edges.to_dict(orient="records"),
            "pruning": prune_info,
            "runtime_note": runtime_note,
        },
    }
    result["n_significant_pairs"] = int(result["stats"]["total_edges"])
    result["n_total_pairs"] = max(int(result["stats"]["num_nodes"]) * max(int(result["stats"]["num_nodes"]) - 1, 1), 1)
    result["density"] = float(result["stats"]["density"])
    result = _to_plain_python(result)
    cache_file = _save_npci_backbone_file(signature, result)
    result["cache_file"] = _relative_cache_path(cache_file)
    _backbone_status(
        "stage2_core",
        f"NPCI Stage2 病灶核心已保存：{len(lesion_core_edges)} 条核心边",
        5,
        core_edge_count=len(lesion_core_edges),
        cache_file=_relative_cache_path(cache_file),
    )
    npci_backbone_cache[cache_key] = result
    return result


def _parse_float_list(raw, default: list[float], low: float, high: float) -> list[float]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, (list, tuple)):
        values = raw
    else:
        values = str(raw).replace(";", ",").split(",")
    out = []
    for item in values:
        try:
            val = float(item)
        except Exception:
            continue
        if np.isfinite(val):
            out.append(max(low, min(high, val)))
    return out or list(default)


def _parse_int_list(raw, default: list[int], low: int, high: int) -> list[int]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, (list, tuple)):
        values = raw
    else:
        values = str(raw).replace(";", ",").split(",")
    out = []
    for item in values:
        try:
            val = int(float(item))
        except Exception:
            continue
        out.append(max(low, min(high, val)))
    return out or list(default)


def _parse_schedule_list(raw) -> list[str]:
    allowed = {
        "constant", "early_pulse", "middle_pulse", "late_pulse",
        "ramp_up", "ramp_decay", "exponential_decay",
    }
    default = [
        "constant", "early_pulse", "middle_pulse", "late_pulse",
        "ramp_up", "ramp_decay", "exponential_decay",
    ]
    if raw is None:
        return default
    values = [str(x).strip() for x in str(raw).replace(";", ",").split(",")]
    out = [x for x in values if x in allowed]
    return out or default


def _df_records(df, limit: int | None = None) -> list[dict]:
    if df is None:
        return []
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return []
    view = df.copy()
    if limit is not None:
        view = view.head(max(0, int(limit)))
    rows = view.replace({np.nan: None}).to_dict(orient="records")
    for row in rows:
        if "node" in row and row["node"] is not None:
            idx = int(row["node"])
            row["node"] = idx
            row["node_label"] = f"ROI {idx + 1}"
    return _to_plain_python(rows)


def _public_policy(policy) -> list[dict]:
    if policy is None:
        return []
    if isinstance(policy, dict):
        items = [policy]
    else:
        items = list(policy)
    public = []
    for item in items:
        row = {}
        for key in ("node", "sign", "strength", "schedule"):
            if key in item:
                row[key] = item[key]
        if "node" in row:
            row["node"] = int(row["node"])
            row["node_label"] = f"ROI {int(row['node']) + 1}"
            row["coord"] = roi_vox[int(row["node"])].tolist()
        public.append(row)
    return _to_plain_python(public)


def _public_scalar_metrics(metrics: dict | None) -> dict:
    if not isinstance(metrics, dict):
        return {}
    out = {}
    for key, value in metrics.items():
        # FC matrices are 360x360 and belong in generated plots/files, not API JSON.
        if key in {"FC_healthy", "FC_lesion", "FC_policy"}:
            continue
        if isinstance(value, np.ndarray):
            if value.size > 64:
                continue
            out[key] = value.tolist()
        elif isinstance(value, (list, tuple)) and len(value) > 64:
            continue
        else:
            out[key] = value
    return _to_plain_python(out)


def _sanitize_npci_stage3_params(params: dict | None = None) -> dict:
    raw = params or {}
    profile = str(raw.get("profile", "fast")).strip().lower() or "fast"
    if profile not in {"fast", "full"}:
        profile = "fast"

    def _int(name: str, default: int, low: int, high: int) -> int:
        try:
            value = int(raw.get(name, default))
        except Exception:
            value = default
        return max(low, min(high, value))

    def _float(name: str, default: float, low: float, high: float) -> float:
        try:
            value = float(raw.get(name, default))
        except Exception:
            value = default
        return max(low, min(high, value))

    def _bool(name: str, default: bool) -> bool:
        value = raw.get(name, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    return {
        "profile": profile,
        "target_count": _int("target_count", 0, 0, ROI_num),
        "target_nodes": _parse_int_list(raw.get("target_nodes"), [], 0, ROI_num - 1) if raw.get("target_nodes") else [],
        "signs": _parse_int_list(raw.get("signs"), [-1, 1], -1, 1),
        "strengths": _parse_float_list(raw.get("strengths"), [0.1, 0.3, 0.6, 1.0], 0.0, 3.0),
        "schedules": _parse_schedule_list(raw.get("schedules") or "constant,early_pulse,ramp_decay"),
        "alpha_traj": _float("alpha_traj", 1.0, 0.0, 5.0),
        "beta_fc": _float("beta_fc", 0.8, 0.0, 5.0),
        "lambda_node": _float("lambda_node", 0.0, 0.0, 5.0),
        "lambda_strength": _float("lambda_strength", 0.05, 0.0, 5.0),
        "lambda_temporal": _float("lambda_temporal", 0.03, 0.0, 5.0),
        "run_greedy": _bool("run_greedy", False),
        "max_interventions": _int("max_interventions", 3, 1, 8),
        "top_m_per_step": _int("top_m_per_step", 12, 1, ROI_num),
        "min_marginal_gain": _float("min_marginal_gain", 1e-4, -1.0, 1.0),
    }


def _run_npci_intervention_recovery(
    strength: float,
    lesion_idx: int,
    backbone_params: dict | None = None,
    stage3_params: dict | None = None,
    force_recompute: bool = False,
) -> dict:
    cleaned_backbone = _sanitize_npci_params(backbone_params)
    cleaned_stage3 = _sanitize_npci_stage3_params(stage3_params)
    signature = _npci_intervention_signature(strength, lesion_idx, cleaned_backbone, cleaned_stage3)
    cache_key = signature["digest"]
    if not force_recompute:
        cached = npci_intervention_cache.get(cache_key)
        if cached is not None:
            cached_result = copy.deepcopy(cached)
            cached_result["cached"] = True
            cached_result["cache_source"] = cached_result.get("cache_source", "memory")
            return cached_result

        cached = _load_npci_intervention_file(signature)
        if cached is not None:
            cached["cache_source"] = "file"
            npci_intervention_cache[cache_key] = copy.deepcopy(cached)
            return cached

    _ensure_npci_runtime_ready()
    modules = get_npci_modules()
    intervention = modules["intervention_optimization"]
    recovery_eval = modules["recovery_evaluation"]

    backbone = _run_npci_backbone(strength, lesion_idx, params=cleaned_backbone)
    local_lpcmci = modules["local_lpcmci"]
    core_edge_rows = backbone.get("npci", {}).get("core_edges", [])
    core_edges_df = pd.DataFrame(core_edge_rows)
    candidate_df = pd.DataFrame()
    if len(core_edges_df) > 0:
        candidate_df = local_lpcmci.build_intervention_candidates_from_core(
            core_edges_df,
            lesion_idx=int(lesion_idx),
            exclude_lesion=True,
        )
    candidate_nodes = candidate_df["node"].astype(int).tolist() if len(candidate_df) else []
    if not candidate_nodes:
        candidate_nodes = (
            backbone.get("npci", {})
            .get("screening", {})
            .get("candidate_nodes", [])
        )
    if not candidate_nodes:
        candidate_nodes = [n.get("index") for n in backbone.get("nodes", []) if "index" in n]
    candidate_nodes = [
        int(n) for n in candidate_nodes
        if n is not None and 0 <= int(n) < ROI_num and int(n) != int(lesion_idx)
    ]
    candidate_nodes = list(dict.fromkeys(candidate_nodes))
    requested_target_nodes = [
        int(n) for n in cleaned_stage3.get("target_nodes", [])
        if n is not None and 0 <= int(n) < ROI_num and int(n) != int(lesion_idx)
    ]
    if requested_target_nodes:
        candidate_nodes = list(dict.fromkeys(requested_target_nodes))
    if int(cleaned_stage3["target_count"]) > 0:
        candidate_nodes = candidate_nodes[: int(cleaned_stage3["target_count"])]
    if not candidate_nodes:
        raise RuntimeError("NPCI Stage3 未找到可用干预候选节点")
    cached_node_results = (
        {}
        if force_recompute
        else _load_npci_stage3_node_cache(signature, candidate_nodes)
    )

    stage3_grid_per_node = (
        len(cleaned_stage3["signs"])
        * len(cleaned_stage3["strengths"])
        * len(cleaned_stage3["schedules"])
    )
    stage3_grid_total = len(candidate_nodes) * stage3_grid_per_node
    estimated_policy_count = stage3_grid_total
    if bool(cleaned_stage3["run_greedy"]):
        estimated_policy_count += int(cleaned_stage3["max_interventions"]) * min(
            int(cleaned_stage3["top_m_per_step"]),
            len(candidate_nodes),
        )

    _update_npci_precompute_status(
        stage="stage3_single",
        progress=0,
        total=max(1, estimated_policy_count),
        profile=str(cleaned_stage3.get("profile", "fast")),
        profile_label="完整验证" if str(cleaned_stage3.get("profile", "fast")) == "full" else "快速筛查",
        candidate_count=len(candidate_nodes),
        candidate_nodes=[int(x) for x in candidate_nodes],
        grid_done=0,
        grid_total=stage3_grid_total,
        grid_per_node=stage3_grid_per_node,
        node_grid_done=0,
        node_grid_total=stage3_grid_per_node,
        cached_node_count=len(cached_node_results),
        saved_node_count=len(cached_node_results),
        stage3_node_cache_dir=_relative_cache_path(_npci_stage3_node_cache_dir(signature)),
        greedy_step=0,
        greedy_done=0,
        greedy_total=0,
        estimated_policy_count=estimated_policy_count,
        message=(
            f"NPCI Stage3 {('完整验证' if str(cleaned_stage3.get('profile', 'fast')) == 'full' else '快速筛查')}启动："
            f"{len(candidate_nodes)} 个候选靶点，约 {estimated_policy_count} 次策略评估，"
            f"已缓存 {len(cached_node_results)} 个"
        ),
        last_heartbeat_at=time.time(),
    )

    def _stage3_progress(update: dict):
        phase = str(update.get("phase", "stage3"))
        current_node = update.get("current_node", update.get("node", update.get("candidate_node")))
        grid_done = int(update.get("grid_done", 0) or 0)
        greedy_done = int(update.get("greedy_done", 0) or 0)
        if phase == "greedy":
            stage = "stage3_greedy"
            msg = (
                f"NPCI Stage3 greedy组合搜索：第 {int(update.get('greedy_step', 0) or 0)} 步，"
                f"{greedy_done}/{int(update.get('greedy_total', 0) or 0)}"
            )
        else:
            stage = "stage3_single"
            if phase == "single_cached":
                msg = (
                    f"NPCI Stage3 读取已完成候选："
                    f"{grid_done}/{stage3_grid_total} 策略组合"
                )
            else:
                msg = (
                    f"NPCI Stage3 单靶点搜索："
                    f"{grid_done}/{stage3_grid_total} 策略组合"
                )
        if phase == "greedy":
            greedy_step = int(update.get("greedy_step", 0) or 0)
            greedy_total = int(update.get("greedy_total", 0) or 0)
            greedy_progress_done = max(0, greedy_step - 1) * max(1, greedy_total) + greedy_done
            overall_progress = stage3_grid_total + greedy_progress_done
            grid_done_for_status = stage3_grid_total
        else:
            greedy_total = int(update.get("greedy_total", 0) or 0)
            overall_progress = grid_done
            grid_done_for_status = grid_done
        _update_npci_precompute_status(
            stage=stage,
            progress=min(max(0, overall_progress), max(1, estimated_policy_count)),
            total=max(1, estimated_policy_count),
            current_node=int(current_node) if current_node is not None else None,
            current_node_label=(
                f"ROI {int(current_node) + 1}" if current_node is not None else ""
            ),
            candidate_index=int(update.get("candidate_index", 0) or 0),
            candidate_count=len(candidate_nodes),
            node_grid_done=int(update.get("node_grid_done", 0) or 0),
            node_grid_total=int(update.get("node_grid_total", stage3_grid_per_node) or stage3_grid_per_node),
            grid_done=grid_done_for_status,
            grid_total=stage3_grid_total,
            grid_per_node=stage3_grid_per_node,
            current_sign=update.get("sign"),
            current_policy_strength=update.get("strength"),
            current_schedule=str(update.get("schedule", "")),
            greedy_step=int(update.get("greedy_step", 0) or 0),
            greedy_done=greedy_done,
            greedy_total=greedy_total,
            cached_node_count=len(cached_node_results),
            message=msg,
            last_heartbeat_at=time.time(),
        )

    saved_node_count = len(cached_node_results)

    def _stage3_node_completed(node_payload: dict):
        nonlocal saved_node_count
        path = _save_npci_stage3_node_cache(signature, node_payload)
        saved_node_count += 1
        node_idx = int(node_payload.get("node"))
        _update_npci_precompute_status(
            saved_node_count=saved_node_count,
            last_saved_node=node_idx,
            last_saved_node_label=f"ROI {node_idx + 1}",
            last_node_cache_file=_relative_cache_path(path),
            message=(
                f"NPCI Stage3 已保存候选靶点 ROI {node_idx + 1} "
                f"({saved_node_count}/{len(candidate_nodes)})"
            ),
            last_heartbeat_at=time.time(),
        )

    # Stage 3: search intervention policies over selected candidate nodes.
    stage3_res = intervention.run_single_node_search(
        model=ANN,
        candidate_nodes=candidate_nodes,
        roi_num=ROI_num,
        using_steps=using_steps,
        lesion_idx=int(lesion_idx),
        horizon=int(cleaned_backbone["horizon"]),
        lesion_strength=float(strength),
        lesion_mode="subtract",
        inputs=inputs_data,
        n_seed_windows=int(cleaned_backbone["n_seed_windows"]),
        signs=tuple(int(x) for x in cleaned_stage3["signs"]),
        strengths=tuple(float(x) for x in cleaned_stage3["strengths"]),
        schedules=tuple(str(x) for x in cleaned_stage3["schedules"]),
        alpha_traj=float(cleaned_stage3["alpha_traj"]),
        beta_fc=float(cleaned_stage3["beta_fc"]),
        lambda_node=float(cleaned_stage3["lambda_node"]),
        lambda_strength=float(cleaned_stage3["lambda_strength"]),
        lambda_temporal=float(cleaned_stage3["lambda_temporal"]),
        batch_size=256,
        run_greedy=bool(cleaned_stage3["run_greedy"]),
        max_interventions=int(cleaned_stage3["max_interventions"]),
        top_m_per_step=int(cleaned_stage3["top_m_per_step"]),
        min_marginal_gain=float(cleaned_stage3["min_marginal_gain"]),
        progress_callback=_stage3_progress,
        cached_node_results=cached_node_results,
        node_result_callback=_stage3_node_completed,
    )

    # Stage 4: summarize recovery under the best Stage 3 policy.
    _update_npci_precompute_status(
        stage="stage4",
        progress=max(1, estimated_policy_count),
        total=max(1, estimated_policy_count),
        message="NPCI Stage4 恢复效果评价中...",
        last_heartbeat_at=time.time(),
    )
    recovery_res = recovery_eval.summarize_recovery(
        healthy_traj=stage3_res["healthy_traj"],
        lesion_traj=stage3_res["lesion_traj"],
        policy_traj=stage3_res["best_policy_traj"],
        policy=stage3_res["best_policy"],
        alpha_traj=float(cleaned_stage3["alpha_traj"]),
        beta_fc=float(cleaned_stage3["beta_fc"]),
        lambda_node=float(cleaned_stage3["lambda_node"]),
        lambda_strength=float(cleaned_stage3["lambda_strength"]),
        lambda_temporal=float(cleaned_stage3["lambda_temporal"]),
        distance_metric="rmse",
    )

    best_policy = _public_policy(stage3_res.get("best_policy"))
    best_target = best_policy[0] if best_policy else None
    single_rows = _df_records(stage3_res.get("single_df"), limit=12)
    greedy_rows = _df_records(stage3_res.get("greedy_history"), limit=12)
    roi_recovery_rows = _df_records(recovery_res.get("roi_recovery_df"), limit=20)
    time_recovery_rows = _df_records(recovery_res.get("time_recovery_df"), limit=None)
    distance_rows = _df_records(recovery_res.get("distance_df"), limit=None)

    result = {
        "available": True,
        "strength": float(strength),
        "lesion_idx": int(lesion_idx),
        "lesion_label": f"ROI {int(lesion_idx) + 1}",
        "params": {
            "backbone": cleaned_backbone,
            "stage3": cleaned_stage3,
        },
        "backbone_summary": {
            "candidate_nodes": candidate_nodes,
            "n_candidate_nodes": len(candidate_nodes),
            "n_core_edges": backbone.get("stats", {}).get("n_core_edges", backbone.get("stats", {}).get("total_edges", 0)),
            "runtime_note": backbone.get("npci", {}).get("runtime_note", ""),
        },
        "stage3": {
            "best_target": best_target,
            "best_policy": best_policy,
            "best_metrics": _public_scalar_metrics(stage3_res.get("best_metrics")),
            "best_objective": float(stage3_res.get("best_objective", np.nan)),
            "top_targets": single_rows,
            "greedy_history": greedy_rows,
            "candidate_nodes": candidate_nodes,
            "candidate_table": _df_records(candidate_df, limit=None),
        },
        "stage4": {
            "metrics": _public_scalar_metrics(recovery_res.get("metrics")),
            "time_recovery": time_recovery_rows,
            "distance": distance_rows,
            "top_roi_recovery": roi_recovery_rows,
            "policy_size": int(recovery_res.get("policy_size", len(best_policy))),
            "policy_nodes": [int(x) for x in recovery_res.get("policy_nodes", [])],
        },
        "cached": False,
        "cache_source": "computed",
        "cache_signature": signature["digest"],
    }
    result = _to_plain_python(result)
    cache_file = _save_npci_intervention_file(signature, result)
    result["cache_file"] = _relative_cache_path(cache_file)
    npci_intervention_cache[cache_key] = copy.deepcopy(result)
    return result


def _npci_backbone_params_from_request() -> dict:
    return {
        "horizon": request.args.get("horizon", NPCI_DEFAULT_PARAMS["horizon"], type=int),
        "n_seed_windows": request.args.get("n_seed_windows", NPCI_DEFAULT_PARAMS["n_seed_windows"], type=int),
        "tau": request.args.get("tau", NPCI_DEFAULT_PARAMS["tau"], type=int),
        "max_candidate_nodes": request.args.get("max_candidate_nodes", NPCI_DEFAULT_PARAMS["max_candidate_nodes"], type=int),
        "min_candidate_nodes": request.args.get("min_candidate_nodes", NPCI_DEFAULT_PARAMS["min_candidate_nodes"], type=int),
        "top_k_direct_lagged": request.args.get("top_k_direct_lagged", NPCI_DEFAULT_PARAMS["top_k_direct_lagged"], type=int),
        "top_k_response": request.args.get("top_k_response", NPCI_DEFAULT_PARAMS["top_k_response"], type=int),
        "top_k_fc": request.args.get("top_k_fc", NPCI_DEFAULT_PARAMS["top_k_fc"], type=int),
        "w_lagged": request.args.get("w_lagged", NPCI_DEFAULT_PARAMS["w_lagged"], type=float),
        "w_response": request.args.get("w_response", NPCI_DEFAULT_PARAMS["w_response"], type=float),
        "w_fc": request.args.get("w_fc", NPCI_DEFAULT_PARAMS["w_fc"], type=float),
        "pc_alpha": request.args.get("pc_alpha", NPCI_DEFAULT_PARAMS["pc_alpha"], type=float),
        "max_core_edges": request.args.get("max_core_edges", NPCI_DEFAULT_PARAMS["max_core_edges"], type=int),
        "max_direct_lesion_edges": request.args.get("max_direct_lesion_edges", NPCI_DEFAULT_PARAMS["max_direct_lesion_edges"], type=int),
        "strong_quantile": request.args.get("strong_quantile", NPCI_DEFAULT_PARAMS["strong_quantile"], type=float),
        "causal_weight": request.args.get("causal_weight", NPCI_DEFAULT_PARAMS["causal_weight"], type=float),
        "fc_weight": request.args.get("fc_weight", NPCI_DEFAULT_PARAMS["fc_weight"], type=float),
    }


def _npci_stage3_params_from_request() -> dict:
    return {
        "profile": request.args.get("profile", "fast"),
        "target_count": request.args.get("target_count", 0, type=int),
        "target_nodes": request.args.get("target_nodes", ""),
        "signs": request.args.get("signs", "-1,1"),
        "strengths": request.args.get("strengths", "0.1,0.3,0.6,1.0"),
        "schedules": request.args.get("schedules", "constant,early_pulse,ramp_decay"),
        "alpha_traj": request.args.get("alpha_traj", 1.0, type=float),
        "beta_fc": request.args.get("beta_fc", 0.8, type=float),
        "lambda_node": request.args.get("lambda_node", 0.0, type=float),
        "lambda_strength": request.args.get("lambda_strength", 0.05, type=float),
        "lambda_temporal": request.args.get("lambda_temporal", 0.03, type=float),
        "run_greedy": request.args.get("run_greedy", "false"),
        "max_interventions": request.args.get("max_interventions", 3, type=int),
        "top_m_per_step": request.args.get("top_m_per_step", 12, type=int),
        "min_marginal_gain": request.args.get("min_marginal_gain", 1e-4, type=float),
    }


def _update_npci_precompute_status(**updates):
    with npci_precompute_lock:
        npci_precompute_status.update(updates)


def _npci_precompute_status_snapshot() -> dict:
    with npci_precompute_lock:
        snapshot = dict(npci_precompute_status)
    now = time.time()
    started_at = snapshot.get("started_at")
    if started_at:
        snapshot["elapsed_seconds"] = int(max(0, now - float(started_at)))
    else:
        snapshot["elapsed_seconds"] = None
    grid_total = int(snapshot.get("grid_total") or 0)
    grid_done = int(snapshot.get("grid_done") or 0)
    greedy_total = int(snapshot.get("greedy_total") or 0)
    greedy_done = int(snapshot.get("greedy_done") or 0)
    if grid_total > 0 and snapshot.get("stage") in {"stage3_single", "stage3_greedy"}:
        snapshot["stage3_percent"] = round(100.0 * min(grid_done, grid_total) / grid_total, 2)
    else:
        snapshot["stage3_percent"] = None
    if greedy_total > 0 and snapshot.get("stage") == "stage3_greedy":
        snapshot["greedy_percent"] = round(100.0 * min(greedy_done, greedy_total) / greedy_total, 2)
    else:
        snapshot["greedy_percent"] = None
    progress = int(snapshot.get("progress") or 0)
    total = int(snapshot.get("total") or 0)
    elapsed = snapshot.get("elapsed_seconds")
    if snapshot.get("is_running") and elapsed and progress > 0 and total > progress:
        rate = progress / max(1, int(elapsed))
        snapshot["estimated_remaining_seconds"] = int((total - progress) / max(rate, 1e-9))
    else:
        snapshot["estimated_remaining_seconds"] = None
    return snapshot


def precompute_npci_intervention_async(
    strength: float,
    node_idx: int,
    backbone_params: dict | None = None,
    stage3_params: dict | None = None,
    force: bool = False,
) -> bool:
    initial_stage3 = _sanitize_npci_stage3_params(stage3_params)
    initial_profile = str(initial_stage3.get("profile", "fast"))
    initial_profile_label = "完整验证" if initial_profile == "full" else "快速筛查"
    with npci_precompute_lock:
        if npci_precompute_status["is_running"]:
            return False
        npci_precompute_status.update({
            "is_running": True,
            "progress": 0,
            "total": 1,
            "strength": float(strength),
            "node_idx": int(node_idx),
            "node_label": f"ROI {int(node_idx) + 1}",
            "profile": initial_profile,
            "profile_label": initial_profile_label,
            "stage": "starting",
            "candidate_count": 0,
            "candidate_nodes": [],
            "grid_done": 0,
            "grid_total": 0,
            "grid_per_node": 0,
            "current_node": None,
            "current_node_label": "",
            "candidate_index": 0,
            "node_grid_done": 0,
            "node_grid_total": 0,
            "current_sign": None,
            "current_policy_strength": None,
            "current_schedule": "",
            "cached_node_count": 0,
            "saved_node_count": 0,
            "last_saved_node": None,
            "last_saved_node_label": "",
            "last_node_cache_file": None,
            "stage3_node_cache_dir": None,
            "backbone_cache_dir": None,
            "stage1_cache_file": None,
            "screening_cache_file": None,
            "local_edges_cache_file": None,
            "local_edge_count": 0,
            "core_edge_count": 0,
            "greedy_step": 0,
            "greedy_done": 0,
            "greedy_total": 0,
            "estimated_policy_count": 0,
            "message": f"NPCI Stage 3/4 {initial_profile_label}启动中...",
            "last_error": "",
            "started_at": time.time(),
            "completed_at": None,
            "last_heartbeat_at": time.time(),
            "cache_file": None,
        })

    def _worker():
        try:
            _update_npci_precompute_status(message="正在计算 NPCI 靶点搜索与恢复评价...")
            result = _run_npci_intervention_recovery(
                strength,
                node_idx,
                backbone_params=backbone_params,
                stage3_params=stage3_params,
                force_recompute=force,
            )
            source = result.get("cache_source", "")
            message = "NPCI 结果已读取保存文件" if source == "file" else "NPCI 结果已预计算并保存"
            candidate_nodes = (
                result.get("backbone_summary", {}).get("candidate_nodes", [])
                if isinstance(result, dict)
                else []
            )
            _update_npci_precompute_status(
                is_running=False,
                progress=1,
                stage="completed",
                candidate_count=len(candidate_nodes),
                candidate_nodes=[int(x) for x in candidate_nodes],
                saved_node_count=len(candidate_nodes),
                completed_at=time.time(),
                last_heartbeat_at=time.time(),
                cache_file=result.get("cache_file"),
                message=message,
            )
        except Exception as exc:
            _update_npci_precompute_status(
                is_running=False,
                stage="failed",
                completed_at=time.time(),
                last_heartbeat_at=time.time(),
                last_error=str(exc),
                message="NPCI 预计算失败",
            )

    threading.Thread(target=_worker, daemon=True).start()
    return True


def _normalize_ec_result_mode(mode: str | None) -> str:
    if mode is None:
        return "none"
    key = str(mode).strip().lower()
    if key in {"", "raw", "npi"}:
        key = "none"
    if key not in EC_RESULT_MODES:
        raise ValueError(f"未知EC结果模式: {mode}")
    return key


def _ec_result_file(mode: str) -> Path:
    return EC_RESULTS_DIR / f"ec_{_normalize_ec_result_mode(mode)}.npz"


def _ec_result_meta_file(mode: str) -> Path:
    return EC_RESULTS_DIR / f"ec_{_normalize_ec_result_mode(mode)}_meta.json"


def _strength_key_tuple(values) -> tuple[float, ...]:
    return tuple(round(float(v), 8) for v in values)


def _configured_strengths() -> list[float]:
    return [float(s) for s in strength_list]


def _ec_strengths_current(result: dict) -> bool:
    return _strength_key_tuple(sorted(result.keys())) == _strength_key_tuple(_configured_strengths())


def _default_ec_meta(mode: str = "unknown") -> dict:
    mode_label = (
        COMPUTE_MODE_LABELS.get(mode)
        if mode in EC_RESULT_MODES
        else "历史 EC（模式未知）"
    )
    return {
        "mode": mode,
        "mode_label": mode_label,
        "screen_first": mode not in {"none", "unknown"},
        "updated_at": None,
        "file": None,
    }


def _load_ec_meta(mode: str | None = None) -> dict:
    global current_ec_meta
    if mode is None:
        if current_ec_meta is not None:
            return dict(current_ec_meta)
        target = ec_meta_file
        defaults = _default_ec_meta()
    else:
        mode = _normalize_ec_result_mode(mode)
        target = _ec_result_meta_file(mode)
        defaults = _default_ec_meta(mode)
    if not os.path.exists(target):
        return defaults
    try:
        with open(target, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return defaults
    defaults.update({k: v for k, v in meta.items() if k in defaults})
    return defaults


def _save_ec_meta(mode: str, mode_label: str, screen_first: bool, target_file: Path | None = None) -> dict:
    mode = _normalize_ec_result_mode(mode)
    meta = {
        "mode": mode,
        "mode_label": mode_label,
        "screen_first": bool(screen_first),
        "updated_at": time.time(),
        "file": str(target_file) if target_file is not None else str(_ec_result_file(mode)),
    }
    with open(_ec_result_meta_file(mode), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _load_ec_result(mode: str) -> tuple[dict, dict]:
    mode = _normalize_ec_result_mode(mode)
    target = _ec_result_file(mode)
    if not os.path.exists(target):
        raise FileNotFoundError(f"{COMPUTE_MODE_LABELS[mode]} 结果尚未计算")
    with np.load(target, allow_pickle=True) as data:
        loaded = {float(k): data[k] for k in data.files}
    if not _ec_strengths_current(loaded):
        raise ValueError(
            f"{COMPUTE_MODE_LABELS[mode]} 结果强度 {sorted(loaded.keys())} "
            f"与当前配置 {_configured_strengths()} 不一致，请重新计算。"
        )
    meta = _load_ec_meta(mode)
    return loaded, meta


def _save_ec_result(mode: str, result: dict, mode_label: str | None = None, screen_first: bool | None = None) -> dict:
    mode = _normalize_ec_result_mode(mode)
    target = _ec_result_file(mode)
    save_dict = {str(s): arr for s, arr in result.items()}
    np.savez(target, **save_dict)
    return _save_ec_meta(
        mode,
        mode_label or COMPUTE_MODE_LABELS[mode],
        mode != "none" if screen_first is None else bool(screen_first),
        target_file=target,
    )


NPI_RELATION_THRESHOLD = float(os.getenv("NPI_RELATION_THRESHOLD", "0.01"))


def _count_npi_perturbation_relations(mat: np.ndarray, threshold: float = NPI_RELATION_THRESHOLD) -> int:
    """
    Count directed NPI perturbation-response relations source→target.

    This follows the NPI interpretation: perturb each source node and count
    target responses with |EC[source,target]| above the visualization threshold.
    Self-relations on the diagonal are excluded.
    """
    arr = np.asarray(mat)
    rel = np.abs(arr) >= float(threshold)
    if rel.ndim == 2 and rel.shape[0] == rel.shape[1]:
        np.fill_diagonal(rel, False)
    return int(np.count_nonzero(rel))


def _edge_count_for_ec_result(result: dict, threshold: float = NPI_RELATION_THRESHOLD) -> tuple[int, dict]:
    per_strength = {}
    total = 0
    for strength, mat in sorted(result.items()):
        count = _count_npi_perturbation_relations(mat, threshold)
        per_strength[str(strength)] = count
        total += count
    return total, per_strength


def _ec_result_reduction_vs_npi(mode: str, threshold: float = NPI_RELATION_THRESHOLD) -> dict:
    mode = _normalize_ec_result_mode(mode)
    if not os.path.exists(_ec_result_file(mode)):
        return {"available": False, "mode": mode, "message": "该模式结果尚未计算。"}
    if not os.path.exists(_ec_result_file("none")):
        return {
            "available": False,
            "mode": mode,
            "baseline_mode": "none",
            "message": "纯 NPI 结果尚未计算，无法统计相对筛除线数。",
        }
    baseline, _ = _load_ec_result("none")
    current, _ = _load_ec_result(mode)
    baseline_total, baseline_per = _edge_count_for_ec_result(baseline, threshold)
    current_total, current_per = _edge_count_for_ec_result(current, threshold)
    per_strength = {}
    for strength, base_count in baseline_per.items():
        mode_count = int(current_per.get(strength, 0))
        removed = max(base_count - mode_count, 0)
        per_strength[strength] = {
            "baseline_edges": base_count,
            "mode_edges": mode_count,
            "removed_edges": removed,
            "removed_pct": round(removed / base_count, 4) if base_count else 0.0,
        }
    removed_total = max(baseline_total - current_total, 0)
    return {
        "available": True,
        "count_type": "directed_npi_perturbation_relations",
        "threshold": float(threshold),
        "mode": mode,
        "baseline_mode": "none",
        "baseline_edges": baseline_total,
        "mode_edges": current_total,
        "removed_edges": removed_total,
        "removed_pct": round(removed_total / baseline_total, 4) if baseline_total else 0.0,
        "per_strength": per_strength,
    }


def _available_ec_results() -> list[dict]:
    rows = []
    for mode in EC_RESULT_MODES:
        target = _ec_result_file(mode)
        if not os.path.exists(target):
            continue
        try:
            with np.load(target, allow_pickle=True) as data:
                result_strengths = [float(k) for k in data.files]
        except Exception:
            continue
        if _strength_key_tuple(result_strengths) != _strength_key_tuple(_configured_strengths()):
            continue
        meta = _load_ec_meta(mode)
        rows.append({
            "mode": mode,
            "mode_label": meta["mode_label"],
            "screen_first": bool(meta["screen_first"]),
            "updated_at": meta["updated_at"],
            "file": str(target),
            "reduction_vs_npi": _ec_result_reduction_vs_npi(mode),
        })
    return rows


def _set_current_ec_result(mode: str, result: dict, meta: dict):
    global ec_dict, current_ec_meta, npi_intervention_score_cache
    global npi_intervention_cache_signature, npci_backbone_cache, npci_intervention_cache
    ec_dict = result
    current_ec_meta = dict(meta)
    npi_intervention_score_cache = None
    npi_intervention_cache_signature = None
    npci_backbone_cache = {}
    npci_intervention_cache = {}
    computation_status["mode"] = current_ec_meta["mode"]
    computation_status["mode_label"] = current_ec_meta["mode_label"]
    computation_status["screen_first"] = bool(current_ec_meta["screen_first"])


def _freeze_params(params: dict) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in params.items()))


def _is_default_params(key: str, params: dict) -> bool:
    return _freeze_params(params) == _freeze_params(DEFAULT_MASK_PARAMS[key])


def _get_mask_by_key(key: str):
    if key == "consensus":
        mask, _ = _compute_consensus_mask()
        return mask
    if key == "ensemble":
        mask, _ = _compute_intervention_ensemble_mask()
        return mask
    mapping = {
        "granger": granger_mask,
        "vargranger": var_granger_mask,
        "pcmci": pcmci_mask,
        "sparsevar": sparse_var_mask,
        "fask": fask_mask,
        "twostep": two_step_mask,
        "fask_local": fask_local_mask,
        "twostep_local": two_step_local_mask,
    }
    return mapping[key]


def _set_mask_by_key(key: str, mask: np.ndarray):
    global granger_mask, var_granger_mask, pcmci_mask, sparse_var_mask, fask_mask
    global two_step_mask, fask_local_mask, two_step_local_mask

    if key == "granger":
        granger_mask = mask
    elif key == "vargranger":
        var_granger_mask = mask
    elif key == "pcmci":
        pcmci_mask = mask
    elif key == "sparsevar":
        sparse_var_mask = mask
    elif key == "fask":
        fask_mask = mask
    elif key == "twostep":
        two_step_mask = mask
    elif key == "fask_local":
        fask_local_mask = mask
    elif key == "twostep_local":
        two_step_local_mask = mask




def _get_screening_state(key: str) -> dict:
    with screening_lock:
        return dict(screening_status[key])


def _update_screening_state(key: str, **updates):
    with screening_lock:
        screening_status[key].update(updates)


def _mark_mask_cached(key: str, completed_at: float | None = None):
    _update_screening_state(
        key,
        is_running=False,
        message=f"{DEFAULT_MASK_LABELS[key]}默认结果已缓存",
        last_error="",
        params=_freeze_params(DEFAULT_MASK_PARAMS[key]),
        completed_at=completed_at if completed_at is not None else time.time(),
    )


def _full_fmri_role_for_mode(key: str, pipeline_mode: str) -> str:
    return "prefilter" if key == pipeline_mode else "ordinary"


def _full_fmri_context_meta(key: str, pipeline_mode: str) -> tuple[str, str]:
    mode = _normalize_pipeline_mode(pipeline_mode)
    return mode, _full_fmri_role_for_mode(key, mode)


def _full_fmri_memory_context_matches(key: str, pipeline_mode: str) -> bool:
    if key not in FMRI_NATIVE_SCREEN_KEYS:
        return True
    mode, role = _full_fmri_context_meta(key, pipeline_mode)
    meta = full_fmri_mask_meta.get(key) or {}
    return meta.get("pipeline_mode") == mode and meta.get("role") == role


def _collect_cached_keys(keys: list | None = None, pipeline_mode: str = "none") -> list:
    target_keys = list(keys or PRECOMPUTE_MASK_KEYS)
    cached_keys = []
    for key in target_keys:
        if key in FMRI_NATIVE_SCREEN_KEYS:
            if (
                _get_mask_by_key(key) is not None
                and _full_fmri_memory_context_matches(key, pipeline_mode)
            ) or _load_single_cached_mask(key, pipeline_mode=pipeline_mode):
                cached_keys.append(key)
        elif _get_mask_by_key(key) is not None:
            cached_keys.append(key)
    return cached_keys


def _resolve_precompute_keys(group: str | None = None, keys: list | None = None) -> list[str]:
    if keys:
        resolved = []
        for key in keys:
            norm = _normalize_mask_key(key)
            if norm is None or norm not in PRECOMPUTE_MASK_KEYS:
                raise ValueError(f"该算法不支持预计算: {key}")
            if norm not in resolved:
                resolved.append(norm)
        if not resolved:
            raise ValueError("预计算算法列表不能为空")
        return resolved

    group_key = (group or "all").strip().lower()
    if group_key in {"all", "full", "全部", "全量"}:
        return list(PRECOMPUTE_MASK_KEYS)
    if group_key in {"standard", "baseline", "classic", "other", "others", "基础", "普通"}:
        return list(STANDARD_PRECOMPUTE_MASK_KEYS)
    if group_key in {"fmri", "native", "fask_twostep", "fask-twostep", "new", "新算法"}:
        return list(FMRI_PRECOMPUTE_MASK_KEYS)
    if group_key in {"fask", "fask_full", "fask-full"}:
        return ["fask"]
    if group_key in {"twostep", "two_step", "two-step", "twostep_full", "twostep-full"}:
        return ["twostep"]
    raise ValueError(f"未知预计算分组: {group}")


def _mask_result_paths(key: str, pipeline_mode: str | None = None, role: str | None = None) -> tuple[Path, Path]:
    if key in FMRI_NATIVE_SCREEN_KEYS and pipeline_mode is not None:
        mode = _normalize_pipeline_mode(pipeline_mode)
        context_role = role or _full_fmri_role_for_mode(key, mode)
        root = FULL_FMRI_MASK_RESULTS_DIR / mode / context_role
        root.mkdir(parents=True, exist_ok=True)
        stem = f"{key}_{context_role}_in_{mode}"
        return root / f"{stem}.npz", root / f"{stem}_meta.json"
    return MASK_RESULTS_DIR / f"{key}.npz", MASK_RESULTS_DIR / f"{key}_meta.json"


def _latest_mask_cache_mtime(keys: list | None = None, pipeline_mode: str = "none") -> float | None:
    mtimes = []
    if os.path.exists(mask_cache_file):
        mtimes.append(os.path.getmtime(mask_cache_file))
    for key in (keys or PRECOMPUTE_MASK_KEYS):
        data_path, meta_path = _mask_result_paths(
            key,
            pipeline_mode=pipeline_mode if key in FMRI_NATIVE_SCREEN_KEYS else None,
        )
        if os.path.exists(data_path):
            mtimes.append(os.path.getmtime(data_path))
        if os.path.exists(meta_path):
            mtimes.append(os.path.getmtime(meta_path))
    return max(mtimes) if mtimes else None


def _save_single_cached_mask(
    key: str,
    mask: np.ndarray,
    params: dict | None = None,
    pipeline_mode: str | None = None,
):
    mode = None
    role = None
    if key in FMRI_NATIVE_SCREEN_KEYS and pipeline_mode is not None:
        mode, role = _full_fmri_context_meta(key, pipeline_mode)
    data_path, meta_path = _mask_result_paths(key, pipeline_mode=mode, role=role)
    mask_bool = np.asarray(mask, dtype=bool)
    params = _sanitize_mask_params(key, params or DEFAULT_MASK_PARAMS[key])
    edge_count = int(mask_bool.sum())
    possible_edges = int(mask_bool.size - min(mask_bool.shape)) if mask_bool.ndim == 2 else int(mask_bool.size)
    meta = {
        "version": MASK_CACHE_VERSION,
        "key": key,
        "label": DEFAULT_MASK_LABELS.get(key, key),
        "params": params,
        "shape": list(mask_bool.shape),
        "edge_count": edge_count,
        "density": edge_count / possible_edges if possible_edges else 0.0,
        "computed_at": time.time(),
        "data_file": _relative_cache_path(data_path),
    }
    if key in FMRI_NATIVE_SCREEN_KEYS and mode is not None:
        meta.update({
            "cache_version": FULL_FMRI_MASK_CACHE_VERSION,
            "pipeline_mode": mode,
            "role": role,
        })
    np.savez_compressed(data_path, mask=mask_bool)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if key in FMRI_NATIVE_SCREEN_KEYS and mode is not None:
        full_fmri_mask_meta[key] = meta


def _load_single_cached_mask(key: str, pipeline_mode: str | None = None) -> bool:
    mode = None
    role = None
    if key in FMRI_NATIVE_SCREEN_KEYS:
        if pipeline_mode is None:
            return False
        mode, role = _full_fmri_context_meta(key, pipeline_mode)
    data_path, meta_path = _mask_result_paths(key, pipeline_mode=mode, role=role)
    if not os.path.exists(data_path) or not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("version", 0)) != MASK_CACHE_VERSION:
            return False
        if key in FMRI_NATIVE_SCREEN_KEYS:
            if int(meta.get("cache_version", 0)) != FULL_FMRI_MASK_CACHE_VERSION:
                return False
            if meta.get("pipeline_mode") != mode:
                return False
            if meta.get("role") != role:
                return False
        meta_params = meta.get("params", {})
        cached_params = (
            tuple(tuple(item) for item in meta_params)
            if isinstance(meta_params, list)
            else _freeze_params(meta_params)
        )
        if cached_params != _freeze_params(DEFAULT_MASK_PARAMS[key]):
            return False
        with np.load(data_path, allow_pickle=False) as data:
            if "mask" not in data.files:
                return False
            mask = data["mask"].astype(bool)
        if mask.shape != (ROI_num, ROI_num):
            return False
        np.fill_diagonal(mask, False)
        _set_mask_by_key(key, mask)
        if key in FMRI_NATIVE_SCREEN_KEYS:
            full_fmri_mask_meta[key] = meta
        _mark_mask_cached(key, completed_at=os.path.getmtime(data_path))
        return True
    except Exception:
        return False


def _save_cached_masks():
    save_dict = {}
    meta = {"version": MASK_CACHE_VERSION, "defaults": DEFAULT_MASK_PARAMS, "computed_keys": []}
    for key in PRECOMPUTE_MASK_KEYS:
        if key in FMRI_NATIVE_SCREEN_KEYS:
            continue
        mask = _get_mask_by_key(key)
        if mask is not None:
            mask_bool = mask.astype(bool)
            _save_single_cached_mask(key, mask_bool)
            meta["computed_keys"].append(key)
            if key in SCREENING_KEYS:
                save_dict[key] = mask_bool
    if save_dict:
        np.savez(mask_cache_file, **save_dict)
        with open(mask_meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_cached_masks():
    loaded_keys = []
    for key in PRECOMPUTE_MASK_KEYS:
        if _load_single_cached_mask(
            key,
            pipeline_mode="none" if key in FMRI_NATIVE_SCREEN_KEYS else None,
        ):
            loaded_keys.append(key)

    if not os.path.exists(mask_cache_file):
        return loaded_keys

    completed_at = os.path.getmtime(mask_cache_file)
    cached_defaults = {}
    if os.path.exists(mask_meta_file):
        try:
            with open(mask_meta_file, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            if int(cached_meta.get("version", 0)) != MASK_CACHE_VERSION:
                return loaded_keys
            cached_defaults = cached_meta.get("defaults", {})
        except Exception:
            cached_defaults = {}
    with np.load(mask_cache_file, allow_pickle=True) as data:
        for key in SCREENING_KEYS:
            if key in FMRI_NATIVE_SCREEN_KEYS:
                continue
            if key in loaded_keys:
                continue
            if key in data.files and _freeze_params(cached_defaults.get(key, {})) == _freeze_params(DEFAULT_MASK_PARAMS[key]):
                _set_mask_by_key(key, data[key].astype(bool))
                loaded_keys.append(key)
                _mark_mask_cached(key, completed_at=completed_at)
    return loaded_keys


def _update_precompute_status(**updates):
    causal_precompute_status.update(updates)


def _precompute_status_snapshot() -> dict:
    """
    Build a live precompute report across every algorithm that should be cached.

    This makes it obvious when the running server is only using an older list,
    when an algorithm failed, or when a cache file is still missing.
    """
    snapshot = dict(causal_precompute_status)
    pipeline_mode = _normalize_pipeline_mode(snapshot.get("pipeline_mode", "none"))
    expected_keys = list(
        snapshot.get("expected_keys") or PRECOMPUTE_MASK_KEYS
    ) if snapshot.get("is_running") else list(PRECOMPUTE_MASK_KEYS)
    computed_keys = []
    cache_files = []
    failed_results = []
    for key in expected_keys:
        if key in FMRI_NATIVE_SCREEN_KEYS:
            if not _full_fmri_memory_context_matches(key, pipeline_mode):
                _load_single_cached_mask(key, pipeline_mode=pipeline_mode)
            mask_available = _get_mask_by_key(key) is not None and _full_fmri_memory_context_matches(key, pipeline_mode)
            data_path, meta_path = _mask_result_paths(key, pipeline_mode=pipeline_mode)
        else:
            if _get_mask_by_key(key) is None:
                _load_single_cached_mask(key)
            mask_available = _get_mask_by_key(key) is not None
            data_path, meta_path = _mask_result_paths(key)
        if mask_available:
            computed_keys.append(key)
            cache_files.append({
                "key": key,
                "label": DEFAULT_MASK_LABELS.get(key, key),
                "data_file": _relative_cache_path(data_path),
                "meta_file": _relative_cache_path(meta_path),
            })
            continue
        state = _get_screening_state(key)
        if state.get("last_error"):
            failed_results.append({
                "key": key,
                "label": DEFAULT_MASK_LABELS.get(key, key),
                "error": state.get("last_error", ""),
                "completed_at": state.get("completed_at"),
            })

    missing_keys = [key for key in expected_keys if key not in computed_keys]
    now = time.time()
    started_at = snapshot.get("started_at")
    current_started_at = snapshot.get("current_started_at")
    if started_at:
        snapshot["elapsed_seconds"] = int(max(0, now - float(started_at)))
    else:
        snapshot["elapsed_seconds"] = None
    if current_started_at:
        snapshot["current_elapsed_seconds"] = int(max(0, now - float(current_started_at)))
    else:
        snapshot["current_elapsed_seconds"] = None
    snapshot.update({
        "total": len(expected_keys),
        "expected_keys": expected_keys,
        "expected_count": len(expected_keys),
        "local_only_keys": list(LOCAL_ONLY_PRECOMPUTE_KEYS),
        "local_only_count": len(LOCAL_ONLY_PRECOMPUTE_KEYS),
        "local_only_labels": [
            DEFAULT_MASK_LABELS.get(key, key)
            for key in LOCAL_ONLY_PRECOMPUTE_KEYS
        ],
        "computed_keys": computed_keys,
        "computed_count": len(computed_keys),
        "missing_keys": missing_keys,
        "missing_count": len(missing_keys),
        "failed_results": failed_results,
        "failed_count": len(failed_results),
        "cache_files": cache_files,
    })
    return snapshot


def _run_default_mask_compute(key: str, signals: list, progress_callback=None) -> np.ndarray:
    return _compute_mask_with_params(
        key,
        signals,
        params=DEFAULT_MASK_PARAMS[key],
        progress_callback=progress_callback,
    )


def _sanitize_mask_params(key: str, params: dict) -> dict:
    cleaned = dict(params)

    def _as_bool(value, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _clamp_int(name: str, low: int, high: int):
        if name in cleaned and cleaned[name] is not None:
            cleaned[name] = max(low, min(high, int(cleaned[name])))

    def _clamp_float(name: str, low: float, high: float):
        if name in cleaned and cleaned[name] is not None:
            cleaned[name] = max(low, min(high, float(cleaned[name])))

    if key == "granger":
        _clamp_int("max_lag", 1, 10)
        _clamp_float("alpha", 1e-4, 0.20)
        _clamp_float("min_effect", 0.0, 1.0)
        if cleaned.get("max_density") is not None:
            cleaned["max_density"] = max(0.001, min(1.0, float(cleaned["max_density"])))
        cleaned["full_tigramite"] = _as_bool(cleaned.get("full_tigramite"), False)
        if cleaned.get("max_candidates_per_target") is not None:
            cleaned["max_candidates_per_target"] = max(1, min(ROI_num - 1, int(cleaned["max_candidates_per_target"])))
    elif key == "vargranger":
        _clamp_int("max_lag", 1, 12)
        _clamp_float("alpha", 1e-4, 0.20)
        _clamp_float("min_effect", 0.0, 1.0)
        if cleaned.get("max_density") is not None:
            cleaned["max_density"] = max(0.001, min(1.0, float(cleaned["max_density"])))
    elif key == "pcmci":
        _clamp_int("max_lag", 1, 5)
        _clamp_float("alpha", 1e-4, 0.20)
        _clamp_float("min_effect", 0.0, 1.0)
        cleaned["full_tigramite"] = _as_bool(cleaned.get("full_tigramite"), False)
        if cleaned.get("max_density") is not None:
            cleaned["max_density"] = max(0.001, min(1.0, float(cleaned["max_density"])))
    elif key == "sparsevar":
        _clamp_int("max_lag", 1, 4)
        _clamp_int("max_candidates_per_target", 1, ROI_num - 1)
        _clamp_float("min_effect", 0.0, 10.0)
        if cleaned.get("max_density") is not None:
            cleaned["max_density"] = max(0.001, min(1.0, float(cleaned["max_density"])))
        if cleaned.get("ridge") is not None:
            cleaned["ridge"] = max(1e-6, float(cleaned["ridge"]))
        _clamp_float("min_session_support_ratio", 0.0, 1.0)
    elif key == "fask":
        _clamp_float("two_cycle_alpha", 1e-6, 0.20)
        _clamp_float("penalty_discount", 0.1, 20.0)
        if cleaned.get("max_density") is not None:
            cleaned["max_density"] = max(0.001, min(1.0, float(cleaned["max_density"])))
        _clamp_int("timeout_seconds", 30, 7200)
    elif key == "twostep":
        if cleaned.get("lambda") is not None:
            cleaned["lambda"] = max(1e-6, float(cleaned["lambda"]))
        _clamp_float("min_effect", 0.0, 10.0)
        if cleaned.get("max_density") is not None:
            cleaned["max_density"] = max(0.001, min(1.0, float(cleaned["max_density"])))
        _clamp_int("timeout_seconds", 30, 7200)
    else:
        raise KeyError(key)

    return cleaned




def _compute_mask_with_params(key: str, signals: list, params: dict | None = None, progress_callback=None) -> np.ndarray:
    """Compute one fMRI-suitable causal mask with bounded parameters."""
    merged = dict(DEFAULT_MASK_PARAMS[key])
    if params:
        for param_key, value in params.items():
            if param_key in merged and value is not None:
                merged[param_key] = value
    merged = _sanitize_mask_params(key, merged)

    if key == "granger":
        return compute_granger_mask(
            signals,
            roi_num=ROI_num,
            max_lag=int(merged["max_lag"]),
            alpha=float(merged["alpha"]),
            min_effect=float(merged["min_effect"]),
            max_density=float(merged["max_density"]) if merged["max_density"] is not None else None,
            max_candidates_per_target=(
                None if merged["max_candidates_per_target"] is None
                else int(merged["max_candidates_per_target"])
            ),
            progress_callback=progress_callback,
        )
    if key == "vargranger":
        return compute_var_granger_mask(
            signals,
            roi_num=ROI_num,
            max_lag=int(merged["max_lag"]),
            alpha=float(merged["alpha"]),
            min_effect=float(merged["min_effect"]),
            max_density=float(merged["max_density"]) if merged["max_density"] is not None else None,
        )
    if key == "pcmci":
        return compute_pcmci_mask(
            signals,
            roi_num=ROI_num,
            max_lag=int(merged["max_lag"]),
            alpha=float(merged["alpha"]),
            min_effect=float(merged["min_effect"]),
            max_density=float(merged["max_density"]) if merged["max_density"] is not None else None,
            full_tigramite=bool(merged.get("full_tigramite", False)),
        )
    if key == "sparsevar":
        return compute_sparse_var_mask(
            signals,
            roi_num=ROI_num,
            max_lag=int(merged["max_lag"]),
            max_candidates_per_target=int(merged["max_candidates_per_target"]),
            min_effect=float(merged["min_effect"]),
            max_density=float(merged["max_density"]) if merged["max_density"] is not None else None,
            ridge=float(merged["ridge"]),
            min_session_support_ratio=float(merged["min_session_support_ratio"]),
        )
    if key == "fask":
        return compute_fask_mask(
            signals,
            roi_num=ROI_num,
            two_cycle_alpha=float(merged["two_cycle_alpha"]),
            penalty_discount=float(merged["penalty_discount"]),
            max_density=float(merged["max_density"]) if merged["max_density"] is not None else None,
            timeout_seconds=int(merged["timeout_seconds"]),
        )
    if key == "twostep":
        return compute_two_step_mask(
            signals,
            roi_num=ROI_num,
            lambda_value=float(merged["lambda"]),
            min_effect=float(merged["min_effect"]),
            max_density=float(merged["max_density"]) if merged["max_density"] is not None else None,
            timeout_seconds=int(merged["timeout_seconds"]),
            progress_callback=progress_callback,
        )
    raise KeyError(key)




def _prepare_pre_npi_screen(
    mode_key: str | None,
    signals: list,
    params: dict | None = None,
    progress_callback=None,
) -> tuple[np.ndarray | None, dict]:
    """
    Build the pre-NPI causal-discovery screen.

    Returns (mask, meta). mask=None means pure NPI, no pre-screen. Consensus and
    Ensemble modes both compute all four temporal screens first; Ensemble cannot
    be a strict pre-NPI estimator because it depends on NPI intervention scores.
    """
    mode_key = _normalize_mask_key(mode_key)
    if mode_key is None:
        return None, {
            "mode": "none",
            "mode_label": COMPUTE_MODE_LABELS["none"],
            "screen_first": False,
            "sources": [],
        }

    if mode_key in SCREENING_KEYS:
        mask = _compute_mask_with_params(
            mode_key,
            signals,
            params=params,
            progress_callback=progress_callback,
        )
        _set_mask_by_key(mode_key, mask)
        cache_params = dict(DEFAULT_MASK_PARAMS[mode_key])
        if isinstance(params, dict):
            for param_key, value in params.items():
                if param_key in cache_params and value is not None:
                    cache_params[param_key] = value
        cache_params = _sanitize_mask_params(mode_key, cache_params)
        if _is_default_params(mode_key, cache_params):
            _mark_mask_cached(mode_key)
            _save_single_cached_mask(
                mode_key,
                mask,
                cache_params,
                pipeline_mode=mode_key if mode_key in FMRI_NATIVE_SCREEN_KEYS else None,
            )
        return mask, {
            "mode": mode_key,
            "mode_label": COMPUTE_MODE_LABELS[mode_key],
            "screen_first": True,
            "sources": [mode_key],
        }

    if mode_key in {"consensus", "ensemble"}:
        for source_key in CONSENSUS_SOURCE_KEYS:
            source_params = params.get(source_key) if isinstance(params, dict) else None
            mask = _compute_mask_with_params(
                source_key,
                signals,
                params=source_params,
                progress_callback=progress_callback,
            )
            _set_mask_by_key(source_key, mask)
        consensus_mask, consensus_meta = _compute_consensus_mask()
        return consensus_mask, {
            "mode": mode_key,
            "mode_label": COMPUTE_MODE_LABELS[mode_key],
            "screen_first": True,
            "sources": consensus_meta["sources"],
            "pre_screen": "consensus",
        }

    raise ValueError(f"不支持的计算模式: {mode_key}")


def precompute_default_causal_masks_async(
    force: bool = False,
    keys: list | None = None,
    pipeline_mode: str = "none",
):
    target_keys = _resolve_precompute_keys(keys=keys)
    pipeline_mode = _normalize_pipeline_mode(pipeline_mode)
    with causal_precompute_lock:
        if causal_precompute_status["is_running"]:
            return False
        causal_precompute_cancel_event.clear()
        _update_precompute_status(
            is_running=True,
            progress=0,
            total=len(target_keys),
            expected_keys=list(target_keys),
            current_key="",
            current_label="",
            current_started_at=None,
            message="正在加载fMRI数据...",
            last_error="",
            started_at=time.time(),
            completed_at=None,
            last_heartbeat_at=time.time(),
            computed_keys=_collect_cached_keys(target_keys, pipeline_mode=pipeline_mode),
            pipeline_mode=pipeline_mode,
        )

    def _worker():
        _update_precompute_status(
            is_running=True,
            progress=0,
            total=len(target_keys),
            expected_keys=list(target_keys),
            current_key="",
            current_label="",
            current_started_at=None,
            message="正在加载fMRI数据...",
            last_error="",
            started_at=time.time(),
            completed_at=None,
            last_heartbeat_at=time.time(),
            computed_keys=_collect_cached_keys(target_keys, pipeline_mode=pipeline_mode),
            pipeline_mode=pipeline_mode,
        )
        try:
            signals = _load_signals()
            if not signals:
                raise RuntimeError("未找到fMRI原始数据文件")

            progress = 0
            errors = []
            for key in target_keys:
                if causal_precompute_cancel_event.is_set():
                    raise RuntimeError("用户已取消因果算法预计算")
                progress += 1
                label = DEFAULT_MASK_LABELS[key]
                current_started_at = time.time()
                context_mode = pipeline_mode if key in FMRI_NATIVE_SCREEN_KEYS else None
                cached_available = (
                    (
                        _get_mask_by_key(key) is not None
                        and _full_fmri_memory_context_matches(key, pipeline_mode)
                    )
                    or _load_single_cached_mask(key, pipeline_mode=context_mode)
                ) if key in FMRI_NATIVE_SCREEN_KEYS else (
                    _get_mask_by_key(key) is not None or _load_single_cached_mask(key)
                )
                if (not force) and cached_available:
                    _update_precompute_status(
                        progress=progress,
                        current_key=key,
                        current_label=label,
                        current_started_at=None,
                        message=f"跳过已缓存结果：{label}",
                        last_heartbeat_at=time.time(),
                        computed_keys=_collect_cached_keys(target_keys, pipeline_mode=pipeline_mode),
                    )
                    continue

                _update_screening_state(
                    key,
                    is_running=True,
                    message=f"{label}默认结果预计算中...",
                    last_error="",
                    params=_freeze_params(DEFAULT_MASK_PARAMS[key]),
                    started_at=time.time(),
                    completed_at=None,
                )
                try:
                    _update_precompute_status(
                        progress=progress,
                        current_key=key,
                        current_label=label,
                        current_started_at=current_started_at,
                        message=f"正在预计算：{label}",
                        last_heartbeat_at=time.time(),
                        computed_keys=_collect_cached_keys(target_keys, pipeline_mode=pipeline_mode),
                    )

                    def _algo_progress(**kwargs):
                        elapsed_seconds = int(kwargs.get("elapsed_seconds", 0))
                        stage = kwargs.get("stage", "running")
                        detail = kwargs.get("detail", "")
                        message = f"正在预计算：{label} · {stage} · 已耗时 {elapsed_seconds}s"
                        if detail:
                            message += f" · {detail}"
                        _update_precompute_status(
                            progress=progress,
                            current_key=key,
                            current_label=label,
                            current_started_at=current_started_at,
                            message=message,
                            last_heartbeat_at=time.time(),
                            computed_keys=_collect_cached_keys(target_keys, pipeline_mode=pipeline_mode),
                        )

                    mask = _run_default_mask_compute(
                        key,
                        signals,
                        progress_callback=_algo_progress,
                    )
                    _set_mask_by_key(key, mask)
                    _mark_mask_cached(key)
                    _save_single_cached_mask(key, mask, pipeline_mode=context_mode)
                    if key in SCREENING_KEYS:
                        _save_cached_masks()
                    _update_precompute_status(
                        progress=progress,
                        current_key=key,
                        current_label=label,
                        current_started_at=None,
                        message=f"已完成：{label}",
                        last_heartbeat_at=time.time(),
                        computed_keys=_collect_cached_keys(target_keys, pipeline_mode=pipeline_mode),
                    )
                except Exception as algo_error:
                    err_text = str(algo_error)
                    errors.append(f"{label}: {err_text}")
                    _update_screening_state(
                        key,
                        is_running=False,
                        message=f"{label}预计算失败: {err_text}",
                        last_error=err_text,
                        params=_freeze_params(DEFAULT_MASK_PARAMS[key]),
                        completed_at=time.time(),
                    )
                    _update_precompute_status(
                        progress=progress,
                        current_key=key,
                        current_label=label,
                        current_started_at=None,
                        message=f"{label}预计算失败，继续下一个算法",
                        last_error=" | ".join(errors),
                        last_heartbeat_at=time.time(),
                        computed_keys=_collect_cached_keys(pipeline_mode=pipeline_mode),
                    )

            if errors:
                _update_precompute_status(
                    is_running=False,
                    current_key="",
                    current_label="",
                    current_started_at=None,
                    message=f"预计算完成，但有 {len(errors)} 个算法失败",
                    last_error=" | ".join(errors),
                    completed_at=time.time(),
                    last_heartbeat_at=time.time(),
                    computed_keys=_collect_cached_keys(target_keys, pipeline_mode=pipeline_mode),
                )
            else:
                _update_precompute_status(
                    is_running=False,
                    current_key="",
                    current_label="",
                    current_started_at=None,
                    message="所选因果算法默认结果已缓存",
                    last_error="",
                    completed_at=time.time(),
                    last_heartbeat_at=time.time(),
                    computed_keys=_collect_cached_keys(target_keys, pipeline_mode=pipeline_mode),
                )
        except Exception as e:
            _update_precompute_status(
                is_running=False,
                message=f"预计算失败: {str(e)}",
                last_error=str(e),
                completed_at=time.time(),
                last_heartbeat_at=time.time(),
                computed_keys=_collect_cached_keys(target_keys, pipeline_mode=pipeline_mode),
            )

    threading.Thread(target=_worker, daemon=True).start()
    return True


def _screening_response_from_mask(key: str, mask: np.ndarray) -> dict:
    data = _mask_summary(mask)
    state = _get_screening_state(key)
    elapsed_seconds = None
    if state["started_at"] and state["completed_at"]:
        elapsed_seconds = max(0.0, float(state["completed_at"]) - float(state["started_at"]))
    data.update({
        "running": False,
        "message": state["message"] or f"{DEFAULT_MASK_LABELS[key]}默认结果已缓存",
        "last_error": state["last_error"],
        "started_at": state["started_at"],
        "completed_at": state["completed_at"],
        "elapsed_seconds": elapsed_seconds,
        "params": _sanitize_mask_params(key, DEFAULT_MASK_PARAMS[key]),
    })
    return data


def _cached_mask_response(key: str, params: dict, pipeline_mode: str | None = None):
    context_mode = _normalize_pipeline_mode(pipeline_mode) if key in FMRI_NATIVE_SCREEN_KEYS else None
    if _is_default_params(key, params):
        if key in FMRI_NATIVE_SCREEN_KEYS:
            if not _full_fmri_memory_context_matches(key, context_mode or "none"):
                _load_single_cached_mask(key, pipeline_mode=context_mode or "none")
        elif _get_mask_by_key(key) is None:
            _load_single_cached_mask(key)
    mask = _get_mask_by_key(key)
    context_ok = (
        _full_fmri_memory_context_matches(key, context_mode or "none")
        if key in FMRI_NATIVE_SCREEN_KEYS
        else True
    )
    if mask is not None and context_ok and _is_default_params(key, params):
        data = _screening_response_from_mask(key, mask)
        data["cached"] = True
        if key in FMRI_NATIVE_SCREEN_KEYS:
            data["meta"] = full_fmri_mask_meta.get(key, {})
        return jsonify(data)

    if causal_precompute_status["is_running"]:
        return jsonify({
            "available": False,
            "running": True,
            "cached": False,
            "message": causal_precompute_status["message"] or "默认结果预计算中...",
            "started_at": causal_precompute_status["started_at"],
            "elapsed_seconds": int(max(0, time.time() - float(causal_precompute_status["started_at"])))
            if causal_precompute_status["started_at"] else None,
            "progress": causal_precompute_status["progress"],
            "total": causal_precompute_status["total"],
            "current_key": causal_precompute_status["current_key"],
            "current_label": causal_precompute_status["current_label"],
        })

    if not _is_default_params(key, params):
        return jsonify({
            "available": False,
            "running": False,
            "cached": False,
            "message": "当前仅支持显示预计算默认参数结果，请先恢复默认参数后查看。",
        })

    state = _get_screening_state(key)
    if state["last_error"]:
        return jsonify({
            "available": False,
            "running": False,
            "cached": False,
            "message": state["message"] or f"{DEFAULT_MASK_LABELS[key]}默认结果预计算失败",
            "last_error": state["last_error"],
            "completed_at": state["completed_at"],
        })

    if key in LOCAL_ONLY_PRECOMPUTE_KEYS:
        return jsonify({
            "available": False,
            "running": False,
            "cached": False,
            "message": f"{DEFAULT_MASK_LABELS[key]} 已接入，但不参与默认全脑预计算；原版 MATLAB 算法在 360 ROI 上耗时很长，建议用于局部/小规模网络或通过 Python SDK 单独调用。",
        })

    return jsonify({
        "available": False,
        "running": False,
        "cached": False,
        "message": "该算法默认结果尚未预计算，请先点击对应分组预计算或“预计算全部算法”。",
    })


def _normalize_mask_key(mask_key: str | None) -> str | None:
    if mask_key is None:
        return None
    key = str(mask_key).strip().lower()
    if key in ("", "none", "npi", "raw"):
        return None
    if (
        key not in SCREENING_KEYS
        and key not in EXPERIMENTAL_MASK_KEYS
        and key not in DERIVED_MASK_KEYS
        and key not in LOCAL_FMRI_MASK_KEYS
    ):
        raise ValueError(f"未知因果算法: {mask_key}")
    return key


def _mask_info(mask_key: str | None) -> dict:
    if mask_key is None:
        return {
            "key": "none",
            "label": "NPI原始结果",
            "active": False,
            "available": True,
            "density": 1.0,
        }

    mask = _get_mask_by_key(mask_key)
    derived_missing = (
        (mask_key == "consensus" and len(_available_consensus_sources()) < len(CONSENSUS_SOURCE_KEYS))
        or (mask_key == "ensemble" and not _ensemble_ready())
    )
    if mask is None or derived_missing:
        raise ValueError(f"{DEFAULT_MASK_LABELS[mask_key]}默认结果尚未预计算")

    info = {
        "key": mask_key,
        "label": DEFAULT_MASK_LABELS[mask_key],
        "active": True,
        "available": True,
        "density": round(float(mask.sum()) / (ROI_num * (ROI_num - 1)), 4),
    }
    if mask_key == "consensus":
        _, meta = _compute_consensus_mask()
        info["sources"] = meta["sources"]
        info["min_score"] = meta["min_score"]
        info["min_support"] = meta["min_support"]
    elif mask_key == "ensemble":
        _, meta = _compute_intervention_ensemble_mask()
        info["sources"] = meta["sources"]
        info["weights"] = meta["weights"]
        info["min_score"] = meta["min_score"]
        info["min_npi_score"] = meta["min_npi_score"]
        info["min_support"] = meta["min_support"]
        info["max_density"] = meta["max_density"]
    elif mask_key in LOCAL_FMRI_MASK_KEYS:
        info.update(local_fmri_mask_meta.get(mask_key, {}))
    return info


def _normalize_pipeline_mode(mode: str | None) -> str:
    mode_key = str(mode or "none").strip().lower()
    if mode_key not in {"none", "fask", "twostep"}:
        mode_key = "none"
    return mode_key


def _local_fmri_role_for_mode(key: str, pipeline_mode: str) -> str:
    return "prefilter" if key == pipeline_mode else "ordinary"


def _local_fmri_mask_paths(
    key: str,
    strength: float,
    node_idx: int,
    max_nodes: int,
    pipeline_mode: str = "none",
) -> tuple[Path, Path]:
    pipeline_mode = _normalize_pipeline_mode(pipeline_mode)
    role = _local_fmri_role_for_mode(key, pipeline_mode)
    root = LOCAL_FMRI_MASK_RESULTS_DIR / pipeline_mode / role
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{key}_{role}_in_{pipeline_mode}_s{str(float(strength)).replace('.', 'p')}_node{node_idx + 1}_k{int(max_nodes)}"
    return (
        root / f"{stem}.npz",
        root / f"{stem}_meta.json",
    )


def _local_roi_indices_for_node(strength: float, node_idx: int, max_nodes: int = 40) -> list[int]:
    max_nodes = max(4, min(int(max_nodes), ROI_num))
    try:
        baseline_ec, _ = _load_ec_result("none")
        ec = np.asarray(baseline_ec[float(strength)])
    except Exception:
        if ec_dict and float(strength) in ec_dict:
            ec = np.asarray(ec_dict[float(strength)])
        else:
            ec = None

    if ec is not None and ec.shape == (ROI_num, ROI_num):
        scores = np.abs(ec[node_idx, :]) + np.abs(ec[:, node_idx])
        scores[node_idx] = np.inf
        selected = np.argsort(scores)[::-1][:max_nodes]
        return sorted(int(i) for i in selected)

    return list(range(min(max_nodes, ROI_num)))


def _embed_local_mask(local_mask: np.ndarray, roi_indices: list[int]) -> np.ndarray:
    full_mask = np.zeros((ROI_num, ROI_num), dtype=bool)
    local_bool = np.asarray(local_mask, dtype=bool)
    for local_i, global_i in enumerate(roi_indices):
        for local_j, global_j in enumerate(roi_indices):
            if local_i != local_j and local_bool[local_i, local_j]:
                full_mask[global_i, global_j] = True
    np.fill_diagonal(full_mask, False)
    return full_mask


def _load_local_fmri_mask_cache(
    key: str,
    strength: float,
    node_idx: int,
    max_nodes: int,
    pipeline_mode: str = "none",
) -> dict | None:
    pipeline_mode = _normalize_pipeline_mode(pipeline_mode)
    data_path, meta_path = _local_fmri_mask_paths(key, strength, node_idx, max_nodes, pipeline_mode=pipeline_mode)
    if not os.path.exists(data_path) or not os.path.exists(meta_path):
        return None
    with np.load(data_path, allow_pickle=False) as data:
        full_mask = data["mask"].astype(bool)
        roi_indices = data["roi_indices"].astype(int).tolist()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if int(meta.get("cache_version", 0)) != LOCAL_FMRI_MASK_CACHE_VERSION:
        return None
    if meta.get("pipeline_mode") != pipeline_mode:
        return None
    if meta.get("role") != _local_fmri_role_for_mode(key, pipeline_mode):
        return None
    _set_mask_by_key(f"{key}_local", full_mask)
    local_fmri_mask_meta[f"{key}_local"] = meta
    return {
        "mask": full_mask,
        "roi_indices": roi_indices,
        "meta": meta,
        "data_path": data_path,
        "meta_path": meta_path,
        "cached": True,
    }


def _compute_local_fmri_mask(
    key: str,
    strength: float,
    node_idx: int,
    max_nodes: int = 40,
    force: bool = False,
    pipeline_mode: str = "none",
) -> dict:
    if key not in FMRI_NATIVE_SCREEN_KEYS:
        raise ValueError("局部筛选只支持 FASK 和 Two-Step")
    if node_idx < 0 or node_idx >= ROI_num:
        raise ValueError("ROI 编号超出范围")
    max_nodes = max(4, min(int(max_nodes), ROI_num))
    pipeline_mode = _normalize_pipeline_mode(pipeline_mode)
    role = _local_fmri_role_for_mode(key, pipeline_mode)

    if not force:
        cached = _load_local_fmri_mask_cache(key, strength, node_idx, max_nodes, pipeline_mode=pipeline_mode)
        if cached is not None:
            return cached

    signals = _load_signals()
    if not signals:
        raise RuntimeError("未找到fMRI原始数据文件")
    roi_indices = _local_roi_indices_for_node(strength, node_idx, max_nodes=max_nodes)
    local_signals = [np.asarray(sig)[:, roi_indices] for sig in signals]
    params = dict(DEFAULT_MASK_PARAMS[key])

    started_at = time.time()
    if key == "fask":
        local_mask = compute_fask_mask(
            local_signals,
            roi_num=len(roi_indices),
            two_cycle_alpha=float(params["two_cycle_alpha"]),
            penalty_discount=float(params["penalty_discount"]),
            max_density=float(params["max_density"]) if params["max_density"] is not None else None,
            timeout_seconds=int(params["timeout_seconds"]),
        )
    else:
        local_mask = compute_two_step_mask(
            local_signals,
            roi_num=len(roi_indices),
            lambda_value=float(params["lambda"]),
            min_effect=float(params["min_effect"]),
            max_density=float(params["max_density"]) if params["max_density"] is not None else None,
            timeout_seconds=int(params["timeout_seconds"]),
        )
    full_mask = _embed_local_mask(local_mask, roi_indices)
    completed_at = time.time()
    data_path, meta_path = _local_fmri_mask_paths(key, strength, node_idx, max_nodes, pipeline_mode=pipeline_mode)
    meta = {
        "cache_version": LOCAL_FMRI_MASK_CACHE_VERSION,
        "key": f"{key}_local",
        "source_algorithm": key,
        "pipeline_mode": pipeline_mode,
        "role": role,
        "label": DEFAULT_MASK_LABELS[f"{key}_local"],
        "strength": float(strength),
        "node_idx": int(node_idx),
        "node_label": f"ROI {node_idx + 1}",
        "max_nodes": int(max_nodes),
        "roi_indices": roi_indices,
        "roi_labels": [f"ROI {i + 1}" for i in roi_indices],
        "edge_count": int(full_mask.sum()),
        "density": round(float(full_mask.sum()) / (ROI_num * (ROI_num - 1)), 6),
        "local_edge_count": int(np.asarray(local_mask, dtype=bool).sum()),
        "local_density": round(float(np.asarray(local_mask, dtype=bool).sum()) / (len(roi_indices) * max(len(roi_indices) - 1, 1)), 6),
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": round(completed_at - started_at, 2),
        "data_file": _relative_cache_path(data_path),
        "meta_file": _relative_cache_path(meta_path),
    }
    np.savez_compressed(
        data_path,
        mask=full_mask,
        local_mask=np.asarray(local_mask, dtype=bool),
        roi_indices=np.asarray(roi_indices, dtype=int),
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _set_mask_by_key(f"{key}_local", full_mask)
    local_fmri_mask_meta[f"{key}_local"] = meta
    return {
        "mask": full_mask,
        "roi_indices": roi_indices,
        "meta": meta,
        "data_path": data_path,
        "meta_path": meta_path,
        "cached": False,
    }


def _apply_mask_to_matrix(ec_matrix: np.ndarray, mask_key: str | None) -> tuple[np.ndarray, dict]:
    info = _mask_info(mask_key)
    if mask_key is None:
        return ec_matrix.copy(), info
    return ec_matrix * _get_mask_by_key(mask_key).astype(ec_matrix.dtype), info


def _edge_count(vec: np.ndarray, exclude_idx: int | None = None) -> int:
    tmp = np.asarray(vec)
    count = int(np.count_nonzero(np.abs(tmp) >= 1e-6))
    if exclude_idx is not None and abs(float(tmp[exclude_idx])) >= 1e-6:
        count -= 1
    return max(count, 0)


def _reduction_stats(original_count: int, filtered_count: int) -> dict:
    removed = max(int(original_count) - int(filtered_count), 0)
    return {
        "original_edges": int(original_count),
        "filtered_edges": int(filtered_count),
        "removed_edges": removed,
        "removed_pct": round(removed / original_count, 4) if original_count else 0.0,
    }


def _mode_reduction_against_npi(mode: str | None = None, threshold: float = NPI_RELATION_THRESHOLD) -> dict:
    mode = _normalize_ec_result_mode(mode or (_load_ec_meta()["mode"] if ec_dict else "none"))
    if mode == "none":
        current_counts = {
            str(s): _count_npi_perturbation_relations(mat, threshold)
            for s, mat in sorted(ec_dict.items())
        }
        total = int(sum(current_counts.values()))
        return {
            "available": True,
            "count_type": "directed_npi_perturbation_relations",
            "threshold": float(threshold),
            "mode": "none",
            "baseline_mode": "none",
            "baseline_edges": total,
            "mode_edges": total,
            "removed_edges": 0,
            "removed_pct": 0.0,
            "per_strength": {
                s: {"baseline_edges": c, "mode_edges": c, "removed_edges": 0, "removed_pct": 0.0}
                for s, c in current_counts.items()
            },
        }
    try:
        baseline, _ = _load_ec_result("none")
    except FileNotFoundError:
        return {
            "available": False,
            "mode": mode,
            "baseline_mode": "none",
            "message": "纯 NPI 结果尚未计算，无法统计相对筛除线数。",
        }

    per_strength = {}
    baseline_total = 0
    mode_total = 0
    for strength, mat in sorted(ec_dict.items()):
        if strength not in baseline:
            continue
        base_count = _count_npi_perturbation_relations(baseline[strength], threshold)
        mode_count = _count_npi_perturbation_relations(mat, threshold)
        removed = max(base_count - mode_count, 0)
        baseline_total += base_count
        mode_total += mode_count
        per_strength[str(strength)] = {
            "baseline_edges": base_count,
            "mode_edges": mode_count,
            "removed_edges": removed,
            "removed_pct": round(removed / base_count, 4) if base_count else 0.0,
        }
    removed_total = max(baseline_total - mode_total, 0)
    return {
        "available": True,
        "count_type": "directed_npi_perturbation_relations",
        "threshold": float(threshold),
        "mode": mode,
        "baseline_mode": "none",
        "baseline_edges": baseline_total,
        "mode_edges": mode_total,
        "removed_edges": removed_total,
        "removed_pct": round(removed_total / baseline_total, 4) if baseline_total else 0.0,
        "per_strength": per_strength,
    }


def _available_consensus_sources() -> list[str]:
    return [key for key in CONSENSUS_SOURCE_KEYS if _get_mask_by_key(key) is not None]


def _compute_consensus_score_matrix() -> tuple[np.ndarray, np.ndarray, list[str]]:
    sources = _available_consensus_sources()
    score = np.zeros((ROI_num, ROI_num), dtype=np.float64)
    support = np.zeros((ROI_num, ROI_num), dtype=np.int16)
    total_weight = sum(CONSENSUS_WEIGHTS[key] for key in sources)
    if not sources or total_weight <= 0:
        return score, support, sources

    for key in sources:
        mask = _get_mask_by_key(key).astype(bool)
        score += (CONSENSUS_WEIGHTS[key] / total_weight) * mask
        support += mask.astype(np.int16)
    np.fill_diagonal(score, 0.0)
    np.fill_diagonal(support, 0)
    return score, support, sources


def _compute_consensus_mask(min_score: float = 0.50, min_support: int = 2) -> tuple[np.ndarray, dict]:
    score, support, sources = _compute_consensus_score_matrix()
    if len(sources) < min_support:
        return np.zeros((ROI_num, ROI_num), dtype=bool), {
            "sources": sources,
            "min_score": min_score,
            "min_support": min_support,
        }
    mask = (score >= min_score) & (support >= min_support)
    np.fill_diagonal(mask, False)
    return mask, {
        "sources": sources,
        "min_score": min_score,
        "min_support": min_support,
    }


def _ensemble_ready() -> bool:
    return bool(ec_dict) and all(_get_mask_by_key(key) is not None for key in CONSENSUS_SOURCE_KEYS)


def _compute_npi_intervention_score_matrix() -> tuple[np.ndarray, dict]:
    """Robust directed NPI prior from perturbation EC matrices across strengths."""
    global npi_intervention_score_cache, npi_intervention_cache_signature

    if not ec_dict:
        return np.zeros((ROI_num, ROI_num), dtype=np.float64), {"strengths": [], "mode": "row_normalized_abs_ec"}

    signature = tuple(
        (float(strength), tuple(np.asarray(mat).shape), id(mat))
        for strength, mat in sorted(ec_dict.items())
    )
    if (
        npi_intervention_score_cache is not None
        and npi_intervention_cache_signature == signature
    ):
        return npi_intervention_score_cache

    score_sum = np.zeros((ROI_num, ROI_num), dtype=np.float64)
    pos_count = np.zeros((ROI_num, ROI_num), dtype=np.int16)
    neg_count = np.zeros((ROI_num, ROI_num), dtype=np.int16)
    used_strengths = []

    for strength in sorted(ec_dict.keys()):
        mat = np.asarray(ec_dict[strength], dtype=np.float64)
        if mat.shape[0] < ROI_num or mat.shape[1] < ROI_num:
            continue
        mat = mat[:ROI_num, :ROI_num].copy()
        np.fill_diagonal(mat, 0.0)
        abs_mat = np.abs(mat)
        row_max = np.max(abs_mat, axis=1, keepdims=True)
        norm = np.divide(abs_mat, row_max, out=np.zeros_like(abs_mat), where=row_max > 1e-12)
        score_sum += norm
        pos_count += (mat > 1e-9).astype(np.int16)
        neg_count += (mat < -1e-9).astype(np.int16)
        used_strengths.append(float(strength))

    if not used_strengths:
        result = (np.zeros((ROI_num, ROI_num), dtype=np.float64), {"strengths": [], "mode": "row_normalized_abs_ec"})
        npi_intervention_score_cache = result
        npi_intervention_cache_signature = signature
        return result

    mean_strength = score_sum / len(used_strengths)
    signed_count = pos_count + neg_count
    sign_consistency = np.divide(
        np.maximum(pos_count, neg_count),
        signed_count,
        out=np.zeros((ROI_num, ROI_num), dtype=np.float64),
        where=signed_count > 0,
    )
    score = mean_strength * sign_consistency
    np.fill_diagonal(score, 0.0)
    result = (score, {
        "strengths": used_strengths,
        "mode": "row_normalized_abs_ec_x_sign_consistency",
    })
    npi_intervention_score_cache = result
    npi_intervention_cache_signature = signature
    return result


def _top_density_mask(score: np.ndarray, base_mask: np.ndarray, max_density: float | None) -> np.ndarray:
    mask = np.asarray(base_mask, dtype=bool).copy()
    np.fill_diagonal(mask, False)
    if max_density is None:
        return mask
    max_edges = int(max(1, round(float(max_density) * ROI_num * (ROI_num - 1))))
    if int(mask.sum()) <= max_edges:
        return mask
    candidates = np.argwhere(mask)
    order = np.argsort(score[mask])[::-1][:max_edges]
    pruned = np.zeros_like(mask, dtype=bool)
    for src, dst in candidates[order]:
        pruned[int(src), int(dst)] = True
    return pruned


def _compute_ensemble_score_matrix() -> tuple[np.ndarray, dict]:
    npi_score, npi_meta = _compute_npi_intervention_score_matrix()
    causal_score, support, sources = _compute_consensus_score_matrix()
    score = (
        ENSEMBLE_WEIGHTS["npi_intervention"] * npi_score
        + ENSEMBLE_WEIGHTS["temporal_consensus"] * causal_score
    )
    np.fill_diagonal(score, 0.0)
    return score, {
        "npi_score": npi_score,
        "causal_score": causal_score,
        "support": support,
        "sources": sources,
        "npi": npi_meta,
        "weights": ENSEMBLE_WEIGHTS,
    }


def _compute_intervention_ensemble_mask(
    min_score: float = ENSEMBLE_DEFAULTS["min_score"],
    min_npi_score: float = ENSEMBLE_DEFAULTS["min_npi_score"],
    min_support: int = ENSEMBLE_DEFAULTS["min_support"],
    max_density: float | None = ENSEMBLE_DEFAULTS["max_density"],
) -> tuple[np.ndarray, dict]:
    score, meta = _compute_ensemble_score_matrix()
    if not ec_dict or len(meta["sources"]) < len(CONSENSUS_SOURCE_KEYS):
        return np.zeros((ROI_num, ROI_num), dtype=bool), {
            **meta,
            "min_score": min_score,
            "min_npi_score": min_npi_score,
            "min_support": min_support,
            "max_density": max_density,
        }
    base = (
        (score >= float(min_score))
        & (meta["npi_score"] >= float(min_npi_score))
        & (meta["support"] >= int(min_support))
    )
    mask = _top_density_mask(score, base, max_density)
    np.fill_diagonal(mask, False)
    return mask, {
        **meta,
        "min_score": min_score,
        "min_npi_score": min_npi_score,
        "min_support": min_support,
        "max_density": max_density,
    }


def _edge_evidence(source_idx: int, target_idx: int) -> dict:
    score, support, sources = _compute_consensus_score_matrix()
    ensemble_score, ensemble_meta = _compute_ensemble_score_matrix()
    algorithms = {}
    for key in CONSENSUS_SOURCE_KEYS:
        mask = _get_mask_by_key(key)
        algorithms[key] = bool(mask[source_idx, target_idx]) if mask is not None else None
    return {
        "consensus_score": round(float(score[source_idx, target_idx]), 4),
        "npi_intervention_score": round(float(ensemble_meta["npi_score"][source_idx, target_idx]), 4),
        "ensemble_score": round(float(ensemble_score[source_idx, target_idx]), 4),
        "support_count": int(support[source_idx, target_idx]),
        "available_count": len(sources),
        "support_ratio": round(float(support[source_idx, target_idx]) / len(sources), 4) if sources else 0.0,
        "algorithms": algorithms,
    }


def _evidence_summary_for_edges(edge_pairs: list[tuple[int, int]]) -> dict:
    if not edge_pairs:
        return {
            "avg_consensus_score": 0.0,
            "avg_npi_intervention_score": 0.0,
            "avg_ensemble_score": 0.0,
            "avg_support_count": 0.0,
            "high_confidence_edges": 0,
            "available_sources": _available_consensus_sources(),
        }
    evidences = [_edge_evidence(i, j) for i, j in edge_pairs]
    return {
        "avg_consensus_score": round(float(np.mean([e["consensus_score"] for e in evidences])), 4),
        "avg_npi_intervention_score": round(float(np.mean([e["npi_intervention_score"] for e in evidences])), 4),
        "avg_ensemble_score": round(float(np.mean([e["ensemble_score"] for e in evidences])), 4),
        "avg_support_count": round(float(np.mean([e["support_count"] for e in evidences])), 2),
        "high_confidence_edges": int(sum(
            e["support_count"] >= 2 and (e["consensus_score"] >= 0.5 or e["ensemble_score"] >= 0.5)
            for e in evidences
        )),
        "available_sources": _available_consensus_sources(),
    }


def _mask_available_for_reliability(key: str) -> bool:
    if key in SCREENING_KEYS:
        return _get_mask_by_key(key) is not None
    if key == "consensus":
        return len(_available_consensus_sources()) == len(CONSENSUS_SOURCE_KEYS)
    if key == "ensemble":
        return _ensemble_ready()
    return False


def _algorithm_reliability_report() -> dict:
    """
    Proxy reliability for screening algorithms on the current NPI dataset.

    This is not supervised accuracy because the real dataset has no ground-truth
    directed graph. It estimates trustworthiness from agreement with NPI
    intervention evidence, agreement with the NPI-Ensemble reference, algorithm
    support, and sparsity control.
    """
    reference_mask, _ = _compute_intervention_ensemble_mask()
    ensemble_score, ensemble_meta = _compute_ensemble_score_matrix()
    npi_score = ensemble_meta["npi_score"]
    causal_score = ensemble_meta["causal_score"]
    support = ensemble_meta["support"]
    available_sources = ensemble_meta["sources"]
    total_pairs = ROI_num * (ROI_num - 1)
    reference_edges = int(reference_mask.sum())
    rows = []

    keys = SCREENING_KEYS + ["consensus", "ensemble"]
    for key in keys:
        if not _mask_available_for_reliability(key):
            rows.append({
                "key": key,
                "label": DEFAULT_MASK_LABELS[key],
                "available": False,
                "message": "结果尚未缓存或依赖数据不足",
            })
            continue

        mask = _get_mask_by_key(key).astype(bool)
        np.fill_diagonal(mask, False)
        edge_count = int(mask.sum())
        density = edge_count / total_pairs if total_pairs else 0.0

        if edge_count:
            overlap = int(np.logical_and(mask, reference_mask).sum())
            union = int(np.logical_or(mask, reference_mask).sum())
            avg_npi = float(np.mean(npi_score[mask]))
            avg_consensus = float(np.mean(causal_score[mask]))
            avg_ensemble = float(np.mean(ensemble_score[mask]))
            avg_support = float(np.mean(support[mask]))
            precision_proxy = overlap / edge_count
            recall_proxy = overlap / reference_edges if reference_edges else 0.0
            jaccard = overlap / union if union else 0.0
        else:
            overlap = 0
            avg_npi = avg_consensus = avg_ensemble = avg_support = 0.0
            precision_proxy = recall_proxy = jaccard = 0.0

        support_ratio = avg_support / len(available_sources) if available_sources else 0.0
        sparsity_score = 1.0 if density <= 0.12 else max(0.0, 1.0 - (density - 0.12) / 0.20)
        reliability_score = (
            0.30 * avg_npi
            + 0.25 * precision_proxy
            + 0.20 * jaccard
            + 0.15 * support_ratio
            + 0.10 * sparsity_score
        )

        rows.append({
            "key": key,
            "label": DEFAULT_MASK_LABELS[key],
            "available": True,
            "edge_count": edge_count,
            "density": round(float(density), 4),
            "avg_npi_intervention_score": round(avg_npi, 4),
            "avg_consensus_score": round(avg_consensus, 4),
            "avg_ensemble_score": round(avg_ensemble, 4),
            "avg_support_count": round(avg_support, 2),
            "support_ratio": round(float(support_ratio), 4),
            "overlap_with_ensemble": overlap,
            "precision_proxy": round(float(precision_proxy), 4),
            "recall_proxy": round(float(recall_proxy), 4),
            "jaccard_with_ensemble": round(float(jaccard), 4),
            "sparsity_score": round(float(sparsity_score), 4),
            "reliability_score": round(float(min(max(reliability_score, 0.0), 1.0)), 4),
        })

    rows.sort(key=lambda row: row.get("reliability_score", -1), reverse=True)
    return {
        "available": bool(ec_dict) and len(available_sources) == len(CONSENSUS_SOURCE_KEYS),
        "reference": "NPI-Ensemble",
        "reference_edges": reference_edges,
        "sources": available_sources,
        "metrics_note": "无真实因果标签时的代理可信度：综合NPI干预一致性、与NPI-Ensemble重叠、算法支持度和稀疏性；不可解释为绝对准确率。",
        "items": rows,
    }


# ══════════════════════════════════════════════════════════════════
#  因果算法科学性速览
# ══════════════════════════════════════════════════════════════════
#
# 约定：所有算法输出 bool mask，mask[source, target] = True 表示 source 在
#       当前算法假设下对 target 有方向性证据。这里的“因果”主要是统计/模型
#       意义上的 directed evidence，不能单独等价为真实神经刺激因果。
#
# 主流程证据源：
#   1. Granger: 完整两两滞后预测 F 检验。方向清楚；不控制第三变量/潜混杂，耗时较长。
#   2. VAR-Granger: 多变量 VAR block 检验。可控制其他 ROI；360 ROI 下受自由度限制。
#   3. PCMCI-Lite: 滞后候选 + 条件独立近似。适合时间序列；为交互速度做了候选截断。
#   4. Sparse VAR-Lite: 候选限制 ridge VAR 系数筛选。偏工程筛选，无严格 p 值解释。
#   5. FASK: Tetrad 非高斯方向判别，可表达反馈结构；通过 Java 后端运行。
#   6. Two-Step: Adaptive Lasso + ICA 循环系统估计；通过 MATLAB 原始实现运行。
#
# 汇总层：
#   Consensus 只做多算法加权一致性，不是新的因果估计器。
#   NPI-Ensemble 融合 NPI 虚拟干预证据和时间序列筛选，是当前推荐的保守可视化掩码。
#


# ══════════════════════════════════════════════════════════════════
#  Shared primitives
# ══════════════════════════════════════════════════════════════════

def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
    """OLS residual sum of squares, robust to rank-deficient X."""
    coef, res, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if len(res) > 0:
        return float(res[0])
    return float(np.sum((y - X @ coef) ** 2))


def _zscore(sig: np.ndarray) -> np.ndarray:
    return (sig - sig.mean(0)) / (sig.std(0) + 1e-8)


def _validate_signals(signals_list: list, roi_num: int, min_rows: int = 3) -> list[np.ndarray]:
    if not signals_list:
        raise ValueError("至少需要 1 个 fMRI session 才能运行因果发现算法")

    valid = []
    for idx, sig in enumerate(signals_list, start=1):
        arr = np.asarray(sig, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"第 {idx} 个 session 必须是二维矩阵，当前形状为 {arr.shape}")
        if arr.shape[1] < roi_num:
            raise ValueError(
                f"第 {idx} 个 session 只有 {arr.shape[1]} 个 ROI，少于请求的 {roi_num} 个"
            )
        if arr.shape[0] < min_rows:
            raise ValueError(
                f"第 {idx} 个 session 只有 {arr.shape[0]} 个时间点，少于算法需要的 {min_rows} 个"
            )
        if not np.isfinite(arr[:, :roi_num]).all():
            raise ValueError(f"第 {idx} 个 session 包含 NaN 或无穷值")
        valid.append(arr[:, :roi_num])
    return valid


def _fdr_bh_mask(p_values: np.ndarray, alpha: float, exclude_diagonal: bool = True) -> np.ndarray:
    """Benjamini-Hochberg FDR correction over directed ROI pairs."""
    p = np.asarray(p_values, dtype=np.float64)
    candidate = np.isfinite(p)
    if exclude_diagonal and p.ndim == 2 and p.shape[0] == p.shape[1]:
        candidate &= ~np.eye(p.shape[0], dtype=bool)

    flat = p[candidate]
    mask = np.zeros(p.shape, dtype=bool)
    if flat.size == 0:
        return mask

    order = np.argsort(flat)
    ranked = flat[order]
    thresholds = alpha * (np.arange(1, flat.size + 1) / flat.size)
    passed = ranked <= thresholds
    if not np.any(passed):
        return mask

    cutoff = ranked[np.where(passed)[0][-1]]
    mask[candidate] = p[candidate] <= cutoff
    if exclude_diagonal and mask.ndim == 2 and mask.shape[0] == mask.shape[1]:
        np.fill_diagonal(mask, False)
    return mask


def _cap_conditioning_neighbors(
    neighbors: np.ndarray,
    corr_abs: np.ndarray,
    i: int,
    j: int,
    max_neighbors: int,
) -> np.ndarray:
    if len(neighbors) <= max_neighbors:
        return neighbors
    scores = np.maximum(corr_abs[i, neighbors], corr_abs[j, neighbors])
    return neighbors[np.argsort(scores)[-max_neighbors:]]


def _lag_matrix(series: np.ndarray, lags: int) -> np.ndarray:
    """Build (T-lags, lags) Toeplitz design matrix of lagged values."""
    T = len(series)
    return np.column_stack([series[lags - k: T - k] for k in range(1, lags + 1)])


def _fisher_combined_pvalues(log_p_sum: np.ndarray, n_sessions: int) -> np.ndarray:
    """
    Fisher's combined probability: chi2 = -2 * sum(log p_k) ~ chi2(2*K).
    """
    if n_sessions <= 0:
        return np.ones_like(log_p_sum, dtype=np.float64)
    chi2 = -2.0 * log_p_sum
    return chi2_dist.sf(chi2, df=2 * n_sessions)


def _fisher_combine(log_p_sum: np.ndarray, n_sessions: int,
                    alpha: float, fdr: bool = True) -> np.ndarray:
    combined_p = _fisher_combined_pvalues(log_p_sum, n_sessions)
    if fdr:
        return _fdr_bh_mask(combined_p, alpha)
    mask = combined_p < alpha
    np.fill_diagonal(mask, False)
    return mask


def _effect_screen_mask(
    p_mask: np.ndarray,
    effect: np.ndarray,
    min_effect: float = 0.0,
    max_density: float | None = None,
) -> np.ndarray:
    """
    Keep statistically significant edges only when their effect size is non-trivial.

    p-values alone can mark tiny fMRI effects significant when many time points are
    pooled, so the default web masks also require a partial-R2/log-ratio effect and
    optionally cap density for readable causal skeletons.
    """
    mask = np.asarray(p_mask, dtype=bool) & np.isfinite(effect) & (effect >= float(min_effect))
    if mask.ndim == 2 and mask.shape[0] == mask.shape[1]:
        np.fill_diagonal(mask, False)

    if max_density is None:
        return mask

    n_rois = mask.shape[0] if mask.ndim == 2 else ROI_num
    n_possible = n_rois * (n_rois - 1)
    max_edges = int(round(float(max_density) * n_possible))
    max_edges = max(0, min(max_edges, n_possible))
    if max_edges <= 0 or int(mask.sum()) <= max_edges:
        return mask

    keep = np.zeros_like(mask, dtype=bool)
    candidate_idx = np.argwhere(mask)
    scores = effect[mask]
    top = np.argsort(scores)[-max_edges:]
    selected = candidate_idx[top]
    keep[selected[:, 0], selected[:, 1]] = True
    return keep


def _combined_standardized_signals(signals_list: list, roi_num: int, min_rows: int = 3) -> np.ndarray:
    valid = _validate_signals(signals_list, roi_num, min_rows=min_rows)
    return np.vstack([_zscore(sig) for sig in valid])


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        try:
            if path and path.exists():
                return path
        except Exception:
            continue
    return None


def _resolve_fask_tetrad_jar() -> Path:
    env_path = os.getenv("NPI_FASK_TETRAD_JAR")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        BASE_DIR / "external_algorithms" / "fask" / "tetrad-lib-7.6.11-SNAPSHOT-shaded.jar",
        Path(r"E:\因果算法\SanchezRomero2019_FASK_TwoStep_Downloads\tetrad\tetrad-lib\target\tetrad-lib-7.6.11-SNAPSHOT-shaded.jar"),
        Path(r"E:\SanchezRomero2019_FASK_TwoStep_Downloads\tetrad\tetrad-lib\target\tetrad-lib-7.6.11-SNAPSHOT-shaded.jar"),
    ])
    jar = _first_existing_path(candidates)
    if jar is None:
        raise RuntimeError("未找到 FASK/Tetrad jar，请设置 NPI_FASK_TETRAD_JAR")
    return jar


def _ensure_fask_runner_compiled(tetrad_jar: Path) -> Path:
    source = FASK_RUNNER_DIR / "FaskCsvRunner.java"
    classes_dir = FASK_RUNNER_DIR / "classes"
    class_file = classes_dir / "FaskCsvRunner.class"
    if not source.exists():
        raise RuntimeError(f"未找到 FASK Java 桥接文件: {source}")
    needs_compile = (
        not class_file.exists()
        or os.path.getmtime(class_file) < os.path.getmtime(source)
    )
    if needs_compile:
        classes_dir.mkdir(parents=True, exist_ok=True)
        javac = shutil.which("javac")
        if not javac:
            raise RuntimeError("未找到 javac，请确认 Java JDK 已安装并加入 PATH")
        proc = subprocess.run(
            [javac, "-encoding", "UTF-8", "-cp", str(tetrad_jar), "-d", str(classes_dir), str(source)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError("FASK Java 桥接编译失败: " + (proc.stderr or proc.stdout).strip())
    return classes_dir


def compute_fask_mask(signals_list: list,
                      two_cycle_alpha: float = 0.01,
                      penalty_discount: float = 2.0,
                      max_density: float | None = 0.10,
                      timeout_seconds: int = 900,
                      roi_num: int = 360) -> np.ndarray:
    """
    Run Tetrad FASK through a Java bridge.

    FASK uses non-Gaussian features to orient feedforward and feedback edges.
    The bridge returns the same mask convention as the rest of this app:
    mask[source, target] = True.
    """
    data = _combined_standardized_signals(signals_list, roi_num, min_rows=10)
    tetrad_jar = _resolve_fask_tetrad_jar()
    classes_dir = _ensure_fask_runner_compiled(tetrad_jar)

    with tempfile.TemporaryDirectory(prefix="fask_", dir=str(EXTERNAL_CAUSAL_TMP_DIR)) as tmp:
        tmp_dir = Path(tmp)
        input_csv = tmp_dir / "signals.csv"
        output_json = tmp_dir / "fask_result.json"
        np.savetxt(input_csv, data, delimiter=",", fmt="%.10g")

        java = shutil.which("java")
        if not java:
            raise RuntimeError("未找到 java，请确认 Java JRE/JDK 已安装并加入 PATH")
        proc = subprocess.run(
            [
                java,
                "-cp",
                f"{classes_dir}{os.pathsep}{tetrad_jar}",
                "FaskCsvRunner",
                str(input_csv),
                str(output_json),
                str(float(two_cycle_alpha)),
                str(float(penalty_discount)),
            ],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(timeout_seconds),
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError("FASK 运行失败: " + detail)
        if not output_json.exists():
            raise RuntimeError("FASK 运行结束但没有生成结果文件")
        with open(output_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

    mask = np.asarray(payload.get("mask"), dtype=bool)
    if mask.shape != (roi_num, roi_num):
        raise RuntimeError(f"FASK 输出形状异常: {mask.shape}")
    effect = mask.astype(float)
    return _effect_screen_mask(mask, effect, 0.0, max_density)


def _resolve_two_step_algorithm_dir() -> Path:
    env_path = os.getenv("NPI_TWO_STEP_ALGO_DIR")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        TWO_STEP_RUNNER_DIR / "Two-Step_Algorithm",
        Path(r"E:\因果算法\SanchezRomero2019_FASK_TwoStep_Downloads\Two-Step\Two-Step_Algorithm"),
        Path(r"E:\SanchezRomero2019_FASK_TwoStep_Downloads\Two-Step\Two-Step_Algorithm"),
    ])
    algo_dir = _first_existing_path(candidates)
    if algo_dir is None:
        raise RuntimeError("未找到 Two-Step MATLAB 源码目录，请设置 NPI_TWO_STEP_ALGO_DIR")
    if not (algo_dir / "two_step_CD.m").exists():
        raise RuntimeError(f"Two-Step 源码目录缺少 two_step_CD.m: {algo_dir}")
    return algo_dir


def _resolve_matlab_exe() -> str:
    env_path = os.getenv("MATLAB_EXE")
    if env_path and Path(env_path).exists():
        return env_path
    matlab = shutil.which("matlab") or shutil.which("matlab.exe")
    if matlab:
        return matlab
    common_roots = [Path(r"C:\Program Files\MATLAB"), Path(r"D:\Program Files\MATLAB"), Path(r"E:\Program Files\MATLAB"), Path(r"F:\Program Files\MATLAB")]
    candidates = []
    for root in common_roots:
        if root.exists():
            candidates.extend(sorted(root.glob(r"R*\bin\matlab.exe"), reverse=True))
    found = _first_existing_path(candidates)
    if found:
        return str(found)
    raise RuntimeError("未找到 MATLAB，请设置 MATLAB_EXE 为 matlab.exe 的完整路径")


def compute_two_step_mask(signals_list: list,
                          lambda_value: float = 2.0,
                          min_effect: float = 0.01,
                          max_density: float | None = 0.10,
                          timeout_seconds: int = 7200,
                          roi_num: int = 360,
                          progress_callback=None) -> np.ndarray:
    """
    Run the original MATLAB Two-Step implementation.

    two_step_CD returns B where B(i,j) means Xj -> Xi. The app convention is
    mask[source, target], so the effect matrix is abs(B).T.
    """
    data = _combined_standardized_signals(signals_list, roi_num, min_rows=10)
    runner = TWO_STEP_RUNNER_DIR / "run_two_step_csv.m"
    if not runner.exists():
        raise RuntimeError(f"未找到 Two-Step MATLAB 桥接文件: {runner}")
    algo_dir = _resolve_two_step_algorithm_dir()
    matlab = _resolve_matlab_exe()

    with tempfile.TemporaryDirectory(prefix="twostep_", dir=str(EXTERNAL_CAUSAL_TMP_DIR)) as tmp:
        tmp_dir = Path(tmp)
        input_csv = tmp_dir / "signals.csv"
        output_csv = tmp_dir / "B_est.csv"
        stdout_log = tmp_dir / "matlab_stdout.log"
        stderr_log = tmp_dir / "matlab_stderr.log"
        np.savetxt(input_csv, data, delimiter=",", fmt="%.10g")

        matlab_cmd = (
            f"addpath('{str(TWO_STEP_RUNNER_DIR)}'); "
            f"run_two_step_csv('{str(input_csv)}','{str(output_csv)}','{str(algo_dir)}',{float(lambda_value)});"
        )
        started_at = time.time()
        with open(stdout_log, "w", encoding="utf-8", errors="replace") as out_f, \
                open(stderr_log, "w", encoding="utf-8", errors="replace") as err_f:
            proc = subprocess.Popen(
                [matlab, "-batch", matlab_cmd],
                cwd=str(BASE_DIR),
                stdout=out_f,
                stderr=err_f,
                text=True,
            )
            global external_causal_current_proc
            with external_causal_proc_lock:
                external_causal_current_proc = proc
            last_callback = 0.0
            try:
                while True:
                    rc = proc.poll()
                    elapsed = time.time() - started_at
                    if causal_precompute_cancel_event.is_set():
                        proc.kill()
                        try:
                            proc.wait(timeout=30)
                        except Exception:
                            pass
                        raise RuntimeError("用户已取消 Two-Step/MATLAB 运行")
                    if progress_callback and (elapsed - last_callback >= 10 or rc is not None):
                        progress_callback(
                            stage="MATLAB运行中" if rc is None else "MATLAB已退出",
                            elapsed_seconds=elapsed,
                            detail=f"ROI {roi_num}",
                        )
                        last_callback = elapsed
                    if rc is not None:
                        break
                    if elapsed > int(timeout_seconds):
                        proc.kill()
                        try:
                            proc.wait(timeout=30)
                        except Exception:
                            pass
                        raise RuntimeError(f"Two-Step/MATLAB 超时：超过 {int(timeout_seconds)} 秒仍未完成")
                    time.sleep(2)
            finally:
                with external_causal_proc_lock:
                    if external_causal_current_proc is proc:
                        external_causal_current_proc = None

        stdout_text = stdout_log.read_text(encoding="utf-8", errors="replace") if stdout_log.exists() else ""
        stderr_text = stderr_log.read_text(encoding="utf-8", errors="replace") if stderr_log.exists() else ""
        if proc.returncode != 0:
            detail = (stderr_text or stdout_text or "").strip()
            raise RuntimeError("Two-Step/MATLAB 运行失败: " + detail)
        if not output_csv.exists():
            raise RuntimeError("Two-Step 运行结束但没有生成 B 矩阵")
        b_est = np.loadtxt(output_csv, delimiter=",")

    b_est = np.asarray(b_est, dtype=np.float64)
    if b_est.shape != (roi_num, roi_num):
        raise RuntimeError(f"Two-Step 输出 B 矩阵形状异常: {b_est.shape}")
    effect = np.abs(b_est).T
    mask = effect >= float(min_effect)
    np.fill_diagonal(mask, False)
    return _effect_screen_mask(mask, effect, min_effect, max_density)


def _partial_corr_pvalue(x: np.ndarray, y: np.ndarray,
                         z: np.ndarray) -> float:
    """
    Fisher-Z partial correlation test.
    H₀: pcor(x, y | z) = 0.  z is (n, k); k=0 → marginal test.
    Returns two-sided p-value.
    """
    n = len(x)
    if z.shape[1] == 0:
        r = float(np.corrcoef(x, y)[0, 1])
    else:
        ones = np.ones((n, 1))
        Z = np.hstack([ones, z])

        def _resid(a):
            c, _, _, _ = np.linalg.lstsq(Z, a, rcond=None)
            return a - Z @ c

        rx, ry = _resid(x), _resid(y)
        denom = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
        r = float(np.dot(rx, ry) / denom) if denom > 1e-12 else 0.0

    r = float(np.clip(r, -1 + 1e-9, 1 - 1e-9))
    k  = z.shape[1]
    df = n - 2 - k
    if df <= 0:
        return 1.0
    z_stat = np.sqrt(df) * 0.5 * np.log((1 + r) / (1 - r))
    return float(2 * norm_dist.sf(abs(z_stat)))


# ══════════════════════════════════════════════════════════════════
#  1. Pairwise Granger causality  (F-test + Fisher pooling)
# ══════════════════════════════════════════════════════════════════

def _granger_test(y: np.ndarray, x: np.ndarray, max_lag: int) -> tuple[float, float]:
    """
    Pairwise Granger causality F-test: H₀ = x does not Granger-cause y.

    Restricted:   y_t = a₀ + Σᵢ aᵢ y_{t-i}
    Unrestricted: y_t = a₀ + Σᵢ aᵢ y_{t-i} + Σᵢ bᵢ x_{t-i}

    F = [(RSS_r - RSS_u) / q] / [RSS_u / (n - k)]
    where q = max_lag restrictions, k = 2·max_lag+1 free params.
    """
    T = len(y)
    if T <= 2 * max_lag + 2:
        return 1.0, 0.0

    n     = T - max_lag
    Ylags = _lag_matrix(y, max_lag)            # (n, max_lag)
    Xlags = _lag_matrix(x, max_lag)
    const = np.ones((n, 1))
    Y     = y[max_lag:]

    rss_r = _ols_rss(np.hstack([const, Ylags]), Y)
    rss_u = _ols_rss(np.hstack([const, Ylags, Xlags]), Y)

    q   = max_lag
    k   = 2 * max_lag + 1
    df2 = n - k
    if df2 <= 0 or rss_u <= 1e-14:
        return 1.0, 0.0

    F = ((rss_r - rss_u) / q) / (rss_u / df2)
    effect = max(0.0, (rss_r - rss_u) / max(rss_r, 1e-14))
    return float(f_dist.sf(max(F, 0.0), q, df2)), float(effect)


def compute_granger_mask(signals_list: list,
                         max_lag: int = 3,
                         alpha: float = 0.05,
                         min_effect: float = 0.001,
                         max_density: float | None = 0.12,
                         max_candidates_per_target: int | None = None,
                         roi_num: int = 360,
                         progress_callback=None) -> np.ndarray:
    """
    Pairwise Granger causality screening.

    By default, for each ordered non-self pair (i→j), runs the F-test across
    every session then combines p-values with Fisher's method.  Returns a
    directed boolean mask.

    max_candidates_per_target is kept only as an optional engineering throttle.
    The default None means full pairwise Granger without candidate pre-screen.

    Reference: Granger (1969), Geweke (1982).

    Scientific role in this app:
        Fast directed lag screen. Use as weak temporal evidence; it can be
        confounded by common drivers because each test only sees one source
        and one target.
    """
    signals_list = _validate_signals(signals_list, roi_num, min_rows=2 * max_lag + 3)
    log_p_sum  = np.zeros((roi_num, roi_num), dtype=np.float64)
    effect_sum = np.zeros((roi_num, roi_num), dtype=np.float64)
    n_sessions = len(signals_list)
    if max_candidates_per_target is None or int(max_candidates_per_target) >= roi_num - 1:
        candidate_mask = np.ones((roi_num, roi_num), dtype=bool)
        np.fill_diagonal(candidate_mask, False)
    else:
        candidate_cap = max(1, min(int(max_candidates_per_target), roi_num - 1))

        # Optional candidate pre-screen: keep sources with strongest lagged
        # association to each target. Disabled by default for full Granger.
        lag_score = np.zeros((roi_num, roi_num), dtype=np.float64)
        for sig in signals_list:
            sig = _zscore(sig)
            for tau in range(1, max_lag + 1):
                x = sig[:-tau, :roi_num]
                y = sig[tau:, :roi_num]
                x = x - x.mean(axis=0, keepdims=True)
                y = y - y.mean(axis=0, keepdims=True)
                denom = np.sqrt(np.sum(x * x, axis=0))[:, None] * np.sqrt(np.sum(y * y, axis=0))[None, :]
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = np.abs((x.T @ y) / np.maximum(denom, 1e-12))
                lag_score += np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(lag_score, -np.inf)

        candidate_mask = np.zeros((roi_num, roi_num), dtype=bool)
        for j in range(roi_num):
            keep = np.argsort(lag_score[:, j])[-candidate_cap:]
            candidate_mask[keep, j] = True
        np.fill_diagonal(candidate_mask, False)

    started_at = time.time()

    for session_idx, sig in enumerate(signals_list, start=1):
        sig = _zscore(sig)
        for i in range(roi_num):
            if progress_callback and (i == 0 or (i + 1) % 12 == 0 or i == roi_num - 1):
                progress_callback(
                    session_idx=session_idx,
                    total_sessions=n_sessions,
                    roi_idx=i + 1,
                    total_rois=roi_num,
                    elapsed_seconds=time.time() - started_at
                )
            for j in range(roi_num):
                if i == j or not candidate_mask[i, j]:
                    continue
                p, effect = _granger_test(sig[:, j], sig[:, i], max_lag)
                log_p_sum[i, j] += np.log(max(p, 1e-300))
                effect_sum[i, j] += effect

    p_mask = _fisher_combine(log_p_sum, n_sessions, alpha)
    return _effect_screen_mask(p_mask & candidate_mask, effect_sum / n_sessions, min_effect, max_density)


# ══════════════════════════════════════════════════════════════════
#  2. Multivariate VAR-Granger  (block F-test, BIC lag selection)
# ══════════════════════════════════════════════════════════════════

def _bic_select_lag(y: np.ndarray, max_lag: int = 8) -> int:
    """
    Select VAR lag order by minimising BIC on univariate AR(p) model for y.
    BIC = n·log(RSS/n) + k·log(n)
    """
    T = len(y)
    best_bic = np.inf
    best_p   = 1
    for p in range(1, max_lag + 1):
        n = T - p
        if n <= p + 1:
            break
        Ylags = _lag_matrix(y, p)
        const = np.ones((n, 1))
        rss   = _ols_rss(np.hstack([const, Ylags]), y[p:])
        k     = p + 1
        bic   = n * np.log(max(rss / n, 1e-14)) + k * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_p   = p
    return best_p


def compute_var_granger_mask(signals_list: list,
                              max_lag: int = 8,
                              alpha: float = 0.05,
                              min_effect: float = 0.001,
                              max_density: float | None = 0.12,
                              roi_num: int = 360) -> np.ndarray:
    """
    Multivariate VAR-Granger causality with BIC lag selection.

    For each target ROI j, fits a lagged model over all ROI_num sources,
    then tests the block of source-i lags via an F-test. The maximum lag is
    capped by available degrees of freedom, which is essential for 360-ROI
    fMRI where a full VAR can otherwise be underdetermined.

    Reference: Lütkepohl (2005) "New Introduction to Multiple Time Series
               Analysis", ch. 2–3.

    Scientific role in this app:
        Stronger temporal screen than pairwise Granger because other ROIs are
        included as nuisance predictors. At 360 ROI it must cap lag order, so
        absence of an edge is not proof of no causal influence.
    """
    signals_list = _validate_signals(signals_list, roi_num, min_rows=max_lag + 3)
    log_p_sum = np.zeros((roi_num, roi_num), dtype=np.float64)
    effect_sum = np.zeros((roi_num, roi_num), dtype=np.float64)
    used_sessions = 0

    for sig in signals_list:
        sig = _zscore(sig)
        T   = sig.shape[0]
        max_feasible_lag = min(max_lag, max(0, (T - 3) // (roi_num + 1)))
        if max_feasible_lag < 1:
            continue
        used_sessions += 1

        # Use a common feasible lag per session so all targets share the same
        # design matrix and the block tests remain comparable.
        p = min(
            max_feasible_lag,
            max(1, int(np.median([_bic_select_lag(sig[:, j], max_feasible_lag) for j in range(roi_num)]))),
        )
        n = T - p
        blocks = [np.ones((n, 1))]
        for k in range(roi_num):
            blocks.append(_lag_matrix(sig[:, k], p))
        X_full = np.hstack(blocks)
        Y_all = sig[p:, :roi_num]

        df2 = n - X_full.shape[1]
        if df2 <= 0:
            used_sessions -= 1
            continue

        beta, _, _, _ = np.linalg.lstsq(X_full, Y_all, rcond=None)
        residual = Y_all - X_full @ beta
        rss = np.sum(residual ** 2, axis=0)
        sigma2 = np.maximum(rss / df2, 1e-14)
        xtx_inv = np.linalg.pinv(X_full.T @ X_full)

        for i in range(roi_num):
            col_start = 1 + i * p
            col_end = col_start + p
            cols = np.arange(col_start, col_end)
            cov_block = xtx_inv[np.ix_(cols, cols)]
            cov_block_inv = np.linalg.pinv(cov_block)

            for j in range(roi_num):
                if i == j:
                    continue
                b = beta[cols, j]
                F = float((b.T @ cov_block_inv @ b) / (p * sigma2[j]))
                p_val = float(f_dist.sf(max(F, 0.0), p, df2))
                log_p_sum[i, j] += np.log(max(p_val, 1e-300))
                effect_sum[i, j] += max(0.0, (F * p) / (F * p + df2)) if np.isfinite(F) else 0.0

    if used_sessions == 0:
        raise ValueError("时间点不足，无法在当前 ROI 数量下拟合多变量 VAR-Granger")

    p_mask = _fisher_combine(log_p_sum, used_sessions, alpha)
    return _effect_screen_mask(p_mask, effect_sum / used_sessions, min_effect, max_density)


def compute_sparse_var_mask(signals_list: list,
                            max_lag: int = 2,
                            max_candidates_per_target: int = 32,
                            min_effect: float = 0.01,
                            max_density: float | None = 0.10,
                            ridge: float = 1.0,
                            min_session_support_ratio: float = 0.25,
                            roi_num: int = 360) -> np.ndarray:
    """
    Candidate-limited sparse VAR screen.

    For each target ROI, select lagged source candidates by temporal
    correlation, fit a ridge-regularized VAR model, and score each source by
    the grouped lag coefficient magnitude. This is not a full Lasso path; it is
    a high-dimensional sparse VAR approximation suitable as a consensus
    evidence source.

    Scientific role in this app:
        Engineering-oriented sparse directed screen. Treat edges as
        coefficient-support evidence, not hypothesis-tested causal claims.
    """
    max_lag = max(1, int(max_lag))
    candidate_cap = max(1, min(int(max_candidates_per_target), roi_num - 1))
    min_session_support_ratio = float(np.clip(min_session_support_ratio, 0.0, 1.0))
    signals_list = _validate_signals(signals_list, roi_num, min_rows=max_lag + 5)
    effect = np.zeros((roi_num, roi_num), dtype=np.float64)
    session_support = np.zeros((roi_num, roi_num), dtype=np.int16)
    used_sessions = 0

    for sig in signals_list:
        sig = _zscore(sig)
        T = len(sig)
        rows = T - max_lag
        if rows <= max_lag + 2:
            continue
        used_sessions += 1

        L = np.column_stack([
            sig[max_lag - tau: T - tau, i]
            for tau in range(1, max_lag + 1)
            for i in range(roi_num)
        ])
        Y = sig[max_lag:, :roi_num]
        Lc = L - L.mean(axis=0, keepdims=True)
        Yc = Y - Y.mean(axis=0, keepdims=True)
        denom = np.sqrt(np.sum(Lc * Lc, axis=0))[:, None] * np.sqrt(np.sum(Yc * Yc, axis=0))[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            lag_corr = np.abs((Lc.T @ Yc) / np.maximum(denom, 1e-12))
        lag_corr = np.nan_to_num(lag_corr, nan=0.0, posinf=0.0, neginf=0.0)

        for j in range(roi_num):
            source_scores = np.zeros(roi_num, dtype=np.float64)
            for i in range(roi_num):
                if i != j:
                    cols = [tau * roi_num + i for tau in range(max_lag)]
                    source_scores[i] = float(np.max(lag_corr[cols, j]))
            candidates = [int(i) for i in np.argsort(source_scores)[-candidate_cap:] if i != j and source_scores[int(i)] > 0]

            cols = []
            source_to_cols = {}
            # Always include target autoregression as nuisance structure.
            target_cols = [tau * roi_num + j for tau in range(max_lag)]
            cols.extend(target_cols)
            for i in candidates:
                c = [tau * roi_num + i for tau in range(max_lag)]
                source_to_cols[i] = list(range(len(cols), len(cols) + len(c)))
                cols.extend(c)

            X = L[:, cols]
            X = np.column_stack([np.ones(rows), X])
            y = Y[:, j]
            penalty = float(ridge) * np.eye(X.shape[1])
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
            coef = beta[1:]
            for i, local_cols in source_to_cols.items():
                group_norm = float(np.linalg.norm(coef[local_cols], ord=2))
                effect[i, j] += group_norm
                if group_norm >= float(min_effect):
                    session_support[i, j] += 1

    if used_sessions == 0:
        raise ValueError("时间点不足，无法运行 Sparse VAR-Lite")

    effect /= used_sessions
    support_ratio = np.divide(
        session_support,
        used_sessions,
        out=np.zeros((roi_num, roi_num), dtype=np.float64),
        where=used_sessions > 0,
    )
    mask = (
        np.isfinite(effect)
        & (effect >= float(min_effect))
        & (support_ratio >= min_session_support_ratio)
    )
    if mask.ndim == 2 and mask.shape[0] == mask.shape[1]:
        np.fill_diagonal(mask, False)
    return _effect_screen_mask(mask, effect, min_effect, max_density)






# ══════════════════════════════════════════════════════════════════
#  7. PCMCI  (time-series PC with Momentary Conditional MI)
# ══════════════════════════════════════════════════════════════════

def compute_pcmci_mask(signals_list: list,
                        max_lag: int = 5,
                        alpha: float = 0.05,
                        min_effect: float = 0.02,
                        max_density: float | None = 0.10,
                        roi_num: int = 360,
                        full_tigramite: bool = False) -> np.ndarray:
    """
    PCMCI: combines the PC skeleton search with a lagged conditional
    independence criterion tailored for autocorrelated time series.

    Two-phase algorithm (Runge et al. 2019):

    Phase 1 — PC₁ (condition-selection):
        For each target j and each lagged predictor (i, τ), τ=1..max_lag,
        iteratively remove predictors whose partial correlation with j
        given the current predictor set is not significant.
        The output is the "parents" set P(j) ⊆ {(i,τ)}.

    Phase 2 — Momentary Conditional Independence (MCI) test:
        For each candidate link (i,τ)→j test:
            i_{t-τ} ⊥ j_t | P(j)\\{(i,τ)}, P(i)_{t-τ}   (MCI test)
        This accounts for the autocorrelation of both source and target,
        dramatically reducing false positives compared to plain PC on
        time series.

    Returns a directed mask: mask[i,j]=True if i Granger/causally precedes
    j at any lag 1..max_lag.

    Reference: Runge, Nowack, Kretschmer, Flaxman & Sejdinovic (2019)
               "Detecting and quantifying causal associations in large
               nonlinear time series datasets", Science Advances.

    Scientific role in this app:
        One of the better-matched temporal screens for fMRI-like time series,
        but this is a partial-correlation, candidate-limited PCMCI-Lite for
        full-brain interactivity rather than a full Tigramite-style PCMCI run.

        full_tigramite=True keeps an explicit escape hatch for small/local
        sub-networks. It is intentionally off for 360-ROI full-brain masks.
    """
    max_lag = max(1, int(max_lag))
    candidate_cap = max(1, min(int(PCMCI_MAX_PARENTS_PER_TARGET), roi_num * max_lag))
    cond_cap = max(0, int(PCMCI_MAX_CONDITIONING_SET))
    pc_max_order = max(0, int(PCMCI_PC_MAX_ORDER))
    if full_tigramite:
        try:
            import tigramite.data_processing as pp
            from tigramite.independence_tests.parcorr import ParCorr
            from tigramite.pcmci import PCMCI

            masks = []
            effects = []
            for sig in _validate_signals(signals_list, roi_num, min_rows=max_lag + 5):
                dataframe = pp.DataFrame(_zscore(sig))
                pcmci = PCMCI(
                    dataframe=dataframe,
                    cond_ind_test=ParCorr(significance="analytic"),
                    verbosity=0,
                )
                result = pcmci.run_pcmci(
                    tau_min=1,
                    tau_max=max_lag,
                    pc_alpha=float(alpha),
                    alpha_level=float(alpha),
                    max_conds_dim=None if cond_cap <= 0 else cond_cap,
                    max_combinations=1,
                    fdr_method="fdr_bh",
                )
                p_matrix = np.asarray(result["p_matrix"], dtype=np.float64)
                val_matrix = np.asarray(result["val_matrix"], dtype=np.float64)
                session_mask = np.any(p_matrix[:, :, 1:max_lag + 1] <= float(alpha), axis=2)
                session_effect = np.max(np.abs(val_matrix[:, :, 1:max_lag + 1]), axis=2)
                np.fill_diagonal(session_mask, False)
                np.fill_diagonal(session_effect, 0.0)
                masks.append(session_mask)
                effects.append(session_effect)

            if masks:
                support = np.mean(np.stack(masks, axis=0), axis=0)
                effect = np.mean(np.stack(effects, axis=0), axis=0)
                mask = (support > 0.0) & (effect >= float(min_effect))
                np.fill_diagonal(mask, False)
                return _effect_screen_mask(mask, effect, min_effect, max_density)
        except Exception as external_error:
            warnings.warn(f"Tigramite PCMCI failed; using built-in PCMCI fallback: {external_error}")

    signals_list = _validate_signals(signals_list, roi_num, min_rows=max_lag + 5)
    log_p_sum = np.zeros((roi_num, roi_num), dtype=np.float64)
    effect_sum = np.zeros((roi_num, roi_num), dtype=np.float64)
    used_sessions = 0

    def _col(i, tau):
        """Column index of (i, tau) in a session-specific lag matrix."""
        return (tau - 1) * roi_num + i

    def _cap_cols(cols, score_vec):
        """Keep the most target-relevant conditioning columns."""
        if cond_cap == 0 or len(cols) <= cond_cap:
            return list(cols)
        ranked = sorted(set(cols), key=lambda c: score_vec[c], reverse=True)
        return ranked[:cond_cap]

    for sig in signals_list:
        sig = _zscore(sig)
        T = len(sig)
        rows = T - max_lag
        if rows <= max_lag + 2:
            continue
        used_sessions += 1

        # Build lagged predictors inside each session only; never connect across run borders.
        L = np.column_stack([
            sig[max_lag - tau: T - tau, i]
            for tau in range(1, max_lag + 1)
            for i in range(roi_num)
        ])

        Y = sig[max_lag:]
        Lc = L - L.mean(axis=0, keepdims=True)
        Yc = Y - Y.mean(axis=0, keepdims=True)
        denom = np.sqrt(np.sum(Lc * Lc, axis=0))[:, None] * np.sqrt(np.sum(Yc * Yc, axis=0))[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            lag_corr = np.abs((Lc.T @ Yc) / np.maximum(denom, 1e-12))
        lag_corr = np.nan_to_num(lag_corr, nan=0.0, posinf=0.0, neginf=0.0)

        # Phase 1: PC1-style parent selection per target.
        # Full PCMCI is too heavy at 360 ROI; this keeps the strongest lagged
        # candidates and performs a shallow PC screen suitable for visualization.
        parents = {}
        for j in range(roi_num):
            score_vec = lag_corr[:, j]
            keep_idx = np.argsort(score_vec)[-candidate_cap:]
            cands = [int(idx) for idx in keep_idx if score_vec[int(idx)] > 0]
            ranked_cands = sorted(cands, key=lambda c: score_vec[c], reverse=True)

            max_order = min(pc_max_order, max(0, rows - 4), len(cands))
            for order in range(max_order + 1):
                to_remove = []
                for c in list(cands):
                    cond_pool = [cc for cc in ranked_cands if cc != c and cc in cands]
                    cond_cols = cond_pool[:min(order, cond_cap)] if order else []
                    z = L[:, cond_cols] if cond_cols else np.empty((rows, 0))
                    p = _partial_corr_pvalue(L[:, c], Y[:, j], z)
                    if p >= alpha:
                        to_remove.append(c)
                for c in to_remove:
                    if c in cands:
                        cands.remove(c)
                ranked_cands = [cc for cc in ranked_cands if cc in cands]
                if not to_remove:
                    break
            parents[j] = cands

        # Phase 2: MCI-style test. For each directed edge, keep the best lag
        # p-value with a simple Bonferroni correction over searched lags.
        p_session = np.ones((roi_num, roi_num), dtype=np.float64)
        for j in range(roi_num):
            parent_set = set(parents[j])
            for c_ij in parent_set:
                i = c_ij % roi_num
                tau = c_ij // roi_num + 1
                if i == j:
                    continue

                pj_minus = [cc for cc in parents[j] if cc != c_ij]
                pi_shifted = []
                for cc in parents[i]:
                    k = cc % roi_num
                    s = cc // roi_num + 1
                    new_tau = s + tau
                    if new_tau <= max_lag:
                        pi_shifted.append(_col(k, new_tau))
                cond_cols = _cap_cols(pj_minus + pi_shifted, lag_corr[:, j])

                z = L[:, cond_cols] if cond_cols else np.empty((rows, 0))
                p = _partial_corr_pvalue(L[:, c_ij], Y[:, j], z)
                p_session[i, j] = min(p_session[i, j], min(1.0, p * max_lag))
                effect_sum[i, j] = max(effect_sum[i, j], float(lag_corr[c_ij, j]))

        log_p_sum += np.log(np.maximum(p_session, 1e-300))

    if used_sessions == 0:
        raise ValueError("时间点不足，无法运行 PCMCI")

    p_mask = _fisher_combine(log_p_sum, used_sessions, alpha)
    return _effect_screen_mask(p_mask, effect_sum, min_effect, max_density)



# ==========================
# 初始化数据
# ==========================
def initialize_data():
    global ec_dict, current_ec_meta, ANN, inputs_data, targets_data, roi_vox, brain_verts, brain_faces, is_initializing

    available_results = _available_ec_results()
    preferred_mode = "none" if any(r["mode"] == "none" for r in available_results) else (
        available_results[0]["mode"] if available_results else None
    )
    if preferred_mode is not None:
        loaded_ec, ec_meta = _load_ec_result(preferred_mode)
        _set_current_ec_result(preferred_mode, loaded_ec, ec_meta)
        print(f"✅ 加载了 {len(ec_dict)} 组EC数据（模式: {ec_meta['mode_label']}）")
    elif os.path.exists(data_file):
        with np.load(data_file, allow_pickle=True) as ec_data:
            ec_dict = {float(k): ec_data[k] for k in ec_data.files}
        if not _ec_strengths_current(ec_dict):
            print(f"⚠ 历史EC数据强度 {sorted(ec_dict.keys())} 与当前配置 {_configured_strengths()} 不一致，已跳过加载")
            ec_dict = {}
            current_ec_meta = _default_ec_meta()
            computation_status["message"] = "EC数据强度配置已变更，请重新计算"
            computation_status["last_error"] = ""
        else:
            current_ec_meta = _load_ec_meta()
            try:
                legacy_mode = _normalize_ec_result_mode(current_ec_meta.get("mode"))
                if not os.path.exists(_ec_result_file(legacy_mode)):
                    current_ec_meta = _save_ec_result(
                        legacy_mode,
                        ec_dict,
                        current_ec_meta.get("mode_label") or COMPUTE_MODE_LABELS[legacy_mode],
                        bool(current_ec_meta.get("screen_first")),
                    )
                    print(f"✅ 已迁移历史EC数据到模式文件: {current_ec_meta['file']}")
            except ValueError:
                pass
            computation_status["mode"] = current_ec_meta["mode"]
            computation_status["mode_label"] = current_ec_meta["mode_label"]
            computation_status["screen_first"] = bool(current_ec_meta["screen_first"])
            print(f"✅ 加载了 {len(ec_dict)} 组历史EC数据")
    else:
        ec_dict = {}
        current_ec_meta = _default_ec_meta()
        print("⚠ 未找到预计算数据")

    loaded_mask_keys = _load_cached_masks()
    if loaded_mask_keys:
        latest_cache_mtime = _latest_mask_cache_mtime(loaded_mask_keys)
        _update_precompute_status(
            is_running=False,
            message=f"已加载 {len(loaded_mask_keys)} 组因果算法缓存",
            last_error="",
            completed_at=latest_cache_mtime,
            computed_keys=loaded_mask_keys,
        )
        print(f"✅ 加载了 {len(loaded_mask_keys)} 组因果算法缓存: {', '.join(loaded_mask_keys)}")
    else:
        _update_precompute_status(
            is_running=False,
            message="未找到因果算法默认缓存",
            last_error="",
            completed_at=None,
            computed_keys=[],
        )
        print("⚠ 未找到因果算法默认缓存")

    file_path = BRAIN_MODEL_FILE
    if os.path.exists(file_path):
        img = nib.load(file_path)
        vol = img.get_fdata()
        vol_smooth = gaussian_filter(vol, sigma=4)
        brain_verts, brain_faces, _, _ = marching_cubes(vol_smooth, level=0.5, step_size=4)
        roi_vox = brain_verts[np.linspace(0, len(brain_verts) - 1, ROI_num, dtype=int)]
        print("✅ 大脑模型加载成功")
    else:
        brain_verts = np.random.rand(1000, 3) * 100
        brain_faces = np.random.randint(0, 1000, (500, 3))
        roi_vox = np.random.rand(ROI_num, 3) * 100
        print("⚠ 使用随机大脑模型")

    is_initializing = False
    print("✅ 初始化完成，服务就绪")


# ==========================
# 异步训练计算
# ==========================
def train_and_compute_async(mode_key: str | None = None, params: dict | None = None):
    global ec_dict, ANN, inputs_data, targets_data, computation_status, \
           granger_mask, var_granger_mask, pcmci_mask, sparse_var_mask, \
           npi_intervention_score_cache, npi_intervention_cache_signature

    try:
        npi = get_npi_module()
        mode_key = _normalize_mask_key(mode_key)
        mode_id = mode_key or "none"
        mode_label = COMPUTE_MODE_LABELS.get(mode_id, DEFAULT_MASK_LABELS.get(mode_id, mode_id))
        screening_steps = 0 if mode_key is None else (len(CONSENSUS_SOURCE_KEYS) if mode_key in {"consensus", "ensemble"} else 1)
        computation_status["is_computing"] = True
        computation_status["progress"] = 0
        computation_status["last_error"] = ""
        computation_status["started_at"] = time.time()
        computation_status["mode"] = mode_id
        computation_status["mode_label"] = mode_label
        computation_status["screen_first"] = mode_key is not None
        # Steps: load -> optional causal screen(s) -> train NPI -> EC passes
        computation_status["total"] = len(strength_list) + 2 + screening_steps
        computation_status["message"] = "加载fMRI数据..."

        signals, inputs, targets = [], [], []
        fmri_files = get_default_fmri_files()
        for f in fmri_files:
            if os.path.exists(f):
                sig = np.loadtxt(f)[30:, :ROI_num]
                inp, tar = npi.multi2one(sig, steps=using_steps)
                signals.append(sig)
                inputs.append(inp)
                targets.append(tar)
        if not signals:
            raise FileNotFoundError(
                f"未找到训练所需的 fMRI 文本数据，请检查目录: {REAL_FMRI_DIR}"
            )

        inputs_data = np.vstack(inputs)
        targets_data = np.vstack(targets)

        computation_status["progress"] += 1

        # ------------------------------------------------------------------
        # Optional causal discovery pre-screen.
        # ------------------------------------------------------------------
        pre_npi_mask = None
        screen_meta = {
            "mode": "none",
            "mode_label": COMPUTE_MODE_LABELS["none"],
            "screen_first": False,
            "sources": [],
        }
        if mode_key is not None:
            def _screen_progress(**kwargs):
                elapsed_seconds = int(kwargs["elapsed_seconds"])
                if "session_idx" in kwargs and "roi_idx" in kwargs:
                    computation_status["message"] = (
                        f"{mode_label} 前置筛查中：session {kwargs['session_idx']}/{kwargs['total_sessions']}，"
                        f"ROI {kwargs['roi_idx']}/{kwargs['total_rois']}，已耗时 {elapsed_seconds}s"
                    )
                else:
                    stage = kwargs.get("stage", "运行中")
                    detail = kwargs.get("detail", "")
                    computation_status["message"] = (
                        f"{mode_label} 前置筛查中：{stage}，已耗时 {elapsed_seconds}s"
                        + (f"，{detail}" if detail else "")
                    )

            computation_status["message"] = f"{mode_label} 前置因果筛查中..."
            print(f"🔍 开始前置因果筛查: {mode_label}")
            pre_npi_mask, screen_meta = _prepare_pre_npi_screen(
                mode_key,
                signals,
                params=params or {},
                progress_callback=_screen_progress,
            )
            n_significant = int(pre_npi_mask.sum())
            n_total = ROI_num * (ROI_num - 1)
            print(f"✅ 前置筛查完成: {n_significant}/{n_total} 对保留 "
                  f"({100 * n_significant / n_total:.1f}%)")
            computation_status["progress"] += screening_steps

        computation_status["message"] = "训练NPI模型..."

        ANN = npi.ANN_MLP(
            input_dim=using_steps * ROI_num,
            hidden_dim=2 * ROI_num,
            latent_dim=int(0.8 * ROI_num),
            output_dim=ROI_num
        )
        ANN, _, _ = npi.train_NN(
            ANN, inputs_data, targets_data,
            batch_size, train_set_proportion,
            num_epochs=100, lr=1e-3, l2=5e-5
        )

        computation_status["progress"] += 1

        # ------------------------------------------------------------------
        # NPI Causal Network Construction (optionally pre-screened)
        # ------------------------------------------------------------------
        new_ec_dict = {}
        for idx, s in enumerate(strength_list):
            if pre_npi_mask is None:
                computation_status["message"] = f"NPI计算强度 {s}（无前置筛查）"
            else:
                computation_status["message"] = f"NPI计算强度 {s}（{screen_meta['mode_label']} 前置筛查已应用）"
            ec = npi.model_EC(ANN, inputs_data, targets_data, pert_strength=s)
            np.fill_diagonal(ec, 0)

            if pre_npi_mask is not None:
                ec = ec * pre_npi_mask.astype(ec.dtype)

            new_ec_dict[s] = ec
            computation_status["progress"] += 1

        saved_meta = _save_ec_result(
            screen_meta["mode"],
            new_ec_dict,
            screen_meta["mode_label"],
            bool(screen_meta["screen_first"]),
        )
        _set_current_ec_result(screen_meta["mode"], new_ec_dict, saved_meta)
        print(f"✅ EC数据已保存（模式: {screen_meta['mode_label']}）")

        computation_status["message"] = f"计算完成！模式：{screen_meta['mode_label']}"
        computation_status["is_computing"] = False
        computation_status["last_error"] = ""

    except Exception as e:
        print(f"❌ 计算错误: {e}")
        computation_status["message"] = f"错误：{str(e)}"
        computation_status["is_computing"] = False
        computation_status["last_error"] = str(e)


def train_and_compute_all_async(params: dict | None = None):
    global ANN, inputs_data, targets_data, computation_status, \
           npi_intervention_score_cache, npi_intervention_cache_signature

    try:
        npi = get_npi_module()
        params = params or {}
        computation_status["is_computing"] = True
        computation_status["progress"] = 0
        computation_status["total"] = 1 + len(BATCH_PRE_NPI_SCREEN_KEYS) + 1 + len(strength_list) + len(BATCH_EC_RESULT_MODES)
        computation_status["message"] = "加载fMRI数据..."
        computation_status["last_error"] = ""
        computation_status["started_at"] = time.time()
        computation_status["mode"] = "all"
        computation_status["mode_label"] = "全部EC结果"
        computation_status["screen_first"] = True
        computation_status["all_modes"] = True

        signals, inputs, targets = [], [], []
        for f in get_default_fmri_files():
            if os.path.exists(f):
                sig = np.loadtxt(f)[30:, :ROI_num]
                inp, tar = npi.multi2one(sig, steps=using_steps)
                signals.append(sig)
                inputs.append(inp)
                targets.append(tar)
        if not signals:
            raise FileNotFoundError(f"未找到训练所需的 fMRI 文本数据，请检查目录: {REAL_FMRI_DIR}")

        inputs_data = np.vstack(inputs)
        targets_data = np.vstack(targets)
        computation_status["progress"] += 1

        masks = {"none": None}
        started_at = time.time()
        for source_key in BATCH_PRE_NPI_SCREEN_KEYS:
            label = DEFAULT_MASK_LABELS[source_key]
            computation_status["message"] = f"计算前置筛查：{label}"

            def _screen_progress(**kwargs):
                elapsed_seconds = int(kwargs.get("elapsed_seconds", 0))
                if "session_idx" in kwargs and "roi_idx" in kwargs:
                    computation_status["message"] = (
                        f"{label} 完整筛查中：session {kwargs['session_idx']}/{kwargs['total_sessions']}，"
                        f"ROI {kwargs['roi_idx']}/{kwargs['total_rois']}，"
                        f"已耗时 {elapsed_seconds}s"
                    )
                else:
                    stage = kwargs.get("stage", "运行中")
                    detail = kwargs.get("detail", "")
                    computation_status["message"] = (
                        f"{label} 完整筛查中：{stage}，已耗时 {elapsed_seconds}s"
                        + (f"，{detail}" if detail else "")
                    )

            mask = _compute_mask_with_params(
                source_key,
                signals,
                params=params.get(source_key) if isinstance(params.get(source_key), dict) else None,
                progress_callback=_screen_progress,
            )
            _set_mask_by_key(source_key, mask)
            masks[source_key] = mask
            _mark_mask_cached(source_key)
            if source_key in FMRI_NATIVE_SCREEN_KEYS:
                _save_single_cached_mask(source_key, mask, pipeline_mode=source_key)
            computation_status["progress"] += 1
            print(f"✅ {label}筛查完成: {int(mask.sum())}/{ROI_num * (ROI_num - 1)}")

        _save_cached_masks()
        consensus_mask, consensus_meta = _compute_consensus_mask()
        masks["consensus"] = consensus_mask
        masks["ensemble"] = consensus_mask

        computation_status["message"] = "训练NPI模型..."
        ANN = npi.ANN_MLP(
            input_dim=using_steps * ROI_num,
            hidden_dim=2 * ROI_num,
            latent_dim=int(0.8 * ROI_num),
            output_dim=ROI_num
        )
        ANN, _, _ = npi.train_NN(
            ANN, inputs_data, targets_data,
            batch_size, train_set_proportion,
            num_epochs=100, lr=1e-3, l2=5e-5
        )
        computation_status["progress"] += 1

        raw_ec_dict = {}
        for s in strength_list:
            computation_status["message"] = f"NPI计算强度 {s}（纯NPI基础结果）"
            ec = npi.model_EC(ANN, inputs_data, targets_data, pert_strength=s)
            np.fill_diagonal(ec, 0)
            raw_ec_dict[s] = ec
            computation_status["progress"] += 1

        saved_meta_by_mode = {}
        for mode in BATCH_EC_RESULT_MODES:
            computation_status["message"] = f"保存EC结果：{COMPUTE_MODE_LABELS[mode]}"
            mask = masks[mode]
            if mask is None:
                result = {s: arr.copy() for s, arr in raw_ec_dict.items()}
            else:
                mask_arr = mask.astype(next(iter(raw_ec_dict.values())).dtype)
                result = {s: arr * mask_arr for s, arr in raw_ec_dict.items()}
            saved_meta_by_mode[mode] = _save_ec_result(
                mode,
                result,
                COMPUTE_MODE_LABELS[mode],
                mode != "none",
            )
            computation_status["progress"] += 1

        _set_current_ec_result("none", raw_ec_dict, saved_meta_by_mode["none"])
        npi_intervention_score_cache = None
        npi_intervention_cache_signature = None
        elapsed = int(time.time() - started_at)
        computation_status["message"] = f"全部EC结果计算完成，用时 {elapsed}s"
        computation_status["mode"] = "none"
        computation_status["mode_label"] = COMPUTE_MODE_LABELS["none"]
        computation_status["screen_first"] = False
        computation_status["is_computing"] = False
        computation_status["last_error"] = ""
        computation_status["all_modes"] = False
        print("✅ 全部EC模式结果已保存")

    except Exception as e:
        print(f"❌ 全部模式计算错误: {e}")
        computation_status["message"] = f"错误：{str(e)}"
        computation_status["is_computing"] = False
        computation_status["last_error"] = str(e)
        computation_status["all_modes"] = False


# ====================== 前端静态文件 ======================
@app.route('/')
def index():
    return send_from_directory(str(TEMPLATES_DIR), 'index.html')


# ====================== 基础API ======================
@app.route("/api/status", methods=['GET'])
def status_api():
    return jsonify(computation_status)


@app.route("/api/available_strengths", methods=['GET'])
def available_strengths_api():
    if is_initializing:
        return jsonify({
            "has_data": False,
            "strengths": [],
            "configured_strengths": _configured_strengths(),
            "stale_strengths": [],
            "strengths_stale": False,
            "initializing": True,
            "ec_results": [],
        })
    ec_meta = _load_ec_meta() if len(ec_dict) > 0 else _default_ec_meta()
    stale_strengths = []
    if not ec_dict and os.path.exists(data_file):
        try:
            with np.load(data_file, allow_pickle=True) as ec_data:
                stale_strengths = sorted(float(k) for k in ec_data.files)
        except Exception:
            stale_strengths = []
    strengths_stale = bool(stale_strengths) and _strength_key_tuple(stale_strengths) != _strength_key_tuple(_configured_strengths())
    return jsonify({
        "has_data": len(ec_dict) > 0,
        "strengths": sorted(list(ec_dict.keys())),
        "configured_strengths": _configured_strengths(),
        "stale_strengths": stale_strengths if strengths_stale else [],
        "strengths_stale": strengths_stale,
        "initializing": False,
        "ec_mode": ec_meta["mode"],
        "ec_mode_label": ec_meta["mode_label"],
        "ec_screen_first": bool(ec_meta["screen_first"]),
        "ec_updated_at": ec_meta["updated_at"],
        "ec_results": _available_ec_results(),
        "ec_reduction_vs_npi": _mode_reduction_against_npi(ec_meta["mode"]) if len(ec_dict) > 0 else {"available": False},
    })


@app.route("/api/ec_results", methods=['GET'])
def ec_results_api():
    return jsonify({
        "current_mode": _load_ec_meta()["mode"] if ec_dict else "none",
        "results": _available_ec_results(),
    })


@app.route("/api/select_ec_result", methods=['POST'])
def select_ec_result_api():
    payload = request.get_json(silent=True) or {}
    try:
        mode = _normalize_ec_result_mode(payload.get("mode", "none"))
        loaded_ec, meta = _load_ec_result(mode)
        _set_current_ec_result(mode, loaded_ec, meta)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"success": False, "message": str(e)}), 404
    return jsonify({
        "success": True,
        "mode": meta["mode"],
        "mode_label": meta["mode_label"],
        "screen_first": bool(meta["screen_first"]),
        "strengths": sorted(list(ec_dict.keys())),
        "ec_reduction_vs_npi": _mode_reduction_against_npi(meta["mode"]),
    })


@app.route("/api/start_computation", methods=['GET', 'POST'])
def start_computation_api():
    payload = request.get_json(silent=True) or {}
    requested_mode = payload.get("mode", request.args.get("mode", "none"))
    requested_params = payload.get("params", {})
    try:
        mode_key = _normalize_mask_key(requested_mode)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    mode_id = mode_key or "none"
    mode_label = COMPUTE_MODE_LABELS.get(mode_id, DEFAULT_MASK_LABELS.get(mode_id, mode_id))
    screening_steps = 0 if mode_key is None else (len(CONSENSUS_SOURCE_KEYS) if mode_key in {"consensus", "ensemble"} else 1)

    with computation_lock:
        if computation_status["is_computing"]:
            return jsonify({"success": False, "message": "计算中，请等待"})
        computation_status["is_computing"] = True
        computation_status["progress"] = 0
        computation_status["total"] = len(strength_list) + 2 + screening_steps
        computation_status["last_error"] = ""
        computation_status["started_at"] = time.time()
        computation_status["mode"] = mode_id
        computation_status["mode_label"] = mode_label
        computation_status["screen_first"] = mode_key is not None
        computation_status["message"] = f"正在启动计算：{mode_label}"
    try:
        get_npi_module()
    except Exception as e:
        with computation_lock:
            computation_status["is_computing"] = False
            computation_status["last_error"] = str(e)
            computation_status["started_at"] = None
            computation_status["message"] = f"错误：{str(e)}"
        return jsonify({
            "success": False,
            "message": f"无法启动计算：{str(e)}。当前环境缺少训练依赖，请先安装对应包。"
        }), 500
    threading.Thread(
        target=train_and_compute_async,
        args=(mode_key, requested_params if isinstance(requested_params, dict) else {}),
        daemon=True,
    ).start()
    return jsonify({"success": True, "message": f"已启动计算：{mode_label}"})


@app.route("/api/start_all_computation", methods=['POST'])
def start_all_computation_api():
    payload = request.get_json(silent=True) or {}
    requested_params = payload.get("params", {})
    with computation_lock:
        if computation_status["is_computing"]:
            return jsonify({
                "success": False,
                "message": f"已有计算任务进行中：{computation_status.get('message', '')}"
            })
        try:
            get_npi_module()
        except RuntimeError as e:
            computation_status["last_error"] = str(e)
            return jsonify({
                "success": False,
                "message": f"无法启动计算：{str(e)}。当前环境缺少训练依赖，请先安装对应包。"
            }), 500
        computation_status["is_computing"] = True
        computation_status["progress"] = 0
        computation_status["total"] = 1
        computation_status["message"] = "正在启动全部EC结果计算"
        computation_status["last_error"] = ""
        computation_status["started_at"] = time.time()
        computation_status["mode"] = "all"
        computation_status["mode_label"] = "全部EC结果"
        computation_status["screen_first"] = True
        computation_status["all_modes"] = True
    threading.Thread(
        target=train_and_compute_all_async,
        args=(requested_params if isinstance(requested_params, dict) else {},),
        daemon=True,
    ).start()
    return jsonify({"success": True, "message": "已启动全部EC结果计算"})


@app.route("/api/granger_mask", methods=['GET'])
def granger_mask_api():
    """
    GET /api/granger_mask
    Optional query params: max_lag (int), alpha (float).

    - With no params (or on startup probe): returns cached mask status.
    - With params: re-runs compute_granger_mask() on the loaded fMRI signals
      using the provided hyper-parameters. Default Granger is full pairwise;
      max_candidates_per_target is accepted only as an explicit engineering
      throttle.
    """
    max_lag = request.args.get('max_lag', None, type=int)
    alpha   = request.args.get('alpha',   None, type=float)
    min_effect = request.args.get('min_effect', None, type=float)
    max_density = request.args.get('max_density', None, type=float)
    max_candidates_per_target = request.args.get('max_candidates_per_target', None, type=int)

    node_idx = request.args.get('node', None, type=int)
    params = {
        "max_lag": max_lag if max_lag is not None else DEFAULT_MASK_PARAMS["granger"]["max_lag"],
        "alpha": alpha if alpha is not None else DEFAULT_MASK_PARAMS["granger"]["alpha"],
        "min_effect": min_effect if min_effect is not None else DEFAULT_MASK_PARAMS["granger"]["min_effect"],
        "max_density": max_density if max_density is not None else DEFAULT_MASK_PARAMS["granger"]["max_density"],
        "max_candidates_per_target": max_candidates_per_target if max_candidates_per_target is not None else DEFAULT_MASK_PARAMS["granger"]["max_candidates_per_target"],
    }

    mask = _get_mask_by_key("granger")
    if mask is None or not _is_default_params("granger", params):
        return _cached_mask_response("granger", params)

    data = _screening_response_from_mask("granger", mask)
    data["cached"] = True
    data["node_mask"] = None
    if node_idx is not None and 0 <= node_idx < ROI_num:
        data["node_mask"] = mask[node_idx].tolist()
    return jsonify(data)


@app.route("/api/npci_backbone/<s>/<int:node_idx>", methods=['GET'])
def npci_backbone_api(s, node_idx):
    try:
        strength = float(s)
    except Exception:
        return jsonify({"error": "强度必须是数字"}), 400
    if node_idx < 0 or node_idx >= ROI_num:
        return jsonify({"error": "节点无效"}), 400

    params = {
        "horizon": request.args.get("horizon", NPCI_DEFAULT_PARAMS["horizon"], type=int),
        "n_seed_windows": request.args.get("n_seed_windows", NPCI_DEFAULT_PARAMS["n_seed_windows"], type=int),
        "tau": request.args.get("tau", NPCI_DEFAULT_PARAMS["tau"], type=int),
        "max_candidate_nodes": request.args.get("max_candidate_nodes", NPCI_DEFAULT_PARAMS["max_candidate_nodes"], type=int),
        "min_candidate_nodes": request.args.get("min_candidate_nodes", NPCI_DEFAULT_PARAMS["min_candidate_nodes"], type=int),
        "top_k_direct_lagged": request.args.get("top_k_direct_lagged", NPCI_DEFAULT_PARAMS["top_k_direct_lagged"], type=int),
        "top_k_response": request.args.get("top_k_response", NPCI_DEFAULT_PARAMS["top_k_response"], type=int),
        "top_k_fc": request.args.get("top_k_fc", NPCI_DEFAULT_PARAMS["top_k_fc"], type=int),
        "w_lagged": request.args.get("w_lagged", NPCI_DEFAULT_PARAMS["w_lagged"], type=float),
        "w_response": request.args.get("w_response", NPCI_DEFAULT_PARAMS["w_response"], type=float),
        "w_fc": request.args.get("w_fc", NPCI_DEFAULT_PARAMS["w_fc"], type=float),
        "pc_alpha": request.args.get("pc_alpha", NPCI_DEFAULT_PARAMS["pc_alpha"], type=float),
        "max_core_edges": request.args.get("max_core_edges", NPCI_DEFAULT_PARAMS["max_core_edges"], type=int),
        "max_direct_lesion_edges": request.args.get("max_direct_lesion_edges", NPCI_DEFAULT_PARAMS["max_direct_lesion_edges"], type=int),
        "strong_quantile": request.args.get("strong_quantile", NPCI_DEFAULT_PARAMS["strong_quantile"], type=float),
        "causal_weight": request.args.get("causal_weight", NPCI_DEFAULT_PARAMS["causal_weight"], type=float),
        "fc_weight": request.args.get("fc_weight", NPCI_DEFAULT_PARAMS["fc_weight"], type=float),
    }
    top_n = request.args.get("top_n", None, type=int)
    filter_type = request.args.get("filter_type", request.args.get("type", "all"))

    try:
        result = _run_npci_backbone(strength, node_idx, params=params)
        if top_n is not None or str(filter_type).lower() != "all":
            result = _filter_npci_result_view(result, top_n=top_n, filter_type=filter_type)
        return jsonify(_to_plain_python(result))
    except Exception as e:
        return jsonify({"error": f"NPCI运行失败: {str(e)}"}), 500


@app.route("/api/npci_intervention/<s>/<int:node_idx>", methods=['GET'])
def npci_intervention_api(s, node_idx):
    try:
        strength = float(s)
    except Exception:
        return jsonify({"error": "强度必须是数字"}), 400
    if node_idx < 0 or node_idx >= ROI_num:
        return jsonify({"error": "节点无效"}), 400

    backbone_params = _npci_backbone_params_from_request()
    stage3_params = _npci_stage3_params_from_request()
    if request.args.get("cached_only", 0, type=int) == 1:
        cleaned_backbone = _sanitize_npci_params(backbone_params)
        cleaned_stage3 = _sanitize_npci_stage3_params(stage3_params)
        signature = _npci_intervention_signature(strength, node_idx, cleaned_backbone, cleaned_stage3)
        cache_key = signature["digest"]
        cached = npci_intervention_cache.get(cache_key)
        if cached is None:
            cached = _load_npci_intervention_file(signature)
            if cached is not None:
                cached["cache_source"] = "file"
                npci_intervention_cache[cache_key] = copy.deepcopy(cached)
        if cached is None:
            return jsonify({
                "available": False,
                "cached": False,
                "running": npci_precompute_status.get("is_running", False),
                "message": "当前 NPCI C/D 结果尚未保存，请先启动 NPCI 预计算。",
                "cache_signature": signature["digest"],
            }), 202
        cached_result = copy.deepcopy(cached)
        cached_result["cached"] = True
        cached_result["cache_source"] = cached_result.get("cache_source", "memory")
        return jsonify(_to_plain_python(cached_result))

    try:
        result = _run_npci_intervention_recovery(
            strength,
            node_idx,
            backbone_params=backbone_params,
            stage3_params=stage3_params,
        )
        return jsonify(_to_plain_python(result))
    except Exception as e:
        return jsonify({"error": f"NPCI干预/恢复运行失败: {str(e)}"}), 500


@app.route("/api/precompute_npci_intervention/<s>/<int:node_idx>", methods=['POST'])
def precompute_npci_intervention_api(s, node_idx):
    try:
        strength = float(s)
    except Exception:
        return jsonify({"success": False, "message": "强度必须是数字"}), 400
    if node_idx < 0 or node_idx >= ROI_num:
        return jsonify({"success": False, "message": "节点无效"}), 400

    force = request.args.get("force", 0, type=int) == 1
    started = precompute_npci_intervention_async(
        strength,
        node_idx,
        backbone_params=_npci_backbone_params_from_request(),
        stage3_params=_npci_stage3_params_from_request(),
        force=force,
    )
    if not started:
        return jsonify({"success": False, "message": "NPCI预计算正在运行，请等待"})
    return jsonify({
        "success": True,
        "message": "已启动当前病灶节点与强度的 NPCI 预计算",
        "strength": strength,
        "node_idx": node_idx,
        "node_label": f"ROI {node_idx + 1}",
    })


@app.route("/api/precompute_npci_status", methods=['GET'])
def precompute_npci_status_api():
    return jsonify(_npci_precompute_status_snapshot())


@app.route("/api/npci_intervention_latest/<s>/<int:node_idx>", methods=['GET'])
def npci_intervention_latest_api(s, node_idx):
    try:
        strength = float(s)
    except Exception:
        return jsonify({"available": False, "message": "强度必须是数字"}), 400
    status = _npci_precompute_status_snapshot()
    if status.get("is_running"):
        return jsonify({
            "available": False,
            "running": True,
            "message": "NPCI 预计算仍在运行",
        }), 202
    if status.get("last_error"):
        return jsonify({
            "available": False,
            "running": False,
            "message": status.get("last_error"),
        }), 500
    requested_profile = request.args.get("profile", None)
    candidate_paths = []
    if (
        status.get("stage") == "completed"
        and status.get("cache_file")
        and int(status.get("node_idx", -1)) == int(node_idx)
        and float(status.get("strength", np.nan)) == float(strength)
        and (not requested_profile or str(status.get("profile", "")) == str(requested_profile))
    ):
        candidate_paths.append(Path(str(status["cache_file"])))

    current_ec = _npci_ec_signature()
    current_mode_tag = _npci_signature_mode({"ec": current_ec})
    strength_tag = str(float(strength)).replace("-", "m").replace(".", "p")
    node_tag = int(node_idx) + 1
    mode_root = _npci_mode_root({"ec": current_ec})
    candidate_paths.extend(mode_root.glob(f"npci_cd_{current_mode_tag}_s{strength_tag}_roi{node_tag}_*.json"))
    candidate_paths = sorted(
        {str(p): p for p in candidate_paths}.values(),
        key=lambda p: (p if p.is_absolute() else BASE_DIR / p).stat().st_mtime
        if (p if p.is_absolute() else BASE_DIR / p).exists()
        else 0,
        reverse=True,
    )
    if not candidate_paths:
        return jsonify({
            "available": False,
            "running": False,
            "message": "暂无当前病灶节点和强度的 NPCI 保存结果",
        }), 404

    last_error = ""
    for candidate in candidate_paths:
        raw_path = Path(str(candidate))
        path = raw_path if raw_path.is_absolute() else BASE_DIR / raw_path
        try:
            resolved = path.resolve()
            data_root = DATA_DIR.resolve()
            if data_root != resolved and data_root not in resolved.parents:
                raise ValueError("缓存路径不在项目数据目录内")
            with open(resolved, "r", encoding="utf-8") as f:
                payload = json.load(f)
            signature = payload.get("signature") or {}
            if payload.get("cache_version") != NPCI_INTERVENTION_CACHE_VERSION:
                continue
            if not isinstance(payload.get("result"), dict):
                continue
            if float(signature.get("strength", np.nan)) != float(strength):
                continue
            if int(signature.get("lesion_idx", -1)) != int(node_idx):
                continue
            stage3 = signature.get("stage3") if isinstance(signature.get("stage3"), dict) else {}
            if requested_profile and str(stage3.get("profile", "")) != str(requested_profile):
                continue
            sig_ec = signature.get("ec") if isinstance(signature.get("ec"), dict) else {}
            if sig_ec.get("mode", "none") != current_ec.get("mode", "none"):
                continue
            result = _to_plain_python(payload["result"])
            result["available"] = True
            result["cached"] = True
            result["cache_source"] = "latest_precompute"
            result["cache_file"] = _relative_cache_path(resolved)
            result["saved_at"] = payload.get("saved_at")
            return jsonify(_to_plain_python(result))
        except Exception as exc:
            last_error = str(exc)
            continue
    message = "未找到与当前 EC 模式匹配的 NPCI 保存结果"
    if last_error:
        message += f": {last_error}"
    return jsonify({
        "available": False,
        "running": False,
        "message": message,
    }), 404


@app.route("/api/npci_stage3_defaults", methods=['GET'])
def npci_stage3_defaults_api():
    return jsonify(_to_plain_python(_sanitize_npci_stage3_params({})))


def _mask_summary(mask: np.ndarray) -> dict:
    """Return the standard {available, n_rois, n_significant_pairs, n_total_pairs, density} dict."""
    n_total = ROI_num * (ROI_num - 1)
    n_sig   = int(mask.sum())
    return {
        "available": True,
        "n_rois": ROI_num,
        "n_significant_pairs": n_sig,
        "n_total_pairs": n_total,
        "density": round(n_sig / n_total, 4),
    }


def _load_signals() -> list:
    """Re-load raw fMRI signals from disk; returns empty list if unavailable."""
    signals = []
    for f in get_default_fmri_files():
        if os.path.exists(f):
            signals.append(np.loadtxt(f)[30:, :ROI_num])
    return signals


@app.route("/api/var_granger_mask", methods=['GET'])
def var_granger_mask_api():
    """
    GET /api/var_granger_mask?max_lag=8&alpha=0.05
    Multivariate VAR-Granger with BIC-selected lag order per target ROI.
    Controls for indirect influences via third variables.
    """
    max_lag = request.args.get('max_lag', DEFAULT_MASK_PARAMS["vargranger"]["max_lag"], type=int)
    alpha   = request.args.get('alpha',   DEFAULT_MASK_PARAMS["vargranger"]["alpha"], type=float)
    min_effect = request.args.get('min_effect', DEFAULT_MASK_PARAMS["vargranger"]["min_effect"], type=float)
    max_density = request.args.get('max_density', DEFAULT_MASK_PARAMS["vargranger"]["max_density"], type=float)
    params = {"max_lag": max_lag, "alpha": alpha, "min_effect": min_effect, "max_density": max_density}
    return _cached_mask_response("vargranger", params)


@app.route("/api/sparse_var_mask", methods=['GET'])
def sparse_var_mask_api():
    max_lag = request.args.get('max_lag', DEFAULT_MASK_PARAMS["sparsevar"]["max_lag"], type=int)
    max_candidates = request.args.get('max_candidates_per_target', DEFAULT_MASK_PARAMS["sparsevar"]["max_candidates_per_target"], type=int)
    min_effect = request.args.get('min_effect', DEFAULT_MASK_PARAMS["sparsevar"]["min_effect"], type=float)
    max_density = request.args.get('max_density', DEFAULT_MASK_PARAMS["sparsevar"]["max_density"], type=float)
    ridge = request.args.get('ridge', DEFAULT_MASK_PARAMS["sparsevar"]["ridge"], type=float)
    min_session_support_ratio = request.args.get('min_session_support_ratio', DEFAULT_MASK_PARAMS["sparsevar"]["min_session_support_ratio"], type=float)
    params = {
        "max_lag": max_lag,
        "max_candidates_per_target": max_candidates,
        "min_effect": min_effect,
        "max_density": max_density,
        "ridge": ridge,
        "min_session_support_ratio": min_session_support_ratio,
    }
    return _cached_mask_response("sparsevar", params)


@app.route("/api/fask_mask", methods=['GET'])
def fask_mask_api():
    two_cycle_alpha = request.args.get('two_cycle_alpha', DEFAULT_MASK_PARAMS["fask"]["two_cycle_alpha"], type=float)
    penalty_discount = request.args.get('penalty_discount', DEFAULT_MASK_PARAMS["fask"]["penalty_discount"], type=float)
    max_density = request.args.get('max_density', DEFAULT_MASK_PARAMS["fask"]["max_density"], type=float)
    timeout_seconds = request.args.get('timeout_seconds', DEFAULT_MASK_PARAMS["fask"]["timeout_seconds"], type=int)
    params = {
        "two_cycle_alpha": two_cycle_alpha,
        "penalty_discount": penalty_discount,
        "max_density": max_density,
        "timeout_seconds": timeout_seconds,
    }
    pipeline_mode = request.args.get("pipeline_mode", "none")
    return _cached_mask_response("fask", params, pipeline_mode=pipeline_mode)


@app.route("/api/two_step_mask", methods=['GET'])
def two_step_mask_api():
    lambda_value = request.args.get('lambda', DEFAULT_MASK_PARAMS["twostep"]["lambda"], type=float)
    min_effect = request.args.get('min_effect', DEFAULT_MASK_PARAMS["twostep"]["min_effect"], type=float)
    max_density = request.args.get('max_density', DEFAULT_MASK_PARAMS["twostep"]["max_density"], type=float)
    timeout_seconds = request.args.get('timeout_seconds', DEFAULT_MASK_PARAMS["twostep"]["timeout_seconds"], type=int)
    params = {
        "lambda": lambda_value,
        "min_effect": min_effect,
        "max_density": max_density,
        "timeout_seconds": timeout_seconds,
    }
    pipeline_mode = request.args.get("pipeline_mode", "none")
    return _cached_mask_response("twostep", params, pipeline_mode=pipeline_mode)


@app.route("/api/local_fmri_mask/<key>/<s>/<int:node_idx>", methods=['GET', 'POST'])
def local_fmri_mask_api(key, s, node_idx):
    try:
        strength = float(s)
        algo_key = str(key).strip().lower().replace("_local", "")
        max_nodes = request.args.get("max_nodes", 40, type=int)
        force = request.args.get("force", 0, type=int) == 1
        pipeline_mode = _normalize_pipeline_mode(request.args.get("pipeline_mode", "none"))
        result = _compute_local_fmri_mask(
            algo_key,
            strength,
            node_idx,
            max_nodes=max_nodes,
            force=force,
            pipeline_mode=pipeline_mode,
        )
        local_key = f"{algo_key}_local"
        local_label = DEFAULT_MASK_LABELS[local_key]
        data = _mask_summary(result["mask"])
        data.update({
            "available": True,
            "running": False,
            "cached": bool(result["cached"]),
            "key": local_key,
            "source_algorithm": algo_key,
            "label": local_label,
            "roi_indices": result["roi_indices"],
            "roi_labels": [f"ROI {int(i) + 1}" for i in result["roi_indices"]],
            "meta": result["meta"],
            "cache_file": _relative_cache_path(result["data_path"]),
            "meta_file": _relative_cache_path(result["meta_path"]),
            "elapsed_seconds": result["meta"].get("elapsed_seconds"),
            "message": f"{local_label} 局部结果已{'读取' if result['cached'] else '计算并保存'}",
        })
        return jsonify(data)
    except Exception as e:
        return jsonify({
            "available": False,
            "running": False,
            "cached": False,
            "message": str(e),
        }), 500


@app.route("/api/consensus_mask", methods=['GET'])
def consensus_mask_api():
    mask, meta = _compute_consensus_mask()
    if len(meta["sources"]) < len(CONSENSUS_SOURCE_KEYS):
        return jsonify({
            "available": False,
            "running": causal_precompute_status["is_running"],
            "cached": False,
            "message": "Consensus需要全部基础证据源缓存，请先预计算基础算法。",
            "sources": meta["sources"],
            "required_sources": CONSENSUS_SOURCE_KEYS,
        })
    data = _mask_summary(mask)
    data.update({
        "cached": True,
        "running": False,
        "message": "Consensus综合筛选结果已就绪",
        "sources": meta["sources"],
        "min_score": meta["min_score"],
        "min_support": meta["min_support"],
    })
    return jsonify(data)


@app.route("/api/ensemble_mask", methods=['GET'])
def ensemble_mask_api():
    mask, meta = _compute_intervention_ensemble_mask()
    if not ec_dict or len(meta["sources"]) < len(CONSENSUS_SOURCE_KEYS):
        return jsonify({
            "available": False,
            "running": causal_precompute_status["is_running"],
            "cached": False,
            "message": "NPI-Ensemble需要NPI EC结果和全部基础因果算法缓存，请先加载数据并预计算基础算法。",
            "sources": meta["sources"],
            "required_sources": CONSENSUS_SOURCE_KEYS,
        })
    data = _mask_summary(mask)
    data.update({
        "cached": True,
        "running": False,
        "message": "NPI-Ensemble定向集成筛选结果已就绪",
        "sources": meta["sources"],
        "weights": meta["weights"],
        "min_score": meta["min_score"],
        "min_npi_score": meta["min_npi_score"],
        "min_support": meta["min_support"],
        "max_density": meta["max_density"],
    })
    return jsonify(data)


@app.route("/api/algorithm_reliability", methods=['GET'])
def algorithm_reliability_api():
    report = _algorithm_reliability_report()
    if not report["available"]:
        report.update({
            "running": causal_precompute_status["is_running"],
            "message": "算法可信度评估需要NPI EC结果和全部基础因果算法缓存。",
        })
    else:
        report.update({
            "running": False,
            "message": "算法可信度代理评估已就绪。",
        })
    return jsonify(report)


@app.route("/api/pcmci_mask", methods=['GET'])
def pcmci_mask_api():
    """
    GET /api/pcmci_mask?max_lag=2&alpha=0.05&full_tigramite=0
    Candidate-limited PCMCI screen for full-brain interactive use.
    full_tigramite=1 is kept only for small/local subnetworks and is not
    used by the frontend default path.
    """
    max_lag = request.args.get('max_lag', DEFAULT_MASK_PARAMS["pcmci"]["max_lag"], type=int)
    alpha   = request.args.get('alpha',   DEFAULT_MASK_PARAMS["pcmci"]["alpha"], type=float)
    min_effect = request.args.get('min_effect', DEFAULT_MASK_PARAMS["pcmci"]["min_effect"], type=float)
    max_density = request.args.get('max_density', DEFAULT_MASK_PARAMS["pcmci"]["max_density"], type=float)
    full_tigramite = request.args.get('full_tigramite', "0")
    params = {
        "max_lag": max_lag,
        "alpha": alpha,
        "min_effect": min_effect,
        "max_density": max_density,
        "full_tigramite": full_tigramite,
    }
    return _cached_mask_response("pcmci", params)


@app.route("/api/brain_model", methods=['GET'])
def brain_model_api():
    return jsonify({
        "verts": brain_verts.round(2).tolist(),
        "faces": brain_faces.tolist(),
        "roi_vox": roi_vox.tolist()
    })


@app.route("/api/precompute_causal_masks", methods=['GET', 'POST'])
def precompute_causal_masks_api():
    force = request.args.get('force', 0, type=int) == 1
    group = request.args.get("group", "all")
    pipeline_mode = _normalize_pipeline_mode(request.args.get("pipeline_mode", "none"))
    payload = request.get_json(silent=True) or {}
    try:
        target_keys = _resolve_precompute_keys(
            group=group,
            keys=payload.get("keys") if isinstance(payload.get("keys"), list) else None,
        )
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    if not precompute_default_causal_masks_async(force=force, keys=target_keys, pipeline_mode=pipeline_mode):
        return jsonify({"success": False, "message": "因果算法预计算中，请等待"})
    return jsonify({
        "success": True,
        "message": "已启动因果算法默认结果预计算",
        "group": group,
        "pipeline_mode": pipeline_mode,
        "expected_keys": target_keys,
    })


@app.route("/api/precompute_causal_status", methods=['GET'])
def precompute_causal_status_api():
    requested_mode = request.args.get("pipeline_mode")
    if requested_mode is not None and not causal_precompute_status.get("is_running"):
        causal_precompute_status["pipeline_mode"] = _normalize_pipeline_mode(requested_mode)
    return jsonify(_precompute_status_snapshot())


@app.route("/api/cancel_causal_precompute", methods=['POST'])
def cancel_causal_precompute_api():
    if not causal_precompute_status.get("is_running"):
        return jsonify({
            "success": True,
            "message": "当前没有正在运行的因果算法预计算",
            "killed_external_process": False,
        })
    causal_precompute_cancel_event.set()
    killed = False
    with external_causal_proc_lock:
        proc = external_causal_current_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                killed = True
            except Exception:
                killed = False
    _update_precompute_status(
        is_running=False,
        current_key="",
        current_label="",
        current_started_at=None,
        message="因果算法预计算已取消",
        last_error="用户已取消因果算法预计算",
        completed_at=time.time(),
        last_heartbeat_at=time.time(),
        computed_keys=_collect_cached_keys(
            pipeline_mode=_normalize_pipeline_mode(causal_precompute_status.get("pipeline_mode", "none"))
        ),
    )
    return jsonify({
        "success": True,
        "message": "已请求停止因果算法预计算",
        "killed_external_process": killed,
    })


@app.route("/api/precomputed_causal_mask/<key>", methods=['GET'])
def precomputed_causal_mask_api(key):
    try:
        mask_key = _normalize_mask_key(key)
    except ValueError as e:
        return jsonify({"available": False, "message": str(e)}), 400
    if mask_key is None or mask_key not in PRECOMPUTE_MASK_KEYS:
        return jsonify({
            "available": False,
            "message": "This algorithm is not part of the default precompute cache.",
        }), 404

    pipeline_mode = _normalize_pipeline_mode(request.args.get("pipeline_mode", "none"))
    context_mode = pipeline_mode if mask_key in FMRI_NATIVE_SCREEN_KEYS else None
    if mask_key in FMRI_NATIVE_SCREEN_KEYS:
        mask_ready = (
            _get_mask_by_key(mask_key) is not None
            and _full_fmri_memory_context_matches(mask_key, pipeline_mode)
        ) or _load_single_cached_mask(mask_key, pipeline_mode=context_mode)
    else:
        mask_ready = _get_mask_by_key(mask_key) is not None or _load_single_cached_mask(mask_key)
    if not mask_ready:
        return jsonify({
            "available": False,
            "cached": False,
            "key": mask_key,
            "message": "No precomputed result file was found for this algorithm.",
        }), 404

    data_path, meta_path = _mask_result_paths(mask_key, pipeline_mode=context_mode)
    data = _screening_response_from_mask(mask_key, _get_mask_by_key(mask_key))
    data.update({
        "available": True,
        "cached": True,
        "key": mask_key,
        "label": DEFAULT_MASK_LABELS.get(mask_key, mask_key),
        "cache_file": _relative_cache_path(data_path),
        "meta_file": _relative_cache_path(meta_path),
    })
    if mask_key in FMRI_NATIVE_SCREEN_KEYS:
        data["meta"] = full_fmri_mask_meta.get(mask_key, {})
    return jsonify(data)


# ====================== 自动组网（POST，最前面） ======================
@app.route('/api/network/<s>', methods=['POST'])
def network_api(s):
    try:
        strength = float(s)
    except:
        return jsonify({"error": "强度必须是数字"}), 400

    if strength not in ec_dict:
        return jsonify({"error": f"强度{strength}未计算，可用: {list(ec_dict.keys())}"}), 404

    data = request.get_json()
    if not data or 'node_indices' not in data:
        return jsonify({"error": "请提供node_indices参数"}), 400

    node_indices = data['node_indices']
    threshold = data.get('threshold', 0.01)
    top_n_per_node = data.get('top_n_per_node', None)
    root_node_idx = data.get('root_node_idx', None)
    direction_mode = data.get('direction_mode', 'root_outgoing' if root_node_idx is not None else 'dominant_pairs')
    preserve_displayed_edges = bool(data.get('preserve_displayed_edges', False))
    try:
        mask_key = _normalize_mask_key(data.get("mask_key"))
        ec_matrix, mask_info = _apply_mask_to_matrix(ec_dict[strength], mask_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if root_node_idx is not None:
        try:
            root_node_idx = int(root_node_idx)
        except (TypeError, ValueError):
            return jsonify({"error": "root_node_idx必须是整数"}), 400
        if root_node_idx < 0 or root_node_idx >= ROI_num:
            return jsonify({"error": f"root_node_idx无效，范围0-{ROI_num - 1}"}), 400

    for idx in node_indices:
        if idx < 0 or idx >= ROI_num:
            return jsonify({"error": f"节点{idx}无效，范围0-{ROI_num - 1}"}), 400

    if direction_mode not in {"root_outgoing", "dominant_pairs", "all_pairs"}:
        return jsonify({"error": "direction_mode仅支持root_outgoing、dominant_pairs或all_pairs"}), 400
    if direction_mode == "root_outgoing" and root_node_idx is None:
        return jsonify({"error": "root_outgoing模式需要root_node_idx"}), 400

    original_ec_matrix = ec_dict[strength]
    nodes_set = set()
    edges = []
    edges_set = set()
    original_edge_count = 0

    edge_threshold = 0.0 if (direction_mode == "root_outgoing" and preserve_displayed_edges) else float(threshold)

    def make_edge(source_idx: int, target_idx: int, ec_value: float) -> dict:
        return {
            "source": int(source_idx),
            "target": int(target_idx),
            "ec": float(ec_value),
            "abs_ec": abs(float(ec_value)),
            "type": "positive" if ec_value > 0 else "negative",
            "source_coord": roi_vox[source_idx].tolist(),
            "target_coord": roi_vox[target_idx].tolist(),
            "evidence": _edge_evidence(int(source_idx), int(target_idx)),
        }

    if direction_mode == "dominant_pairs":
        nodes_set.update(int(idx) for idx in node_indices)
        node_list = [int(idx) for idx in node_indices]
        for pos, source_a in enumerate(node_list):
            for source_b in node_list[pos + 1:]:
                if source_a == source_b:
                    continue

                original_ab = original_ec_matrix[source_a][source_b]
                original_ba = original_ec_matrix[source_b][source_a]
                if abs(original_ab) >= edge_threshold:
                    original_edge_count += 1
                if abs(original_ba) >= edge_threshold:
                    original_edge_count += 1

                candidates = []
                ec_ab = ec_matrix[source_a][source_b]
                ec_ba = ec_matrix[source_b][source_a]
                if abs(ec_ab) >= edge_threshold:
                    candidates.append(make_edge(source_a, source_b, ec_ab))
                if abs(ec_ba) >= edge_threshold:
                    candidates.append(make_edge(source_b, source_a, ec_ba))
                if not candidates:
                    continue

                candidates.sort(key=lambda x: x["abs_ec"], reverse=True)
                edge = candidates[0]
                edge_key = (edge["source"], edge["target"])
                if edge_key not in edges_set:
                    edges_set.add(edge_key)
                    edges.append(edge)

    source_indices = [root_node_idx] if direction_mode == "root_outgoing" else node_indices
    target_indices = node_indices

    if direction_mode != "dominant_pairs":
        for source_idx in source_indices:
            nodes_set.add(source_idx)
            ec_row = ec_matrix[source_idx]
            original_row = original_ec_matrix[source_idx]

            candidate_edges = []
            for target_idx in target_indices:
                if target_idx == source_idx:
                    continue
                if abs(original_row[target_idx]) >= edge_threshold:
                    original_edge_count += 1
                ec_value = ec_row[target_idx]
                if abs(ec_value) >= edge_threshold:
                    candidate_edges.append(make_edge(int(source_idx), int(target_idx), ec_value))
                    nodes_set.add(target_idx)

            if top_n_per_node is not None and len(candidate_edges) > top_n_per_node:
                candidate_edges.sort(key=lambda x: x['abs_ec'], reverse=True)
                candidate_edges = candidate_edges[:top_n_per_node]

            for edge in candidate_edges:
                edge_key = (edge['source'], edge['target'])
                if edge_key not in edges_set:
                    edges_set.add(edge_key)
                    edges.append(edge)

    nodes = []
    for idx in sorted(nodes_set):
        nodes.append({
            "label": f"ROI {idx + 1}",
            "index": int(idx),
            "coord": roi_vox[idx].tolist(),
            "out_degree": sum(1 for e in edges if e['source'] == idx),
            "in_degree": sum(1 for e in edges if e['target'] == idx),
            "total_degree": sum(1 for e in edges if e['source'] == idx or e['target'] == idx)
        })

    edges_positive = []
    edges_negative = []
    for edge in edges:
        if edge['type'] == 'positive':
            edges_positive.append({
                "from": edge['source_coord'],
                "to": edge['target_coord'],
                "source_label": f"ROI {edge['source'] + 1}",
                "target_label": f"ROI {edge['target'] + 1}",
                "strength": edge['ec'],
                "evidence": edge["evidence"],
            })
        else:
            edges_negative.append({
                "from": edge['source_coord'],
                "to": edge['target_coord'],
                "source_label": f"ROI {edge['source'] + 1}",
                "target_label": f"ROI {edge['target'] + 1}",
                "strength": edge['ec'],
                "evidence": edge["evidence"],
            })

    all_ec_values = [abs(e['ec']) for e in edges]
    reduction = _reduction_stats(original_edge_count, len(edges))

    return jsonify({
        "strength": strength,
        "mask": mask_info,
        "requested_nodes": node_indices,
        "root_node_idx": root_node_idx,
        "direction_mode": direction_mode,
        "preserve_displayed_edges": preserve_displayed_edges,
        "nodes": nodes,
        "edges": edges,
        "edges_positive": edges_positive,
        "edges_negative": edges_negative,
        "stats": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "num_positive_edges": len(edges_positive),
            "num_negative_edges": len(edges_negative),
            "avg_abs_ec": round(float(np.mean(all_ec_values)), 4) if all_ec_values else 0,
            "max_abs_ec": round(float(max(all_ec_values)), 4) if all_ec_values else 0,
            "min_abs_ec": round(float(min(all_ec_values)), 4) if all_ec_values else 0,
            "network_density": round(len(edges) / (len(nodes) * (len(nodes) - 1)), 4) if len(nodes) > 1 else 0,
            "evidence_summary": _evidence_summary_for_edges([(e["source"], e["target"]) for e in edges]),
            "mode_reduction_vs_npi": _mode_reduction_against_npi(),
            **reduction,
        }
    })


# ====================== 最强节点（/top路由，放通用路由前） ======================
@app.route('/api/causal_backbone/<s>/<int:node_idx>/top', methods=['GET'])
def causal_backbone_top_api(s, node_idx):
    try:
        strength = float(s)
    except:
        return jsonify({"error": "强度必须是数字"}), 400

    if strength not in ec_dict:
        return jsonify({"error": f"强度{strength}未计算"}), 404
    if node_idx < 0 or node_idx >= ROI_num:
        return jsonify({"error": f"节点{node_idx}无效"}), 400

    top_n = request.args.get('top_n', 5, type=int)
    filter_type = request.args.get('type', 'all')
    try:
        mask_key = _normalize_mask_key(request.args.get("mask_key"))
        _, mask_info = _apply_mask_to_matrix(ec_dict[strength], mask_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    original_ec = ec_dict[strength][node_idx].copy()
    ec = original_ec.copy()
    pert_coord = roi_vox[node_idx].tolist()

    if filter_type == 'positive':
        original_ec[original_ec < 0] = 0
        ec[ec < 0] = 0
    elif filter_type == 'negative':
        original_ec[original_ec > 0] = 0
        ec[ec > 0] = 0

    if mask_key is not None:
        ec = ec * _get_mask_by_key(mask_key)[node_idx].astype(ec.dtype)

    abs_ec = np.abs(ec)
    abs_ec[node_idx] = -1
    top_indices = np.argsort(abs_ec)[-top_n:][::-1]
    original_abs_ec = np.abs(original_ec)
    original_abs_ec[node_idx] = -1
    original_top_indices = np.argsort(original_abs_ec)[-top_n:][::-1]
    original_count = int(np.count_nonzero(original_abs_ec[original_top_indices] >= 1e-6))

    nodes = []
    edges_positive = []
    edges_negative = []

    for idx in top_indices:
        v = ec[idx]
        if abs(v) < 1e-6:
            continue

        coord = roi_vox[idx].tolist()
        evidence = _edge_evidence(node_idx, int(idx))
        nodes.append({
            "label": f"ROI {idx + 1}",
            "index": int(idx),
            "ec": round(float(v), 4),
            "abs_ec": round(float(abs(v)), 4),
            "type": "positive" if v > 0 else "negative",
            "coord": coord,
            "evidence": evidence,
        })

        if v > 0:
            edges_positive.append({
                "from": pert_coord,
                "to": coord,
                "strength": round(float(v), 4),
                "evidence": evidence,
            })
        else:
            edges_negative.append({
                "from": pert_coord,
                "to": coord,
                "strength": round(float(abs(v)), 4),
                "evidence": evidence,
            })

    displayed_pairs = [(node_idx, n["index"]) for n in nodes]
    return jsonify({
        "strength": strength,
        "mask": mask_info,
        "pert_node_label": f"ROI {node_idx + 1}",
        "pert_coord": pert_coord,
        "nodes": nodes,
        "edges_positive": edges_positive,
        "edges_negative": edges_negative,
        "stats": {
            "top_n": len(nodes),
            "num_positive": len(edges_positive),
            "num_negative": len(edges_negative),
            "max_abs_ec": round(float(np.max(np.abs(ec[top_indices]))), 4) if len(top_indices) > 0 else 0,
            "min_abs_ec": round(float(np.min(np.abs(ec[top_indices]))), 4) if len(top_indices) > 0 else 0,
            "evidence_summary": _evidence_summary_for_edges(displayed_pairs),
            "mode_reduction_vs_npi": _mode_reduction_against_npi(),
            **_reduction_stats(original_count, len(edges_positive) + len(edges_negative)),
        }
    })


# ====================== 因果骨架（通用路由，放最后） ======================
@app.route('/api/causal_backbone/<s>/<int:node_idx>', methods=['GET'])
def causal_backbone_api(s, node_idx):
    try:
        strength = float(s)
    except:
        return jsonify({"error": "强度必须是数字"}), 400

    if strength not in ec_dict:
        return jsonify({"error": f"强度{strength}未计算"}), 404
    if node_idx < 0 or node_idx >= ROI_num:
        return jsonify({"error": "节点无效"}), 400

    top_n = request.args.get('top_n', None, type=int)
    threshold = request.args.get('threshold', 0.01, type=float)
    filter_type = request.args.get('filter_type', 'all')
    try:
        mask_key = _normalize_mask_key(request.args.get("mask_key"))
        _, mask_info = _apply_mask_to_matrix(ec_dict[strength], mask_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    original_ec = ec_dict[strength][node_idx].copy()
    ec = original_ec.copy()
    pert_coord = roi_vox[node_idx].tolist()

    if filter_type == 'positive':
        original_ec[original_ec < 0] = 0
        ec[ec < 0] = 0
    elif filter_type == 'negative':
        original_ec[original_ec > 0] = 0
        ec[ec > 0] = 0

    original_ec[np.abs(original_ec) < threshold] = 0
    ec[np.abs(ec) < threshold] = 0
    if mask_key is not None:
        ec = ec * _get_mask_by_key(mask_key)[node_idx].astype(ec.dtype)

    if top_n is not None:
        original_abs_ec = np.abs(original_ec)
        original_abs_ec[node_idx] = -1
        original_top_indices = np.argsort(original_abs_ec)[-top_n:][::-1]
        original_top_mask = np.zeros_like(original_ec)
        original_top_mask[original_top_indices] = 1
        original_ec = original_ec * original_top_mask

        abs_ec = np.abs(ec)
        abs_ec[node_idx] = -1
        top_indices = np.argsort(abs_ec)[-top_n:][::-1]
        mask = np.zeros_like(ec)
        mask[top_indices] = 1
        ec = ec * mask

    nodes = []
    edges_pos = []
    edges_neg = []

    for i in range(ROI_num):
        v = ec[i]
        if abs(v) < 1e-6:
            continue
        coord = roi_vox[i].tolist()
        evidence = _edge_evidence(node_idx, i)
        nodes.append({
            "label": f"ROI {i + 1}",
            "index": int(i),
            "ec": round(float(v), 4),
            "type": "positive" if v > 0 else "negative",
            "coord": coord,
            "evidence": evidence,
        })
        if v > 0:
            edges_pos.append({
                "from": pert_coord,
                "to": coord,
                "strength": round(float(v), 4),
                "evidence": evidence,
            })
        else:
            edges_neg.append({
                "from": pert_coord,
                "to": coord,
                "strength": round(float(abs(v)), 4),
                "evidence": evidence,
            })

    pos_count = sum(1 for n in nodes if n["ec"] > 0)
    neg_count = len(nodes) - pos_count
    vals = [abs(n["ec"]) for n in nodes]
    reduction = _reduction_stats(
        _edge_count(original_ec, exclude_idx=node_idx),
        len(edges_pos) + len(edges_neg),
    )

    return jsonify({
        "strength": strength,
        "mask": mask_info,
        "pert_node_label": f"ROI {node_idx + 1}",
        "pert_coord": pert_coord,
        "nodes": nodes,
        "edges_positive": edges_pos,
        "edges_negative": edges_neg,
        "stats": {
            "num_nodes": len(nodes),
            "num_positive": pos_count,
            "num_negative": neg_count,
            "total_edges": len(edges_pos) + len(edges_neg),
            "max_ec": round(max(vals), 4) if vals else 0,
            "min_ec": round(min(vals), 4) if vals else 0,
            "avg_abs_ec": round(float(np.mean(vals)), 4) if vals else 0,
            "global_response": round(float(np.mean(np.abs(ec_dict[strength][node_idx]))), 3),
            "evidence_summary": _evidence_summary_for_edges([(node_idx, n["index"]) for n in nodes]),
            "mode_reduction_vs_npi": _mode_reduction_against_npi(),
            **reduction,
        }
    })


# ==========================
# 启动服务
# ==========================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 脑扰动可视化系统启动中...")
    print("=" * 50)
    threading.Thread(target=initialize_data, daemon=True).start()
    app.run(
        host=os.getenv("NPI_HOST", "127.0.0.1"),
        port=int(os.getenv("NPI_PORT", "5002")),
        debug=False,
        use_reloader=False,
        threaded=True,
    )
