"""
Multi-lesion ablation study for lesion-response / lagged-causal / FC-pruning pipeline.

Purpose
-------
This script implements the experimental target, ablation variants, and multi-lesion
workflow discussed for the paper:

    Full method
    w/o response prior
    w/o LPCMCI
    Random ranking

Key fairness rule
-----------------
All variants call the same Stage-3 function: `run_single_node_search()`.
Only Stage-2 candidate/core-network construction is changed through `stage2_variant_fn`.

What you need to provide in your notebook
-----------------------------------------
1. `stage1_provider(lesion_idx) -> stage1_res`
   Returns a dict containing at least:
       seed_inputs, healthy_traj, lesion_traj
   and preferably lesion_response_df.

2. `stage2_variant_fn(variant_cfg, lesion_idx, stage1_res, context) -> stage2_res`
   Returns a dict containing at least:
       candidate_nodes
   and optionally:
       lesion_core_edges, candidate_df, screen_res, candidate_universe,
       n_core_edges, n_direct_lesion_edges, mean_abs_fc_core.

3. Existing model/config variables:
       ANN, ROI_num, using_steps, empirical_FC, roi_names, BEST_CONFIG

Recommended lesion list
-----------------------
    lesion_list = [5, 23, 47, 50, 51, 226, 231, 247]
"""

from __future__ import annotations

import os
import copy
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from intervention_optimization import run_single_node_search


# =========================================================
# 0. Experiment target and ablation variant definitions
# =========================================================

EXPERIMENT_TARGET = {
    "name": "multi_lesion_module_ablation",
    "lesion_list": [5, 23, 47, 50, 51, 226, 231, 247],
    "goal": (
        "Evaluate whether lesion-response prior, local LPCMCI, and candidate ranking "
        "affect compact candidate construction, lesion-centered causal-core validity, "
        "and intervention-node consistency across multiple lesion nodes."
    ),
    "fairness_rule": (
        "All variants use the same ANN surrogate, seed windows, healthy/lesion rollouts, "
        "Stage-3 intervention search space, objective function, and cost weights. "
        "Only Stage-2 candidate/core construction is changed."
    ),
}


@dataclass(frozen=True)
class AblationVariant:
    """Configuration for one ablation variant."""
    variant: str
    use_response_prior: bool
    use_lpcmci: bool
    prune_mode: str
    causal_weight: Optional[float]
    fc_weight: Optional[float]
    ranking_mode: str
    is_random: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_ablation_variants(best_config: Optional[Dict[str, Any]] = None) -> List[AblationVariant]:
    """
    Build the recommended 4 internal ablation groups.

    Parameters
    ----------
    best_config:
        BEST_CONFIG from your main experiment. The function will use its causal/fc
        weights if available; otherwise defaults to causal_weight=0.7, fc_weight=0.3.
    """
    best_config = best_config or {}

    def _is_missing_config_value(value):
        """Return True for None/NaN config values."""
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    def _safe_config_float(names, default):
        """Read first non-missing config value and convert to float."""
        for name in names:
            if name in best_config and not _is_missing_config_value(best_config[name]):
                return float(best_config[name])
        return float(default)

    # Full method should inherit the pruning weights selected by the main
    # stage-wise grid search. Support both explicit causal_weight/fc_weight
    # and prune_weight_pair=(causal_weight, fc_weight).
    if (
        "causal_weight" in best_config
        and "fc_weight" in best_config
        and not _is_missing_config_value(best_config["causal_weight"])
        and not _is_missing_config_value(best_config["fc_weight"])
    ):
        full_causal_w = float(best_config["causal_weight"])
        full_fc_w = float(best_config["fc_weight"])
    elif (
        "prune_causal_weight" in best_config
        and "prune_fc_weight" in best_config
        and not _is_missing_config_value(best_config["prune_causal_weight"])
        and not _is_missing_config_value(best_config["prune_fc_weight"])
    ):
        full_causal_w = float(best_config["prune_causal_weight"])
        full_fc_w = float(best_config["prune_fc_weight"])
    elif "prune_weight_pair" in best_config and not _is_missing_config_value(best_config["prune_weight_pair"]):
        full_causal_w, full_fc_w = best_config["prune_weight_pair"]
        full_causal_w = float(full_causal_w)
        full_fc_w = float(full_fc_w)
    else:
        full_causal_w = _safe_config_float(["causal_weight", "prune_causal_weight"], 0.7)
        full_fc_w = _safe_config_float(["fc_weight", "prune_fc_weight"], 0.3)

    return [
        AblationVariant(
            variant="Full method",
            use_response_prior=True,
            use_lpcmci=True,
            prune_mode="causal_fc_fusion",
            causal_weight=full_causal_w,
            fc_weight=full_fc_w,
            ranking_mode="full",
            description="Complete lesion-response + lagged/LPCMCI + causal-FC constrained method.",
        ),
        AblationVariant(
            variant="w/o response prior",
            use_response_prior=False,
            use_lpcmci=True,
            prune_mode="causal_fc_fusion",
            causal_weight=full_causal_w,
            fc_weight=full_fc_w,
            ranking_mode="no_response",
            description="Remove Stage-1 lesion-response score from Stage-2 candidate scoring/ranking.",
        ),
        AblationVariant(
            variant="w/o LPCMCI",
            use_response_prior=True,
            use_lpcmci=False,
            prune_mode="none",
            causal_weight=None,
            fc_weight=None,
            ranking_mode="response_lagged_fc",
            description="Skip local LPCMCI and build candidates from response/lagged/FC evidence only.",
        ),
        AblationVariant(
            variant="Random ranking",
            use_response_prior=True,
            use_lpcmci=True,
            prune_mode="causal_fc_fusion",
            causal_weight=full_causal_w,
            fc_weight=full_fc_w,
            ranking_mode="random",
            is_random=True,
            description="Randomly sample/rank candidate nodes from the same broad candidate universe.",
        ),
    ]

def ablation_design_table(best_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Return a paper-facing table of the ablation design."""
    rows = []
    for v in build_ablation_variants(best_config):
        rows.append({
            "variant": v.variant,
            "Stage1 response prior": "yes" if v.use_response_prior else "no",
            "LPCMCI": "yes" if v.use_lpcmci else "no",
            "FC pruning": v.prune_mode,
            "causal_weight": v.causal_weight,
            "fc_weight": v.fc_weight,
            "candidate ranking": v.ranking_mode,
            "random repeats": "yes" if v.is_random else "no",
            "description": v.description,
        })
    return pd.DataFrame(rows)


# =========================================================
# 1. General helpers
# =========================================================

def unique_nodes(nodes: Iterable[Any], roi_num: int, exclude: Optional[Iterable[int]] = None) -> List[int]:
    """Return valid, unique ROI indices while preserving order."""
    exclude = set(int(x) for x in (exclude or []))
    out, seen = [], set()
    for n in nodes:
        try:
            n = int(n)
        except Exception:
            continue
        if n < 0 or n >= int(roi_num):
            continue
        if n in exclude or n in seen:
            continue
        out.append(n)
        seen.add(n)
    return out


def policy_nodes(policy: Any) -> List[int]:
    """Extract node indices from a Stage-3 policy list."""
    if not isinstance(policy, list):
        return []
    out = []
    for item in policy:
        if isinstance(item, dict) and "node" in item:
            out.append(int(item["node"]))
    return out


def policy_summary(policy: Any) -> List[Dict[str, Any]]:
    """Compact, CSV-friendly policy representation."""
    if not isinstance(policy, list):
        return []
    out = []
    for item in policy:
        if isinstance(item, dict):
            out.append({
                "node": int(item.get("node")) if item.get("node") is not None else None,
                "sign": int(item.get("sign")) if item.get("sign") is not None else None,
                "strength": float(item.get("strength")) if item.get("strength") is not None else None,
                "schedule": str(item.get("schedule")) if item.get("schedule") is not None else None,
            })
    return out


def _safe_float(x: Any, default: float = np.nan) -> float:
    """Convert to float; None or invalid values become default."""
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def metrics_to_fields(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    metrics = metrics or {}
    keys = [
        "objective", "traj_recovery", "fc_recovery",
        "D_lesion_traj", "D_policy_traj", "D_lesion_fc", "D_policy_fc",
        "node_cost", "strength_cost", "temporal_cost",
    ]
    return {k: _safe_float(metrics.get(k, np.nan)) for k in keys}


def map_roi_names(nodes: Any, roi_names: Optional[Sequence[str]]) -> Any:
    if roi_names is None or not isinstance(nodes, list):
        return nodes
    names = []
    for n in nodes:
        try:
            n = int(n)
            names.append(roi_names[n] if n < len(roi_names) else f"ROI_{n}")
        except Exception:
            names.append(str(n))
    return names


def infer_core_stats(stage2_res: Dict[str, Any], lesion_idx: int, empirical_FC: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Infer core-network statistics from stage2_res if available."""
    out = {
        "n_core_edges": np.nan,
        "n_direct_lesion_edges": np.nan,
        "mean_abs_fc_core": np.nan,
    }

    # Prefer explicitly provided values.
    for k in out.keys():
        if k in stage2_res:
            out[k] = stage2_res[k]

    edges = stage2_res.get("lesion_core_edges", None)
    if isinstance(edges, pd.DataFrame) and len(edges) > 0:
        out["n_core_edges"] = len(edges)
        if {"src", "dst"}.issubset(edges.columns):
            src = edges["src"].astype(int)
            dst = edges["dst"].astype(int)
            out["n_direct_lesion_edges"] = int(((src == int(lesion_idx)) | (dst == int(lesion_idx))).sum())

            if empirical_FC is not None:
                try:
                    fc = np.asarray(empirical_FC, dtype=float)
                    vals = []
                    for s, d in zip(src, dst):
                        if 0 <= int(s) < fc.shape[0] and 0 <= int(d) < fc.shape[1]:
                            vals.append(abs(float(fc[int(s), int(d)])))
                    if vals:
                        out["mean_abs_fc_core"] = float(np.mean(vals))
                except Exception:
                    pass

    return out


# =========================================================
# 2. Stage adapters
# =========================================================

def default_stage2_variant_adapter(
    variant_cfg: AblationVariant,
    lesion_idx: int,
    stage1_res: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Adapter template for your Stage-2 implementation.

    Replace this function in your notebook with your actual Stage-2 call.
    The returned dictionary must contain at least:
        candidate_nodes: list[int]

    Recommended optional keys:
        lesion_core_edges, candidate_df, screen_res, candidate_universe,
        n_core_edges, n_direct_lesion_edges, mean_abs_fc_core.

    Example mapping you should implement in your notebook:
        - Full method: normal Stage2 config.
        - w/o response prior: set response weights to 0 / remove response score.
        - w/o LPCMCI: skip LPCMCI and rank by response/lagged/FC only.
        - Causal-only pruning: causal_weight=1.0, fc_weight=0.0.
        - FC-only pruning: causal_weight=0.0, fc_weight=1.0.
        - Random ranking: use Full method Stage2 to obtain candidate_universe.
    """
    raise NotImplementedError(
        "Please provide `stage2_variant_fn`. It should build candidate_nodes/core edges "
        "for the given AblationVariant. See `make_stage2_variant_adapter_template()` below."
    )


def make_stage2_variant_adapter_template() -> str:
    """Return a template string that can be copied into the notebook and filled in."""
    return r'''
def my_stage2_variant_fn(variant_cfg, lesion_idx, stage1_res, context):
    """
    Replace `run_your_stage2_pipeline(...)` with your real Stage2 function.
    It must return candidate_nodes and optional core-network outputs.
    """
    BEST_CONFIG = context["BEST_CONFIG"]
    roi_num = context["roi_num"]
    empirical_FC = context.get("empirical_FC", None)

    # Start from main Stage2 config and only change the ablated module.
    cfg = dict(BEST_CONFIG)

    cfg["use_response_prior"] = variant_cfg.use_response_prior
    cfg["use_lpcmci"] = variant_cfg.use_lpcmci
    cfg["ranking_mode"] = variant_cfg.ranking_mode

    if variant_cfg.causal_weight is not None:
        cfg["causal_weight"] = variant_cfg.causal_weight
    if variant_cfg.fc_weight is not None:
        cfg["fc_weight"] = variant_cfg.fc_weight

    # A typical implementation pattern:
    # if variant_cfg.variant == "w/o response prior":
    #     cfg["w_response"] = 0.0
    #     cfg["w_spread"] = 0.0
    #     cfg["w_damage"] = 0.0
    #
    # if variant_cfg.variant == "w/o LPCMCI":
    #     cfg["run_lpcmci"] = False
    #     cfg["use_causal_core"] = False
    #
    # if variant_cfg.variant == "Causal-only pruning":
    #     cfg["causal_weight"] = 1.0
    #     cfg["fc_weight"] = 0.0
    #
    # if variant_cfg.variant == "FC-only pruning":
    #     cfg["causal_weight"] = 0.0
    #     cfg["fc_weight"] = 1.0

    # ------------------------------------------------------------------
    # Replace this block with your actual Stage2 function call.
    # ------------------------------------------------------------------
    stage2_res = run_your_stage2_pipeline(
        lesion_idx=lesion_idx,
        stage1_res=stage1_res,
        roi_num=roi_num,
        empirical_FC=empirical_FC,
        config=cfg,
    )

    # Required return key.
    candidate_nodes = stage2_res["candidate_nodes"]

    return {
        "candidate_nodes": candidate_nodes,
        "candidate_universe": stage2_res.get("candidate_universe", stage2_res.get("all_candidate_nodes", candidate_nodes)),
        "lesion_core_edges": stage2_res.get("lesion_core_edges", None),
        "candidate_df": stage2_res.get("candidate_df", None),
        "screen_res": stage2_res.get("screen_res", None),
        "n_core_edges": stage2_res.get("n_core_edges", np.nan),
        "n_direct_lesion_edges": stage2_res.get("n_direct_lesion_edges", np.nan),
        "mean_abs_fc_core": stage2_res.get("mean_abs_fc_core", np.nan),
    }
'''

# =========================================================
# 2b. Concrete Stage-2 adapter for the uploaded notebook cells
# =========================================================


def _is_missing_config_value(value: Any) -> bool:
    """Return True for None/NaN config values."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _none_if_missing_config_value(value: Any) -> Any:
    """Convert None/NaN config values to None and keep valid values unchanged."""
    return None if _is_missing_config_value(value) else value


def _stage2_cfg_value(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """
    Read a Stage-2 config value with a safe default.

    If cfg contains key but its value is None/NaN, treat it as missing.
    This avoids float(None) / int(None) errors for ordinary parameters.
    Parameters for which None has semantic meaning, such as alpha_level,
    should be handled explicitly rather than through this helper.
    """
    val = cfg.get(key, default)
    return default if _is_missing_config_value(val) else val


def _as_node_score_df(screen_res: Dict[str, Any]) -> pd.DataFrame:
    if not isinstance(screen_res, dict) or "node_score_df" not in screen_res:
        raise KeyError("screen_res must contain node_score_df")
    df = screen_res["node_score_df"].copy()
    if "node" not in df.columns and "roi" in df.columns:
        df = df.rename(columns={"roi": "node"})
    if "node" not in df.columns:
        raise KeyError("screen_res['node_score_df'] must contain node or roi")
    return df


def build_candidates_from_screening_for_ablation(
    screen_res: Dict[str, Any],
    lesion_idx: int,
    roi_num: int,
    empirical_FC: Optional[np.ndarray] = None,
    roi_names: Optional[Sequence[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    top_m: int = 20,
    use_response_prior: bool = True,
    ranking_mode: str = "response_lagged_fc",
) -> pd.DataFrame:
    """
    Build intervention candidates when LPCMCI is ablated.

    This is used for the `w/o LPCMCI` variant. It ranks selected screening nodes
    using lagged screening score, optional Stage-1 response score, and optional FC.
    """
    cfg = cfg or {}
    df = _as_node_score_df(screen_res)

    if "selected" in df.columns:
        df = df[df["selected"].astype(bool)].copy()

    df = df[df["node"].astype(int) != int(lesion_idx)].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["node", "candidate_score"])

    def _norm(series):
        vals = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if vals.size == 0:
            return vals
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if hi - lo < 1e-12:
            return np.zeros_like(vals, dtype=float)
        return (vals - lo) / (hi - lo)

    # Lagged evidence from screening.
    if "lagged_direct_score" in df.columns:
        lagged_raw = df["lagged_direct_score"]
    elif "final_screen_score" in df.columns:
        lagged_raw = df["final_screen_score"]
    else:
        lagged_raw = pd.Series(np.zeros(len(df)), index=df.index)

    # Stage-1 lesion-response evidence.
    if use_response_prior and "lesion_response_prior" in df.columns:
        response_raw = df["lesion_response_prior"]
    elif use_response_prior and "lesion_response_norm" in df.columns:
        response_raw = df["lesion_response_norm"]
    else:
        response_raw = pd.Series(np.zeros(len(df)), index=df.index)

    # FC evidence. Prefer screening's fc column; otherwise derive from empirical_FC.
    if "fc_lesion_abs" in df.columns:
        fc_raw = df["fc_lesion_abs"]
    elif empirical_FC is not None:
        fc = np.asarray(empirical_FC, dtype=float)
        fc_raw = df["node"].astype(int).map(lambda n: abs(float(fc[int(lesion_idx), int(n)])))
    else:
        fc_raw = pd.Series(np.zeros(len(df)), index=df.index)

    df["lagged_norm_ablation"] = _norm(lagged_raw)
    df["response_norm_ablation"] = _norm(response_raw)
    df["fc_norm_ablation"] = _norm(fc_raw)

    # Use independent ablation weights when provided; otherwise fall back to Stage-2 screening weights.
    w_lagged = float(cfg.get("wo_lpcmci_w_lagged", cfg.get("w_lagged", 1.0)))
    w_response = float(cfg.get("wo_lpcmci_w_response", cfg.get("w_response", 0.6))) if use_response_prior else 0.0
    w_fc = float(cfg.get("wo_lpcmci_w_fc", cfg.get("w_fc", 0.0)))

    if ranking_mode == "no_response":
        w_response = 0.0
    elif ranking_mode == "lagged_only":
        w_response = 0.0
        w_fc = 0.0
    elif ranking_mode == "fc_only":
        w_lagged = 0.0
        w_response = 0.0
        w_fc = 1.0

    df["candidate_score"] = (
        w_lagged * df["lagged_norm_ablation"]
        + w_response * df["response_norm_ablation"]
        + w_fc * df["fc_norm_ablation"]
    )

    keep_cols = [
        "node", "candidate_score",
        "lagged_norm_ablation", "response_norm_ablation", "fc_norm_ablation",
    ]
    for c in ["roi_name", "lagged_role", "final_screen_score", "lagged_direct_score", "lesion_response_prior", "fc_lesion_abs"]:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    out = df[keep_cols].sort_values("candidate_score", ascending=False).reset_index(drop=True)

    if roi_names is not None and "roi_name" not in out.columns:
        out.insert(
            out.columns.get_loc("node") + 1,
            "roi_name",
            out["node"].astype(int).map(lambda i: roi_names[int(i)] if int(i) < len(roi_names) else f"ROI_{int(i)}"),
        )

    return out


def normalize_stage2_output_for_ablation(
    stage2_res: Dict[str, Any],
    lesion_idx: int,
    roi_num: int,
    empirical_FC: Optional[np.ndarray] = None,
    top_m: int = 20,
) -> Dict[str, Any]:
    """Normalize Stage-2 output to the schema expected by the ablation runner."""
    if stage2_res is None:
        raise ValueError("stage2_res is None")

    stage2_res = dict(stage2_res)

    if "candidate_df" in stage2_res and isinstance(stage2_res["candidate_df"], pd.DataFrame):
        cand_df = stage2_res["candidate_df"].copy()
        if "node" not in cand_df.columns and "roi" in cand_df.columns:
            cand_df = cand_df.rename(columns={"roi": "node"})
        if "candidate_score" in cand_df.columns:
            cand_df = cand_df.sort_values("candidate_score", ascending=False).reset_index(drop=True)
        stage2_res["candidate_df"] = cand_df
    else:
        cand_df = None

    if "candidate_nodes" in stage2_res and stage2_res["candidate_nodes"] is not None:
        candidate_nodes = stage2_res["candidate_nodes"]
    elif cand_df is not None and "node" in cand_df.columns:
        candidate_nodes = cand_df["node"].astype(int).head(int(top_m)).tolist()
    else:
        raise ValueError("Stage2 output must contain candidate_nodes or candidate_df with node column")

    stage2_res["candidate_nodes"] = unique_nodes(candidate_nodes, roi_num=roi_num, exclude=[lesion_idx])[: int(top_m)]

    # A broad universe is important for Random ranking. Prefer screened candidate nodes.
    universe = stage2_res.get("candidate_universe", None)
    if universe is None and isinstance(stage2_res.get("screen_res"), dict):
        universe = stage2_res["screen_res"].get("candidate_nodes", None)
    if universe is None and cand_df is not None and "node" in cand_df.columns:
        universe = cand_df["node"].astype(int).tolist()
    if universe is None:
        universe = stage2_res["candidate_nodes"]

    stage2_res["candidate_universe"] = unique_nodes(universe, roi_num=roi_num, exclude=[lesion_idx])

    core_stats = infer_core_stats(stage2_res, lesion_idx=lesion_idx, empirical_FC=empirical_FC)
    stage2_res.update(core_stats)

    return stage2_res


def run_stage2_from_uploaded_pipeline(
    variant_cfg: AblationVariant,
    lesion_idx: int,
    stage1_res: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Concrete Stage-2 implementation matching the user's original notebook cells.

    It uses:
        run_lagged_candidate_screening
        run_candidate_subnetwork_lpcmci
        lpcmci_to_edge_df
        extract_lesion_neighborhood
        prune_edges_by_lpcmci_fc_constraints
        build_intervention_candidates_from_core

    Required in context:
        signals, BEST_CONFIG, roi_num, empirical_FC, top_m.
    """
    from local_lpcmci import (
        run_lagged_candidate_screening,
        run_candidate_subnetwork_lpcmci,
        lpcmci_to_edge_df,
        extract_lesion_neighborhood,
        prune_edges_by_lpcmci_fc_constraints,
        build_intervention_candidates_from_core,
    )

    cfg = dict(context["BEST_CONFIG"])
    roi_num = int(context["roi_num"])
    top_m = int(context.get("top_m", 20))
    empirical_FC = context.get("empirical_FC", None)
    roi_names = context.get("roi_names", None)

    signals = context.get("signals", None)
    if signals is None:
        # Fallback for notebook use: if `signals` exists in globals(), use it.
        signals = globals().get("signals", None)
    if signals is None:
        raise ValueError(
            "signals is required for Stage2. Pass signals=signals to "
            "run_multi_lesion_ablation_study(...), or add it to context."
        )

    # --------------------------------------------------
    # 1. Configure the ablated module
    # --------------------------------------------------
    use_response_prior = bool(variant_cfg.use_response_prior)
    use_lpcmci = bool(variant_cfg.use_lpcmci)

    causal_weight = variant_cfg.causal_weight
    fc_weight = variant_cfg.fc_weight
    if causal_weight is None:
        causal_weight = float(cfg.get("causal_weight", 0.7))
    if fc_weight is None:
        fc_weight = float(cfg.get("fc_weight", 0.3))

    # Stage1 response prior removal means it is removed both from top-k union
    # and from score weighting.
    if use_response_prior:
        lesion_response_prior = stage1_res
        top_k_response = int(_stage2_cfg_value(cfg, "top_k_response", 30))
        w_response = float(_stage2_cfg_value(cfg, "w_response", 0.6))
    else:
        lesion_response_prior = None
        top_k_response = 0
        w_response = 0.0

    # Keep FC out of the main screening stage, matching the original notebook.
    screening_empirical_fc = None
    top_k_fc = int(_stage2_cfg_value(cfg, "top_k_fc", 0))
    w_fc = float(_stage2_cfg_value(cfg, "w_fc", 0.0))

    # If LPCMCI is removed, optional FC evidence can be enabled for fallback ranking.
    # Defaults follow the main config and do not inject arbitrary weights.
    if not use_lpcmci and bool(cfg.get("wo_lpcmci_use_fc_in_screening", False)):
        screening_empirical_fc = empirical_FC
        top_k_fc = int(cfg.get("wo_lpcmci_top_k_fc", top_k_fc))
        w_fc = float(cfg.get("wo_lpcmci_w_fc", w_fc))

    # --------------------------------------------------
    # 2. Whole-brain high-recall lagged screening
    # --------------------------------------------------
    screen_res = run_lagged_candidate_screening(
        signals=signals,
        lesion_idx=lesion_idx,
        roi_names=roi_names,
        lesion_response_prior=lesion_response_prior,
        empirical_FC=screening_empirical_fc,

        tau=int(_stage2_cfg_value(cfg, "tau", 1)),
        max_candidate_nodes=int(_stage2_cfg_value(cfg, "max_candidate_nodes", 80)),
        min_candidate_nodes=int(_stage2_cfg_value(cfg, "min_candidate_nodes", 30)),
        top_k_direct_lagged=int(_stage2_cfg_value(cfg, "top_k_direct_lagged", 40)),
        top_k_response=top_k_response,
        top_k_fc=top_k_fc,

        w_lagged=float(_stage2_cfg_value(cfg, "w_lagged", 1.0)),
        w_response=w_response,
        w_fc=w_fc,

        standardize=_stage2_cfg_value(cfg, "standardize", "auto"),
    )

    candidate_universe = unique_nodes(screen_res.get("candidate_nodes", []), roi_num=roi_num, exclude=[lesion_idx])

    # --------------------------------------------------
    # 3. w/o LPCMCI: build candidate nodes directly from screening table
    # --------------------------------------------------
    if not use_lpcmci:
        candidate_df = build_candidates_from_screening_for_ablation(
            screen_res=screen_res,
            lesion_idx=lesion_idx,
            roi_num=roi_num,
            empirical_FC=empirical_FC,
            roi_names=roi_names,
            cfg=cfg,
            top_m=top_m,
            use_response_prior=use_response_prior,
            ranking_mode=variant_cfg.ranking_mode,
        )
        candidate_nodes = candidate_df["node"].astype(int).head(top_m).tolist() if len(candidate_df) else []

        return normalize_stage2_output_for_ablation({
            "candidate_nodes": candidate_nodes,
            "candidate_universe": candidate_universe,
            "screen_res": screen_res,
            "candidate_df": candidate_df,
            "lesion_core_edges": pd.DataFrame(),
            "local_edge_df": pd.DataFrame(),
            "lesion_sub_edges": pd.DataFrame(),
            "prune_info": {"reason": "w/o LPCMCI; candidates from screening only"},
            "n_core_edges": 0,
            "n_direct_lesion_edges": 0,
            "mean_abs_fc_core": np.nan,
        }, lesion_idx=lesion_idx, roi_num=roi_num, empirical_FC=empirical_FC, top_m=top_m)

    # --------------------------------------------------
    # 4. Local LPCMCI on screened candidate subnetwork
    # --------------------------------------------------
    lpcmci_results, node_map = run_candidate_subnetwork_lpcmci(
        signals=signals,
        candidate_nodes=screen_res["candidate_nodes"],
        tau_min=int(_stage2_cfg_value(cfg, "tau_min", 1)),
        tau_max=int(_stage2_cfg_value(cfg, "tau_max", _stage2_cfg_value(cfg, "tau", 1))),
        pc_alpha=float(_stage2_cfg_value(cfg, "pc_alpha", 0.05)),
        verbosity=int(_stage2_cfg_value(cfg, "verbosity", 0)),
        standardize=_stage2_cfg_value(cfg, "standardize", "auto"),
        max_cond_px=int(_stage2_cfg_value(cfg, "max_cond_px", 1)),
        max_p_global=int(_stage2_cfg_value(cfg, "max_p_global", 2)),
        max_p_non_ancestral=int(_stage2_cfg_value(cfg, "max_p_non_ancestral", 2)),
        max_q_global=int(_stage2_cfg_value(cfg, "max_q_global", 5)),
        max_pds_set=int(_stage2_cfg_value(cfg, "max_pds_set", 8)),
    )

    # In the main grid-search result, alpha_level may be None/NaN by design.
    # This means: do not filter LPCMCI edges at conversion time; apply
    # prune_alpha_level later during FC-constrained pruning.
    alpha_level_cfg = _none_if_missing_config_value(cfg.get("alpha_level", None))
    if alpha_level_cfg is not None:
        alpha_level_cfg = float(alpha_level_cfg)

    local_edge_df = lpcmci_to_edge_df(
        lpcmci_results,
        node_map=node_map,
        include_contemp=bool(_stage2_cfg_value(cfg, "include_contemp", False)),
        lag_only=bool(_stage2_cfg_value(cfg, "lag_only", True)),
        alpha_level=alpha_level_cfg,
    )

    if local_edge_df is None or len(local_edge_df) == 0:
        # Robust fallback: keep experiment running, but mark that no causal core was found.
        candidate_df = build_candidates_from_screening_for_ablation(
            screen_res=screen_res,
            lesion_idx=lesion_idx,
            roi_num=roi_num,
            empirical_FC=empirical_FC,
            roi_names=roi_names,
            cfg=cfg,
            top_m=top_m,
            use_response_prior=use_response_prior,
            ranking_mode="no_lpcmci_fallback",
        )
        candidate_nodes = candidate_df["node"].astype(int).head(top_m).tolist() if len(candidate_df) else []
        return normalize_stage2_output_for_ablation({
            "candidate_nodes": candidate_nodes,
            "candidate_universe": candidate_universe,
            "screen_res": screen_res,
            "candidate_df": candidate_df,
            "lesion_core_edges": pd.DataFrame(),
            "local_edge_df": pd.DataFrame(),
            "lesion_sub_edges": pd.DataFrame(),
            "prune_info": {"reason": "empty LPCMCI edge table; fallback to screening candidates"},
            "n_core_edges": 0,
            "n_direct_lesion_edges": 0,
            "mean_abs_fc_core": np.nan,
        }, lesion_idx=lesion_idx, roi_num=roi_num, empirical_FC=empirical_FC, top_m=top_m)

    lesion_nodes, lesion_sub_edges, sub_info = extract_lesion_neighborhood(
        edge_df=local_edge_df,
        lesion_idx=lesion_idx,
        min_nodes=int(_stage2_cfg_value(cfg, "min_lesion_sub_nodes", 8)),
        max_nodes=int(_stage2_cfg_value(cfg, "max_lesion_sub_nodes", 30)),
        direction=str(_stage2_cfg_value(cfg, "lesion_neighborhood_direction", "both")),
        expand_to_second_order=bool(_stage2_cfg_value(cfg, "expand_to_second_order", True)),
    )

    # --------------------------------------------------
    # 5. FC-constrained / causal-only / FC-only pruning
    # --------------------------------------------------
    lesion_core_edges, prune_info = prune_edges_by_lpcmci_fc_constraints(
        sub_edges=lesion_sub_edges,
        empirical_FC=empirical_FC,
        lesion_idx=lesion_idx,
        max_edges=int(_stage2_cfg_value(cfg, "max_edges", 80)),
        alpha_level=float(_stage2_cfg_value(cfg, "prune_alpha_level", _stage2_cfg_value(cfg, "alpha_level", 0.05))),
        strong_quantile=float(_stage2_cfg_value(cfg, "strong_quantile", 0.85)),
        causal_weight=float(causal_weight),
        fc_weight=float(fc_weight),
        max_direct_lesion_edges=int(_stage2_cfg_value(cfg, "max_direct_lesion_edges", 35)),
    )

    candidate_df = build_intervention_candidates_from_core(
        lesion_core_edges,
        lesion_idx=lesion_idx,
        exclude_lesion=True,
    )

    if roi_names is not None and len(candidate_df) > 0 and "roi_name" not in candidate_df.columns:
        candidate_df.insert(
            candidate_df.columns.get_loc("node") + 1,
            "roi_name",
            candidate_df["node"].astype(int).map(lambda i: roi_names[int(i)] if int(i) < len(roi_names) else f"ROI_{int(i)}"),
        )

    candidate_nodes = candidate_df["node"].astype(int).head(top_m).drop_duplicates().tolist() if len(candidate_df) else []

    # Fallback if the core produces too few/no nodes.
    if len(candidate_nodes) == 0:
        fallback_df = build_candidates_from_screening_for_ablation(
            screen_res=screen_res,
            lesion_idx=lesion_idx,
            roi_num=roi_num,
            empirical_FC=empirical_FC,
            roi_names=roi_names,
            cfg=cfg,
            top_m=top_m,
            use_response_prior=use_response_prior,
            ranking_mode="core_empty_fallback",
        )
        candidate_df = fallback_df
        candidate_nodes = fallback_df["node"].astype(int).head(top_m).tolist() if len(fallback_df) else []

    # More detailed edge counts.
    n_direct_before = np.nan
    if isinstance(lesion_sub_edges, pd.DataFrame) and len(lesion_sub_edges) > 0 and {"src", "dst"}.issubset(lesion_sub_edges.columns):
        n_direct_before = int(
            ((lesion_sub_edges["src"].astype(int) == int(lesion_idx)) |
             (lesion_sub_edges["dst"].astype(int) == int(lesion_idx))).sum()
        )

    direct_core = pd.DataFrame()
    if isinstance(lesion_core_edges, pd.DataFrame) and len(lesion_core_edges) > 0 and {"src", "dst"}.issubset(lesion_core_edges.columns):
        direct_core = lesion_core_edges[
            (lesion_core_edges["src"].astype(int) == int(lesion_idx)) |
            (lesion_core_edges["dst"].astype(int) == int(lesion_idx))
        ].copy()

    return normalize_stage2_output_for_ablation({
        "candidate_nodes": candidate_nodes,
        "candidate_universe": candidate_universe,
        "screen_res": screen_res,
        "lpcmci_results": lpcmci_results,
        "node_map": node_map,
        "local_edge_df": local_edge_df,
        "lesion_nodes": lesion_nodes,
        "lesion_sub_edges": lesion_sub_edges,
        "lesion_core_edges": lesion_core_edges,
        "direct_lesion_edges": direct_core,
        "candidate_df": candidate_df,
        "sub_info": sub_info,
        "prune_info": prune_info,
        "n_core_edges": len(lesion_core_edges) if isinstance(lesion_core_edges, pd.DataFrame) else 0,
        "n_direct_lesion_edges": len(direct_core),
        "n_direct_lesion_edges_before_prune": n_direct_before,
        "mean_abs_fc_core": infer_core_stats({"lesion_core_edges": lesion_core_edges}, lesion_idx, empirical_FC).get("mean_abs_fc_core", np.nan),
    }, lesion_idx=lesion_idx, roi_num=roi_num, empirical_FC=empirical_FC, top_m=top_m)


# Notebook-friendly alias expected by the ablation runner.
my_stage2_variant_fn = run_stage2_from_uploaded_pipeline


def make_stage1_provider_from_inputs(
    model: Any,
    inputs: np.ndarray,
    roi_num: int,
    using_steps: int,
    roi_names: Optional[Sequence[str]],
    best_config: Dict[str, Any],
) -> Callable[[int], Dict[str, Any]]:
    """
    Convenience helper when Stage-1 results for the 8 lesions have not been precomputed.
    It computes `compute_lesion_response(...)` on demand.
    """
    from lesion_response import compute_lesion_response

    cache: Dict[int, Dict[str, Any]] = {}

    def _provider(lesion_idx: int) -> Dict[str, Any]:
        lesion_idx_int = int(lesion_idx)
        if lesion_idx_int not in cache:
            cache[lesion_idx_int] = compute_lesion_response(
                model=model,
                inputs=inputs,
                roi_num=roi_num,
                using_steps=using_steps,
                lesion_idx=lesion_idx_int,
                roi_names=roi_names,
                horizon=int(best_config.get("horizon", 20)),
                n_seed_windows=int(best_config.get("n_seed_windows", best_config.get("n_seed", 64))),
                lesion_strength=float(best_config.get("lesion_strength", 0.5)),
                lesion_mode=str(best_config.get("lesion_mode", "subtract")),
                random_state=int(best_config.get("random_state", 123)),
                batch_size=int(best_config.get("batch_size", 256)),
            )
        return cache[lesion_idx_int]

    return _provider



def get_stage1_for_lesion(
    lesion_idx: int,
    stage1_provider: Union[Callable[[int], Dict[str, Any]], Dict[int, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Get or compute Stage-1 result for a lesion."""
    if callable(stage1_provider):
        stage1_res = stage1_provider(int(lesion_idx))
    elif isinstance(stage1_provider, dict):
        stage1_res = stage1_provider[int(lesion_idx)]
    else:
        raise TypeError("stage1_provider must be a callable or a dict keyed by lesion_idx.")

    required = ["seed_inputs", "healthy_traj", "lesion_traj"]
    missing = [k for k in required if k not in stage1_res]
    if missing:
        raise KeyError(f"stage1_res for lesion {lesion_idx} misses required keys: {missing}")
    return stage1_res


# =========================================================
# 3. Stage-3 consistent evaluation
# =========================================================

def run_stage3_for_candidates(
    model: Any,
    candidate_nodes: Sequence[int],
    stage1_res: Dict[str, Any],
    lesion_idx: int,
    roi_num: int,
    using_steps: int,
    best_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the exact same Stage3 entry point for every ablation variant."""
    candidate_nodes = unique_nodes(candidate_nodes, roi_num=roi_num, exclude=[])
    if len(candidate_nodes) == 0:
        raise ValueError(f"Empty candidate_nodes for lesion {lesion_idx}.")

    horizon = int(best_config.get("horizon", stage1_res["healthy_traj"].shape[1]))

    return run_single_node_search(
        model=model,
        candidate_nodes=candidate_nodes,
        roi_num=roi_num,
        using_steps=using_steps,
        lesion_idx=lesion_idx,

        seed_inputs=stage1_res["seed_inputs"],
        healthy_traj=stage1_res["healthy_traj"],
        lesion_traj=stage1_res["lesion_traj"],

        horizon=horizon,
        lesion_strength=best_config["lesion_strength"],
        lesion_mode=best_config["lesion_mode"],

        signs=best_config["signs"],
        strengths=best_config["strengths"],
        schedules=best_config["schedules"],

        alpha_traj=best_config["alpha_traj"],
        beta_fc=best_config["beta_fc"],
        lambda_node=best_config["lambda_node"],
        lambda_strength=best_config["lambda_strength"],
        lambda_temporal=best_config["lambda_temporal"],

        batch_size=best_config["batch_size"],

        run_greedy=best_config["run_greedy"],
        max_interventions=best_config["max_interventions"],
        top_m_per_step=best_config["top_m_per_step"],
        min_marginal_gain=best_config["min_marginal_gain"],
        force_accept_first=best_config["force_accept_first"],
    )


def extract_best_stage3_result(stage3_res: Dict[str, Any], prefer_multi: bool = True) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract the best policy/metrics from Stage3.

    If prefer_multi=True and greedy returned best_multi_policy/metrics, use multi;
    otherwise fall back to best_single_policy/metrics.
    """
    if prefer_multi and stage3_res.get("best_multi_policy", None) is not None and stage3_res.get("best_multi_metrics", None) is not None:
        return "multi", stage3_res.get("best_multi_policy", []), stage3_res.get("best_multi_metrics", {})
    return "single", stage3_res.get("best_single_policy", []), stage3_res.get("best_single_metrics", {})


def build_ablation_row(
    lesion_idx: int,
    lesion_name: Optional[str],
    variant_cfg: AblationVariant,
    stage2_res: Dict[str, Any],
    stage3_res: Dict[str, Any],
    roi_names: Optional[Sequence[str]] = None,
    repeat_id: Optional[int] = None,
    status: str = "ok",
    prefer_multi: bool = True,
    empirical_FC: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build one output row from Stage2 and Stage3 results."""
    selected_mode, policy, metrics = extract_best_stage3_result(stage3_res, prefer_multi=prefer_multi)
    nodes = policy_nodes(policy)
    core_stats = infer_core_stats(stage2_res, lesion_idx=lesion_idx, empirical_FC=empirical_FC)

    row = {
        "lesion_idx": int(lesion_idx),
        "lesion_name": lesion_name if lesion_name is not None else f"ROI_{int(lesion_idx)}",
        "variant": variant_cfg.variant,
        "repeat_id": repeat_id,
        "selected_mode": selected_mode,
        "status": status,
        "use_response_prior": variant_cfg.use_response_prior,
        "use_lpcmci": variant_cfg.use_lpcmci,
        "prune_mode": variant_cfg.prune_mode,
        "causal_weight": variant_cfg.causal_weight,
        "fc_weight": variant_cfg.fc_weight,
        "ranking_mode": variant_cfg.ranking_mode,
        "candidate_nodes": stage2_res.get("candidate_nodes", []),
        "n_candidate_nodes": len(stage2_res.get("candidate_nodes", [])),
        "policy": policy_summary(policy),
        "policy_nodes": nodes,
        "policy_roi_names": map_roi_names(nodes, roi_names),
        "policy_size": len(nodes),
        "best_node": nodes[0] if nodes else None,
    }
    row.update(core_stats)
    row.update(metrics_to_fields(metrics))
    return row


# =========================================================
# 3b. Random-intervention baseline utilities
# =========================================================

def _as_list_from_config(value: Any, default: Sequence[Any]) -> List[Any]:
    """Convert a config value to a non-empty list."""
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple, np.ndarray)):
        out = list(value)
    else:
        out = [value]
    return out if len(out) > 0 else list(default)


def _sample_random_intervention_unit(
    rng: np.random.Generator,
    roi_num: int,
    lesion_idx: int,
    signs: Sequence[Any],
    strengths: Sequence[Any],
    schedules: Sequence[Any],
    node_pool: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Sample one single-node random intervention unit."""
    if node_pool is None:
        nodes = [i for i in range(int(roi_num)) if int(i) != int(lesion_idx)]
    else:
        nodes = unique_nodes(node_pool, roi_num=roi_num, exclude=[lesion_idx])
    if len(nodes) == 0:
        raise ValueError("Random-intervention baseline has an empty node pool.")

    node = int(rng.choice(nodes))
    sign = int(rng.choice(list(signs)))
    strength = float(rng.choice(list(strengths)))
    schedule = str(rng.choice(list(schedules)))
    return {
        "node": node,
        "sign": sign,
        "strength": strength,
        "schedule": schedule,
        "node_pool_size": len(nodes),
    }


def run_stage3_for_fixed_single_intervention(
    model: Any,
    intervention_unit: Dict[str, Any],
    stage1_res: Dict[str, Any],
    lesion_idx: int,
    roi_num: int,
    using_steps: int,
    best_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate one fixed single-node intervention using the same Stage-3 evaluator.

    This is not a Stage-3 search: candidate_nodes, signs, strengths, and schedules
    are all singletons, and greedy expansion is disabled. The call reuses the same
    rollout and metric computation as the optimized methods.
    """
    fixed_config = dict(best_config)
    fixed_config["signs"] = [int(intervention_unit["sign"])]
    fixed_config["strengths"] = [float(intervention_unit["strength"])]
    fixed_config["schedules"] = [str(intervention_unit["schedule"])]
    fixed_config["run_greedy"] = False
    fixed_config["max_interventions"] = 1
    fixed_config["top_m_per_step"] = 1
    fixed_config["force_accept_first"] = True

    return run_stage3_for_candidates(
        model=model,
        candidate_nodes=[int(intervention_unit["node"])],
        stage1_res=stage1_res,
        lesion_idx=lesion_idx,
        roi_num=roi_num,
        using_steps=using_steps,
        best_config=fixed_config,
    )


def run_random_intervention_baseline_for_lesion(
    model: Any,
    lesion_idx: int,
    lesion_name: Optional[str],
    roi_num: int,
    using_steps: int,
    stage1_res: Dict[str, Any],
    best_config: Dict[str, Any],
    roi_names: Optional[Sequence[str]] = None,
    empirical_FC: Optional[np.ndarray] = None,
    repeats: int = 50,
    random_state: int = 123,
    node_pool: Optional[Sequence[int]] = None,
    signs: Optional[Sequence[Any]] = None,
    strengths: Optional[Sequence[Any]] = None,
    schedules: Optional[Sequence[Any]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """
    Run the direct Random-intervention baseline for one lesion node.

    Each repeat samples a node, sign, strength, and temporal schedule uniformly
    from the specified pools and evaluates that single fixed policy. The per-lesion
    output reports the mean performance over repeats.
    """
    variant_cfg = AblationVariant(
        variant="Random-intervention baseline",
        use_response_prior=False,
        use_lpcmci=False,
        prune_mode="random_intervention",
        causal_weight=None,
        fc_weight=None,
        ranking_mode="random_policy",
        is_random=True,
        description="Direct random intervention without candidate ranking or Stage-3 optimization.",
    )

    signs = _as_list_from_config(signs, best_config.get("signs", [-1, 1]))
    strengths = _as_list_from_config(strengths, best_config.get("strengths", [0.2]))
    schedules = _as_list_from_config(schedules, best_config.get("schedules", ["constant"]))

    rng = np.random.default_rng(int(random_state) + int(lesion_idx) * 1009)
    repeat_rows: List[Dict[str, Any]] = []
    raw_repeats: Dict[int, Any] = {}

    for repeat_id in range(1, int(repeats) + 1):
        unit = _sample_random_intervention_unit(
            rng=rng,
            roi_num=roi_num,
            lesion_idx=lesion_idx,
            signs=signs,
            strengths=strengths,
            schedules=schedules,
            node_pool=node_pool,
        )
        stage2_res = {
            "candidate_nodes": [int(unit["node"])],
            "candidate_universe": unique_nodes(node_pool if node_pool is not None else range(int(roi_num)), roi_num=roi_num, exclude=[lesion_idx]),
            "candidate_df": pd.DataFrame([{
                "rank": 1,
                "node": int(unit["node"]),
                "roi_name": roi_names[int(unit["node"])] if roi_names is not None and int(unit["node"]) < len(roi_names) else f"ROI_{int(unit['node'])}",
                "candidate_score": np.nan,
                "random_sign": int(unit["sign"]),
                "random_strength": float(unit["strength"]),
                "random_schedule": str(unit["schedule"]),
            }]),
            "lesion_core_edges": pd.DataFrame(),
            "n_core_edges": 0,
            "n_direct_lesion_edges": 0,
            "mean_abs_fc_core": np.nan,
        }
        stage3_res = run_stage3_for_fixed_single_intervention(
            model=model,
            intervention_unit=unit,
            stage1_res=stage1_res,
            lesion_idx=lesion_idx,
            roi_num=roi_num,
            using_steps=using_steps,
            best_config=best_config,
        )
        row = build_ablation_row(
            lesion_idx=lesion_idx,
            lesion_name=lesion_name,
            variant_cfg=variant_cfg,
            stage2_res=stage2_res,
            stage3_res=stage3_res,
            roi_names=roi_names,
            repeat_id=repeat_id,
            status="ok",
            prefer_multi=False,
            empirical_FC=empirical_FC,
        )
        row.update({
            "random_node": int(unit["node"]),
            "random_sign": int(unit["sign"]),
            "random_strength": float(unit["strength"]),
            "random_schedule": str(unit["schedule"]),
            "random_node_pool_size": int(unit["node_pool_size"]),
        })
        repeat_rows.append(row)
        raw_repeats[repeat_id] = {"unit": unit, "stage3_res": stage3_res}

    repeat_df = pd.DataFrame(repeat_rows)

    agg_row: Dict[str, Any] = {
        "lesion_idx": int(lesion_idx),
        "lesion_name": lesion_name if lesion_name is not None else f"ROI_{int(lesion_idx)}",
        "variant": "Random-intervention baseline",
        "repeat_id": None,
        "selected_mode": "random_repeat_mean",
        "status": "ok",
        "use_response_prior": False,
        "use_lpcmci": False,
        "prune_mode": "random_intervention",
        "causal_weight": None,
        "fc_weight": None,
        "ranking_mode": "random_policy",
        "candidate_nodes": "random_intervention_repeats",
        "n_candidate_nodes": 1.0,
        "policy": "random_intervention_repeats",
        "policy_nodes": "random_intervention_repeats",
        "policy_roi_names": "random_intervention_repeats",
        "policy_size": 1.0,
        "best_node": None,
        "n_core_edges": 0,
        "n_direct_lesion_edges": 0,
        "mean_abs_fc_core": np.nan,
        "random_intervention_repeats": int(repeats),
        "random_node_pool_size": int(repeat_df["random_node_pool_size"].iloc[0]) if len(repeat_df) else np.nan,
    }

    metric_cols = [
        "objective", "traj_recovery", "fc_recovery",
        "D_lesion_traj", "D_policy_traj", "D_lesion_fc", "D_policy_fc",
        "node_cost", "strength_cost", "temporal_cost",
    ]
    for m in metric_cols:
        if m in repeat_df.columns:
            vals = pd.to_numeric(repeat_df[m], errors="coerce")
            agg_row[m] = float(vals.mean())
            agg_row[f"{m}_std_within_lesion"] = float(vals.std())

    raw = {"repeat_outputs": raw_repeats}
    return agg_row, repeat_df, raw


# =========================================================
# 4. Multi-lesion ablation workflow
# =========================================================

def _resolve_random_pool_size(
    random_pool_size: Union[int, str, None],
    full_candidate_size: int,
    universe_size: int,
    max_interventions: int,
) -> int:
    """
    Resolve random candidate pool size.

    `match_full` is used only if the candidate universe is larger than the full pool;
    otherwise it falls back to max_interventions to avoid sampling the full set and
    producing an order-only randomization that Stage3 will ignore.
    """
    if isinstance(random_pool_size, int):
        k = int(random_pool_size)
    elif random_pool_size == "match_full":
        if universe_size > full_candidate_size:
            k = int(full_candidate_size)
        else:
            k = int(max_interventions)
    elif random_pool_size is None:
        k = int(max_interventions)
    else:
        raise ValueError("random_pool_size must be int, 'match_full', or None.")
    return max(1, min(k, int(universe_size)))


def run_multi_lesion_ablation_study(
    model: Any,
    lesion_list: Sequence[int],
    roi_num: int,
    using_steps: int,
    empirical_FC: Optional[np.ndarray],
    best_config: Dict[str, Any],
    stage1_provider: Union[Callable[[int], Dict[str, Any]], Dict[int, Dict[str, Any]]],
    stage2_variant_fn: Callable[[AblationVariant, int, Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    signals: Optional[np.ndarray] = None,
    roi_names: Optional[Sequence[str]] = None,
    variants: Optional[List[AblationVariant]] = None,
    top_m: int = 20,
    random_repeats: int = 30,
    random_state: int = 123,
    random_pool_size: Union[int, str, None] = None,
    out_dir: str = "./ablation_multi_lesion",
    prefer_multi: bool = True,
    success_traj_threshold: float = 0.0,
    success_fc_threshold: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run the full multi-lesion ablation study.

    Returns
    -------
    per_lesion_df:
        One row per lesion and deterministic variant. Random ranking is stored as
        one per-lesion mean row.
    random_repeat_df:
        Random-ranking repeat-level rows.
    summary_df:
        Cross-lesion mean/std summary.
    core_summary_df:
        Core-network structural summary.
    raw_outputs:
        Nested dict containing selected Stage2/Stage3 outputs for debugging.
    """
    os.makedirs(out_dir, exist_ok=True)
    variants = variants or build_ablation_variants(best_config)
    rng = np.random.default_rng(int(random_state))

    # Common context passed into the user-defined Stage2 adapter.
    context = {
        "BEST_CONFIG": best_config,
        "roi_num": int(roi_num),
        "using_steps": int(using_steps),
        "empirical_FC": empirical_FC,
        "signals": signals,
        "roi_names": roi_names,
        "model": model,
        "top_m": int(top_m),
        "random_repeats": int(random_repeats),
        "random_state": int(random_state),
        "out_dir": out_dir,
    }

    per_lesion_rows: List[Dict[str, Any]] = []
    random_rows: List[Dict[str, Any]] = []
    core_rows: List[Dict[str, Any]] = []
    raw_outputs: Dict[str, Any] = {}

    # Split deterministic variants and random variant.
    det_variants = [v for v in variants if not v.is_random]
    random_variants = [v for v in variants if v.is_random]
    random_variant = random_variants[0] if len(random_variants) > 0 else None

    for lesion_idx in lesion_list:
        lesion_idx = int(lesion_idx)
        lesion_name = roi_names[lesion_idx] if roi_names is not None and lesion_idx < len(roi_names) else f"ROI_{lesion_idx}"

        print("=" * 100)
        print(f"[Ablation] lesion_idx={lesion_idx}, lesion_name={lesion_name}")
        print("=" * 100)

        stage1_res = get_stage1_for_lesion(lesion_idx, stage1_provider)
        raw_outputs.setdefault(lesion_idx, {"stage1_res": stage1_res, "variants": {}})

        full_stage2_for_random: Optional[Dict[str, Any]] = None
        full_candidate_nodes: List[int] = []

        # Deterministic variants.
        for variant_cfg in det_variants:
            print(f"[Variant] {variant_cfg.variant}")
            try:
                stage2_res = stage2_variant_fn(variant_cfg, lesion_idx, stage1_res, context)
                candidate_nodes = unique_nodes(stage2_res.get("candidate_nodes", []), roi_num=roi_num, exclude=[lesion_idx])
                stage2_res = dict(stage2_res)
                stage2_res["candidate_nodes"] = candidate_nodes

                if len(candidate_nodes) == 0:
                    raise ValueError("empty candidate_nodes returned by stage2_variant_fn")

                stage3_res = run_stage3_for_candidates(
                    model=model,
                    candidate_nodes=candidate_nodes,
                    stage1_res=stage1_res,
                    lesion_idx=lesion_idx,
                    roi_num=roi_num,
                    using_steps=using_steps,
                    best_config=best_config,
                )

                row = build_ablation_row(
                    lesion_idx=lesion_idx,
                    lesion_name=lesion_name,
                    variant_cfg=variant_cfg,
                    stage2_res=stage2_res,
                    stage3_res=stage3_res,
                    roi_names=roi_names,
                    repeat_id=None,
                    status="ok",
                    prefer_multi=prefer_multi,
                    empirical_FC=empirical_FC,
                )
                per_lesion_rows.append(row)
                core_rows.append({k: row.get(k) for k in [
                    "lesion_idx", "lesion_name", "variant", "n_candidate_nodes",
                    "n_core_edges", "n_direct_lesion_edges", "mean_abs_fc_core", "best_node", "policy_nodes",
                ]})

                raw_outputs[lesion_idx]["variants"][variant_cfg.variant] = {
                    "stage2_res": stage2_res,
                    "stage3_res": stage3_res,
                }

                if variant_cfg.variant == "Full method":
                    full_stage2_for_random = stage2_res
                    full_candidate_nodes = candidate_nodes

            except Exception as e:
                warnings.warn(f"Ablation variant failed: lesion={lesion_idx}, variant={variant_cfg.variant}, error={e}")
                fail_row = {
                    "lesion_idx": lesion_idx,
                    "lesion_name": lesion_name,
                    "variant": variant_cfg.variant,
                    "repeat_id": None,
                    "selected_mode": None,
                    "status": "failed",
                    "error": str(e),
                    "candidate_nodes": [],
                    "n_candidate_nodes": 0,
                }
                fail_row.update(variant_cfg.to_dict())
                per_lesion_rows.append(fail_row)

        # Random ranking variant: sample from broad universe, then same Stage3.
        if random_variant is not None:
            print(f"[Variant] {random_variant.variant} ({random_repeats} repeats)")

            if full_stage2_for_random is None:
                # Fall back to calling Stage2 for random variant, but this should still return a broad universe.
                try:
                    full_stage2_for_random = stage2_variant_fn(random_variant, lesion_idx, stage1_res, context)
                    full_candidate_nodes = unique_nodes(full_stage2_for_random.get("candidate_nodes", []), roi_num=roi_num, exclude=[lesion_idx])
                except Exception as e:
                    warnings.warn(f"Cannot build random universe for lesion {lesion_idx}: {e}")
                    full_stage2_for_random = None

            if full_stage2_for_random is not None:
                universe = full_stage2_for_random.get(
                    "candidate_universe",
                    full_stage2_for_random.get("all_candidate_nodes", full_stage2_for_random.get("candidate_nodes", [])),
                )
                universe = unique_nodes(universe, roi_num=roi_num, exclude=[lesion_idx])

                if len(universe) == 0:
                    warnings.warn(f"Random ranking skipped for lesion {lesion_idx}: empty candidate universe.")
                else:
                    k = _resolve_random_pool_size(
                        random_pool_size=random_pool_size,
                        full_candidate_size=len(full_candidate_nodes),
                        universe_size=len(universe),
                        max_interventions=int(best_config["max_interventions"]),
                    )

                    if len(universe) <= len(full_candidate_nodes) and k == len(universe):
                        warnings.warn(
                            f"Random ranking for lesion {lesion_idx} may be uninformative: "
                            f"universe size ({len(universe)}) equals random pool size ({k}). "
                            "Provide a larger `candidate_universe` in stage2_res."
                        )

                    repeat_rows_for_lesion = []
                    for repeat_id in range(1, int(random_repeats) + 1):
                        try:
                            random_nodes = rng.choice(universe, size=k, replace=False).tolist()
                            random_nodes = unique_nodes(random_nodes, roi_num=roi_num, exclude=[lesion_idx])

                            # Stage2 structural stats inherited from full Stage2; only ranking/selection is random.
                            random_stage2_res = dict(full_stage2_for_random)
                            random_stage2_res["candidate_nodes"] = random_nodes
                            random_stage2_res["random_universe_size"] = len(universe)

                            stage3_res = run_stage3_for_candidates(
                                model=model,
                                candidate_nodes=random_nodes,
                                stage1_res=stage1_res,
                                lesion_idx=lesion_idx,
                                roi_num=roi_num,
                                using_steps=using_steps,
                                best_config=best_config,
                            )

                            row = build_ablation_row(
                                lesion_idx=lesion_idx,
                                lesion_name=lesion_name,
                                variant_cfg=random_variant,
                                stage2_res=random_stage2_res,
                                stage3_res=stage3_res,
                                roi_names=roi_names,
                                repeat_id=repeat_id,
                                status="ok",
                                prefer_multi=prefer_multi,
                                empirical_FC=empirical_FC,
                            )
                            row["random_pool_size"] = k
                            row["random_universe_size"] = len(universe)
                            random_rows.append(row)
                            repeat_rows_for_lesion.append(row)

                        except Exception as e:
                            warnings.warn(f"Random repeat failed: lesion={lesion_idx}, repeat={repeat_id}, error={e}")

                    # Aggregate random repeats into one per-lesion row.
                    if len(repeat_rows_for_lesion) > 0:
                        rep_df = pd.DataFrame(repeat_rows_for_lesion)
                        agg_row = {
                            "lesion_idx": lesion_idx,
                            "lesion_name": lesion_name,
                            "variant": random_variant.variant,
                            "repeat_id": None,
                            "selected_mode": "repeat_mean",
                            "status": "ok",
                            "use_response_prior": random_variant.use_response_prior,
                            "use_lpcmci": random_variant.use_lpcmci,
                            "prune_mode": random_variant.prune_mode,
                            "causal_weight": random_variant.causal_weight,
                            "fc_weight": random_variant.fc_weight,
                            "ranking_mode": random_variant.ranking_mode,
                            "candidate_nodes": "random repeats",
                            "n_candidate_nodes": float(pd.to_numeric(rep_df["n_candidate_nodes"], errors="coerce").mean()),
                            "policy": "random repeats",
                            "policy_nodes": "random repeats",
                            "policy_roi_names": "random repeats",
                            "policy_size": float(pd.to_numeric(rep_df["policy_size"], errors="coerce").mean()),
                            "best_node": None,
                            "n_core_edges": full_stage2_for_random.get("n_core_edges", np.nan),
                            "n_direct_lesion_edges": full_stage2_for_random.get("n_direct_lesion_edges", np.nan),
                            "mean_abs_fc_core": full_stage2_for_random.get("mean_abs_fc_core", np.nan),
                            "random_pool_size": k,
                            "random_universe_size": len(universe),
                        }
                        for m in [
                            "objective", "traj_recovery", "fc_recovery",
                            "D_lesion_traj", "D_policy_traj", "D_lesion_fc", "D_policy_fc",
                            "node_cost", "strength_cost", "temporal_cost",
                        ]:
                            if m in rep_df.columns:
                                vals = pd.to_numeric(rep_df[m], errors="coerce")
                                agg_row[m] = float(vals.mean())
                                agg_row[f"{m}_std_within_lesion"] = float(vals.std())
                        per_lesion_rows.append(agg_row)
                        core_rows.append({k0: agg_row.get(k0) for k0 in [
                            "lesion_idx", "lesion_name", "variant", "n_candidate_nodes",
                            "n_core_edges", "n_direct_lesion_edges", "mean_abs_fc_core", "best_node", "policy_nodes",
                        ]})

    per_lesion_df = pd.DataFrame(per_lesion_rows)
    random_repeat_df = pd.DataFrame(random_rows)
    core_summary_df = pd.DataFrame(core_rows)

    # Success flag at per-lesion level.
    if len(per_lesion_df) > 0:
        per_lesion_df["success"] = (
            (pd.to_numeric(per_lesion_df.get("traj_recovery"), errors="coerce") > float(success_traj_threshold))
            & (pd.to_numeric(per_lesion_df.get("fc_recovery"), errors="coerce") > float(success_fc_threshold))
            & (per_lesion_df.get("status") == "ok")
        )

    summary_df = summarize_ablation_results(per_lesion_df)

    # Save outputs.
    per_lesion_path = os.path.join(out_dir, "ablation_per_lesion_results.csv")
    random_path = os.path.join(out_dir, "ablation_random_repeats.csv")
    summary_path = os.path.join(out_dir, "ablation_summary_mean_std.csv")
    core_path = os.path.join(out_dir, "ablation_core_network_summary.csv")

    per_lesion_df.to_csv(per_lesion_path, index=False)
    random_repeat_df.to_csv(random_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    core_summary_df.to_csv(core_path, index=False)

    print("Saved:")
    print("  ", per_lesion_path)
    print("  ", random_path)
    print("  ", summary_path)
    print("  ", core_path)

    return per_lesion_df, random_repeat_df, summary_df, core_summary_df, raw_outputs


# =========================================================
# 5. Summary and paper table helpers
# =========================================================

def summarize_ablation_results(per_lesion_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-lesion mean/std summary for each ablation variant."""
    if per_lesion_df is None or len(per_lesion_df) == 0:
        return pd.DataFrame()

    df = per_lesion_df.copy()
    df = df[df.get("status", "ok") == "ok"].copy()

    metrics = [
        "objective", "traj_recovery", "fc_recovery",
        "D_policy_traj", "D_policy_fc",
        "policy_size", "n_candidate_nodes", "n_core_edges", "n_direct_lesion_edges",
    ]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for variant, sub in df.groupby("variant"):
        row = {
            "variant": variant,
            "n_lesions": int(sub["lesion_idx"].nunique()) if "lesion_idx" in sub.columns else len(sub),
        }
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce")
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std())
        if "success" in sub.columns:
            row["success_rate"] = float(sub["success"].mean())
        rows.append(row)

    out = pd.DataFrame(rows)
    order = [
        "Full method",
        "w/o response prior",
        "w/o LPCMCI",
        "Causal-only pruning",
        "FC-only pruning",
        "Random ranking",
        "LNM-FC baseline",
        "DSA-stimulation baseline",
        "NCT-state-transition baseline",
        "Target-control baseline",
        "Tracking-control baseline",
        "PCMCI+ causal-core baseline",
    ]
    if "variant" in out.columns:
        order_map = {name: i for i, name in enumerate(order)}
        out["_variant_order"] = out["variant"].map(lambda x: order_map.get(str(x), len(order)))
        out = out.sort_values(["_variant_order", "variant"]).drop(columns=["_variant_order"]).reset_index(drop=True)
    return out


def make_ablation_paper_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compact table for manuscript."""
    keep = [
        "variant", "n_lesions",
        "objective_mean", "objective_std",
        "traj_recovery_mean", "traj_recovery_std",
        "fc_recovery_mean", "fc_recovery_std",
        "policy_size_mean", "policy_size_std",
        "success_rate",
    ]
    cols = [c for c in keep if c in summary_df.columns]
    return summary_df[cols].copy()


def format_mean_std_table(summary_df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Return a formatted mean±std table for reporting."""
    df = summary_df.copy()
    rows = []
    for _, r in df.iterrows():
        row = {"variant": r["variant"], "n_lesions": r.get("n_lesions", np.nan)}
        for m in ["objective", "traj_recovery", "fc_recovery", "policy_size"]:
            mean_col, std_col = f"{m}_mean", f"{m}_std"
            if mean_col in df.columns:
                mean = r.get(mean_col, np.nan)
                std = r.get(std_col, np.nan)
                if pd.notna(std):
                    row[m] = f"{mean:.{decimals}f} ± {std:.{decimals}f}"
                else:
                    row[m] = f"{mean:.{decimals}f}"
        if "success_rate" in df.columns:
            row["success_rate"] = f"{r.get('success_rate', np.nan):.{decimals}f}"
        rows.append(row)
    return pd.DataFrame(rows)


def make_compact_internal_ablation_table(
    per_lesion_df: pd.DataFrame,
    reference_variant: str = "Full method",
) -> pd.DataFrame:
    """
    Build the internal candidate-construction sensitivity table.

    Main-text metrics:
        Candidate nodes
        Core edges
        Direct lesion edges
        Best-node overlap

    Random ranking does not independently construct a lesion-centered causal core,
    so its core-edge metrics are set to NaN and should be displayed as "--".
    """
    keep_variants = [
        "Full method",
        "w/o response prior",
        "w/o LPCMCI",
        "Random ranking",
    ]

    if per_lesion_df is None or len(per_lesion_df) == 0:
        return pd.DataFrame()

    df = per_lesion_df[per_lesion_df["variant"].astype(str).isin(keep_variants)].copy()

    if "status" in df.columns:
        df = df[df["status"].astype(str) == "ok"].copy()

    rows = []
    for variant, sub in df.groupby("variant", dropna=False):
        row = {
            "variant": str(variant),
            "n_lesions": int(sub["lesion_idx"].nunique()) if "lesion_idx" in sub.columns else len(sub),
        }
        for m in ["n_candidate_nodes", "n_core_edges", "n_direct_lesion_edges"]:
            if m in sub.columns:
                vals = pd.to_numeric(sub[m], errors="coerce")
                row[f"{m}_mean"] = float(vals.mean())
                row[f"{m}_std"] = float(vals.std())
        rows.append(row)

    out = pd.DataFrame(rows)

    overlap_df = summarize_best_node_overlap(df, reference_variant=reference_variant)
    if overlap_df is not None and len(overlap_df) > 0:
        out = out.merge(
            overlap_df[["variant", "best_node_overlap_rate"]],
            on="variant",
            how="left",
        )
    else:
        out["best_node_overlap_rate"] = np.nan

    order = {
        "Full method": 0,
        "w/o response prior": 1,
        "w/o LPCMCI": 2,
        "Random ranking": 3,
    }
    out["_order"] = out["variant"].map(lambda x: order.get(str(x), len(order)))
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    mask_random = out["variant"].astype(str) == "Random ranking"
    for col in [
        "n_core_edges_mean",
        "n_core_edges_std",
        "n_direct_lesion_edges_mean",
        "n_direct_lesion_edges_std",
    ]:
        if col in out.columns:
            out.loc[mask_random, col] = np.nan

    return out


def format_compact_internal_ablation_table(
    internal_table: pd.DataFrame,
    decimals: int = 3,
    dash: str = "--",
) -> pd.DataFrame:
    """Format the compact internal ablation table for manuscript reporting."""
    if internal_table is None or len(internal_table) == 0:
        return pd.DataFrame()

    rows = []
    for _, r in internal_table.iterrows():
        def _mean_std(mean_col: str, std_col: str) -> str:
            mean = r.get(mean_col, np.nan)
            std = r.get(std_col, np.nan)
            if pd.notna(mean) and pd.notna(std):
                return f"{float(mean):.1f} ± {float(std):.1f}"
            return dash

        row = {
            "Variant": r["variant"],
            "Candidate nodes": _mean_std("n_candidate_nodes_mean", "n_candidate_nodes_std"),
            "Core edges": _mean_std("n_core_edges_mean", "n_core_edges_std"),
            "Direct lesion edges": _mean_std("n_direct_lesion_edges_mean", "n_direct_lesion_edges_std"),
            "Best-node overlap": (
                dash if str(r["variant"]) == "Full method"
                else f"{float(r['best_node_overlap_rate']):.{decimals}f}"
                if pd.notna(r.get("best_node_overlap_rate", np.nan))
                else dash
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_compact_internal_ablation_outputs(
    per_lesion_df: pd.DataFrame,
    out_dir: str = "./ablation_multi_lesion_compact_internal",
    reference_variant: str = "Full method",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save internal ablation outputs separately from baseline-comparison outputs.
    """
    os.makedirs(out_dir, exist_ok=True)
    internal_table = make_compact_internal_ablation_table(
        per_lesion_df,
        reference_variant=reference_variant,
    )
    formatted = format_compact_internal_ablation_table(internal_table)

    internal_table.to_csv(os.path.join(out_dir, "internal_structure_ablation_table_raw.csv"), index=False)
    formatted.to_csv(os.path.join(out_dir, "internal_structure_ablation_table_formatted.csv"), index=False)

    return internal_table, formatted


def build_baseline_comparison_results(
    ablation_per_lesion_df: pd.DataFrame,
    literature_per_lesion_df: pd.DataFrame,
    baseline_variants: Optional[Sequence[str]] = None,
    reference_variant: str = "Full method",
    out_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build baseline-comparison outputs separately from internal ablation outputs.

    This keeps only:
        Full method from the internal ablation run
        selected external baselines from the literature-baseline run
    """
    baseline_variants = list(baseline_variants or LITERATURE_BASELINE_ORDER)

    rows = []
    if ablation_per_lesion_df is not None and len(ablation_per_lesion_df) > 0:
        full = ablation_per_lesion_df[
            ablation_per_lesion_df["variant"].astype(str) == str(reference_variant)
        ].copy()
        if len(full) > 0:
            full["baseline_group"] = "proposed"
            rows.append(full)

    if literature_per_lesion_df is not None and len(literature_per_lesion_df) > 0:
        lit = literature_per_lesion_df[
            literature_per_lesion_df["variant"].astype(str).isin(baseline_variants)
        ].copy()
        if len(lit) > 0:
            rows.append(lit)

    baseline_df = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    summary_df = summarize_ablation_results(baseline_df)
    formatted_df = format_mean_std_table(summary_df)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        baseline_df.to_csv(os.path.join(out_dir, "baseline_comparison_per_lesion_results.csv"), index=False)
        summary_df.to_csv(os.path.join(out_dir, "baseline_comparison_summary_mean_std.csv"), index=False)
        formatted_df.to_csv(os.path.join(out_dir, "baseline_comparison_formatted.csv"), index=False)

    return baseline_df, summary_df, formatted_df


def plot_baseline_objective_only(
    summary_df: pd.DataFrame,
    variants: Optional[Sequence[str]] = None,
    out_dir: str = "./baseline_comparison",
    save_name: str = "baseline_objective_comparison.png",
    title: str = "Candidate-budget-matched baseline comparison",
) -> str:
    """Plot only the objective bar chart for baseline comparison."""
    os.makedirs(out_dir, exist_ok=True)
    df = summary_df.copy()

    if variants is not None:
        variants = list(variants)
        df = df[df["variant"].astype(str).isin(variants)].copy()
        order = {v: i for i, v in enumerate(variants)}
        df["_order"] = df["variant"].map(lambda x: order.get(str(x), len(order)))
        df = df.sort_values("_order").drop(columns="_order")
    elif "variant" in df.columns:
        order = list(LITERATURE_BASELINE_ORDER)
        order_map = {v: i for i, v in enumerate(order)}
        df["_order"] = df["variant"].map(lambda x: order_map.get(str(x), len(order)))
        df = df.sort_values("_order").drop(columns="_order")

    mean_col = "objective_mean"
    std_col = "objective_std"
    if mean_col not in df.columns:
        raise ValueError("summary_df must contain objective_mean.")

    x = np.arange(len(df))
    y = pd.to_numeric(df[mean_col], errors="coerce").values
    yerr = pd.to_numeric(df[std_col], errors="coerce").fillna(0.0).values if std_col in df.columns else None

    plt.close("all")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x, y, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"].astype(str).tolist(), rotation=25, ha="right")
    ax.set_ylabel("Objective")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()

    save_path = os.path.join(out_dir, save_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return save_path


# =========================================================
# 6. Plot helpers
# =========================================================

def plot_ablation_bar(
    summary_df: pd.DataFrame,
    metric: str = "objective",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot cross-lesion mean ± std for one metric."""
    plt.close("all")
    df = summary_df.copy()
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    if mean_col not in df.columns:
        raise ValueError(f"{mean_col} not found in summary_df.")

    x = np.arange(len(df))
    y = pd.to_numeric(df[mean_col], errors="coerce").values
    yerr = pd.to_numeric(df[std_col], errors="coerce").fillna(0.0).values if std_col in df.columns else None

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(x, y, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"].tolist(), rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Multi-lesion ablation: {metric}")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_ablation_three_metrics(
    summary_df: pd.DataFrame,
    metrics: Sequence[str] = ("objective", "traj_recovery", "fc_recovery"),
    out_dir: str = "./ablation_multi_lesion",
):
    """Plot objective/traj_recovery/fc_recovery as separate manuscript-style figures."""
    os.makedirs(out_dir, exist_ok=True)
    for metric in metrics:
        plot_ablation_bar(
            summary_df,
            metric=metric,
            save_path=os.path.join(out_dir, f"ablation_{metric}.png"),
        )


def plot_policy_distance_ablation(
    summary_df: pd.DataFrame,
    metrics: Sequence[str] = ("D_policy_traj", "D_policy_fc"),
    out_dir: str = "./ablation_multi_lesion",
):
    """Plot residual distance after intervention; lower is better."""
    os.makedirs(out_dir, exist_ok=True)
    for metric in metrics:
        if f"{metric}_mean" in summary_df.columns:
            plot_ablation_bar(
                summary_df,
                metric=metric,
                save_path=os.path.join(out_dir, f"ablation_{metric}.png"),
                title=f"Multi-lesion ablation residual distance: {metric}",
            )


# =========================================================
# 7. Minimal run template
# =========================================================

def print_usage_template():
    """Print a notebook usage template."""
    print(r'''
# 1. Choose lesions from the previous robustness experiment.
lesion_list = [5, 23, 47, 50, 51, 226, 231, 247]

# 2. Define Stage1 provider. Use cache if you already saved Stage1 results.
def stage1_provider(lesion_idx):
    # Option A: return cached results
    return stage1_res_by_lesion[lesion_idx]

    # Option B: compute Stage1 here
    # return run_stage1_for_one_lesion(lesion_idx, ...)

# 3. Define Stage2 variant adapter.
print(make_stage2_variant_adapter_template())
# Copy the printed template and replace run_your_stage2_pipeline(...) with your real Stage2 call.

# 4. Run ablation.
per_lesion_df, random_repeat_df, summary_df, core_summary_df, raw_outputs = run_multi_lesion_ablation_study(
    model=ANN,
    lesion_list=lesion_list,
    roi_num=ROI_num,
    using_steps=using_steps,
    empirical_FC=empirical_FC,
    best_config=BEST_CONFIG,
    stage1_provider=stage1_provider,
    stage2_variant_fn=my_stage2_variant_fn,
    signals=signals,
    roi_names=roi_names,
    top_m=20,
    random_repeats=30,
    random_state=BEST_CONFIG.get("random_state", 123),
    random_pool_size="match_full",
    out_dir="./ablation_multi_lesion",
)

# 5. Paper table and figures.
paper_table = make_ablation_paper_table(summary_df)
formatted_table = format_mean_std_table(summary_df)
plot_ablation_three_metrics(summary_df, out_dir="./ablation_multi_lesion")
plot_policy_distance_ablation(summary_df, out_dir="./ablation_multi_lesion")
''')


# =========================================================
# 8. Literature-based baseline comparison for Section 4.6.2
# =========================================================

@dataclass(frozen=True)
class LiteratureBaselineVariant:
    """Configuration for one literature-based baseline."""
    variant: str
    short_name: str
    literature_basis: str
    candidate_score: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


LITERATURE_BASELINE_ORDER = [
    "Full method",
    "Global-FC hub baseline",
    "NCT-state-transition baseline",
    "Target-control baseline",
    "Tracking-control baseline",
    "Random-intervention baseline",
]


def build_literature_baseline_variants() -> List[LiteratureBaselineVariant]:
    """
    Return the main-text candidate-budget-matched baselines.

    Main-text setting:
        Global-FC hub
        NCT-state-transition
        Target-control
        Tracking-control
        Random-intervention

    Random-intervention is a direct random-policy lower-bound baseline.
    Lagged-only and PCMCI+ are kept as optional functions in this script, but
    they are not included in the default main-text baseline list.
    """
    return [
        LiteratureBaselineVariant(
            variant="Global-FC hub baseline",
            short_name="Global-FC hub",
            literature_basis="Global functional-connectivity hub lower-bound baseline",
            candidate_score="sum_j |FC_{node,j}|",
            description=(
                "Rank nodes by whole-brain functional-connectivity strength. "
                "This tests whether globally connected hub nodes are sufficient for "
                "lesion-conditioned intervention."
            ),
        ),
        LiteratureBaselineVariant(
            variant="NCT-state-transition baseline",
            short_name="NCT-state-transition",
            literature_basis="Network control theory / optimal brain state transition: Gu et al.",
            candidate_score="negative lesion-to-health transition energy",
            description="Select nodes with low control energy from lesion state to healthy target state.",
        ),
        LiteratureBaselineVariant(
            variant="Target-control baseline",
            short_name="Target-control",
            literature_basis="Target controllability: Manjunatha et al.",
            candidate_score="negative target-response-domain control energy",
            description="Use the lesion-response domain as target subnetwork and rank input nodes by target-control energy.",
        ),
        LiteratureBaselineVariant(
            variant="Tracking-control baseline",
            short_name="Tracking-control",
            literature_basis="Optimal tracking control: Dong et al.",
            candidate_score="negative lesion-to-healthy trajectory tracking cost",
            description="Rank nodes by the cost of tracking the healthy trajectory from lesion initial states.",
        ),
        LiteratureBaselineVariant(
            variant="Random-intervention baseline",
            short_name="Random-intervention",
            literature_basis="Random intervention lower-bound baseline",
            candidate_score="random node/sign/strength/schedule; no candidate ranking",
            description=(
                "Directly sample intervention nodes and intervention parameters without "
                "candidate ranking or Stage-3 optimization. This tests whether the "
                "optimized lesion-centered strategy is better than unstructured random adjustment."
            ),
        ),
    ]


def build_optional_lagged_only_baseline_variant() -> LiteratureBaselineVariant:
    """Return the optional Lagged-only baseline; not used in the main-text default run."""
    return LiteratureBaselineVariant(
        variant="Lagged-only lesion-proximal baseline",
        short_name="Lagged-only",
        literature_basis="Lesion-proximal first-order lagged association lower-bound baseline",
        candidate_score="|corr(x_l(t), x_i(t+1))| + |corr(x_i(t), x_l(t+1))|",
        description=(
            "Rank nodes by first-order temporal association with the lesion node. "
            "This tests whether local lagged association alone can replace "
            "lesion-response-driven causal-core construction."
        ),
    )


def build_optional_pcmciplus_baseline_variant() -> LiteratureBaselineVariant:
    """Return the optional PCMCI+ baseline; not used in the main-text default run."""
    return LiteratureBaselineVariant(
        variant="PCMCI+ causal-core baseline",
        short_name="PCMCI+ causal-core",
        literature_basis="PCMCI+ time-series causal discovery: Runge",
        candidate_score="lesion-proximal incident causal strength",
        description="General time-series causal-discovery baseline without lesion-response-driven LPCMCI core fusion.",
    )


def literature_baseline_design_table(
    baselines: Optional[List[LiteratureBaselineVariant]] = None,
    include_pcmciplus: bool = False,
) -> pd.DataFrame:
    """
    Paper-facing design table for the currently selected literature baselines.

    By default this table follows the main-text setting and excludes PCMCI+.
    Pass include_pcmciplus=True or provide an explicit baselines list to report
    the optional PCMCI+ baseline.
    """
    if baselines is None:
        baselines = build_literature_baseline_variants()
        if include_pcmciplus:
            baselines = baselines + [build_optional_pcmciplus_baseline_variant()]

    rows = []
    for v in baselines:
        rows.append({
            "variant": v.variant,
            "short_name": v.short_name,
            "literature_basis": v.literature_basis,
            "candidate_score": v.candidate_score,
            "description": v.description,
        })
    return pd.DataFrame(rows)

def _pseudo_ablation_variant_from_literature(v: LiteratureBaselineVariant) -> AblationVariant:
    """Build a pseudo AblationVariant so existing Stage-3 row builders can be reused."""
    return AblationVariant(
        variant=v.variant,
        use_response_prior=False,
        use_lpcmci=False,
        prune_mode="literature_baseline",
        causal_weight=None,
        fc_weight=None,
        ranking_mode=v.short_name,
        is_random=False,
        description=v.description,
    )


# -----------------------------
# 8.1 Shared numerical helpers
# -----------------------------

def _as_2d_time_by_node(signals: np.ndarray, roi_num: Optional[int] = None) -> np.ndarray:
    """Coerce signals into a 2-D array with shape (T, N)."""
    arr = np.asarray(signals, dtype=float)
    if arr.ndim == 2:
        # Prefer (T, N). If it is (N, T), transpose when roi_num makes this clear.
        if roi_num is not None and arr.shape[0] == int(roi_num) and arr.shape[1] != int(roi_num):
            arr = arr.T
        return arr
    if arr.ndim == 3:
        # Common forms: (subjects/runs, T, N) or (T, windows, N). Flatten leading axes except last.
        if roi_num is not None and arr.shape[-1] == int(roi_num):
            return arr.reshape(-1, arr.shape[-1])
        if roi_num is not None and arr.shape[1] == int(roi_num):
            return np.swapaxes(arr, 1, 2).reshape(-1, int(roi_num))
    raise ValueError(f"Cannot coerce signals with shape {arr.shape} to (T, N).")


def _zscore_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def compute_empirical_fc_from_signals(signals: np.ndarray, roi_num: Optional[int] = None) -> np.ndarray:
    """Compute Pearson FC from a healthy fMRI time series."""
    x = _as_2d_time_by_node(signals, roi_num=roi_num)
    x = _zscore_columns(x)
    fc = np.corrcoef(x, rowvar=False)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(fc, 0.0)
    return fc


def estimate_linear_transition_matrix(
    signals: np.ndarray,
    roi_num: int,
    ridge: float = 1e-2,
    standardize: bool = True,
    stable_normalize: bool = True,
    stability_eps: float = 0.05,
) -> np.ndarray:
    """
    Estimate a first-order linear transition matrix A from healthy fMRI signals.

    The fitted model is x(t+1) = A x(t) + error. The returned matrix is optionally
    spectral-radius normalized for finite-horizon network-control baselines.
    """
    x = _as_2d_time_by_node(signals, roi_num=roi_num)
    if standardize:
        x = _zscore_columns(x)

    X = x[:-1].T  # N x (T-1)
    Y = x[1:].T   # N x (T-1)
    n = X.shape[0]
    reg = float(ridge) * np.eye(n)
    A = Y @ X.T @ np.linalg.pinv(X @ X.T + reg)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    if stable_normalize:
        try:
            rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        except Exception:
            rho = 0.0
        if np.isfinite(rho) and rho > 0:
            A = A / (rho + float(stability_eps))
    return A.astype(float)


def _matrix_powers(A: np.ndarray, horizon: int) -> List[np.ndarray]:
    """Return [I, A, A^2, ..., A^horizon]."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    powers = [np.eye(n)]
    for _ in range(int(horizon)):
        powers.append(A @ powers[-1])
    return powers


def _single_input_reachability_matrix_from_powers(powers: Sequence[np.ndarray], node: int, horizon: int) -> np.ndarray:
    """
    Build C_i = [A^{H-1}e_i, A^{H-2}e_i, ..., e_i] for terminal-state control.
    Shape: (N, H).
    """
    cols = []
    for k in range(int(horizon)):
        cols.append(powers[int(horizon) - 1 - k][:, int(node)])
    return np.stack(cols, axis=1)


def _regularized_low_rank_energy(delta: np.ndarray, C: np.ndarray, eps: float = 1e-3) -> float:
    """
    Compute delta^T (C C^T + eps I)^(-1) delta efficiently using Woodbury.

    C has shape (D, H), with H usually much smaller than D.
    """
    delta = np.asarray(delta, dtype=float).reshape(-1)
    C = np.asarray(C, dtype=float)
    eps = float(eps)
    if C.size == 0:
        return float(np.dot(delta, delta) / eps)
    Ct_delta = C.T @ delta
    small = np.eye(C.shape[1]) + (C.T @ C) / eps
    try:
        solved = np.linalg.solve(small, Ct_delta)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(small) @ Ct_delta
    energy = (np.dot(delta, delta) / eps) - (np.dot(Ct_delta, solved) / (eps ** 2))
    return float(max(energy, 0.0))


def _topk_from_scores(
    scores: Dict[int, float],
    roi_num: int,
    lesion_idx: int,
    top_k: int,
    descending: bool = True,
) -> List[int]:
    """Return valid Top-K nodes from a score dictionary, excluding the lesion node."""
    items = []
    for node, score in scores.items():
        try:
            node_i = int(node)
            score_f = float(score)
        except Exception:
            continue
        if node_i == int(lesion_idx) or node_i < 0 or node_i >= int(roi_num):
            continue
        if not np.isfinite(score_f):
            continue
        items.append((node_i, score_f))
    items.sort(key=lambda x: x[1], reverse=bool(descending))
    return unique_nodes([n for n, _ in items], roi_num=roi_num, exclude=[lesion_idx])[: int(top_k)]


def _scores_to_candidate_df(
    scores: Dict[int, float],
    candidate_nodes: Sequence[int],
    score_name: str,
    roi_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Create a candidate_df compatible with the existing ablation outputs."""
    rows = []
    for rank, node in enumerate(candidate_nodes, start=1):
        node = int(node)
        rows.append({
            "rank": rank,
            "node": node,
            "roi_name": roi_names[node] if roi_names is not None and node < len(roi_names) else f"ROI_{node}",
            "candidate_score": float(scores.get(node, np.nan)),
            score_name: float(scores.get(node, np.nan)),
        })
    return pd.DataFrame(rows)


def _extract_response_scores(stage1_res: Dict[str, Any], roi_num: int) -> np.ndarray:
    """
    Extract lesion-response scores from a Stage-1 result. Falls back to mean
    absolute healthy-lesion trajectory difference if no response table is found.
    """
    for key in ["lesion_response", "response_abs", "response_scores", "lesion_response_scores"]:
        if key in stage1_res:
            arr = np.asarray(stage1_res[key], dtype=float).reshape(-1)
            if arr.size >= int(roi_num):
                return arr[: int(roi_num)]

    for key in ["lesion_response_df", "response_df"]:
        df = stage1_res.get(key, None)
        if isinstance(df, pd.DataFrame):
            node_col = "node" if "node" in df.columns else ("roi" if "roi" in df.columns else None)
            score_col = None
            for c in ["response_abs", "lesion_response", "response_score", "score"]:
                if c in df.columns:
                    score_col = c
                    break
            if node_col is not None and score_col is not None:
                out = np.zeros(int(roi_num), dtype=float)
                for _, r in df.iterrows():
                    try:
                        out[int(r[node_col])] = float(r[score_col])
                    except Exception:
                        pass
                return out

    healthy = np.asarray(stage1_res["healthy_traj"], dtype=float)
    lesion = np.asarray(stage1_res["lesion_traj"], dtype=float)
    # Expected shape: B x H x N.
    if healthy.shape != lesion.shape:
        raise ValueError("healthy_traj and lesion_traj must have the same shape.")
    return np.nanmean(np.abs(lesion - healthy), axis=(0, 1)).reshape(-1)[: int(roi_num)]


def _trajectory_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute distance between two trajectory tensors."""
    return float(np.nanmean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _stage1_initial_states(stage1_res: Dict[str, Any], roi_num: int, using_steps: Optional[int] = None) -> np.ndarray:
    """Extract seed-level initial states from Stage-1 seed inputs."""
    seed = np.asarray(stage1_res["seed_inputs"], dtype=float)
    if seed.ndim == 3:
        return seed[:, -1, : int(roi_num)]
    if seed.ndim == 2:
        if seed.shape[1] == int(roi_num):
            return seed[:, : int(roi_num)]
        if using_steps is not None and seed.shape[1] == int(using_steps) * int(roi_num):
            return seed.reshape(seed.shape[0], int(using_steps), int(roi_num))[:, -1, :]
    raise ValueError(f"Cannot extract initial states from seed_inputs with shape {seed.shape}.")


def _stage1_terminal_states(stage1_res: Dict[str, Any], kind: str = "healthy") -> np.ndarray:
    """Extract seed-level terminal states from healthy/lesion trajectory."""
    key = f"{kind}_traj"
    z = np.asarray(stage1_res[key], dtype=float)
    if z.ndim != 3:
        raise ValueError(f"{key} must have shape (B, H, N), got {z.shape}.")
    return z[:, -1, :]


def _ensure_linear_control_matrix(context: Dict[str, Any], roi_num: int) -> np.ndarray:
    """Get or estimate the shared linear dynamics matrix for control baselines."""
    if "control_A" in context and context["control_A"] is not None:
        return np.asarray(context["control_A"], dtype=float)
    if "A_control" in context and context["A_control"] is not None:
        return np.asarray(context["A_control"], dtype=float)
    signals = context.get("signals", None)
    if signals is None:
        raise ValueError("signals or precomputed control_A is required for control baselines.")
    A = estimate_linear_transition_matrix(
        signals=signals,
        roi_num=roi_num,
        ridge=float(context.get("control_ridge", 1e-2)),
        standardize=bool(context.get("control_standardize", True)),
        stable_normalize=bool(context.get("control_stable_normalize", True)),
        stability_eps=float(context.get("control_stability_eps", 0.05)),
    )
    context["control_A"] = A
    return A


# -----------------------------
# 8.2 Model rollout for DSA baseline
# -----------------------------

def _predict_next_state(model: Any, windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    Predict next ROI state from history windows for several common model APIs.

    Important implementation detail
    -------------------------------
    The paper surrogate used in the notebook is an MLP whose input is usually
    flattened as (B, using_steps * roi_num).  During DSA rollout the temporary
    window is naturally stored as (B, using_steps, roi_num).  Passing this 3-D
    tensor directly to a PyTorch Linear layer with in_features=using_steps*roi_num
    causes the error:

        mat1 and mat2 shapes cannot be multiplied (... x roi_num and
        using_steps*roi_num x hidden_dim)

    Therefore this helper tries the flattened representation first for 3-D
    windows, and falls back to the original representation only if the model
    genuinely expects a temporal 3-D input.
    """
    x0 = np.asarray(windows, dtype=np.float32)
    candidates = []
    if x0.ndim == 3:
        candidates.append(x0.reshape(x0.shape[0], -1))  # MLP-compatible shape
        candidates.append(x0)                           # sequence-model fallback
    else:
        candidates.append(x0)

    last_error: Optional[Exception] = None

    for x in candidates:
        out_chunks = []

        # Keras-like API.
        if hasattr(model, "predict"):
            try:
                for start in range(0, x.shape[0], int(batch_size)):
                    pred = model.predict(x[start:start + int(batch_size)], verbose=0)
                    out_chunks.append(np.asarray(pred, dtype=float))
                return np.concatenate(out_chunks, axis=0)
            except Exception as e:
                last_error = e
                continue

        # PyTorch-like or callable API.
        try:
            import torch
            is_torch_module = isinstance(model, torch.nn.Module)
        except Exception:
            torch = None
            is_torch_module = False

        if is_torch_module:
            try:
                device = next(model.parameters()).device
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    for start in range(0, x.shape[0], int(batch_size)):
                        xb = torch.as_tensor(x[start:start + int(batch_size)], dtype=torch.float32, device=device)
                        pred = model(xb)
                        if isinstance(pred, (tuple, list)):
                            pred = pred[0]
                        out_chunks.append(pred.detach().cpu().numpy().astype(float))
                if was_training:
                    model.train()
                return np.concatenate(out_chunks, axis=0)
            except Exception as e:
                try:
                    if 'was_training' in locals() and was_training:
                        model.train()
                except Exception:
                    pass
                last_error = e
                continue

        if callable(model):
            try:
                for start in range(0, x.shape[0], int(batch_size)):
                    pred = model(x[start:start + int(batch_size)])
                    out_chunks.append(np.asarray(pred, dtype=float))
                return np.concatenate(out_chunks, axis=0)
            except Exception as e:
                last_error = e
                continue

    if last_error is not None:
        raise last_error
    raise TypeError("model must be a Keras-like, PyTorch-like, or callable predictor.")

    if callable(model):
        for start in range(0, x.shape[0], int(batch_size)):
            pred = model(x[start:start + int(batch_size)])
            out_chunks.append(np.asarray(pred, dtype=float))
        return np.concatenate(out_chunks, axis=0)

    raise TypeError("model must be a Keras-like, PyTorch-like, or callable predictor.")


def rollout_fixed_stimulation_under_lesion(
    model: Any,
    seed_inputs: np.ndarray,
    roi_num: int,
    using_steps: int,
    lesion_idx: int,
    stim_node: int,
    stim_sign: int,
    stim_strength: float,
    horizon: int,
    lesion_strength: float,
    lesion_mode: str = "subtract",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Generate trajectories with persistent lesion perturbation plus fixed node stimulation.

    This is used only for DSA-style candidate ranking; the final policy is still
    selected by the common Stage-3 optimizer.
    """
    win = np.asarray(seed_inputs, dtype=np.float32).copy()
    if win.ndim == 2 and win.shape[1] == int(using_steps) * int(roi_num):
        win = win.reshape(win.shape[0], int(using_steps), int(roi_num))
    if win.ndim != 3:
        raise ValueError(f"seed_inputs must have shape (B, using_steps, roi_num), got {win.shape}.")

    B = win.shape[0]
    traj = np.zeros((B, int(horizon), int(roi_num)), dtype=float)
    lesion_mode = str(lesion_mode).lower()

    for h in range(int(horizon)):
        x_in = win.copy()
        # Apply lesion to the last state in the input window.
        if lesion_mode == "subtract":
            x_in[:, -1, int(lesion_idx)] -= float(lesion_strength)
        elif lesion_mode == "zero":
            x_in[:, -1, int(lesion_idx)] = 0.0
        elif lesion_mode == "scale":
            x_in[:, -1, int(lesion_idx)] *= (1.0 - float(lesion_strength))
        else:
            # Default to subtract to match the manuscript setting.
            x_in[:, -1, int(lesion_idx)] -= float(lesion_strength)

        x_in[:, -1, int(stim_node)] += float(stim_sign) * float(stim_strength)
        pred = _predict_next_state(model, x_in, batch_size=batch_size)
        pred = np.asarray(pred, dtype=float)
        if pred.ndim > 2:
            pred = pred.reshape(pred.shape[0], -1)
        pred = pred[:, : int(roi_num)]
        traj[:, h, :] = pred
        win = np.concatenate([win[:, 1:, :], pred[:, None, :].astype(np.float32)], axis=1)
    return traj


# -----------------------------
# 8.3 Baseline-specific candidate builders
# -----------------------------

def build_global_fc_hub_candidates(
    lesion_idx: int,
    roi_num: int,
    empirical_FC: Optional[np.ndarray],
    signals: Optional[np.ndarray] = None,
    roi_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Global-FC hub lower-bound baseline.

    Score each node by its whole-brain FC strength:
        s_i = sum_{j != i} |FC_{i,j}|

    This baseline is not lesion-conditioned except for excluding the lesion node
    from the candidate list.
    """
    if empirical_FC is None:
        if signals is None:
            raise ValueError("empirical_FC or signals is required for Global-FC hub baseline.")
        empirical_FC = compute_empirical_fc_from_signals(signals, roi_num=roi_num)

    fc = np.asarray(empirical_FC, dtype=float)
    if fc.ndim != 2:
        raise ValueError(f"empirical_FC must be 2-D, got shape {fc.shape}.")
    if fc.shape[0] < int(roi_num) or fc.shape[1] < int(roi_num):
        raise ValueError(f"empirical_FC shape {fc.shape} is smaller than roi_num={roi_num}.")

    abs_fc = np.abs(fc[: int(roi_num), : int(roi_num)]).astype(float)
    np.fill_diagonal(abs_fc, 0.0)
    scores = {
        i: float(np.nansum(abs_fc[int(i), :]))
        for i in range(int(roi_num))
        if int(i) != int(lesion_idx)
    }

    candidate_nodes = _topk_from_scores(scores, roi_num, lesion_idx, top_k, descending=True)
    return {
        "candidate_nodes": candidate_nodes,
        "candidate_universe": unique_nodes(list(scores.keys()), roi_num, exclude=[lesion_idx]),
        "candidate_df": _scores_to_candidate_df(scores, candidate_nodes, "global_fc_hub_score", roi_names),
        "lesion_core_edges": pd.DataFrame(),
        "n_core_edges": 0,
        "n_direct_lesion_edges": 0,
        "baseline_scores": scores,
    }


def build_lagged_only_lesion_proximal_candidates(
    lesion_idx: int,
    roi_num: int,
    signals: np.ndarray,
    roi_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Lagged-only lesion-proximal lower-bound baseline.

    Score each node by first-order temporal association with the lesion node:
        s_i = |corr(x_l(t), x_i(t+1))| + |corr(x_i(t), x_l(t+1))|

    It is lesion-centered and dynamic, but it does not use LPCMCI conditional
    independence tests, lesion-response priors, or FC-constrained pruning.
    """
    if signals is None:
        raise ValueError("signals is required for Lagged-only lesion-proximal baseline.")

    X = _as_2d_time_by_node(signals, roi_num=roi_num)
    X = _zscore_columns(X)
    lesion_idx = int(lesion_idx)

    scores: Dict[int, float] = {}
    forward_scores: Dict[int, float] = {}
    backward_scores: Dict[int, float] = {}

    for node in range(int(roi_num)):
        if node == lesion_idx:
            continue
        try:
            lesion_to_node = abs(float(np.corrcoef(X[:-1, lesion_idx], X[1:, node])[0, 1]))
        except Exception:
            lesion_to_node = 0.0
        try:
            node_to_lesion = abs(float(np.corrcoef(X[:-1, node], X[1:, lesion_idx])[0, 1]))
        except Exception:
            node_to_lesion = 0.0

        lesion_to_node = float(np.nan_to_num(lesion_to_node, nan=0.0, posinf=0.0, neginf=0.0))
        node_to_lesion = float(np.nan_to_num(node_to_lesion, nan=0.0, posinf=0.0, neginf=0.0))
        forward_scores[node] = lesion_to_node
        backward_scores[node] = node_to_lesion
        scores[node] = lesion_to_node + node_to_lesion

    candidate_nodes = _topk_from_scores(scores, roi_num, lesion_idx, top_k, descending=True)
    cand_df = _scores_to_candidate_df(scores, candidate_nodes, "lagged_only_score", roi_names)
    if len(cand_df) > 0:
        cand_df["lesion_to_node_lagged_abs_corr"] = cand_df["node"].astype(int).map(forward_scores)
        cand_df["node_to_lesion_lagged_abs_corr"] = cand_df["node"].astype(int).map(backward_scores)

    return {
        "candidate_nodes": candidate_nodes,
        "candidate_universe": unique_nodes(list(scores.keys()), roi_num, exclude=[lesion_idx]),
        "candidate_df": cand_df,
        "lesion_core_edges": pd.DataFrame(),
        "n_core_edges": 0,
        "n_direct_lesion_edges": 0,
        "baseline_scores": scores,
        "lagged_forward_scores": forward_scores,
        "lagged_backward_scores": backward_scores,
    }


def build_lnm_fc_candidates(
    lesion_idx: int,
    roi_num: int,
    empirical_FC: Optional[np.ndarray],
    signals: Optional[np.ndarray] = None,
    roi_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """B1: Lesion Network Mapping inspired static lesion-to-brain FC baseline."""
    if empirical_FC is None:
        if signals is None:
            raise ValueError("empirical_FC or signals is required for LNM-FC baseline.")
        empirical_FC = compute_empirical_fc_from_signals(signals, roi_num=roi_num)
    fc = np.asarray(empirical_FC, dtype=float)
    scores = {j: abs(float(fc[int(lesion_idx), j])) for j in range(int(roi_num)) if j != int(lesion_idx)}
    candidate_nodes = _topk_from_scores(scores, roi_num, lesion_idx, top_k, descending=True)
    return {
        "candidate_nodes": candidate_nodes,
        "candidate_universe": unique_nodes(list(scores.keys()), roi_num, exclude=[lesion_idx]),
        "candidate_df": _scores_to_candidate_df(scores, candidate_nodes, "lnm_fc_score", roi_names),
        "lesion_core_edges": pd.DataFrame(),
        "n_core_edges": 0,
        "n_direct_lesion_edges": 0,
        "baseline_scores": scores,
    }


def build_dsa_stimulation_candidates(
    model: Any,
    lesion_idx: int,
    roi_num: int,
    using_steps: int,
    stage1_res: Dict[str, Any],
    best_config: Dict[str, Any],
    roi_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
    stim_strength: Optional[float] = None,
    signs: Sequence[int] = (-1, 1),
    candidate_pool: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """B2: Dynamic Sensitivity Analysis inspired fixed-stimulation baseline."""
    healthy = np.asarray(stage1_res["healthy_traj"], dtype=float)
    lesion = np.asarray(stage1_res["lesion_traj"], dtype=float)
    base_dist = _trajectory_distance(lesion, healthy)

    if stim_strength is None:
        strengths = best_config.get("strengths", [0.2])
        try:
            stim_strength = float(np.median(np.asarray(strengths, dtype=float)))
        except Exception:
            stim_strength = 0.2

    nodes = candidate_pool if candidate_pool is not None else list(range(int(roi_num)))
    nodes = unique_nodes(nodes, roi_num=roi_num, exclude=[lesion_idx])

    scores: Dict[int, float] = {}
    best_signs: Dict[int, int] = {}
    for node in nodes:
        best_gain = -np.inf
        best_s = None
        for sign in signs:
            stim_traj = rollout_fixed_stimulation_under_lesion(
                model=model,
                seed_inputs=stage1_res["seed_inputs"],
                roi_num=roi_num,
                using_steps=using_steps,
                lesion_idx=lesion_idx,
                stim_node=node,
                stim_sign=int(sign),
                stim_strength=float(stim_strength),
                horizon=int(best_config.get("horizon", healthy.shape[1])),
                lesion_strength=float(best_config.get("lesion_strength", 0.5)),
                lesion_mode=str(best_config.get("lesion_mode", "subtract")),
                batch_size=int(best_config.get("batch_size", 256)),
            )
            gain = base_dist - _trajectory_distance(stim_traj, healthy)
            if gain > best_gain:
                best_gain = float(gain)
                best_s = int(sign)
        scores[int(node)] = float(best_gain)
        best_signs[int(node)] = int(best_s) if best_s is not None else 0

    candidate_nodes = _topk_from_scores(scores, roi_num, lesion_idx, top_k, descending=True)
    cand_df = _scores_to_candidate_df(scores, candidate_nodes, "dsa_recovery_gain", roi_names)
    if len(cand_df) > 0:
        cand_df["dsa_best_fixed_sign"] = cand_df["node"].astype(int).map(best_signs)
        cand_df["dsa_fixed_strength"] = float(stim_strength)
    return {
        "candidate_nodes": candidate_nodes,
        "candidate_universe": nodes,
        "candidate_df": cand_df,
        "lesion_core_edges": pd.DataFrame(),
        "n_core_edges": 0,
        "n_direct_lesion_edges": 0,
        "baseline_scores": scores,
        "dsa_best_signs": best_signs,
        "dsa_fixed_strength": float(stim_strength),
    }


def build_nct_state_transition_candidates(
    lesion_idx: int,
    roi_num: int,
    using_steps: int,
    stage1_res: Dict[str, Any],
    context: Dict[str, Any],
    roi_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """B3: Network-control state-transition energy baseline."""
    A = _ensure_linear_control_matrix(context, roi_num=roi_num)
    H = int(context.get("control_horizon", context.get("BEST_CONFIG", {}).get("horizon", stage1_res["healthy_traj"].shape[1])))
    H = min(H, int(stage1_res["healthy_traj"].shape[1]))
    eps = float(context.get("control_energy_eps", 1e-3))
    powers = _matrix_powers(A, H)

    x0 = _stage1_initial_states(stage1_res, roi_num=roi_num, using_steps=using_steps)
    xH = _stage1_terminal_states(stage1_res, kind="healthy")
    x0 = x0[:, : int(roi_num)]
    xH = xH[:, : int(roi_num)]

    scores: Dict[int, float] = {}
    energies: Dict[int, float] = {}
    for node in range(int(roi_num)):
        if node == int(lesion_idx):
            continue
        C = _single_input_reachability_matrix_from_powers(powers, node=node, horizon=H)
        e_vals = []
        for b in range(x0.shape[0]):
            delta = xH[b] - (powers[H] @ x0[b])
            e_vals.append(_regularized_low_rank_energy(delta, C, eps=eps))
        energy = float(np.nanmean(e_vals))
        energies[node] = energy
        scores[node] = -energy

    candidate_nodes = _topk_from_scores(scores, roi_num, lesion_idx, top_k, descending=True)
    cand_df = _scores_to_candidate_df(scores, candidate_nodes, "nct_state_transition_score", roi_names)
    if len(cand_df) > 0:
        cand_df["nct_state_transition_energy"] = cand_df["node"].astype(int).map(energies)
    return {
        "candidate_nodes": candidate_nodes,
        "candidate_universe": unique_nodes(range(int(roi_num)), roi_num, exclude=[lesion_idx]),
        "candidate_df": cand_df,
        "lesion_core_edges": pd.DataFrame(),
        "n_core_edges": 0,
        "n_direct_lesion_edges": 0,
        "baseline_scores": scores,
        "transition_energies": energies,
    }


def build_target_control_candidates(
    lesion_idx: int,
    roi_num: int,
    using_steps: int,
    stage1_res: Dict[str, Any],
    context: Dict[str, Any],
    roi_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """B4: Target-control baseline using the lesion-response domain as target subnetwork."""
    A = _ensure_linear_control_matrix(context, roi_num=roi_num)
    H = int(context.get("control_horizon", context.get("BEST_CONFIG", {}).get("horizon", stage1_res["healthy_traj"].shape[1])))
    H = min(H, int(stage1_res["healthy_traj"].shape[1]))
    eps = float(context.get("control_energy_eps", 1e-3))
    target_m = int(context.get("target_control_m", top_k))
    powers = _matrix_powers(A, H)

    response_scores = _extract_response_scores(stage1_res, roi_num=roi_num)
    target_scores = {j: float(response_scores[j]) for j in range(int(roi_num)) if j != int(lesion_idx)}
    target_nodes = _topk_from_scores(target_scores, roi_num, lesion_idx, target_m, descending=True)
    if len(target_nodes) == 0:
        raise ValueError("Target-control baseline could not build a non-empty response-domain target set.")

    x0 = _stage1_initial_states(stage1_res, roi_num=roi_num, using_steps=using_steps)
    xH = _stage1_terminal_states(stage1_res, kind="healthy")
    x0 = x0[:, : int(roi_num)]
    xH = xH[:, : int(roi_num)]

    scores: Dict[int, float] = {}
    energies: Dict[int, float] = {}
    for node in range(int(roi_num)):
        if node == int(lesion_idx):
            continue
        C_full = _single_input_reachability_matrix_from_powers(powers, node=node, horizon=H)
        C_target = C_full[target_nodes, :]
        e_vals = []
        for b in range(x0.shape[0]):
            delta_full = xH[b] - (powers[H] @ x0[b])
            delta_target = delta_full[target_nodes]
            e_vals.append(_regularized_low_rank_energy(delta_target, C_target, eps=eps))
        energy = float(np.nanmean(e_vals))
        energies[node] = energy
        scores[node] = -energy

    candidate_nodes = _topk_from_scores(scores, roi_num, lesion_idx, top_k, descending=True)
    cand_df = _scores_to_candidate_df(scores, candidate_nodes, "target_control_score", roi_names)
    if len(cand_df) > 0:
        cand_df["target_control_energy"] = cand_df["node"].astype(int).map(energies)
    return {
        "candidate_nodes": candidate_nodes,
        "candidate_universe": unique_nodes(range(int(roi_num)), roi_num, exclude=[lesion_idx]),
        "candidate_df": cand_df,
        "lesion_core_edges": pd.DataFrame(),
        "n_core_edges": 0,
        "n_direct_lesion_edges": 0,
        "baseline_scores": scores,
        "target_control_energies": energies,
        "target_nodes": target_nodes,
        "target_roi_names": map_roi_names(target_nodes, roi_names),
    }


def build_tracking_control_candidates(
    lesion_idx: int,
    roi_num: int,
    using_steps: int,
    stage1_res: Dict[str, Any],
    context: Dict[str, Any],
    roi_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """B5: Trajectory tracking-control baseline."""
    A = _ensure_linear_control_matrix(context, roi_num=roi_num)
    H = int(context.get("control_horizon", context.get("BEST_CONFIG", {}).get("horizon", stage1_res["healthy_traj"].shape[1])))
    H = min(H, int(stage1_res["healthy_traj"].shape[1]))
    lambda_u = float(context.get("tracking_lambda_u", 1e-2))
    powers = _matrix_powers(A, H)

    x0 = _stage1_initial_states(stage1_res, roi_num=roi_num, using_steps=using_steps)
    healthy = np.asarray(stage1_res["healthy_traj"], dtype=float)[:, :H, : int(roi_num)]
    x0 = x0[:, : int(roi_num)]

    scores: Dict[int, float] = {}
    costs: Dict[int, float] = {}
    n = int(roi_num)

    # Build the H-step lower-triangular control matrix node by node.
    for node in range(n):
        if node == int(lesion_idx):
            continue
        # M has shape (H*N, H), column k affects times h >= k+1 through A^{h-k}e_i.
        M = np.zeros((H * n, H), dtype=float)
        for h in range(H):
            for k in range(h + 1):
                M[h * n:(h + 1) * n, k] = powers[h - k][:, node]
        MtM = M.T @ M
        reg = lambda_u * np.eye(H)
        try:
            inv_term = np.linalg.inv(MtM + reg)
        except np.linalg.LinAlgError:
            inv_term = np.linalg.pinv(MtM + reg)
        Mt = M.T

        seed_costs = []
        for b in range(x0.shape[0]):
            free = np.stack([powers[h + 1] @ x0[b] for h in range(H)], axis=0).reshape(-1)
            target = healthy[b].reshape(-1)
            rhs = target - free
            u_star = inv_term @ (Mt @ rhs)
            residual = M @ u_star - rhs
            cost = float(np.dot(residual, residual) + lambda_u * np.dot(u_star, u_star))
            seed_costs.append(cost)
        cost_i = float(np.nanmean(seed_costs))
        costs[node] = cost_i
        scores[node] = -cost_i

    candidate_nodes = _topk_from_scores(scores, roi_num, lesion_idx, top_k, descending=True)
    cand_df = _scores_to_candidate_df(scores, candidate_nodes, "tracking_control_score", roi_names)
    if len(cand_df) > 0:
        cand_df["tracking_control_cost"] = cand_df["node"].astype(int).map(costs)
    return {
        "candidate_nodes": candidate_nodes,
        "candidate_universe": unique_nodes(range(int(roi_num)), roi_num, exclude=[lesion_idx]),
        "candidate_df": cand_df,
        "lesion_core_edges": pd.DataFrame(),
        "n_core_edges": 0,
        "n_direct_lesion_edges": 0,
        "baseline_scores": scores,
        "tracking_costs": costs,
    }


def build_pcmciplus_causal_core_candidates(
    lesion_idx: int,
    roi_num: int,
    stage1_res: Dict[str, Any],
    context: Dict[str, Any],
    roi_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    B6: PCMCI+ causal-core baseline.

    By default this uses tigramite's PCMCI.run_pcmciplus if available. If tigramite
    is not installed, set context['allow_pcmciplus_fallback']=True to use a lagged-
    correlation proxy so the rest of the pipeline can be tested.
    """
    signals = context.get("signals", None)
    if signals is None:
        raise ValueError("signals is required for PCMCI+ baseline.")
    X = _as_2d_time_by_node(signals, roi_num=roi_num)
    X = _zscore_columns(X)

    # Candidate subsystem: lesion + top lagged-proximal nodes. This avoids all-ROI PCMCI+ cost.
    max_nodes = int(context.get("pcmci_plus_max_nodes", context.get("BEST_CONFIG", {}).get("max_candidate_nodes", 50)))
    tau_max = int(context.get("pcmci_plus_tau_max", context.get("BEST_CONFIG", {}).get("tau_max", 1)))
    pc_alpha = float(context.get("pcmci_plus_pc_alpha", context.get("BEST_CONFIG", {}).get("pc_alpha", 0.01)))
    eta = float(context.get("pcmci_plus_incident_eta", 0.25))

    lag_scores = {}
    for j in range(int(roi_num)):
        if j == int(lesion_idx):
            continue
        try:
            in_score = abs(float(np.corrcoef(X[:-1, j], X[1:, int(lesion_idx)])[0, 1]))
            out_score = abs(float(np.corrcoef(X[:-1, int(lesion_idx)], X[1:, j])[0, 1]))
            lag_scores[j] = np.nan_to_num(max(in_score, out_score), nan=0.0)
        except Exception:
            lag_scores[j] = 0.0
    subsystem_nodes = [int(lesion_idx)] + _topk_from_scores(lag_scores, roi_num, lesion_idx, max_nodes - 1, descending=True)
    subsystem_nodes = unique_nodes(subsystem_nodes, roi_num=roi_num, exclude=[])
    subX = X[:, subsystem_nodes]
    local_index = {node: k for k, node in enumerate(subsystem_nodes)}

    # Run PCMCI+ if tigramite is available.
    edge_strengths: Dict[Tuple[int, int], float] = {}
    try:
        from tigramite.data_processing import DataFrame as TigramiteDataFrame
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI

        dataframe = TigramiteDataFrame(subX)
        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
        val_matrix = results.get("val_matrix", None)
        graph = results.get("graph", None)
        if val_matrix is None:
            raise RuntimeError("PCMCI+ returned no val_matrix.")

        n_sub = len(subsystem_nodes)
        # val_matrix convention is commonly [source, target, tau]. Use non-zero/marked graph entries if available.
        for src_local in range(n_sub):
            for dst_local in range(n_sub):
                if src_local == dst_local:
                    continue
                vals = []
                for tau in range(val_matrix.shape[2]):
                    keep = True
                    if graph is not None:
                        try:
                            g = str(graph[src_local, dst_local, tau])
                            keep = (g not in ["", "--", "---", "0"])
                        except Exception:
                            keep = True
                    if keep:
                        vals.append(abs(float(val_matrix[src_local, dst_local, tau])))
                if vals:
                    src = subsystem_nodes[src_local]
                    dst = subsystem_nodes[dst_local]
                    edge_strengths[(src, dst)] = float(np.nanmax(vals))
    except Exception as e:
        if not bool(context.get("allow_pcmciplus_fallback", False)):
            raise ImportError(
                "PCMCI+ baseline requires tigramite. Install tigramite or set "
                "allow_pcmciplus_fallback=True for a lagged-correlation proxy. "
                f"Original error: {e}"
            )
        # Fallback: pairwise lagged-correlation proxy, useful only for pipeline testing.
        warnings.warn(f"PCMCI+ unavailable; using lagged-correlation fallback. Error: {e}")
        for src in subsystem_nodes:
            for dst in subsystem_nodes:
                if src == dst:
                    continue
                vals = []
                for tau in range(1, tau_max + 1):
                    try:
                        vals.append(abs(float(np.corrcoef(X[:-tau, src], X[tau:, dst])[0, 1])))
                    except Exception:
                        pass
                if vals:
                    edge_strengths[(src, dst)] = float(np.nan_to_num(np.nanmax(vals), nan=0.0))

    # Lesion-proximal plus incident causal strength.
    scores: Dict[int, float] = {}
    for node in subsystem_nodes:
        if node == int(lesion_idx):
            continue
        direct = edge_strengths.get((int(lesion_idx), node), 0.0) + edge_strengths.get((node, int(lesion_idx)), 0.0)
        incident = 0.0
        for other in subsystem_nodes:
            if other in [node, int(lesion_idx)]:
                continue
            incident += edge_strengths.get((other, node), 0.0) + edge_strengths.get((node, other), 0.0)
        scores[node] = float(direct + eta * incident)

    candidate_nodes = _topk_from_scores(scores, roi_num, lesion_idx, top_k, descending=True)
    cand_df = _scores_to_candidate_df(scores, candidate_nodes, "pcmciplus_causal_score", roi_names)

    edge_df = pd.DataFrame([
        {"src": int(s), "dst": int(d), "abs_val": float(v), "val": float(v), "tau": np.nan}
        for (s, d), v in edge_strengths.items()
    ])
    return {
        "candidate_nodes": candidate_nodes,
        "candidate_universe": unique_nodes(subsystem_nodes, roi_num=roi_num, exclude=[lesion_idx]),
        "candidate_df": cand_df,
        "lesion_core_edges": edge_df,
        "n_core_edges": len(edge_df),
        "n_direct_lesion_edges": int(((edge_df.get("src", pd.Series(dtype=int)) == int(lesion_idx)) | (edge_df.get("dst", pd.Series(dtype=int)) == int(lesion_idx))).sum()) if len(edge_df) > 0 else 0,
        "baseline_scores": scores,
        "pcmciplus_edge_strengths": edge_strengths,
        "pcmciplus_subsystem_nodes": subsystem_nodes,
    }


def build_literature_baseline_candidates(
    baseline_cfg: LiteratureBaselineVariant,
    lesion_idx: int,
    stage1_res: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Dispatch candidate construction for one literature-based baseline."""
    roi_num = int(context["roi_num"])
    using_steps = int(context["using_steps"])
    top_k = int(context.get("top_m", 20))
    roi_names = context.get("roi_names", None)
    empirical_FC = context.get("empirical_FC", None)
    signals = context.get("signals", None)
    best_config = context.get("BEST_CONFIG", {})
    model = context.get("model", None)

    name = baseline_cfg.variant
    if name == "Global-FC hub baseline":
        return build_global_fc_hub_candidates(
            lesion_idx=lesion_idx,
            roi_num=roi_num,
            empirical_FC=empirical_FC,
            signals=signals,
            roi_names=roi_names,
            top_k=top_k,
        )
    if name == "Lagged-only lesion-proximal baseline":
        return build_lagged_only_lesion_proximal_candidates(
            lesion_idx=lesion_idx,
            roi_num=roi_num,
            signals=signals,
            roi_names=roi_names,
            top_k=top_k,
        )
    if name == "LNM-FC baseline":
        return build_lnm_fc_candidates(
            lesion_idx=lesion_idx,
            roi_num=roi_num,
            empirical_FC=empirical_FC,
            signals=signals,
            roi_names=roi_names,
            top_k=top_k,
        )
    if name == "DSA-stimulation baseline":
        if model is None:
            raise ValueError("model is required for DSA-stimulation baseline.")
        return build_dsa_stimulation_candidates(
            model=model,
            lesion_idx=lesion_idx,
            roi_num=roi_num,
            using_steps=using_steps,
            stage1_res=stage1_res,
            best_config=best_config,
            roi_names=roi_names,
            top_k=top_k,
            stim_strength=context.get("dsa_stim_strength", None),
            signs=context.get("dsa_signs", (-1, 1)),
            candidate_pool=context.get("dsa_candidate_pool", None),
        )
    if name == "NCT-state-transition baseline":
        return build_nct_state_transition_candidates(
            lesion_idx=lesion_idx,
            roi_num=roi_num,
            using_steps=using_steps,
            stage1_res=stage1_res,
            context=context,
            roi_names=roi_names,
            top_k=top_k,
        )
    if name == "Target-control baseline":
        return build_target_control_candidates(
            lesion_idx=lesion_idx,
            roi_num=roi_num,
            using_steps=using_steps,
            stage1_res=stage1_res,
            context=context,
            roi_names=roi_names,
            top_k=top_k,
        )
    if name == "Tracking-control baseline":
        return build_tracking_control_candidates(
            lesion_idx=lesion_idx,
            roi_num=roi_num,
            using_steps=using_steps,
            stage1_res=stage1_res,
            context=context,
            roi_names=roi_names,
            top_k=top_k,
        )
    if name == "Random-intervention baseline":
        raise ValueError(
            "Random-intervention baseline is a direct random-policy baseline and must be "
            "handled by the literature-baseline workflow, not by candidate construction."
        )
    if name == "PCMCI+ causal-core baseline":
        return build_pcmciplus_causal_core_candidates(
            lesion_idx=lesion_idx,
            roi_num=roi_num,
            stage1_res=stage1_res,
            context=context,
            roi_names=roi_names,
            top_k=top_k,
        )
    raise ValueError(f"Unknown literature baseline variant: {name}")


# -----------------------------
# 8.4 Literature-baseline workflow
# -----------------------------

def run_multi_lesion_literature_baseline_study(
    model: Any,
    lesion_list: Sequence[int],
    roi_num: int,
    using_steps: int,
    empirical_FC: Optional[np.ndarray],
    best_config: Dict[str, Any],
    stage1_provider: Union[Callable[[int], Dict[str, Any]], Dict[int, Dict[str, Any]]],
    signals: Optional[np.ndarray] = None,
    roi_names: Optional[Sequence[str]] = None,
    baselines: Optional[List[LiteratureBaselineVariant]] = None,
    top_m: int = 20,
    out_dir: str = "./ablation_literature_baselines",
    prefer_multi: bool = True,
    success_traj_threshold: float = 0.0,
    success_fc_threshold: float = 0.0,
    control_A: Optional[np.ndarray] = None,
    control_ridge: float = 1e-2,
    control_energy_eps: float = 1e-3,
    control_horizon: Optional[int] = None,
    target_control_m: Optional[int] = None,
    tracking_lambda_u: float = 1e-2,
    dsa_stim_strength: Optional[float] = None,
    dsa_signs: Sequence[int] = (-1, 1),
    dsa_candidate_pool: Optional[Sequence[int]] = None,
    pcmci_plus_max_nodes: Optional[int] = None,
    pcmci_plus_tau_max: Optional[int] = None,
    pcmci_plus_pc_alpha: Optional[float] = None,
    allow_pcmciplus_fallback: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run the six literature-based baselines across multiple lesion nodes.

    Returns
    -------
    per_lesion_df:
        One row per lesion and literature baseline.
    summary_df:
        Cross-lesion mean/std summary.
    candidate_table_df:
        Candidate-node details for each lesion and baseline.
    raw_outputs:
        Nested dict with stage2-like candidate outputs and Stage-3 outputs.
    """
    os.makedirs(out_dir, exist_ok=True)
    baselines = baselines or build_literature_baseline_variants()

    context = {
        "BEST_CONFIG": best_config,
        "roi_num": int(roi_num),
        "using_steps": int(using_steps),
        "empirical_FC": empirical_FC,
        "signals": signals,
        "roi_names": roi_names,
        "model": model,
        "top_m": int(top_m),
        "out_dir": out_dir,
        "control_A": control_A,
        "control_ridge": float(control_ridge),
        "control_energy_eps": float(control_energy_eps),
        "control_horizon": int(control_horizon) if control_horizon is not None else int(best_config.get("horizon", 20)),
        "target_control_m": int(target_control_m) if target_control_m is not None else int(top_m),
        "tracking_lambda_u": float(tracking_lambda_u),
        "dsa_stim_strength": dsa_stim_strength,
        "dsa_signs": dsa_signs,
        "dsa_candidate_pool": dsa_candidate_pool,
        "pcmci_plus_max_nodes": pcmci_plus_max_nodes if pcmci_plus_max_nodes is not None else int(best_config.get("max_candidate_nodes", 50)),
        "pcmci_plus_tau_max": pcmci_plus_tau_max if pcmci_plus_tau_max is not None else int(best_config.get("tau_max", 1)),
        "pcmci_plus_pc_alpha": pcmci_plus_pc_alpha if pcmci_plus_pc_alpha is not None else float(best_config.get("pc_alpha", 0.01)),
        "allow_pcmciplus_fallback": bool(allow_pcmciplus_fallback),
    }

    per_lesion_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    raw_outputs: Dict[str, Any] = {}

    for lesion_idx in lesion_list:
        lesion_idx = int(lesion_idx)
        lesion_name = roi_names[lesion_idx] if roi_names is not None and lesion_idx < len(roi_names) else f"ROI_{lesion_idx}"
        print("=" * 100)
        print(f"[Literature baseline] lesion_idx={lesion_idx}, lesion_name={lesion_name}")
        print("=" * 100)

        stage1_res = get_stage1_for_lesion(lesion_idx, stage1_provider)
        raw_outputs.setdefault(lesion_idx, {"stage1_res": stage1_res, "baselines": {}})

        for baseline_cfg in baselines:
            print(f"[Baseline] {baseline_cfg.variant}")
            pseudo_variant = _pseudo_ablation_variant_from_literature(baseline_cfg)
            try:
                stage2_res = build_literature_baseline_candidates(
                    baseline_cfg=baseline_cfg,
                    lesion_idx=lesion_idx,
                    stage1_res=stage1_res,
                    context=context,
                )
                stage2_res = normalize_stage2_output_for_ablation(
                    stage2_res,
                    lesion_idx=lesion_idx,
                    roi_num=roi_num,
                    empirical_FC=empirical_FC,
                    top_m=top_m,
                )
                candidate_nodes = stage2_res["candidate_nodes"]
                if len(candidate_nodes) == 0:
                    raise ValueError("empty candidate_nodes returned by literature baseline")

                # Candidate details for output table.
                cand_df = stage2_res.get("candidate_df", None)
                if isinstance(cand_df, pd.DataFrame) and len(cand_df) > 0:
                    for _, rr in cand_df.iterrows():
                        candidate_rows.append({
                            "lesion_idx": lesion_idx,
                            "lesion_name": lesion_name,
                            "variant": baseline_cfg.variant,
                            "rank": int(rr.get("rank", len(candidate_rows) + 1)) if pd.notna(rr.get("rank", np.nan)) else np.nan,
                            "node": int(rr.get("node")) if pd.notna(rr.get("node", np.nan)) else np.nan,
                            "roi_name": rr.get("roi_name", None),
                            "candidate_score": _safe_float(rr.get("candidate_score", np.nan)),
                        })

                stage3_res = run_stage3_for_candidates(
                    model=model,
                    candidate_nodes=candidate_nodes,
                    stage1_res=stage1_res,
                    lesion_idx=lesion_idx,
                    roi_num=roi_num,
                    using_steps=using_steps,
                    best_config=best_config,
                )

                row = build_ablation_row(
                    lesion_idx=lesion_idx,
                    lesion_name=lesion_name,
                    variant_cfg=pseudo_variant,
                    stage2_res=stage2_res,
                    stage3_res=stage3_res,
                    roi_names=roi_names,
                    repeat_id=None,
                    status="ok",
                    prefer_multi=prefer_multi,
                    empirical_FC=empirical_FC,
                )
                row["baseline_group"] = "literature"
                row["literature_basis"] = baseline_cfg.literature_basis
                row["candidate_score_definition"] = baseline_cfg.candidate_score
                per_lesion_rows.append(row)

                raw_outputs[lesion_idx]["baselines"][baseline_cfg.variant] = {
                    "stage2_res": stage2_res,
                    "stage3_res": stage3_res,
                }

            except Exception as e:
                warnings.warn(f"Literature baseline failed: lesion={lesion_idx}, baseline={baseline_cfg.variant}, error={e}")
                fail_row = {
                    "lesion_idx": lesion_idx,
                    "lesion_name": lesion_name,
                    "variant": baseline_cfg.variant,
                    "status": f"failed: {e}",
                    "baseline_group": "literature",
                    "literature_basis": baseline_cfg.literature_basis,
                    "candidate_score_definition": baseline_cfg.candidate_score,
                }
                for key in metrics_to_fields({}).keys():
                    fail_row[key] = np.nan
                per_lesion_rows.append(fail_row)

    per_lesion_df = pd.DataFrame(per_lesion_rows)
    candidate_table_df = pd.DataFrame(candidate_rows)

    if len(per_lesion_df) > 0:
        per_lesion_df["success"] = (
            (pd.to_numeric(per_lesion_df.get("traj_recovery"), errors="coerce") > float(success_traj_threshold))
            & (pd.to_numeric(per_lesion_df.get("fc_recovery"), errors="coerce") > float(success_fc_threshold))
            & (per_lesion_df.get("status") == "ok")
        )

    summary_df = summarize_ablation_results(per_lesion_df)

    per_lesion_path = os.path.join(out_dir, "literature_baseline_per_lesion_results.csv")
    candidate_path = os.path.join(out_dir, "literature_baseline_candidate_table.csv")
    summary_path = os.path.join(out_dir, "literature_baseline_summary_mean_std.csv")
    design_path = os.path.join(out_dir, "literature_baseline_design_table.csv")

    per_lesion_df.to_csv(per_lesion_path, index=False)
    candidate_table_df.to_csv(candidate_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    literature_baseline_design_table().to_csv(design_path, index=False)

    print("Saved:")
    print("  ", per_lesion_path)
    print("  ", candidate_path)
    print("  ", summary_path)
    print("  ", design_path)

    return per_lesion_df, summary_df, candidate_table_df, raw_outputs


# -----------------------------
# 8.5 Combined summaries and plotting
# -----------------------------

def combine_ablation_and_literature_results(
    ablation_per_lesion_df: pd.DataFrame,
    literature_per_lesion_df: pd.DataFrame,
    out_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combine internal module ablation rows and literature-baseline rows."""
    df_list = []
    if ablation_per_lesion_df is not None and len(ablation_per_lesion_df) > 0:
        a = ablation_per_lesion_df.copy()
        if "baseline_group" not in a.columns:
            a["baseline_group"] = "internal_ablation"
        df_list.append(a)
    if literature_per_lesion_df is not None and len(literature_per_lesion_df) > 0:
        l = literature_per_lesion_df.copy()
        if "baseline_group" not in l.columns:
            l["baseline_group"] = "literature"
        df_list.append(l)
    combined = pd.concat(df_list, ignore_index=True, sort=False) if df_list else pd.DataFrame()
    summary = summarize_ablation_results(combined)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        combined.to_csv(os.path.join(out_dir, "combined_4_6_per_lesion_results.csv"), index=False)
        summary.to_csv(os.path.join(out_dir, "combined_4_6_summary_mean_std.csv"), index=False)
    return combined, summary


def best_node_overlap_with_reference(
    per_lesion_df: pd.DataFrame,
    reference_variant: str = "Full method",
) -> pd.DataFrame:
    """Compute whether each variant selects the same best node as a reference method."""
    if per_lesion_df is None or len(per_lesion_df) == 0:
        return pd.DataFrame()
    df = per_lesion_df.copy()
    ok = df[df.get("status", "ok") == "ok"].copy()
    ref = ok[ok["variant"] == reference_variant][["lesion_idx", "best_node"]].rename(columns={"best_node": "reference_best_node"})
    out = ok.merge(ref, on="lesion_idx", how="left")
    out["same_best_node_as_reference"] = out["best_node"].astype("object") == out["reference_best_node"].astype("object")
    return out[["lesion_idx", "lesion_name", "variant", "best_node", "reference_best_node", "same_best_node_as_reference"]]


def summarize_best_node_overlap(
    per_lesion_df: pd.DataFrame,
    reference_variant: str = "Full method",
) -> pd.DataFrame:
    """Return overlap ratio for each variant against the reference best node."""
    detail = best_node_overlap_with_reference(per_lesion_df, reference_variant=reference_variant)
    if len(detail) == 0:
        return pd.DataFrame()
    rows = []
    for variant, sub in detail.groupby("variant"):
        if variant == reference_variant:
            continue
        vals = sub["same_best_node_as_reference"].astype(float)
        rows.append({
            "variant": variant,
            "n_lesions": int(sub["lesion_idx"].nunique()),
            "best_node_overlap_rate": float(vals.mean()),
        })
    return pd.DataFrame(rows)


def candidate_hit_rate_against_reference(
    per_lesion_df: pd.DataFrame,
    reference_variant: str = "Full method",
) -> pd.DataFrame:
    """Check whether each variant's candidate set contains the reference best node."""
    if per_lesion_df is None or len(per_lesion_df) == 0:
        return pd.DataFrame()
    df = per_lesion_df.copy()
    ok = df[df.get("status", "ok") == "ok"].copy()
    ref = ok[ok["variant"] == reference_variant][["lesion_idx", "best_node"]].rename(columns={"best_node": "reference_best_node"})
    out = ok.merge(ref, on="lesion_idx", how="left")

    def _contains(row):
        nodes = row.get("candidate_nodes", [])
        ref_node = row.get("reference_best_node", None)
        if ref_node is None or pd.isna(ref_node):
            return False
        if isinstance(nodes, str):
            return False
        try:
            return int(ref_node) in [int(x) for x in nodes]
        except Exception:
            return False

    out["candidate_contains_reference_best_node"] = out.apply(_contains, axis=1)
    rows = []
    for variant, sub in out.groupby("variant"):
        vals = sub["candidate_contains_reference_best_node"].astype(float)
        rows.append({
            "variant": variant,
            "n_lesions": int(sub["lesion_idx"].nunique()),
            "candidate_hit_rate": float(vals.mean()),
        })
    return pd.DataFrame(rows)


def plot_best_node_overlap_heatmap(
    per_lesion_df: pd.DataFrame,
    reference_variant: str = "Full method",
    save_path: Optional[str] = None,
):
    """Plot lesion-by-variant heatmap of best-node agreement with a reference method."""
    plt.close("all")
    detail = best_node_overlap_with_reference(per_lesion_df, reference_variant=reference_variant)
    if detail is None or len(detail) == 0:
        raise ValueError("No overlap detail available.")
    detail = detail[detail["variant"] != reference_variant].copy()
    pivot = detail.pivot_table(
        index="lesion_idx",
        columns="variant",
        values="same_best_node_as_reference",
        aggfunc="mean",
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pivot.columns)), 4.8))
    im = ax.imshow(pivot.values.astype(float), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index.tolist()])
    ax.set_xlabel("Variant")
    ax.set_ylabel("Lesion node")
    ax.set_title(f"Best-node agreement with {reference_variant}")
    fig.colorbar(im, ax=ax, label="same best node")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Override the earlier summarizer so literature baselines keep their names and order.
def summarize_ablation_results(per_lesion_df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[no-redef]
    """Cross-lesion mean/std summary for internal and literature variants."""
    if per_lesion_df is None or len(per_lesion_df) == 0:
        return pd.DataFrame()

    df = per_lesion_df.copy()
    df = df[df.get("status", "ok") == "ok"].copy()

    metrics = [
        "objective", "traj_recovery", "fc_recovery",
        "D_policy_traj", "D_policy_fc",
        "policy_size", "n_candidate_nodes", "n_core_edges", "n_direct_lesion_edges",
    ]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for variant, sub in df.groupby("variant", dropna=False):
        row = {
            "variant": variant,
            "n_lesions": int(sub["lesion_idx"].nunique()) if "lesion_idx" in sub.columns else len(sub),
        }
        if "baseline_group" in sub.columns:
            groups = [str(x) for x in sub["baseline_group"].dropna().unique().tolist()]
            row["baseline_group"] = groups[0] if len(groups) == 1 else ";".join(groups)
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce")
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std())
        if "success" in sub.columns:
            row["success_rate"] = float(sub["success"].mean())
        rows.append(row)

    out = pd.DataFrame(rows)
    order = [
        "Full method",
        "w/o response prior",
        "w/o LPCMCI",
        "Causal-only pruning",
        "FC-only pruning",
        "Random ranking",
    ] + LITERATURE_BASELINE_ORDER
    if "variant" in out.columns:
        order_map = {name: i for i, name in enumerate(order)}
        out["_order"] = out["variant"].map(lambda x: order_map.get(str(x), len(order_map) + 1))
        out = out.sort_values(["_order", "variant"]).drop(columns=["_order"]).reset_index(drop=True)
    return out


def print_literature_baseline_usage_template():
    """Print notebook code for running the new 4.6.2 literature baselines."""
    print(r'''
# 4.6.2 Literature-based baseline comparison
# Run after your original 4.6 internal ablation has produced per_lesion_df and summary_df.

literature_per_lesion_df, literature_summary_df, literature_candidate_df, literature_raw_outputs = \
    run_multi_lesion_literature_baseline_study(
        model=ANN,
        lesion_list=lesion_list,
        roi_num=ROI_num,
        using_steps=using_steps,
        empirical_FC=empirical_FC,
        best_config=BEST_CONFIG_ABLATION,
        stage1_provider=stage1_provider,
        signals=signals,
        roi_names=roi_names,
        top_m=20,
        out_dir="./ablation_multi_lesion_literature",

        # Shared control-baseline settings
        control_ridge=1e-2,
        control_energy_eps=1e-3,
        control_horizon=BEST_CONFIG_ABLATION.get("horizon", 20),
        target_control_m=20,
        tracking_lambda_u=1e-2,

        # DSA baseline: fixed stimulation only, before unified Stage-3 search
        dsa_stim_strength=0.2,
        dsa_signs=(-1, 1),

        # PCMCI+ baseline. If tigramite is unavailable, keep False for real experiments.
        allow_pcmciplus_fallback=False,
    )

# Combine original internal ablation and six literature baselines.
combined_4_6_df, combined_4_6_summary_df = combine_ablation_and_literature_results(
    ablation_per_lesion_df=per_lesion_df,
    literature_per_lesion_df=literature_per_lesion_df,
    out_dir="./ablation_multi_lesion_combined",
)

# Paper tables
literature_design_table = literature_baseline_design_table()
literature_paper_table = make_ablation_paper_table(literature_summary_df)
combined_paper_table = make_ablation_paper_table(combined_4_6_summary_df)
combined_formatted_table = format_mean_std_table(combined_4_6_summary_df)

# Figures: same style as original 4.6
plot_ablation_three_metrics(
    combined_4_6_summary_df,
    out_dir="./ablation_multi_lesion_combined",
)
plot_policy_distance_ablation(
    combined_4_6_summary_df,
    out_dir="./ablation_multi_lesion_combined",
)

# Best-node consistency and candidate hit-rate tables
best_node_overlap_detail = best_node_overlap_with_reference(combined_4_6_df, reference_variant="Full method")
best_node_overlap_summary = summarize_best_node_overlap(combined_4_6_df, reference_variant="Full method")
candidate_hit_summary = candidate_hit_rate_against_reference(combined_4_6_df, reference_variant="Full method")

best_node_overlap_detail.to_csv("./ablation_multi_lesion_combined/best_node_overlap_detail.csv", index=False)
best_node_overlap_summary.to_csv("./ablation_multi_lesion_combined/best_node_overlap_summary.csv", index=False)
candidate_hit_summary.to_csv("./ablation_multi_lesion_combined/candidate_hit_rate_summary.csv", index=False)

plot_best_node_overlap_heatmap(
    combined_4_6_df,
    reference_variant="Full method",
    save_path="./ablation_multi_lesion_combined/best_node_overlap_heatmap.png",
)
''')


# =========================================================
# 8.6 Baseline-level checkpoint workflow
# =========================================================

def _safe_baseline_filename(name: str) -> str:
    """Convert a baseline name to a filesystem-safe filename stem."""
    import re
    stem = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")
    return stem.lower() or "baseline"


def _read_csv_if_exists(path: str) -> pd.DataFrame:
    """Read CSV if it exists and is non-empty; otherwise return an empty DataFrame."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _atomic_to_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """Write CSV through a temporary file to reduce corruption risk after interruption."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=index)
    os.replace(tmp_path, path)


def _drop_pair_rows(rows: List[Dict[str, Any]], lesion_idx: int, variant: str) -> List[Dict[str, Any]]:
    """Remove existing rows for one (lesion_idx, variant) pair before replacing it."""
    out = []
    for r in rows:
        try:
            same_lesion = int(r.get("lesion_idx")) == int(lesion_idx)
        except Exception:
            same_lesion = False
        same_variant = str(r.get("variant")) == str(variant)
        if not (same_lesion and same_variant):
            out.append(r)
    return out


def _checkpoint_literature_outputs(
    out_dir: str,
    per_lesion_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    completed_baseline: Optional[str] = None,
    save_per_baseline_files: bool = True,
    success_traj_threshold: float = 0.0,
    success_fc_threshold: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save all accumulated literature-baseline rows and optional per-baseline files."""
    os.makedirs(out_dir, exist_ok=True)

    per_lesion_df = pd.DataFrame(per_lesion_rows)
    candidate_table_df = pd.DataFrame(candidate_rows)

    if len(per_lesion_df) > 0:
        per_lesion_df["success"] = (
            (pd.to_numeric(per_lesion_df.get("traj_recovery"), errors="coerce") > float(success_traj_threshold))
            & (pd.to_numeric(per_lesion_df.get("fc_recovery"), errors="coerce") > float(success_fc_threshold))
            & (per_lesion_df.get("status") == "ok")
        )

    summary_df = summarize_ablation_results(per_lesion_df)

    per_lesion_path = os.path.join(out_dir, "literature_baseline_per_lesion_results.csv")
    candidate_path = os.path.join(out_dir, "literature_baseline_candidate_table.csv")
    summary_path = os.path.join(out_dir, "literature_baseline_summary_mean_std.csv")
    design_path = os.path.join(out_dir, "literature_baseline_design_table.csv")

    _atomic_to_csv(per_lesion_df, per_lesion_path, index=False)
    _atomic_to_csv(candidate_table_df, candidate_path, index=False)
    _atomic_to_csv(summary_df, summary_path, index=False)
    _atomic_to_csv(literature_baseline_design_table(), design_path, index=False)

    if save_per_baseline_files and completed_baseline is not None and len(per_lesion_df) > 0:
        stem = _safe_baseline_filename(completed_baseline)
        baseline_dir = os.path.join(out_dir, "per_baseline")
        os.makedirs(baseline_dir, exist_ok=True)

        one_per = per_lesion_df[per_lesion_df["variant"].astype(str) == str(completed_baseline)].copy()
        one_cand = candidate_table_df[candidate_table_df["variant"].astype(str) == str(completed_baseline)].copy() \
            if len(candidate_table_df) > 0 and "variant" in candidate_table_df.columns else pd.DataFrame()
        one_summary = summarize_ablation_results(one_per)

        _atomic_to_csv(one_per, os.path.join(baseline_dir, f"{stem}_per_lesion_results.csv"), index=False)
        _atomic_to_csv(one_cand, os.path.join(baseline_dir, f"{stem}_candidate_table.csv"), index=False)
        _atomic_to_csv(one_summary, os.path.join(baseline_dir, f"{stem}_summary_mean_std.csv"), index=False)

    return per_lesion_df, summary_df, candidate_table_df


def _build_fc_top_pool_for_lesion(
    empirical_FC: Optional[np.ndarray],
    lesion_idx: int,
    roi_num: int,
    pool_size: int,
) -> Optional[List[int]]:
    """Build an FC Top-M pool for DSA candidate pre-filtering."""
    if empirical_FC is None:
        return None
    fc = np.asarray(empirical_FC, dtype=float)
    if fc.ndim != 2 or fc.shape[0] <= int(lesion_idx):
        return None
    scores = np.abs(fc[int(lesion_idx), :]).astype(float)
    scores[int(lesion_idx)] = -np.inf
    nodes = np.argsort(scores)[::-1].tolist()
    return unique_nodes(nodes, roi_num=roi_num, exclude=[lesion_idx])[: int(pool_size)]


def run_multi_lesion_literature_baseline_study_by_baseline_checkpointed(
    model: Any,
    lesion_list: Sequence[int],
    roi_num: int,
    using_steps: int,
    empirical_FC: Optional[np.ndarray],
    best_config: Dict[str, Any],
    stage1_provider: Union[Callable[[int], Dict[str, Any]], Dict[int, Dict[str, Any]]],
    signals: Optional[np.ndarray] = None,
    roi_names: Optional[Sequence[str]] = None,
    baselines: Optional[List[LiteratureBaselineVariant]] = None,
    top_m: int = 20,
    out_dir: str = "./ablation_literature_baselines_by_baseline",
    prefer_multi: bool = True,
    success_traj_threshold: float = 0.0,
    success_fc_threshold: float = 0.0,
    control_A: Optional[np.ndarray] = None,
    control_ridge: float = 1e-2,
    control_energy_eps: float = 1e-3,
    control_horizon: Optional[int] = None,
    target_control_m: Optional[int] = None,
    tracking_lambda_u: float = 1e-2,
    dsa_stim_strength: Optional[float] = None,
    dsa_signs: Sequence[int] = (-1, 1),
    dsa_candidate_pool: Optional[Sequence[int]] = None,
    dsa_candidate_pool_by_lesion: Optional[Dict[int, Sequence[int]]] = None,
    dsa_fc_pool_size: Optional[int] = None,
    pcmci_plus_max_nodes: Optional[int] = None,
    pcmci_plus_tau_max: Optional[int] = None,
    pcmci_plus_pc_alpha: Optional[float] = None,
    allow_pcmciplus_fallback: bool = False,
    random_intervention_repeats: int = 50,
    random_intervention_seed: Optional[int] = None,
    random_intervention_node_pool: Optional[Sequence[int]] = None,
    random_intervention_signs: Optional[Sequence[Any]] = None,
    random_intervention_strengths: Optional[Sequence[Any]] = None,
    random_intervention_schedules: Optional[Sequence[Any]] = None,
    resume: bool = True,
    skip_failed_when_resume: bool = False,
    save_per_baseline_files: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run literature baselines with baseline-level checkpoints.

    Loop order:
        for baseline in baselines:
            for lesion in lesion_list:
                run one lesion under this baseline
            save checkpoint for the completed baseline

    This matches the requirement: eight lesion nodes run one baseline, then save once.
    """
    import time

    os.makedirs(out_dir, exist_ok=True)
    baselines = baselines or build_literature_baseline_variants()

    per_lesion_path = os.path.join(out_dir, "literature_baseline_per_lesion_results.csv")
    candidate_path = os.path.join(out_dir, "literature_baseline_candidate_table.csv")
    random_repeat_path = os.path.join(out_dir, "random_intervention_repeat_results.csv")

    existing_per = _read_csv_if_exists(per_lesion_path) if resume else pd.DataFrame()
    existing_cand = _read_csv_if_exists(candidate_path) if resume else pd.DataFrame()
    existing_random = _read_csv_if_exists(random_repeat_path) if resume else pd.DataFrame()

    per_lesion_rows: List[Dict[str, Any]] = existing_per.to_dict("records") if len(existing_per) > 0 else []
    candidate_rows: List[Dict[str, Any]] = existing_cand.to_dict("records") if len(existing_cand) > 0 else []
    random_intervention_rows: List[Dict[str, Any]] = existing_random.to_dict("records") if len(existing_random) > 0 else []
    raw_outputs: Dict[str, Any] = {}

    context_base = {
        "BEST_CONFIG": best_config,
        "roi_num": int(roi_num),
        "using_steps": int(using_steps),
        "empirical_FC": empirical_FC,
        "signals": signals,
        "roi_names": roi_names,
        "model": model,
        "top_m": int(top_m),
        "out_dir": out_dir,
        "control_A": control_A,
        "control_ridge": float(control_ridge),
        "control_energy_eps": float(control_energy_eps),
        "control_horizon": int(control_horizon) if control_horizon is not None else int(best_config.get("horizon", 20)),
        "target_control_m": int(target_control_m) if target_control_m is not None else int(top_m),
        "tracking_lambda_u": float(tracking_lambda_u),
        "dsa_stim_strength": dsa_stim_strength,
        "dsa_signs": dsa_signs,
        "dsa_candidate_pool": dsa_candidate_pool,
        "pcmci_plus_max_nodes": pcmci_plus_max_nodes if pcmci_plus_max_nodes is not None else int(best_config.get("max_candidate_nodes", 50)),
        "pcmci_plus_tau_max": pcmci_plus_tau_max if pcmci_plus_tau_max is not None else int(best_config.get("tau_max", 1)),
        "pcmci_plus_pc_alpha": pcmci_plus_pc_alpha if pcmci_plus_pc_alpha is not None else float(best_config.get("pc_alpha", 0.01)),
        "allow_pcmciplus_fallback": bool(allow_pcmciplus_fallback),
    }

    def _row_is_done(row: Dict[str, Any]) -> bool:
        if str(row.get("status")) == "ok":
            return True
        return bool(skip_failed_when_resume) and str(row.get("status", "")).startswith("failed")

    def _has_done_pair(lesion_idx: int, variant: str) -> bool:
        for r in per_lesion_rows:
            try:
                same_lesion = int(r.get("lesion_idx")) == int(lesion_idx)
            except Exception:
                same_lesion = False
            if same_lesion and str(r.get("variant")) == str(variant) and _row_is_done(r):
                return True
        return False

    for baseline_cfg in baselines:
        variant_name = baseline_cfg.variant
        baseline_t0 = time.time()

        done_flags = [_has_done_pair(int(l), variant_name) for l in lesion_list]
        if resume and all(done_flags):
            print("=" * 100)
            print(f"[Baseline checkpoint] SKIP completed baseline: {variant_name}")
            print("=" * 100)
            continue

        print("=" * 100)
        print(f"[Baseline checkpoint] START baseline: {variant_name}")
        print("=" * 100)

        pseudo_variant = _pseudo_ablation_variant_from_literature(baseline_cfg)

        for lesion_idx in lesion_list:
            lesion_idx = int(lesion_idx)
            lesion_name = roi_names[lesion_idx] if roi_names is not None and lesion_idx < len(roi_names) else f"ROI_{lesion_idx}"

            if resume and _has_done_pair(lesion_idx, variant_name):
                print(f"  [Skip] lesion={lesion_idx}, baseline={variant_name} already completed.")
                continue

            print("-" * 80)
            print(f"  [Run] baseline={variant_name}, lesion_idx={lesion_idx}, lesion_name={lesion_name}")
            lesion_t0 = time.time()

            # Replace any old failed/partial row for this pair before re-running.
            per_lesion_rows = _drop_pair_rows(per_lesion_rows, lesion_idx, variant_name)
            candidate_rows = _drop_pair_rows(candidate_rows, lesion_idx, variant_name)
            random_intervention_rows = _drop_pair_rows(random_intervention_rows, lesion_idx, variant_name)

            try:
                stage1_res = get_stage1_for_lesion(lesion_idx, stage1_provider)
                raw_outputs.setdefault(lesion_idx, {"stage1_res": stage1_res, "baselines": {}})

                context = dict(context_base)

                # Optional per-lesion DSA candidate pool.
                if variant_name == "DSA-stimulation baseline":
                    if dsa_candidate_pool_by_lesion is not None and int(lesion_idx) in dsa_candidate_pool_by_lesion:
                        context["dsa_candidate_pool"] = list(dsa_candidate_pool_by_lesion[int(lesion_idx)])
                    elif dsa_fc_pool_size is not None:
                        pool = _build_fc_top_pool_for_lesion(
                            empirical_FC=empirical_FC,
                            lesion_idx=lesion_idx,
                            roi_num=roi_num,
                            pool_size=int(dsa_fc_pool_size),
                        )
                        if pool is not None:
                            context["dsa_candidate_pool"] = pool

                if variant_name == "Random-intervention baseline":
                    print(f"    - running random interventions: repeats={int(random_intervention_repeats)} ...")
                    t_random = time.time()
                    seed_base = (
                        int(random_intervention_seed)
                        if random_intervention_seed is not None
                        else int(best_config.get("random_state", 123))
                    )
                    agg_row, repeat_df, random_raw = run_random_intervention_baseline_for_lesion(
                        model=model,
                        lesion_idx=lesion_idx,
                        lesion_name=lesion_name,
                        roi_num=roi_num,
                        using_steps=using_steps,
                        stage1_res=stage1_res,
                        best_config=best_config,
                        roi_names=roi_names,
                        empirical_FC=empirical_FC,
                        repeats=int(random_intervention_repeats),
                        random_state=seed_base,
                        node_pool=random_intervention_node_pool,
                        signs=random_intervention_signs,
                        strengths=random_intervention_strengths,
                        schedules=random_intervention_schedules,
                    )
                    agg_row["baseline_group"] = "random_intervention"
                    agg_row["literature_basis"] = baseline_cfg.literature_basis
                    agg_row["candidate_score_definition"] = baseline_cfg.candidate_score
                    agg_row["success"] = (
                        _safe_float(agg_row.get("traj_recovery"), np.nan) > float(success_traj_threshold)
                        and _safe_float(agg_row.get("fc_recovery"), np.nan) > float(success_fc_threshold)
                        and agg_row.get("status") == "ok"
                    )
                    per_lesion_rows.append(agg_row)

                    if repeat_df is not None and len(repeat_df) > 0:
                        repeat_df = repeat_df.copy()
                        repeat_df["baseline_group"] = "random_intervention"
                        random_intervention_rows.extend(repeat_df.to_dict("records"))
                        for _, rr in repeat_df.iterrows():
                            candidate_rows.append({
                                "lesion_idx": lesion_idx,
                                "lesion_name": lesion_name,
                                "variant": variant_name,
                                "repeat_id": int(rr.get("repeat_id")) if pd.notna(rr.get("repeat_id", np.nan)) else np.nan,
                                "rank": np.nan,
                                "node": int(rr.get("random_node")) if pd.notna(rr.get("random_node", np.nan)) else np.nan,
                                "roi_name": roi_names[int(rr.get("random_node"))] if roi_names is not None and pd.notna(rr.get("random_node", np.nan)) and int(rr.get("random_node")) < len(roi_names) else None,
                                "candidate_score": np.nan,
                                "random_sign": rr.get("random_sign", np.nan),
                                "random_strength": rr.get("random_strength", np.nan),
                                "random_schedule": rr.get("random_schedule", None),
                            })

                    raw_outputs[lesion_idx]["baselines"][variant_name] = {
                        "random_repeat_outputs": random_raw,
                    }
                    print(f"    - random interventions done: time={time.time() - t_random:.1f}s")
                    print(f"    - lesion done: total={time.time() - lesion_t0:.1f}s")
                    continue

                print("    - building candidates ...")
                t_stage2 = time.time()
                stage2_res = build_literature_baseline_candidates(
                    baseline_cfg=baseline_cfg,
                    lesion_idx=lesion_idx,
                    stage1_res=stage1_res,
                    context=context,
                )
                stage2_res = normalize_stage2_output_for_ablation(
                    stage2_res,
                    lesion_idx=lesion_idx,
                    roi_num=roi_num,
                    empirical_FC=empirical_FC,
                    top_m=top_m,
                )
                candidate_nodes = stage2_res["candidate_nodes"]
                print(f"    - candidates built: n={len(candidate_nodes)}, time={time.time() - t_stage2:.1f}s")

                if len(candidate_nodes) == 0:
                    raise ValueError("empty candidate_nodes returned by literature baseline")

                cand_df = stage2_res.get("candidate_df", None)
                if isinstance(cand_df, pd.DataFrame) and len(cand_df) > 0:
                    for _, rr in cand_df.iterrows():
                        candidate_rows.append({
                            "lesion_idx": lesion_idx,
                            "lesion_name": lesion_name,
                            "variant": variant_name,
                            "rank": int(rr.get("rank", len(candidate_rows) + 1)) if pd.notna(rr.get("rank", np.nan)) else np.nan,
                            "node": int(rr.get("node")) if pd.notna(rr.get("node", np.nan)) else np.nan,
                            "roi_name": rr.get("roi_name", None),
                            "candidate_score": _safe_float(rr.get("candidate_score", np.nan)),
                        })

                print("    - running Stage 3 ...")
                t_stage3 = time.time()
                stage3_res = run_stage3_for_candidates(
                    model=model,
                    candidate_nodes=candidate_nodes,
                    stage1_res=stage1_res,
                    lesion_idx=lesion_idx,
                    roi_num=roi_num,
                    using_steps=using_steps,
                    best_config=best_config,
                )
                print(f"    - Stage 3 done: time={time.time() - t_stage3:.1f}s")

                row = build_ablation_row(
                    lesion_idx=lesion_idx,
                    lesion_name=lesion_name,
                    variant_cfg=pseudo_variant,
                    stage2_res=stage2_res,
                    stage3_res=stage3_res,
                    roi_names=roi_names,
                    repeat_id=None,
                    status="ok",
                    prefer_multi=prefer_multi,
                    empirical_FC=empirical_FC,
                )
                row["baseline_group"] = "literature"
                row["literature_basis"] = baseline_cfg.literature_basis
                row["candidate_score_definition"] = baseline_cfg.candidate_score
                row["success"] = (
                    _safe_float(row.get("traj_recovery"), np.nan) > float(success_traj_threshold)
                    and _safe_float(row.get("fc_recovery"), np.nan) > float(success_fc_threshold)
                    and row.get("status") == "ok"
                )
                per_lesion_rows.append(row)

                raw_outputs[lesion_idx]["baselines"][variant_name] = {
                    "stage2_res": stage2_res,
                    "stage3_res": stage3_res,
                }

                print(f"    - lesion done: total={time.time() - lesion_t0:.1f}s")

            except Exception as e:
                warnings.warn(f"Literature baseline failed: lesion={lesion_idx}, baseline={variant_name}, error={e}")
                fail_row = {
                    "lesion_idx": lesion_idx,
                    "lesion_name": lesion_name,
                    "variant": variant_name,
                    "status": f"failed: {e}",
                    "baseline_group": "literature",
                    "literature_basis": baseline_cfg.literature_basis,
                    "candidate_score_definition": baseline_cfg.candidate_score,
                    "success": False,
                }
                for key in metrics_to_fields({}).keys():
                    fail_row[key] = np.nan
                per_lesion_rows.append(fail_row)

        # Save once after all lesion nodes have been attempted for this baseline.
        per_lesion_df, summary_df, candidate_table_df = _checkpoint_literature_outputs(
            out_dir=out_dir,
            per_lesion_rows=per_lesion_rows,
            candidate_rows=candidate_rows,
            completed_baseline=variant_name,
            save_per_baseline_files=save_per_baseline_files,
            success_traj_threshold=success_traj_threshold,
            success_fc_threshold=success_fc_threshold,
        )
        if len(random_intervention_rows) > 0:
            _atomic_to_csv(pd.DataFrame(random_intervention_rows), random_repeat_path, index=False)

        n_ok = int(((per_lesion_df["variant"].astype(str) == variant_name) & (per_lesion_df["status"] == "ok")).sum()) \
            if len(per_lesion_df) > 0 else 0
        print(f"[Baseline checkpoint] SAVED baseline={variant_name}, ok_lesions={n_ok}/{len(lesion_list)}, "
              f"elapsed={time.time() - baseline_t0:.1f}s")

    per_lesion_df, summary_df, candidate_table_df = _checkpoint_literature_outputs(
        out_dir=out_dir,
        per_lesion_rows=per_lesion_rows,
        candidate_rows=candidate_rows,
        completed_baseline=None,
        save_per_baseline_files=False,
        success_traj_threshold=success_traj_threshold,
        success_fc_threshold=success_fc_threshold,
    )
    if len(random_intervention_rows) > 0:
        _atomic_to_csv(pd.DataFrame(random_intervention_rows), random_repeat_path, index=False)

    print("Saved final baseline-level checkpoint outputs:")
    print("  ", os.path.join(out_dir, "literature_baseline_per_lesion_results.csv"))
    print("  ", os.path.join(out_dir, "literature_baseline_candidate_table.csv"))
    print("  ", os.path.join(out_dir, "literature_baseline_summary_mean_std.csv"))
    print("  ", os.path.join(out_dir, "literature_baseline_design_table.csv"))
    print("  ", random_repeat_path)

    return per_lesion_df, summary_df, candidate_table_df, raw_outputs


def print_literature_baseline_by_baseline_checkpoint_usage_template():
    """Print notebook code for baseline-level checkpoint execution."""
    print("""
# 4.6.2 Literature-based baseline comparison: baseline-level checkpoint version
# Loop order: run 8 lesion nodes for one baseline, save once, then move to the next baseline.

literature_per_lesion_df, literature_summary_df, literature_candidate_df, literature_raw_outputs = \\
    run_multi_lesion_literature_baseline_study_by_baseline_checkpointed(
        model=ANN,
        lesion_list=lesion_list,
        roi_num=ROI_num,
        using_steps=using_steps,
        empirical_FC=empirical_FC,
        best_config=BEST_CONFIG_ABLATION,
        stage1_provider=stage1_provider,
        signals=signals,
        roi_names=roi_names,
        top_m=20,
        out_dir="./ablation_multi_lesion_literature_by_baseline",

        # Shared control-baseline settings
        control_ridge=1e-2,
        control_energy_eps=1e-3,
        control_horizon=BEST_CONFIG_ABLATION.get("horizon", 20),
        target_control_m=20,
        tracking_lambda_u=1e-2,

        # DSA baseline: fixed stimulation only.
        # dsa_fc_pool_size=60 is optional but strongly recommended for speed.
        dsa_stim_strength=0.2,
        dsa_signs=(-1, 1),
        dsa_fc_pool_size=60,

        # PCMCI+ baseline.
        pcmci_plus_max_nodes=30,
        pcmci_plus_tau_max=1,
        pcmci_plus_pc_alpha=0.01,
        allow_pcmciplus_fallback=False,

        # Checkpoint behavior
        resume=True,
        skip_failed_when_resume=False,
        save_per_baseline_files=True,
    )

# Build baseline-comparison outputs separately from internal ablation outputs.
baseline_df, baseline_summary_df, baseline_formatted_df = build_baseline_comparison_results(
    ablation_per_lesion_df=per_lesion_df,
    literature_per_lesion_df=literature_per_lesion_df,
    out_dir="./ablation_multi_lesion_baseline_comparison",
)

internal_structure_table, internal_structure_formatted = save_compact_internal_ablation_outputs(
    per_lesion_df,
    out_dir="./ablation_multi_lesion_compact_internal",
)
""")


# =========================================================
# 9. Candidate-budget-matched literature baseline workflow
# =========================================================

def build_key_ablation_variants(
    best_config: Optional[Dict[str, Any]] = None,
    include_random: bool = True,
    include_pruning_controls: bool = False,
) -> List[AblationVariant]:
    """
    Build a compact set of internal ablation variants for paper-constrained runs.

    Recommended main-text set:
        Full method
        w/o response prior
        w/o LPCMCI
        Random ranking

    This helper now mirrors the default internal ablation list.
    `include_pruning_controls` is kept for backward-compatible notebook calls,
    but no extra pruning-control variants are returned by default.
    """
    variants = build_ablation_variants(best_config)

    keep = [
        "Full method",
        "w/o response prior",
        "w/o LPCMCI",
    ]
    if include_pruning_controls:
        keep += [
            "Causal-only pruning",
            "FC-only pruning",
        ]
    if include_random:
        keep += ["Random ranking"]

    keep_set = set(keep)
    return [v for v in variants if v.variant in keep_set]


def build_literature_baseline_variants_no_pcmciplus() -> List[LiteratureBaselineVariant]:
    """
    Backward-compatible alias for the main-text baseline list.

    Current default baselines:
        Global-FC hub
        NCT-state-transition
        Target-control
        Tracking-control
        Random-intervention
    """
    return build_literature_baseline_variants()


def _parse_candidate_nodes_len(x: Any) -> Optional[int]:
    """Best-effort length extraction from a candidate_nodes field."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return len(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "random repeats"}:
            return None
        try:
            import ast
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple, set)):
                return len(val)
        except Exception:
            return None
    return None


def extract_full_candidate_budget(
    per_lesion_df: pd.DataFrame,
    reference_variant: str = "Full method",
    min_k: int = 1,
    max_k: Optional[int] = None,
    fallback_k: int = 20,
) -> Dict[int, int]:
    """
    Extract per-lesion candidate budgets from the Full method.

    For each lesion node l:
        K_l = |C_{Full,l}|
    """
    if per_lesion_df is None or len(per_lesion_df) == 0:
        raise ValueError("per_lesion_df is empty; cannot extract Full-method candidate budgets.")

    df = per_lesion_df.copy()
    if "variant" not in df.columns or "lesion_idx" not in df.columns:
        raise KeyError("per_lesion_df must contain variant and lesion_idx columns.")

    if "status" in df.columns:
        sub = df[(df["variant"].astype(str) == str(reference_variant)) & (df["status"].astype(str) == "ok")].copy()
    else:
        sub = df[df["variant"].astype(str) == str(reference_variant)].copy()

    if len(sub) == 0:
        raise ValueError(f"No rows found for reference_variant={reference_variant!r}.")

    budgets: Dict[int, int] = {}
    for _, row in sub.iterrows():
        lesion_idx = int(row["lesion_idx"])

        k_val = None
        if "n_candidate_nodes" in row.index:
            try:
                val = float(row["n_candidate_nodes"])
                if np.isfinite(val) and val > 0:
                    k_val = int(round(val))
            except Exception:
                k_val = None

        if k_val is None and "candidate_nodes" in row.index:
            k_val = _parse_candidate_nodes_len(row.get("candidate_nodes"))

        if k_val is None or k_val <= 0:
            k_val = int(fallback_k)

        k_val = max(int(min_k), int(k_val))
        if max_k is not None:
            k_val = min(int(max_k), int(k_val))
        budgets[lesion_idx] = int(k_val)

    return budgets


def run_multi_lesion_literature_baseline_study_full_size_matched_by_baseline_checkpointed(
    model: Any,
    lesion_list: Sequence[int],
    roi_num: int,
    using_steps: int,
    empirical_FC: Optional[np.ndarray],
    best_config: Dict[str, Any],
    stage1_provider: Union[Callable[[int], Dict[str, Any]], Dict[int, Dict[str, Any]]],
    full_method_per_lesion_df: Optional[pd.DataFrame] = None,
    candidate_budget_by_lesion: Optional[Dict[int, int]] = None,
    signals: Optional[np.ndarray] = None,
    roi_names: Optional[Sequence[str]] = None,
    baselines: Optional[List[LiteratureBaselineVariant]] = None,
    include_pcmciplus: bool = False,
    fallback_top_m: int = 20,
    out_dir: str = "./ablation_literature_full_size_matched_by_baseline",
    prefer_multi: bool = True,
    success_traj_threshold: float = 0.0,
    success_fc_threshold: float = 0.0,
    control_A: Optional[np.ndarray] = None,
    control_ridge: float = 1e-2,
    control_energy_eps: float = 1e-3,
    control_horizon: Optional[int] = None,
    target_control_m: Optional[int] = None,
    tracking_lambda_u: float = 1e-2,
    dsa_stim_strength: Optional[float] = None,
    dsa_signs: Sequence[int] = (-1, 1),
    dsa_candidate_pool: Optional[Sequence[int]] = None,
    dsa_candidate_pool_by_lesion: Optional[Dict[int, Sequence[int]]] = None,
    dsa_fc_pool_size: Optional[int] = 60,
    pcmci_plus_max_nodes: Optional[int] = None,
    pcmci_plus_tau_max: Optional[int] = None,
    pcmci_plus_pc_alpha: Optional[float] = None,
    allow_pcmciplus_fallback: bool = False,
    random_intervention_repeats: int = 50,
    random_intervention_seed: Optional[int] = None,
    random_intervention_node_pool: Optional[Sequence[int]] = None,
    random_intervention_signs: Optional[Sequence[Any]] = None,
    random_intervention_strengths: Optional[Sequence[Any]] = None,
    random_intervention_schedules: Optional[Sequence[Any]] = None,
    resume: bool = True,
    skip_failed_when_resume: bool = False,
    save_per_baseline_files: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run literature baselines under a Full-method candidate-budget-matched setting.

    For each lesion l:
        K_l = |C_{Full,l}|

    Each literature baseline uses its own ranking but only takes Top-K_l nodes:
        C_{baseline,l}^{K_l} = TopK_{K_l}(R_{baseline,l})

    Loop order:
        for baseline in baselines:
            for lesion in lesion_list:
                use top_m = K_l
            save checkpoint once after the baseline finishes all lesions

    By default, PCMCI+ is excluded to save time.
    """
    import time

    os.makedirs(out_dir, exist_ok=True)

    if baselines is None:
        baselines = build_literature_baseline_variants() if include_pcmciplus else build_literature_baseline_variants_no_pcmciplus()

    if candidate_budget_by_lesion is None:
        if full_method_per_lesion_df is None:
            raise ValueError(
                "Provide either full_method_per_lesion_df=per_lesion_df or "
                "candidate_budget_by_lesion={lesion_idx: K_l}."
            )
        candidate_budget_by_lesion = extract_full_candidate_budget(
            full_method_per_lesion_df,
            reference_variant="Full method",
            min_k=1,
            max_k=int(fallback_top_m),
            fallback_k=int(fallback_top_m),
        )

    candidate_budget_by_lesion = {int(k): int(v) for k, v in candidate_budget_by_lesion.items()}

    per_lesion_path = os.path.join(out_dir, "literature_baseline_per_lesion_results.csv")
    candidate_path = os.path.join(out_dir, "literature_baseline_candidate_table.csv")
    random_repeat_path = os.path.join(out_dir, "random_intervention_repeat_results.csv")

    existing_per = _read_csv_if_exists(per_lesion_path) if resume else pd.DataFrame()
    existing_cand = _read_csv_if_exists(candidate_path) if resume else pd.DataFrame()
    existing_random = _read_csv_if_exists(random_repeat_path) if resume else pd.DataFrame()

    per_lesion_rows: List[Dict[str, Any]] = existing_per.to_dict("records") if len(existing_per) > 0 else []
    candidate_rows: List[Dict[str, Any]] = existing_cand.to_dict("records") if len(existing_cand) > 0 else []
    random_intervention_rows: List[Dict[str, Any]] = existing_random.to_dict("records") if len(existing_random) > 0 else []
    raw_outputs: Dict[str, Any] = {}

    context_base = {
        "BEST_CONFIG": best_config,
        "roi_num": int(roi_num),
        "using_steps": int(using_steps),
        "empirical_FC": empirical_FC,
        "signals": signals,
        "roi_names": roi_names,
        "model": model,
        "out_dir": out_dir,
        "control_A": control_A,
        "control_ridge": float(control_ridge),
        "control_energy_eps": float(control_energy_eps),
        "control_horizon": int(control_horizon) if control_horizon is not None else int(best_config.get("horizon", 20)),
        "tracking_lambda_u": float(tracking_lambda_u),
        "dsa_stim_strength": dsa_stim_strength,
        "dsa_signs": dsa_signs,
        "dsa_candidate_pool": dsa_candidate_pool,
        "pcmci_plus_max_nodes": pcmci_plus_max_nodes if pcmci_plus_max_nodes is not None else int(best_config.get("max_candidate_nodes", 50)),
        "pcmci_plus_tau_max": pcmci_plus_tau_max if pcmci_plus_tau_max is not None else int(best_config.get("tau_max", 1)),
        "pcmci_plus_pc_alpha": pcmci_plus_pc_alpha if pcmci_plus_pc_alpha is not None else float(best_config.get("pc_alpha", 0.01)),
        "allow_pcmciplus_fallback": bool(allow_pcmciplus_fallback),
    }

    def _row_is_done(row: Dict[str, Any]) -> bool:
        if str(row.get("status")) == "ok":
            return True
        return bool(skip_failed_when_resume) and str(row.get("status", "")).startswith("failed")

    def _has_done_pair(lesion_idx: int, variant: str, k_l: int) -> bool:
        for r in per_lesion_rows:
            try:
                same_lesion = int(r.get("lesion_idx")) == int(lesion_idx)
            except Exception:
                same_lesion = False
            same_variant = str(r.get("variant")) == str(variant)
            same_budget = True
            if "candidate_budget_k" in r and not pd.isna(r.get("candidate_budget_k")):
                try:
                    same_budget = int(float(r.get("candidate_budget_k"))) == int(k_l)
                except Exception:
                    same_budget = False
            if same_lesion and same_variant and same_budget and _row_is_done(r):
                return True
        return False

    for baseline_cfg in baselines:
        variant_name = baseline_cfg.variant
        baseline_t0 = time.time()

        done_flags = [
            _has_done_pair(int(l), variant_name, int(candidate_budget_by_lesion.get(int(l), fallback_top_m)))
            for l in lesion_list
        ]
        if resume and all(done_flags):
            print("=" * 100)
            print(f"[Full-size matched checkpoint] SKIP completed baseline: {variant_name}")
            print("=" * 100)
            continue

        print("=" * 100)
        print(f"[Full-size matched checkpoint] START baseline: {variant_name}")
        print("=" * 100)

        pseudo_variant = _pseudo_ablation_variant_from_literature(baseline_cfg)

        for lesion_idx in lesion_list:
            lesion_idx = int(lesion_idx)
            lesion_name = roi_names[lesion_idx] if roi_names is not None and lesion_idx < len(roi_names) else f"ROI_{lesion_idx}"
            k_l = int(candidate_budget_by_lesion.get(lesion_idx, int(fallback_top_m)))
            k_l = max(1, min(k_l, int(roi_num) - 1))

            if resume and _has_done_pair(lesion_idx, variant_name, k_l):
                print(f"  [Skip] lesion={lesion_idx}, baseline={variant_name}, K_l={k_l} already completed.")
                continue

            print("-" * 80)
            print(f"  [Run] baseline={variant_name}, lesion_idx={lesion_idx}, lesion_name={lesion_name}, K_l={k_l}")
            lesion_t0 = time.time()

            per_lesion_rows = _drop_pair_rows(per_lesion_rows, lesion_idx, variant_name)
            candidate_rows = _drop_pair_rows(candidate_rows, lesion_idx, variant_name)
            random_intervention_rows = _drop_pair_rows(random_intervention_rows, lesion_idx, variant_name)

            try:
                stage1_res = get_stage1_for_lesion(lesion_idx, stage1_provider)
                raw_outputs.setdefault(lesion_idx, {"stage1_res": stage1_res, "baselines": {}})

                context = dict(context_base)
                context["top_m"] = int(k_l)
                context["target_control_m"] = int(target_control_m) if target_control_m is not None else int(k_l)

                if variant_name == "DSA-stimulation baseline":
                    if dsa_candidate_pool_by_lesion is not None and lesion_idx in dsa_candidate_pool_by_lesion:
                        context["dsa_candidate_pool"] = list(dsa_candidate_pool_by_lesion[lesion_idx])
                    elif dsa_fc_pool_size is not None:
                        pool_size = max(int(dsa_fc_pool_size), int(k_l))
                        pool = _build_fc_top_pool_for_lesion(
                            empirical_FC=empirical_FC,
                            lesion_idx=lesion_idx,
                            roi_num=roi_num,
                            pool_size=pool_size,
                        )
                        if pool is not None:
                            context["dsa_candidate_pool"] = pool

                if variant_name == "Random-intervention baseline":
                    print(f"    - running random interventions: repeats={int(random_intervention_repeats)} ...")
                    t_random = time.time()
                    seed_base = (
                        int(random_intervention_seed)
                        if random_intervention_seed is not None
                        else int(best_config.get("random_state", 123))
                    )
                    agg_row, repeat_df, random_raw = run_random_intervention_baseline_for_lesion(
                        model=model,
                        lesion_idx=lesion_idx,
                        lesion_name=lesion_name,
                        roi_num=roi_num,
                        using_steps=using_steps,
                        stage1_res=stage1_res,
                        best_config=best_config,
                        roi_names=roi_names,
                        empirical_FC=empirical_FC,
                        repeats=int(random_intervention_repeats),
                        random_state=seed_base,
                        node_pool=random_intervention_node_pool,
                        signs=random_intervention_signs,
                        strengths=random_intervention_strengths,
                        schedules=random_intervention_schedules,
                    )
                    agg_row["baseline_group"] = "random_intervention"
                    agg_row["literature_basis"] = baseline_cfg.literature_basis
                    agg_row["candidate_score_definition"] = baseline_cfg.candidate_score
                    agg_row["candidate_budget_k"] = np.nan
                    agg_row["candidate_budget_rule"] = "not_applicable_random_policy"
                    agg_row["success"] = (
                        _safe_float(agg_row.get("traj_recovery"), np.nan) > float(success_traj_threshold)
                        and _safe_float(agg_row.get("fc_recovery"), np.nan) > float(success_fc_threshold)
                        and agg_row.get("status") == "ok"
                    )
                    per_lesion_rows.append(agg_row)

                    if repeat_df is not None and len(repeat_df) > 0:
                        repeat_df = repeat_df.copy()
                        repeat_df["baseline_group"] = "random_intervention"
                        repeat_df["candidate_budget_rule"] = "not_applicable_random_policy"
                        random_intervention_rows.extend(repeat_df.to_dict("records"))
                        for _, rr in repeat_df.iterrows():
                            candidate_rows.append({
                                "lesion_idx": lesion_idx,
                                "lesion_name": lesion_name,
                                "variant": variant_name,
                                "repeat_id": int(rr.get("repeat_id")) if pd.notna(rr.get("repeat_id", np.nan)) else np.nan,
                                "rank": np.nan,
                                "node": int(rr.get("random_node")) if pd.notna(rr.get("random_node", np.nan)) else np.nan,
                                "roi_name": roi_names[int(rr.get("random_node"))] if roi_names is not None and pd.notna(rr.get("random_node", np.nan)) and int(rr.get("random_node")) < len(roi_names) else None,
                                "candidate_score": np.nan,
                                "random_sign": rr.get("random_sign", np.nan),
                                "random_strength": rr.get("random_strength", np.nan),
                                "random_schedule": rr.get("random_schedule", None),
                                "candidate_budget_k": np.nan,
                                "candidate_budget_rule": "not_applicable_random_policy",
                            })

                    raw_outputs[lesion_idx]["baselines"][variant_name] = {
                        "random_repeat_outputs": random_raw,
                        "candidate_budget_k": None,
                    }
                    print(f"    - random interventions done: time={time.time() - t_random:.1f}s")
                    print(f"    - lesion done: total={time.time() - lesion_t0:.1f}s")
                    continue

                print("    - building candidates ...")
                t_stage2 = time.time()
                stage2_res = build_literature_baseline_candidates(
                    baseline_cfg=baseline_cfg,
                    lesion_idx=lesion_idx,
                    stage1_res=stage1_res,
                    context=context,
                )
                stage2_res = normalize_stage2_output_for_ablation(
                    stage2_res,
                    lesion_idx=lesion_idx,
                    roi_num=roi_num,
                    empirical_FC=empirical_FC,
                    top_m=k_l,
                )
                candidate_nodes = stage2_res["candidate_nodes"]
                print(f"    - candidates built: n={len(candidate_nodes)}, time={time.time() - t_stage2:.1f}s")

                if len(candidate_nodes) == 0:
                    raise ValueError("empty candidate_nodes returned by literature baseline")

                cand_df = stage2_res.get("candidate_df", None)
                if isinstance(cand_df, pd.DataFrame) and len(cand_df) > 0:
                    for _, rr in cand_df.iterrows():
                        candidate_rows.append({
                            "lesion_idx": lesion_idx,
                            "lesion_name": lesion_name,
                            "variant": variant_name,
                            "rank": int(rr.get("rank", len(candidate_rows) + 1)) if pd.notna(rr.get("rank", np.nan)) else np.nan,
                            "node": int(rr.get("node")) if pd.notna(rr.get("node", np.nan)) else np.nan,
                            "roi_name": rr.get("roi_name", None),
                            "candidate_score": _safe_float(rr.get("candidate_score", np.nan)),
                            "candidate_budget_k": int(k_l),
                            "candidate_budget_rule": "matched_to_full_method_n_candidate_nodes",
                        })

                print("    - running Stage 3 ...")
                t_stage3 = time.time()
                stage3_res = run_stage3_for_candidates(
                    model=model,
                    candidate_nodes=candidate_nodes,
                    stage1_res=stage1_res,
                    lesion_idx=lesion_idx,
                    roi_num=roi_num,
                    using_steps=using_steps,
                    best_config=best_config,
                )
                print(f"    - Stage 3 done: time={time.time() - t_stage3:.1f}s")

                row = build_ablation_row(
                    lesion_idx=lesion_idx,
                    lesion_name=lesion_name,
                    variant_cfg=pseudo_variant,
                    stage2_res=stage2_res,
                    stage3_res=stage3_res,
                    roi_names=roi_names,
                    repeat_id=None,
                    status="ok",
                    prefer_multi=prefer_multi,
                    empirical_FC=empirical_FC,
                )
                row["baseline_group"] = "literature_full_size_matched"
                row["literature_basis"] = baseline_cfg.literature_basis
                row["candidate_score_definition"] = baseline_cfg.candidate_score
                row["candidate_budget_k"] = int(k_l)
                row["candidate_budget_rule"] = "matched_to_full_method_n_candidate_nodes"
                row["success"] = (
                    _safe_float(row.get("traj_recovery"), np.nan) > float(success_traj_threshold)
                    and _safe_float(row.get("fc_recovery"), np.nan) > float(success_fc_threshold)
                    and row.get("status") == "ok"
                )
                per_lesion_rows.append(row)

                raw_outputs[lesion_idx]["baselines"][variant_name] = {
                    "stage2_res": stage2_res,
                    "stage3_res": stage3_res,
                    "candidate_budget_k": int(k_l),
                }

                print(f"    - lesion done: total={time.time() - lesion_t0:.1f}s")

            except Exception as e:
                warnings.warn(f"Full-size matched literature baseline failed: lesion={lesion_idx}, baseline={variant_name}, K_l={k_l}, error={e}")
                fail_row = {
                    "lesion_idx": lesion_idx,
                    "lesion_name": lesion_name,
                    "variant": variant_name,
                    "status": f"failed: {e}",
                    "baseline_group": "literature_full_size_matched",
                    "literature_basis": baseline_cfg.literature_basis,
                    "candidate_score_definition": baseline_cfg.candidate_score,
                    "candidate_budget_k": int(k_l),
                    "candidate_budget_rule": "matched_to_full_method_n_candidate_nodes",
                    "success": False,
                }
                for key in metrics_to_fields({}).keys():
                    fail_row[key] = np.nan
                per_lesion_rows.append(fail_row)

        per_lesion_df, summary_df, candidate_table_df = _checkpoint_literature_outputs(
            out_dir=out_dir,
            per_lesion_rows=per_lesion_rows,
            candidate_rows=candidate_rows,
            completed_baseline=variant_name,
            save_per_baseline_files=save_per_baseline_files,
            success_traj_threshold=success_traj_threshold,
            success_fc_threshold=success_fc_threshold,
        )
        if len(random_intervention_rows) > 0:
            _atomic_to_csv(pd.DataFrame(random_intervention_rows), random_repeat_path, index=False)

        n_ok = int(((per_lesion_df["variant"].astype(str) == variant_name) & (per_lesion_df["status"] == "ok")).sum())             if len(per_lesion_df) > 0 else 0
        print(f"[Full-size matched checkpoint] SAVED baseline={variant_name}, ok_lesions={n_ok}/{len(lesion_list)}, "
              f"elapsed={time.time() - baseline_t0:.1f}s")

    per_lesion_df, summary_df, candidate_table_df = _checkpoint_literature_outputs(
        out_dir=out_dir,
        per_lesion_rows=per_lesion_rows,
        candidate_rows=candidate_rows,
        completed_baseline=None,
        save_per_baseline_files=False,
        success_traj_threshold=success_traj_threshold,
        success_fc_threshold=success_fc_threshold,
    )
    if len(random_intervention_rows) > 0:
        _atomic_to_csv(pd.DataFrame(random_intervention_rows), random_repeat_path, index=False)

    print("Saved final full-size-matched literature baseline outputs:")
    print("  ", os.path.join(out_dir, "literature_baseline_per_lesion_results.csv"))
    print("  ", os.path.join(out_dir, "literature_baseline_candidate_table.csv"))
    print("  ", os.path.join(out_dir, "literature_baseline_summary_mean_std.csv"))
    print("  ", os.path.join(out_dir, "literature_baseline_design_table.csv"))
    print("  ", random_repeat_path)

    return per_lesion_df, summary_df, candidate_table_df, raw_outputs


def plot_baseline_three_metrics_grouped(
    baseline_summary_df,
    variants=None,
    out_dir="./ablation_multi_lesion_baseline_comparison_random_intervention",
    save_name="baseline_three_metrics_grouped.png",
    title="Candidate-budget-matched baseline comparison",
    figsize=(14, 6),
    rotation=25,
):
    """
    在一张图中画 baseline 的三项指标分组柱状图：
    objective / traj_recovery / fc_recovery

    参数
    ----
    baseline_summary_df : pd.DataFrame
        必须包含以下列：
        - variant
        - objective_mean, objective_std
        - traj_recovery_mean, traj_recovery_std
        - fc_recovery_mean, fc_recovery_std

    variants : list[str] or None
        指定绘图顺序；如果为 None，则使用当前表中的顺序。
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = baseline_summary_df.copy()

    # 指定顺序
    if variants is not None:
        order_map = {v: i for i, v in enumerate(variants)}
        df = df[df["variant"].isin(variants)].copy()
        df["_order"] = df["variant"].map(order_map)
        df = df.sort_values("_order").drop(columns="_order")
    else:
        df = df.reset_index(drop=True)

    required_cols = [
        "variant",
        "objective_mean", "objective_std",
        "traj_recovery_mean", "traj_recovery_std",
        "fc_recovery_mean", "fc_recovery_std",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"baseline_summary_df 缺少列: {missing}")

    x = np.arange(len(df))
    width = 0.24

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x - width,
        df["objective_mean"].values,
        width,
        yerr=df["objective_std"].values,
        capsize=4,
        label="Objective"
    )

    ax.bar(
        x,
        df["traj_recovery_mean"].values,
        width,
        yerr=df["traj_recovery_std"].values,
        capsize=4,
        label="Trajectory recovery"
    )

    ax.bar(
        x + width,
        df["fc_recovery_mean"].values,
        width,
        yerr=df["fc_recovery_std"].values,
        capsize=4,
        label="FC recovery"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"].tolist(), rotation=rotation, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, save_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return fig, ax, save_path
