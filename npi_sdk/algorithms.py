"""Direct access to causal-discovery algorithm functions."""

from __future__ import annotations

from .client import NPIClient


def get_algorithm_functions(project_root=None):
    """Return fMRI-suitable causal algorithm functions from the engine."""
    engine = NPIClient(project_root, auto_initialize=False).engine
    return {
        "granger": engine.compute_granger_mask,
        "vargranger": engine.compute_var_granger_mask,
        "pcmci": engine.compute_pcmci_mask,
        "sparsevar": engine.compute_sparse_var_mask,
        "fask": engine.compute_fask_mask,
        "twostep": engine.compute_two_step_mask,
    }


def get_cached_mask_functions(project_root=None, *, pipeline_mode="none"):
    """Return SDK mask-callables that follow the web UI cache rules.

    The returned functions accept an optional ``params`` dictionary and call
    :meth:`NPIClient.compute_mask`, so FASK/Two-Step full-brain caches are kept
    separate for ``none``, ``fask``, and ``twostep`` pipeline modes.
    """
    client = NPIClient(project_root)

    def _make(name):
        return lambda params=None, **kwargs: client.compute_mask(
            name,
            params=params,
            pipeline_mode=kwargs.pop("pipeline_mode", pipeline_mode),
            **kwargs,
        )

    return {
        "granger": _make("granger"),
        "vargranger": _make("vargranger"),
        "pcmci": _make("pcmci"),
        "sparsevar": _make("sparsevar"),
        "fask": _make("fask"),
        "twostep": _make("twostep"),
        "consensus": _make("consensus"),
        "ensemble": _make("ensemble"),
    }
