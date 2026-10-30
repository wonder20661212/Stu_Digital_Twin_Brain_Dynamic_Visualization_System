"""Smoke-test the causal discovery routines on real slices and synthetic data."""

from __future__ import annotations

import numpy as np

import app


def _assert_mask(name: str, mask: np.ndarray, roi_num: int) -> None:
    if mask.shape != (roi_num, roi_num):
        raise AssertionError(f"{name}: expected {(roi_num, roi_num)}, got {mask.shape}")
    if mask.dtype != bool:
        raise AssertionError(f"{name}: expected bool mask, got {mask.dtype}")
    if np.diag(mask).any():
        raise AssertionError(f"{name}: diagonal must be empty")
    print(f"{name}: shape={mask.shape}, edges={int(mask.sum())}")


def _load_real_slice(roi_num: int = 8, rows: int = 190) -> list[np.ndarray]:
    signals = []
    for path in app.get_default_fmri_files()[:2]:
        signals.append(np.loadtxt(path)[30:30 + rows, :roi_num])
    return signals


def _synthetic_chain(n: int = 600) -> list[np.ndarray]:
    rng = np.random.default_rng(4)
    x = rng.normal(size=n)
    y = np.zeros(n)
    z = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.75 * x[t - 1] + 0.15 * y[t - 1] + rng.normal(scale=0.4)
        z[t] = 0.8 * y[t - 1] + rng.normal(scale=0.4)
    return [np.column_stack([x, y, z])]


def _synthetic_var_network(roi_num: int = 8, n: int = 900, sessions: int = 3) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(42)
    coeffs = {
        (0, 1): 0.55,
        (1, 2): 0.50,
        (0, 3): 0.45,
        (4, 5): -0.45,
        (4, 6): 0.35,
        (6, 7): 0.40,
    }
    A = np.zeros((roi_num, roi_num), dtype=float)
    truth = np.zeros((roi_num, roi_num), dtype=bool)
    for (src, dst), coef in coeffs.items():
        A[src, dst] = coef
        truth[src, dst] = True

    signals = []
    for _ in range(sessions):
        x = np.zeros((n, roi_num), dtype=float)
        noise = rng.normal(scale=0.55, size=(n, roi_num))
        for t in range(1, n):
            x[t] = 0.20 * x[t - 1] + x[t - 1] @ A + noise[t]
        signals.append(x[100:])
    return signals, truth


def _score_against_truth(name: str, mask: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    pred = np.asarray(mask, dtype=bool).copy()
    np.fill_diagonal(pred, False)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"{name}-truth: tp={tp}, fp={fp}, fn={fn}, precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
    return precision, recall, f1


def main() -> None:
    roi_num = 8
    real_signals = _load_real_slice(roi_num=roi_num)
    algorithms = [
        ("granger", app.compute_granger_mask, {"max_lag": 2, "alpha": 0.05, "min_effect": 0.001, "max_density": 0.5, "max_candidates_per_target": 4, "roi_num": roi_num}),
        ("vargranger", app.compute_var_granger_mask, {"max_lag": 3, "alpha": 0.05, "min_effect": 0.001, "max_density": 0.5, "roi_num": roi_num}),
        ("pcmci", app.compute_pcmci_mask, {"max_lag": 2, "alpha": 0.05, "roi_num": roi_num}),
        ("sparsevar", app.compute_sparse_var_mask, {"max_lag": 2, "max_candidates_per_target": 4, "min_effect": 0.001, "max_density": 0.5, "roi_num": roi_num}),
        ("pc", app.compute_pc_mask, {"alpha": 0.05, "roi_num": roi_num}),
        ("fci", app.compute_fci_mask, {"alpha": 0.05, "max_path": 1, "roi_num": roi_num}),
    ]
    for name, func, kwargs in algorithms:
        mask = func(real_signals, **kwargs)
        _assert_mask(name, mask, roi_num)
        if name in {"granger", "vargranger"} and mask.sum() > int(0.5 * roi_num * (roi_num - 1)):
            raise AssertionError(f"{name}: max_density screen did not cap edge count")

    chain = _synthetic_chain()
    chain_tests = [
        ("granger-chain", app.compute_granger_mask, {"max_lag": 1, "alpha": 0.05, "roi_num": 3, "max_density": 0.5, "max_candidates_per_target": 2}),
        ("vargranger-chain", app.compute_var_granger_mask, {"max_lag": 1, "alpha": 0.05, "roi_num": 3, "max_density": 0.5}),
        # The app's default PCMCI density cap is tuned for 360-ROI visualization.
        # Relax it here so the 3-node synthetic chain can keep both true edges.
        ("pcmci-chain", app.compute_pcmci_mask, {"max_lag": 1, "alpha": 0.05, "roi_num": 3, "max_density": 0.5}),
        ("sparsevar-chain", app.compute_sparse_var_mask, {"max_lag": 1, "roi_num": 3, "max_density": 0.34, "max_candidates_per_target": 2, "min_effect": 0.001}),
    ]
    for name, func, extra in chain_tests:
        mask = func(chain, **extra)
        _assert_mask(name, mask, 3)
        if not (mask[0, 1] and mask[1, 2]):
            raise AssertionError(f"{name}: failed to recover expected 0->1->2 chain")

    var_signals, truth = _synthetic_var_network()
    truth_tests = [
        ("granger", app.compute_granger_mask, {"max_lag": 1, "alpha": 0.01, "min_effect": 0.01, "max_density": 0.2, "max_candidates_per_target": 3, "roi_num": 8}, 0.80, 0.80),
        ("vargranger", app.compute_var_granger_mask, {"max_lag": 2, "alpha": 0.01, "min_effect": 0.01, "max_density": 0.2, "roi_num": 8}, 0.90, 0.90),
        ("pcmci", app.compute_pcmci_mask, {"max_lag": 1, "alpha": 0.01, "roi_num": 8}, 0.90, 0.90),
        ("sparsevar", app.compute_sparse_var_mask, {"max_lag": 1, "min_effect": 0.01, "max_density": 0.10, "max_candidates_per_target": 3, "roi_num": 8}, 0.90, 0.80),
    ]
    truth_masks = {}
    for name, func, kwargs, min_precision, min_recall in truth_tests:
        mask = func(var_signals, **kwargs)
        truth_masks[name] = mask
        precision, recall, _ = _score_against_truth(name, mask, truth)
        if precision < min_precision or recall < min_recall:
            raise AssertionError(
                f"{name}: synthetic VAR benchmark failed "
                f"(precision={precision:.3f}, recall={recall:.3f})"
            )

    old_roi_num = app.ROI_num
    old_masks = {key: app._get_mask_by_key(key) for key in app.SCREENING_KEYS}
    old_ec_dict = app.ec_dict
    old_npi_cache = app.npi_intervention_score_cache
    old_npi_signature = app.npi_intervention_cache_signature
    try:
        app.ROI_num = 8
        for key, mask in truth_masks.items():
            app._set_mask_by_key(key, mask)
        consensus, meta = app._compute_consensus_mask()
        _assert_mask("consensus", consensus, 8)
        precision, recall, _ = _score_against_truth("consensus", consensus, truth)
        if precision < 0.90 or recall < 0.80:
            raise AssertionError(
                f"consensus: synthetic VAR benchmark failed "
                f"(precision={precision:.3f}, recall={recall:.3f}, sources={meta['sources']})"
            )

        npi_ec = truth.astype(float)
        npi_ec += 0.03 * np.ones_like(npi_ec)
        np.fill_diagonal(npi_ec, 0.0)
        app.ec_dict = {0.2: npi_ec, 0.4: 1.5 * npi_ec}
        app.npi_intervention_score_cache = None
        app.npi_intervention_cache_signature = None
        ensemble, meta = app._compute_intervention_ensemble_mask()
        _assert_mask("ensemble", ensemble, 8)
        precision, recall, _ = _score_against_truth("ensemble", ensemble, truth)
        if precision < 0.90 or recall < 0.80:
            raise AssertionError(
                f"ensemble: synthetic VAR benchmark failed "
                f"(precision={precision:.3f}, recall={recall:.3f}, sources={meta['sources']})"
            )
    finally:
        app.ROI_num = old_roi_num
        app.ec_dict = old_ec_dict
        app.npi_intervention_score_cache = old_npi_cache
        app.npi_intervention_cache_signature = old_npi_signature
        for key, mask in old_masks.items():
            app._set_mask_by_key(key, mask)

    print("causal algorithm validation ok")


if __name__ == "__main__":
    main()
