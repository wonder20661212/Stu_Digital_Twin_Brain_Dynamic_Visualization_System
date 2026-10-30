"""Smoke-test the LiNGAM-family causal discovery routines."""

from __future__ import annotations

import time
import traceback

import numpy as np

import app


def _synthetic_sessions(
    roi_num: int = 4,
    n_rows: int = 90,
    sessions: int = 2,
) -> list[np.ndarray]:
    rng = np.random.default_rng(123)
    out = []
    for _ in range(sessions):
        x = np.zeros((n_rows, roi_num), dtype=np.float64)
        noise = rng.laplace(size=(n_rows, roi_num)) * 0.4
        for t in range(1, n_rows):
            x[t, 0] = 0.25 * x[t - 1, 0] + noise[t, 0]
            x[t, 1] = 0.55 * x[t - 1, 0] + 0.20 * x[t - 1, 1] + noise[t, 1]
            x[t, 2] = 0.45 * x[t - 1, 1] + 0.20 * x[t - 1, 2] + noise[t, 2]
            x[t, 3] = 0.35 * x[t - 1, 0] + 0.20 * x[t - 1, 3] + noise[t, 3]
        out.append(x[10:])
    return out


def _small_params(key: str) -> dict:
    params = dict(app.DEFAULT_MASK_PARAMS[key])
    if "max_density" in params:
        params["max_density"] = 0.5
    if key == "icalingam":
        params["max_iter"] = 200
    elif key == "varmalingam":
        params["max_iter"] = 10
    elif key == "abiclingam":
        params["max_outer"] = 5
    elif key == "highdimlingam":
        params["J"] = 3
        params["K"] = 2
    return params


def _assert_mask(key: str, mask: np.ndarray, roi_num: int) -> None:
    if not isinstance(mask, np.ndarray):
        raise AssertionError(f"{key}: returned {type(mask).__name__}, not ndarray")
    if mask.shape != (roi_num, roi_num):
        raise AssertionError(f"{key}: expected {(roi_num, roi_num)}, got {mask.shape}")
    if mask.dtype != bool:
        raise AssertionError(f"{key}: expected bool mask, got {mask.dtype}")
    if np.diag(mask).any():
        raise AssertionError(f"{key}: diagonal must be empty")


def main() -> None:
    roi_num = 4
    old_roi_num = app.ROI_num
    app.ROI_num = roi_num
    signals = _synthetic_sessions(roi_num=roi_num)
    keys = [
        "lingam",
        "dirlingam",
        "plingam",
        "pairlingam",
        "lofs",
        "imageslofs",
        "psilingam",
        "bilingam",
        "lvlingam",
        *app.OFFICIAL_LINGAM_PACKAGE_KEYS,
    ]

    failures = []
    try:
        for key in keys:
            started = time.time()
            try:
                mask = app._compute_mask_with_params(key, signals, params=_small_params(key))
                _assert_mask(key, mask, roi_num)
                print(
                    f"{key}: OK shape={mask.shape} edges={int(mask.sum())} "
                    f"time={time.time() - started:.2f}s"
                )
            except Exception as exc:
                failures.append((key, exc, traceback.format_exc()))
                print(f"{key}: FAIL {type(exc).__name__}: {exc} time={time.time() - started:.2f}s")
    finally:
        app.ROI_num = old_roi_num

    if failures:
        print("\nFailures:")
        for key, exc, tb in failures:
            print(f"\n[{key}] {type(exc).__name__}: {exc}")
            print(tb.rstrip())
        raise SystemExit(1)

    client = app.app.test_client()
    api_paths = [
        "/api/pairwise_lingam_mask",
        "/api/psi_lingam_mask",
        "/api/lofs_mask",
        "/api/images_lofs_mask",
        "/api/precomputed_causal_mask/pairlingam",
        "/api/precomputed_causal_mask/psilingam",
        "/api/precomputed_causal_mask/lofs",
        "/api/precomputed_causal_mask/imageslofs",
    ]
    for path in api_paths:
        response = client.get(path)
        payload = response.get_json() or {}
        if response.status_code != 200 or payload.get("available") is False:
            raise AssertionError(
                f"{path}: status={response.status_code}, message={payload.get('message')}"
            )
        print(
            f"{path}: OK status={response.status_code} "
            f"cached={payload.get('cached')} edges={payload.get('edge_count')}"
        )

    print("lingam algorithm validation ok")


if __name__ == "__main__":
    main()
