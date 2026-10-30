"""High-level SDK client for the NPI visualizer and causal algorithms.

The SDK intentionally reuses the existing project engine in ``app.py`` so the
web UI and Python API stay consistent.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .exceptions import NPIError


class NPIClient:
    """Convenient Python interface for bundled NPI EC data and algorithms.

    Parameters
    ----------
    project_root:
        Path to the project root containing ``app.py`` and bundled resources.
        Defaults to ``NPI_PROJECT_ROOT`` when set, otherwise the installed
        package root.
    auto_initialize:
        When true, load cached EC data, cached causal masks, and brain geometry
        on construction.
    """

    _module_lock = threading.Lock()

    def __init__(self, project_root: Optional[Union[str, os.PathLike[str]]] = None, *, auto_initialize: bool = True):
        self.project_root = self._resolve_project_root(project_root)
        self._engine = self._load_engine()
        self._initialized = False
        if auto_initialize:
            self.initialize()

    @property
    def engine(self):
        """Return the loaded ``app.py`` module for advanced use."""
        return self._engine

    def initialize(self, *, force: bool = False) -> "NPIClient":
        """Load cached EC results, causal masks, and brain geometry."""
        if self._initialized and not force:
            return self
        self._engine.initialize_data()
        self._initialized = True
        return self

    def status(self) -> dict[str, Any]:
        """Return the visualizer status payload."""
        return self._get_json("/api/status")

    def available_strengths(self) -> list[float]:
        """Return available perturbation strengths."""
        data = self._get_json("/api/available_strengths")
        return [float(x) for x in data.get("strengths", [])]

    def available_ec_results(self) -> list[dict[str, Any]]:
        """Return saved EC result modes and metadata."""
        return self._get_json("/api/ec_results").get("results", [])

    def select_ec_result(self, mode: str) -> dict[str, Any]:
        """Select an EC result mode such as ``none``, ``pcmci``, or ``ensemble``."""
        return self._post_json("/api/select_ec_result", {"mode": mode})

    def get_ec_matrix(self, strength: float, *, copy: bool = True) -> np.ndarray:
        """Return the EC matrix for one perturbation strength."""
        self.initialize()
        key = self._resolve_strength_key(strength)
        mat = self._engine.ec_dict[key]
        return mat.copy() if copy else mat

    def get_all_ec_matrices(self, *, copy: bool = True) -> dict[float, np.ndarray]:
        """Return all loaded EC matrices keyed by perturbation strength."""
        self.initialize()
        return {float(k): (v.copy() if copy else v) for k, v in self._engine.ec_dict.items()}

    def load_signals(self) -> list[np.ndarray]:
        """Load raw fMRI sessions from the configured project data directory."""
        return self._engine._load_signals()

    def load_training_data(
        self,
        *,
        roi_num: Optional[int] = None,
        steps: Optional[int] = None,
        skip_rows: int = 30,
        fmri_files: Optional[list[Union[str, os.PathLike[str]]]] = None,
    ) -> dict[str, Any]:
        """Load fMRI sessions and convert them into NPI training windows."""
        npi = self._get_training_module()
        roi_num = int(roi_num or self._engine.ROI_num)
        steps = int(steps or self._engine.using_steps)
        files = [Path(p) for p in fmri_files] if fmri_files else self._engine.get_default_fmri_files()

        signals = []
        inputs = []
        targets = []
        used_files = []
        for path in files:
            if not path.exists():
                continue
            sig = np.loadtxt(path)[int(skip_rows):, :roi_num]
            inp, tar = npi.multi2one(sig, steps=steps)
            signals.append(sig)
            inputs.append(inp)
            targets.append(tar)
            used_files.append(str(path))

        if not signals:
            raise NPIError(f"No fMRI training files were found under: {self._engine.REAL_FMRI_DIR}")

        return {
            "signals": signals,
            "inputs": np.vstack(inputs),
            "targets": np.vstack(targets),
            "files": used_files,
            "roi_num": roi_num,
            "steps": steps,
        }

    def create_surrogate_model(
        self,
        *,
        model_type: str = "mlp",
        roi_num: Optional[int] = None,
        steps: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
    ):
        """Create an NPI surrogate-brain model.

        Supported model types are ``mlp``, ``cnn``, ``rnn``, and ``var``.
        """
        npi = self._get_training_module()
        roi_num = int(roi_num or self._engine.ROI_num)
        steps = int(steps or self._engine.using_steps)
        hidden_dim = int(hidden_dim or (2 * roi_num))
        latent_dim = int(latent_dim or (0.8 * roi_num))
        key = str(model_type).strip().lower()

        if key == "mlp":
            return npi.ANN_MLP(
                input_dim=steps * roi_num,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                output_dim=roi_num,
            )
        if key == "cnn":
            return npi.ANN_CNN(
                in_channels=roi_num,
                hidden_channels=hidden_dim,
                out_channels=latent_dim,
                data_length=steps,
            )
        if key == "rnn":
            return npi.ANN_RNN(
                input_dim=roi_num,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                output_dim=roi_num,
                data_length=steps,
            )
        if key == "var":
            return npi.ANN_VAR(input_dim=steps * roi_num, output_dim=roi_num)
        raise NPIError(f"Unsupported surrogate model type: {model_type}")

    def train_surrogate_model(
        self,
        training_data: Optional[dict[str, Any]] = None,
        *,
        model: Optional[Any] = None,
        model_type: str = "mlp",
        roi_num: Optional[int] = None,
        steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        train_set_proportion: Optional[float] = None,
        num_epochs: int = 100,
        lr: float = 1e-3,
        l2: float = 5e-5,
    ) -> dict[str, Any]:
        """Train an NPI surrogate-brain model and return model plus losses."""
        npi = self._get_training_module()
        data = training_data or self.load_training_data(roi_num=roi_num, steps=steps)
        roi_num = int(roi_num or data["roi_num"])
        steps = int(steps or data["steps"])
        model = model or self.create_surrogate_model(model_type=model_type, roi_num=roi_num, steps=steps)

        trained, train_loss, test_loss = npi.train_NN(
            model,
            data["inputs"],
            data["targets"],
            batch_size=int(batch_size or self._engine.batch_size),
            train_set_proportion=float(train_set_proportion or self._engine.train_set_proportion),
            num_epochs=int(num_epochs),
            lr=float(lr),
            l2=float(l2),
        )
        return {
            "model": trained,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "training_data": data,
            "roi_num": roi_num,
            "steps": steps,
        }

    def compute_npi_ec(
        self,
        model: Any,
        training_data: dict[str, Any],
        *,
        strengths: Optional[list[float]] = None,
        mask: Optional[np.ndarray] = None,
        zero_diagonal: bool = True,
    ) -> dict[float, np.ndarray]:
        """Compute NPI effective-connectivity matrices from a trained model."""
        npi = self._get_training_module()
        strengths = [float(x) for x in (strengths or self._engine.strength_list)]
        out: dict[float, np.ndarray] = {}
        for strength in strengths:
            ec = npi.model_EC(model, training_data["inputs"], training_data["targets"], pert_strength=strength)
            if zero_diagonal:
                np.fill_diagonal(ec, 0.0)
            if mask is not None:
                ec = ec * np.asarray(mask, dtype=ec.dtype)
            out[float(strength)] = ec
        return out

    def train_and_compute_ec(
        self,
        *,
        mode: Optional[str] = None,
        mask_params: Optional[dict[str, Any]] = None,
        model_type: str = "mlp",
        roi_num: Optional[int] = None,
        steps: Optional[int] = None,
        strengths: Optional[list[float]] = None,
        num_epochs: int = 100,
        batch_size: Optional[int] = None,
        train_set_proportion: Optional[float] = None,
        lr: float = 1e-3,
        l2: float = 5e-5,
        save: bool = False,
    ) -> dict[str, Any]:
        """Run the full NPI training and EC computation pipeline.

        ``mode`` may be ``None``/``none`` for pure NPI or one of the supported
        causal masks, such as ``pcmci`` or ``consensus``.
        """
        data = self.load_training_data(roi_num=roi_num, steps=steps)
        mask = None
        mode_key = self._normalize_algorithm(mode or "none")
        if mode_key is not None:
            pipeline_mode = mode_key if mode_key in self._engine.FMRI_NATIVE_SCREEN_KEYS else "none"
            mask = self.compute_mask(mode_key, mask_params, pipeline_mode=pipeline_mode)

        trained = self.train_surrogate_model(
            data,
            model_type=model_type,
            roi_num=data["roi_num"],
            steps=data["steps"],
            batch_size=batch_size,
            train_set_proportion=train_set_proportion,
            num_epochs=num_epochs,
            lr=lr,
            l2=l2,
        )
        ec = self.compute_npi_ec(
            trained["model"],
            data,
            strengths=strengths,
            mask=mask,
        )
        result = {
            "mode": mode_key or "none",
            "model": trained["model"],
            "train_loss": trained["train_loss"],
            "test_loss": trained["test_loss"],
            "training_data": data,
            "ec": ec,
        }
        if save:
            result["meta"] = self.save_ec_results(ec, mode=mode_key or "none")
        return result

    def save_ec_results(
        self,
        ec_results: dict[float, np.ndarray],
        *,
        mode: str = "none",
        mode_label: Optional[str] = None,
        screen_first: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Save EC matrices using the same format as the web visualizer."""
        mode_key = self._engine._normalize_ec_result_mode(mode)
        label = mode_label or self._engine.COMPUTE_MODE_LABELS.get(mode_key, mode_key)
        meta = self._engine._save_ec_result(
            mode_key,
            ec_results,
            label,
            bool(screen_first if screen_first is not None else mode_key != "none"),
        )
        self._engine._set_current_ec_result(mode_key, ec_results, meta)
        return meta

    def compute_mask(
        self,
        algorithm: str,
        params: Optional[dict[str, Any]] = None,
        *,
        pipeline_mode: str = "none",
        prefer_cache: bool = True,
        save_default_cache: bool = True,
    ) -> np.ndarray:
        """Compute or load a causal-discovery mask.

        Supported algorithms are the fMRI time-series screens:
        ``granger``, ``vargranger``, ``pcmci``, ``sparsevar``, ``fask``,
        ``twostep``, plus the derived ``consensus`` and ``ensemble`` masks.

        ``pipeline_mode`` follows the web UI modes: ``none``, ``fask``, or
        ``twostep``. It only changes cache selection for the native fMRI
        algorithms, so FASK/Two-Step pre-screen results and ordinary algorithm
        results do not overwrite or read each other.
        """
        self.initialize()
        key = self._normalize_algorithm(algorithm)
        if key is None:
            raise NPIError("The 'none' mode has no causal mask.")
        pipeline_mode = self._normalize_pipeline_mode(pipeline_mode)

        if key == "consensus":
            mask, _ = self._engine._compute_consensus_mask()
            return mask.astype(bool)
        if key == "ensemble":
            mask, _ = self._engine._compute_intervention_ensemble_mask()
            return mask.astype(bool)

        is_default = params is None or self._engine._is_default_params(
            key,
            {**self._engine.DEFAULT_MASK_PARAMS[key], **params},
        )

        if prefer_cache and is_default:
            if key in self._engine.FMRI_NATIVE_SCREEN_KEYS:
                self._engine._load_single_cached_mask(key, pipeline_mode=pipeline_mode)
                cached = self._engine._get_mask_by_key(key)
                if cached is not None and not self._engine._full_fmri_memory_context_matches(key, pipeline_mode):
                    cached = None
            else:
                self._engine._load_cached_masks()
                cached = self._engine._get_mask_by_key(key)
            if cached is not None:
                return cached.astype(bool)

        signals = self.load_signals()
        if not signals:
            raise NPIError("No raw fMRI signal files were found for mask computation.")

        mask = self._engine._compute_mask_with_params(key, signals, params)
        self._engine._set_mask_by_key(key, mask)
        if save_default_cache and key in self._engine.SCREENING_KEYS and is_default:
            self._engine._mark_mask_cached(key)
            if key in self._engine.FMRI_NATIVE_SCREEN_KEYS:
                self._engine._save_single_cached_mask(key, mask, pipeline_mode=pipeline_mode)
            else:
                self._engine._save_cached_masks()
        return mask.astype(bool)

    def precompute_default_masks(
        self,
        *,
        force: bool = False,
        group: str = "standard",
        keys: Optional[list[str]] = None,
        pipeline_mode: str = "none",
    ) -> dict[str, dict[str, Any]]:
        """Synchronously compute and save default causal masks.

        By default this runs the Python-native standard algorithms only:
        Granger, VAR-Granger, PCMCI, and Sparse VAR. Pass ``group="fask"`` or
        explicit ``keys=["fask"]`` to run FASK. Two-Step is not run unless it
        is explicitly requested, because it depends on the external MATLAB
        environment in the current system.
        """
        self.initialize()
        pipeline_mode = self._normalize_pipeline_mode(pipeline_mode)
        target_keys = self._resolve_precompute_keys(group=group, keys=keys)
        signals = self.load_signals()
        if not signals:
            raise NPIError("No raw fMRI signal files were found for mask precomputation.")

        out: dict[str, dict[str, Any]] = {}
        if not force:
            self._engine._load_cached_masks()

        for key in target_keys:
            if key in self._engine.FMRI_NATIVE_SCREEN_KEYS:
                if not force:
                    self._engine._load_single_cached_mask(key, pipeline_mode=pipeline_mode)
                mask = None if force else self._engine._get_mask_by_key(key)
                if mask is not None and not self._engine._full_fmri_memory_context_matches(key, pipeline_mode):
                    mask = None
            else:
                mask = None if force else self._engine._get_mask_by_key(key)
            if mask is None:
                mask = self._engine._run_default_mask_compute(key, signals)
                self._engine._set_mask_by_key(key, mask)
                self._engine._mark_mask_cached(key)
                if key in self._engine.FMRI_NATIVE_SCREEN_KEYS:
                    self._engine._save_single_cached_mask(key, mask, pipeline_mode=pipeline_mode)
                else:
                    self._engine._save_cached_masks()
            out[key] = self.mask_summary(mask)
        return out

    def precompute_causal_masks(
        self,
        *,
        group: str = "standard",
        keys: Optional[list[str]] = None,
        force: bool = False,
        pipeline_mode: str = "none",
    ) -> dict[str, Any]:
        """Start the same background causal-mask precompute job used by the UI."""
        query = {
            "group": group,
            "force": 1 if force else 0,
            "pipeline_mode": self._normalize_pipeline_mode(pipeline_mode),
        }
        payload = {"keys": keys} if keys else {}
        response = self._client().post("/api/precompute_causal_masks", query_string=query, json=payload)
        return self._decode_response(response)

    def precompute_causal_status(self, *, pipeline_mode: str = "none") -> dict[str, Any]:
        """Return causal-mask precompute status for one UI pipeline mode."""
        return self._get_json(
            "/api/precompute_causal_status",
            {"pipeline_mode": self._normalize_pipeline_mode(pipeline_mode)},
        )

    def algorithm_reliability(self) -> dict[str, Any]:
        """Return the visualizer's causal-algorithm reliability summary."""
        return self._get_json("/api/algorithm_reliability")

    def precomputed_causal_mask(self, key: str, *, pipeline_mode: str = "none") -> dict[str, Any]:
        """Return a precomputed causal mask payload by key."""
        return self._get_json(
            f"/api/precomputed_causal_mask/{key}",
            {"pipeline_mode": self._normalize_pipeline_mode(pipeline_mode)},
        )

    def local_fmri_mask(
        self,
        algorithm: str,
        strength: float,
        node_idx: int,
        *,
        max_nodes: int = 40,
        pipeline_mode: str = "none",
        force: bool = False,
    ) -> dict[str, Any]:
        """Compute or load a node-centered local FASK/Two-Step mask.

        This mirrors the C-stage local screening controls in the web UI.
        """
        key = self._normalize_algorithm(algorithm)
        if key not in {"fask", "twostep"}:
            raise NPIError("Local fMRI masks only support 'fask' and 'twostep'.")
        query = {
            "max_nodes": int(max_nodes),
            "pipeline_mode": self._normalize_pipeline_mode(pipeline_mode),
            "force": 1 if force else 0,
        }
        return self._get_json(
            f"/api/local_fmri_mask/{key}/{float(strength)}/{int(node_idx)}",
            query,
        )

    def mask_summary(self, mask: np.ndarray) -> dict[str, Any]:
        """Return edge-count and density statistics for a mask."""
        arr = np.asarray(mask, dtype=bool)
        n_total = int(arr.shape[0] * max(arr.shape[1] - 1, 1))
        n_edges = int(arr.sum())
        return {
            "shape": list(arr.shape),
            "edges": n_edges,
            "density": round(n_edges / n_total, 6) if n_total else 0.0,
        }

    def causal_backbone(
        self,
        strength: float,
        node_idx: int,
        *,
        top_n: Optional[int] = None,
        threshold: float = 0.01,
        filter_type: str = "all",
        mask_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Return the causal-backbone payload for one perturbation node."""
        query: dict[str, Any] = {"threshold": threshold, "filter_type": filter_type}
        if top_n is not None:
            query["top_n"] = top_n
        if mask_key:
            query["mask_key"] = mask_key
        return self._get_json(f"/api/causal_backbone/{float(strength)}/{int(node_idx)}", query)

    def top_backbone(
        self,
        strength: float,
        node_idx: int,
        *,
        top_n: int = 5,
        filter_type: str = "all",
        mask_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Return the top-N causal-backbone payload for one perturbation node."""
        query: dict[str, Any] = {"top_n": top_n, "type": filter_type}
        if mask_key:
            query["mask_key"] = mask_key
        return self._get_json(f"/api/causal_backbone/{float(strength)}/{int(node_idx)}/top", query)

    def top_backbone_target_nodes(
        self,
        strength: float,
        node_idx: int,
        *,
        top_n: int = 5,
        filter_type: str = "all",
        mask_key: Optional[str] = None,
    ) -> list[int]:
        """Return top-N backbone ROI indices for use as NPCI Stage C/D targets.

        This mirrors the web UI flow: run the B-stage strongest-node screening
        first, then pass the resulting nodes to ``npci_intervention`` or
        ``precompute_npci_intervention`` through ``target_nodes``.
        """
        data = self.top_backbone(
            strength,
            node_idx,
            top_n=top_n,
            filter_type=filter_type,
            mask_key=mask_key,
        )
        nodes = data.get("nodes", [])
        return [
            int(item["index"])
            for item in nodes
            if isinstance(item, dict) and item.get("index") is not None
        ]

    def build_network(
        self,
        strength: float,
        node_indices: list[int],
        *,
        threshold: float = 0.01,
        top_n_per_node: Optional[int] = None,
        root_node_idx: Optional[int] = None,
        direction_mode: Optional[str] = None,
        preserve_displayed_edges: bool = False,
        mask_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build a compact network among selected ROI nodes."""
        payload: dict[str, Any] = {
            "node_indices": [int(x) for x in node_indices],
            "threshold": threshold,
            "preserve_displayed_edges": preserve_displayed_edges,
        }
        if top_n_per_node is not None:
            payload["top_n_per_node"] = int(top_n_per_node)
        if root_node_idx is not None:
            payload["root_node_idx"] = int(root_node_idx)
        if direction_mode is not None:
            payload["direction_mode"] = direction_mode
        if mask_key:
            payload["mask_key"] = mask_key
        return self._post_json(f"/api/network/{float(strength)}", payload)

    def npci_stage3_defaults(self) -> dict[str, Any]:
        """Return the bundled NPCI Stage 3 default strengths/profiles."""
        return self._get_json("/api/npci_stage3_defaults")

    def npci_backbone(
        self,
        strength: float,
        node_idx: int,
        params: Optional[dict[str, Any]] = None,
        *,
        top_n: Optional[int] = None,
        filter_type: str = "all",
    ) -> dict[str, Any]:
        """Return the NPCI Stage A/B lesion-response and causal-backbone payload."""
        query = dict(params or {})
        if top_n is not None:
            query["top_n"] = int(top_n)
        if filter_type:
            query["filter_type"] = filter_type
        return self._get_json(f"/api/npci_backbone/{float(strength)}/{int(node_idx)}", query)

    def npci_intervention(
        self,
        strength: float,
        node_idx: int,
        backbone_params: Optional[dict[str, Any]] = None,
        stage3_params: Optional[dict[str, Any]] = None,
        *,
        cached_only: bool = False,
        target_nodes: Optional[Union[str, list[int], tuple[int, ...]]] = None,
    ) -> dict[str, Any]:
        """Return the NPCI Stage C/D intervention-target and recovery payload.

        ``target_nodes`` optionally restricts Stage C/D to an explicit ROI list,
        for example the output of ``top_backbone_target_nodes``. ROI indices are
        zero-based, matching the JSON API and Python arrays.
        """
        query = {}
        query.update(backbone_params or {})
        query.update(stage3_params or {})
        if target_nodes is not None:
            query["target_nodes"] = self._format_target_nodes(target_nodes)
        if cached_only:
            query["cached_only"] = 1
        return self._get_json(f"/api/npci_intervention/{float(strength)}/{int(node_idx)}", query)

    def precompute_npci_intervention(
        self,
        strength: float,
        node_idx: int,
        backbone_params: Optional[dict[str, Any]] = None,
        stage3_params: Optional[dict[str, Any]] = None,
        *,
        force: bool = False,
        target_nodes: Optional[Union[str, list[int], tuple[int, ...]]] = None,
    ) -> dict[str, Any]:
        """Start background NPCI Stage C/D precomputation for one lesion node.

        Pass ``target_nodes`` to reproduce the web UI behavior where the B-stage
        strongest-node screening becomes the C/D candidate set.
        """
        query = {}
        query.update(backbone_params or {})
        query.update(stage3_params or {})
        if target_nodes is not None:
            query["target_nodes"] = self._format_target_nodes(target_nodes)
        if force:
            query["force"] = 1
        response = self._client().post(
            f"/api/precompute_npci_intervention/{float(strength)}/{int(node_idx)}",
            query_string=query,
        )
        return self._decode_response(response)

    def precompute_npci_status(self) -> dict[str, Any]:
        """Return current NPCI precompute progress/status."""
        return self._get_json("/api/precompute_npci_status")

    def run_web_app(self, *, host: str = "127.0.0.1", port: int = 5002, block: bool = True):
        """Start the Flask visualizer using the SDK-loaded engine."""
        self.initialize()
        if block:
            return self._engine.app.run(host=host, port=int(port), debug=False, use_reloader=False)

        thread = threading.Thread(
            target=self._engine.app.run,
            kwargs={"host": host, "port": int(port), "debug": False, "use_reloader": False},
            daemon=True,
        )
        thread.start()
        return thread

    def _load_engine(self):
        app_path = self.project_root / "app.py"
        if not app_path.exists():
            raise NPIError(f"Cannot find app.py under project root: {self.project_root}")
        module_name = f"_npi_sdk_engine_{abs(hash(str(app_path)))}"
        with self._module_lock:
            if module_name in sys.modules:
                return sys.modules[module_name]
            sys.path.insert(0, str(self.project_root))
            spec = importlib.util.spec_from_file_location(module_name, app_path)
            if spec is None or spec.loader is None:
                raise NPIError(f"Cannot load engine module from: {app_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

    def _get_training_module(self):
        try:
            return self._engine.get_npi_module()
        except RuntimeError as exc:
            raise NPIError(
                "NPI training requires PyTorch. Reinstall the full SDK with dependencies, "
                "or run `python -m pip install torch>=2.2,<3`. If you are working from "
                "source, run `python -m pip install -r requirements.txt` before "
                "calling training APIs."
            ) from exc

    def _resolve_project_root(self, project_root: Optional[Union[str, os.PathLike[str]]]) -> Path:
        if project_root:
            return Path(project_root).resolve()
        env_root = os.getenv("NPI_PROJECT_ROOT")
        if env_root:
            return Path(env_root).resolve()
        source_root = Path(__file__).resolve().parents[1]
        if (source_root / "app.py").exists() and (source_root / "templates").exists():
            return source_root
        bundled_root = Path(__file__).resolve().parent / "_engine"
        if bundled_root.exists():
            return bundled_root
        return source_root

    def _client(self):
        self.initialize()
        return self._engine.app.test_client()

    def _get_json(self, path: str, query: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        response = self._client().get(path, query_string=query or {})
        return self._decode_response(response)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client().post(path, json=payload)
        return self._decode_response(response)

    def _decode_response(self, response) -> dict[str, Any]:
        data = response.get_json(silent=True)
        if response.status_code >= 400:
            if isinstance(data, dict):
                message = data.get("error") or data.get("message")
            else:
                message = response.get_data(as_text=True)
            raise NPIError(f"Request failed ({response.status_code}): {message}")
        if not isinstance(data, dict):
            raise NPIError(f"Expected JSON object, got: {type(data).__name__}")
        return data

    def _normalize_pipeline_mode(self, mode: Optional[str]) -> str:
        value = str(mode or "none").strip().lower()
        if value not in {"none", "fask", "twostep"}:
            raise NPIError(f"Unsupported pipeline mode: {mode}. Use 'none', 'fask', or 'twostep'.")
        return value

    def _format_target_nodes(self, target_nodes: Union[str, list[int], tuple[int, ...]]) -> str:
        if isinstance(target_nodes, str):
            return target_nodes
        cleaned: list[int] = []
        for node in target_nodes:
            value = int(node)
            if value < 0:
                raise NPIError("target_nodes must use zero-based non-negative ROI indices.")
            if value not in cleaned:
                cleaned.append(value)
        return ",".join(str(x) for x in cleaned)

    def _normalize_algorithm(self, algorithm: str) -> Optional[str]:
        try:
            return self._engine._normalize_mask_key(algorithm)
        except ValueError as exc:
            raise NPIError(str(exc)) from exc

    def _resolve_precompute_keys(
        self,
        *,
        group: Optional[str] = None,
        keys: Optional[list[str]] = None,
    ) -> list[str]:
        try:
            return self._engine._resolve_precompute_keys(group=group, keys=keys)
        except ValueError as exc:
            raise NPIError(str(exc)) from exc

    def _resolve_strength_key(self, strength: float) -> float:
        self.initialize()
        value = float(strength)
        if value in self._engine.ec_dict:
            return value
        for key in self._engine.ec_dict:
            if abs(float(key) - value) < 1e-9:
                return key
        raise NPIError(f"Strength {strength} is not loaded. Available: {sorted(self._engine.ec_dict)}")
