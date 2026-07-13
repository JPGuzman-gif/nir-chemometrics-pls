"""NIR PLS predictor — load manifest + models and predict fuel properties."""

import json
from pathlib import Path

import joblib
import numpy as np

from nir_pls.domain import check_applicability
from nir_pls.preprocess import WL_CUTOFF, apply_preproc


class NIRPredictor:
    """Predict diesel fuel properties from NIR spectra."""

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        manifest_path = self.models_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Run: python scripts/build_manifest.py"
            )

        with manifest_path.open(encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.wavelengths = np.asarray(self.manifest["wavelengths"], dtype=float)
        self.wl_mask = self.wavelengths <= WL_CUTOFF
        self.ad = self.manifest["applicability_domain"]
        self._property_configs = {p["property"]: p for p in self.manifest["properties"]}
        self._models: dict[str, object] = {}

    def _load_model(self, prop: str):
        if prop not in self._models:
            cfg = self._property_configs[prop]
            path = self.models_dir / cfg["model_file"]
            self._models[prop] = joblib.load(path)
        return self._models[prop]

    def _validate_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        x = np.asarray(spectrum, dtype=float).reshape(-1)
        if x.shape[0] != self.wavelengths.shape[0]:
            raise ValueError(
                f"Expected {self.wavelengths.shape[0]} wavelengths, got {x.shape[0]}"
            )
        return x

    def _predict_one(
        self,
        spectrum: np.ndarray,
        *,
        properties: list[str] | None = None,
        include_weak: bool = False,
    ) -> dict:
        x = self._validate_spectrum(spectrum)
        x_batch = x.reshape(1, -1)

        applicability = check_applicability(x, self.ad)
        predictions: dict[str, float] = {}
        flags: dict[str, dict] = {}

        props = properties or list(self._property_configs.keys())
        for prop in props:
            if prop not in self._property_configs:
                raise ValueError(f"Unknown property: {prop!r}")

            cfg = self._property_configs[prop]
            flag = {
                "deployable": cfg["deployable"],
                "quality_tier": cfg["quality_tier"],
                "rpd": cfg["rpd"],
            }

            if not cfg["deployable"] and not include_weak:
                flag["skipped"] = True
                flags[prop] = flag
                continue

            col_means = np.asarray(cfg["col_means"], dtype=float)
            X_prep = apply_preproc(
                cfg["preproc"],
                x_batch,
                col_means=col_means,
                wl_mask=self.wl_mask,
            )
            pls = self._load_model(prop)
            pred = float(pls.predict(X_prep).ravel()[0])
            predictions[prop] = pred
            flags[prop] = flag

        return {
            "predictions": predictions,
            "flags": flags,
            "applicability": applicability,
        }

    def predict(
        self,
        spectrum: np.ndarray,
        *,
        properties: list[str] | None = None,
        include_weak: bool = False,
    ) -> dict:
        """Predict properties for a single spectrum (shape n_wavelengths,)."""
        return self._predict_one(
            spectrum,
            properties=properties,
            include_weak=include_weak,
        )

    def predict_batch(
        self,
        X: np.ndarray,
        *,
        sample_ids: list[str] | None = None,
        properties: list[str] | None = None,
        include_weak: bool = False,
    ) -> list[dict]:
        """Predict properties for multiple spectra (shape n_samples, n_wavelengths)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_wavelengths)")

        results = []
        for i in range(X.shape[0]):
            entry = self._predict_one(
                X[i],
                properties=properties,
                include_weak=include_weak,
            )
            if sample_ids is not None:
                entry["sample_id"] = sample_ids[i]
            results.append(entry)
        return results

    @property
    def property_names(self) -> list[str]:
        return list(self._property_configs.keys())

    @property
    def deployable_properties(self) -> list[str]:
        return [p for p, c in self._property_configs.items() if c["deployable"]]
