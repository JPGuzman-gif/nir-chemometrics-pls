"""Tests for NIR PLS inference pipeline."""

import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nir_pls.io import load_aligned_dataset, load_spectra
from nir_pls.predictor import NIRPredictor
from nir_pls.preprocess import PREPROC_NAMES, WL_CUTOFF, apply_preproc, calibration_col_means


MANIFEST_PATH = ROOT / "models" / "manifest.json"
MODELS_DIR = ROOT / "models"
SPEC_PATH = ROOT / "data" / "diesel_spec.csv"
PROP_PATH = ROOT / "data" / "diesel_prop.csv"


def _ensure_manifest() -> None:
    if not MANIFEST_PATH.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "build_manifest",
            ROOT / "scripts" / "build_manifest.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        manifest = mod.build_manifest(
            MODELS_DIR,
            SPEC_PATH,
            PROP_PATH,
            ROOT / "reports" / "model_summary.csv",
        )
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


@unittest.skipUnless(SPEC_PATH.exists(), "calibration data not available")
class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_manifest()
        cls.sample_ids, cls.wavelengths, cls.X, cls.prop_names, cls.Y = load_aligned_dataset(
            SPEC_PATH, PROP_PATH
        )
        cls.wl_mask = cls.wavelengths <= WL_CUTOFF
        with MANIFEST_PATH.open(encoding="utf-8") as f:
            cls.manifest = json.load(f)
        cls.predictor = NIRPredictor(MODELS_DIR)

    def test_preproc_shapes(self):
        x = self.X[:5]
        for name in PREPROC_NAMES:
            out = apply_preproc(name, x, wl_mask=self.wl_mask)
            if name == "SNV+SG ≤1450":
                self.assertEqual(out.shape[1], int(self.wl_mask.sum()))
            else:
                self.assertEqual(out.shape, (5, self.X.shape[1]))

    def test_manifest_loads(self):
        self.assertEqual(len(self.manifest["properties"]), 7)
        for entry in self.manifest["properties"]:
            self.assertEqual(len(entry["col_means"]), entry["n_wavelengths"])

    def test_round_trip(self):
        """Predictor output matches direct joblib + preprocessing path."""
        idx = 0
        spectrum = self.X[idx]
        result = self.predictor.predict(spectrum, include_weak=True)

        for entry in self.manifest["properties"]:
            prop = entry["property"]
            col_means = np.asarray(entry["col_means"], dtype=float)
            X_prep = apply_preproc(
                entry["preproc"],
                spectrum.reshape(1, -1),
                col_means=col_means,
                wl_mask=self.wl_mask,
            )
            pls = joblib.load(MODELS_DIR / entry["model_file"])
            expected = float(pls.predict(X_prep).ravel()[0])
            actual = result["predictions"][prop]
            self.assertAlmostEqual(actual, expected, places=5)

    def test_ad_flags_outlier(self):
        outlier_ids = set(self.manifest["applicability_domain"]["outlier_sample_ids"])
        self.assertGreater(len(outlier_ids), 0)

        outlier_id = next(iter(outlier_ids))
        idx = np.where(self.sample_ids == outlier_id)[0][0]
        result = self.predictor.predict(self.X[idx])
        self.assertFalse(result["applicability"]["in_domain"])

    def test_skips_weak_by_default(self):
        result = self.predictor.predict(self.X[0])
        self.assertNotIn("CN", result["predictions"])
        self.assertNotIn("FLASH", result["predictions"])
        self.assertIn("CN", result["flags"])
        self.assertTrue(result["flags"]["CN"]["skipped"])

        with_weak = self.predictor.predict(self.X[0], include_weak=True)
        self.assertIn("CN", with_weak["predictions"])
        self.assertIn("FLASH", with_weak["predictions"])


if __name__ == "__main__":
    unittest.main()
