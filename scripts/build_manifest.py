#!/usr/bin/env python3
"""Build models/manifest.json from training data and model_summary.csv."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nir_pls.domain import compute_ad_limits
from nir_pls.io import load_aligned_dataset, load_spectra
from nir_pls.preprocess import WL_CUTOFF, calibration_col_means, n_wavelengths_after_preproc


def quality_tier(rpd: float) -> tuple[str, bool]:
    """Map RPD to quality tier and deployable flag per README thresholds."""
    if rpd < 1.5:
        return "not_usable", False
    if rpd < 2.0:
        return "rough_screening", False
    if rpd < 2.5:
        return "quantitative_screening", True
    return "good_quantitative", True


def build_manifest(
    models_dir: Path,
    spec_path: Path,
    prop_path: Path,
    summary_path: Path,
) -> dict:
    sample_ids, wavelengths, X, prop_names, Y = load_aligned_dataset(spec_path, prop_path)
    summary = pd.read_csv(summary_path)

    wl_mask = wavelengths <= WL_CUTOFF
    ad = compute_ad_limits(X)
    ad["outlier_sample_ids"] = sample_ids[ad.pop("outlier_indices")].tolist()

    # Cohort masks (notebook Cell C)
    outlier_set = set(ad["outlier_sample_ids"])
    outlier_mask = np.array([sid in outlier_set for sid in sample_ids])
    cohorts = {
        "all": np.ones(len(X), dtype=bool),
        "no_T2": ~outlier_mask,
    }

    properties = []
    for _, row in summary.iterrows():
        prop = row["Property"]
        preproc = row["Preproc"]
        cohort = row["Cohort"]
        rpd = float(row["RPD"])
        tier, deployable = quality_tier(rpd)

        X_cohort = X[cohorts[cohort]]
        col_means = calibration_col_means(preproc, X_cohort, wl_mask=wl_mask)
        n_wl = n_wavelengths_after_preproc(preproc, X.shape[1], wl_mask=wl_mask)

        properties.append({
            "property": prop,
            "preproc": preproc,
            "n_components": int(row["LVs"]),
            "cohort": cohort,
            "col_means": col_means.tolist(),
            "n_wavelengths": n_wl,
            "wl_cutoff": WL_CUTOFF if preproc == "SNV+SG ≤1450" else None,
            "rpd": rpd,
            "r2_test": float(row["R2_test"]),
            "rmsep": float(row["RMSEP"]),
            "deployable": deployable,
            "quality_tier": tier,
            "model_file": f"pls_{prop}.joblib",
        })

    return {
        "version": 1,
        "wavelengths": wavelengths.tolist(),
        "wl_cutoff": WL_CUTOFF,
        "n_samples": int(len(X)),
        "properties": properties,
        "applicability_domain": ad,
    }


def main() -> None:
    models_dir = ROOT / "models"
    spec_path = ROOT / "data" / "diesel_spec.csv"
    prop_path = ROOT / "data" / "diesel_prop.csv"
    summary_path = ROOT / "reports" / "model_summary.csv"
    out_path = models_dir / "manifest.json"

    manifest = build_manifest(models_dir, spec_path, prop_path, summary_path)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  {len(manifest['properties'])} properties")
    print(f"  {len(manifest['applicability_domain']['outlier_sample_ids'])} T² outliers")


if __name__ == "__main__":
    main()
