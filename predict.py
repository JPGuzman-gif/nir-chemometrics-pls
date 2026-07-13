#!/usr/bin/env python3
"""CLI for NIR diesel property prediction."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nir_pls.io import load_spectra
from nir_pls.predictor import NIRPredictor


def _parse_properties(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [p.strip() for p in value.split(",") if p.strip()]


def _format_table(results: list[dict]) -> str:
    lines = []
    for r in results:
        sid = r.get("sample_id", "—")
        ad = r["applicability"]
        domain = "in-domain" if ad["in_domain"] else "OUT-OF-DOMAIN"
        lines.append(f"Sample: {sid}  |  T²={ad['t2']:.2f} (limit {ad['t2_limit']:.2f})  |  {domain}")

        if not r["predictions"]:
            lines.append("  (no predictions — all properties skipped or filtered)")
        for prop, val in sorted(r["predictions"].items()):
            flag = r["flags"][prop]
            tier = flag["quality_tier"]
            note = "" if flag["deployable"] else " [weak model]"
            lines.append(f"  {prop:<8} {val:>12.4g}  ({tier}){note}")
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict diesel fuel properties from NIR spectra")
    parser.add_argument("spectra", type=Path, help="Path to diesel_spec.csv-format file")
    parser.add_argument("--sample-id", help="Predict a single sample ID")
    parser.add_argument("--properties", help="Comma-separated property list (default: deployable only)")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models", help="Models directory")
    parser.add_argument("--include-weak", action="store_true", help="Include CN and FLASH (RPD < 2.0)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output JSON")
    args = parser.parse_args()

    sample_ids, _wavelengths, X = load_spectra(args.spectra)
    properties = _parse_properties(args.properties)

    if args.sample_id:
        matches = np.where(sample_ids == str(args.sample_id))[0]
        if len(matches) == 0:
            print(f"Sample ID not found: {args.sample_id}", file=sys.stderr)
            return 1
        idx = matches[0]
        sample_ids = sample_ids[idx : idx + 1]
        X = X[idx : idx + 1]

    predictor = NIRPredictor(args.models_dir)
    results = predictor.predict_batch(
        X,
        sample_ids=sample_ids.tolist(),
        properties=properties,
        include_weak=args.include_weak,
    )

    if args.as_json:
        print(json.dumps(results, indent=2))
    else:
        print(_format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
