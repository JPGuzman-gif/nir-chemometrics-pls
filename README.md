# nir-chemometrics-pls

NIR chemometrics project: PLS regression models for diesel fuel property prediction from near-infrared spectra.

## Dataset

- **Spectra** (`data/diesel_spec.csv`): 784 samples × 401 wavelengths (750–1550 nm)
- **Properties** (`data/diesel_prop.csv`): 7 fuel properties — BP50, CN, D4052, FLASH, FREEZE, TOTAL, VISC
- Properties have varying missing-value counts (e.g. CN n≈381 vs others n≈395)

## Workflow

Run [nir_diesel_pls.ipynb](nir_diesel_pls.ipynb) top-to-bottom:

| Cell | Content |
|------|---------|
| 0 | Imports |
| 1 | Data loading (`X`, `Y`, `X_raw`, `Y_raw`, `wavelengths`, `prop_names`) |
| 2 | Shared preamble — PCA, Hotelling T², Q residuals, correlation spectra |
| 3–7 | EDA figures (`01`–`05`) |
| 8 (A) | Spectral preprocessing — SNV, SG derivative, mean centering |
| 9 (B) | LV selection — 10-fold CV, 1–SE rule, MAX_LV=25 (`06`) |
| 10 (C) | Cohort sensitivity — all vs exclude T² outliers, 5 preprocessing methods (`07`) |
| 11 (D) | Per-property winner selection (`08`) |
| 12 (E) | Final PLS models — 80/20 hold-out validation (`09`, `10`) |
| 13 (F) | Model interpretation — regression vectors, loadings, VIP (`11`) |
| 14 (G) | Summary export — `reports/model_summary.csv`, `models/pls_*.joblib` |

## Key EDA findings

- **Property groups**: BP50, D4052, FLASH, FREEZE, VISC are highly intercorrelated (heavy-hydrocarbon axis); CN is mostly independent.
- **Outliers**: 67 samples flagged by Hotelling T² only (chemically extreme, not spectral artifacts). Sensitivity analysis compares keeping vs excluding them.
- **Spectral variance**: dominated by 1450–1550 nm O-H region; optional truncation at 1450 nm tested as 5th preprocessing candidate.

## Outputs

```
reports/
  figures/01_EDA.png … 11_loadings_vip.png
  model_summary.csv
models/
  pls_BP50.joblib … pls_VISC.joblib
  manifest.json          # inference config (generated)
src/nir_pls/             # inference package
predict.py               # CLI entry point
```

## Inference (Phase 11)

After running the notebook through Cell G, build the inference manifest once:

```bash
python scripts/build_manifest.py
```

Predict properties from a spectrum file:

```bash
python predict.py data/diesel_spec.csv --sample-id 17366
python predict.py data/diesel_spec.csv --properties VISC,BP50 --json
python predict.py data/diesel_spec.csv --include-weak   # include CN, FLASH
```

Run inference tests:

```bash
python -m unittest tests.test_inference -v
```

### Deployable properties (from hold-out RPD)

| Property | RPD | Tier | Deploy? |
|----------|-----|------|---------|
| D4052 | 14.86 | Good quantitative | Yes |
| TOTAL | 10.21 | Good quantitative | Yes |
| BP50 | 4.52 | Good quantitative | Yes |
| VISC | 3.52 | Good quantitative | Yes |
| FREEZE | 2.30 | Quantitative screening | Yes (caution) |
| CN | 1.60 | Rough screening | No (use `--include-weak`) |
| FLASH | 1.53 | Not usable | No (use `--include-weak`) |

### Applicability domain

Each prediction includes Hotelling T² and Q residual checks against the calibration set (PCA, 10 LVs, 95% limits). Samples flagged **OUT-OF-DOMAIN** were chemically extreme in training (67 T² outliers) — treat predictions with caution.

## Environment

```bash
conda env create -f environment.yml
conda activate nir-pls
jupyter notebook nir_diesel_pls.ipynb
```

## RPD interpretation

| RPD | Quality |
|-----|---------|
| < 1.5 | Not usable |
| 1.5–2.0 | Rough screening |
| 2.0–2.5 | Quantitative screening |
| > 2.5 | Good quantitative |
