"""Load diesel NIR spectra and properties from CSV files."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_spectra(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load spectra from diesel_spec.csv format.

    Returns (sample_ids, wavelengths, X) where X has shape (n_samples, n_wavelengths).
    """
    path = Path(path)
    X_raw = pd.read_csv(path, header=None, skiprows=10)
    X_raw = X_raw.dropna(how="all")
    X_raw = X_raw.set_index(1)
    X_raw = X_raw.drop(columns=[0])
    X_raw.index = X_raw.index.astype(str)
    X_raw = X_raw.dropna(axis=1, how="all")

    wav_row = pd.read_csv(path, header=None, skiprows=9, nrows=1)
    wavelengths = wav_row.iloc[0, 2:].dropna().astype(float).values

    n = min(len(wavelengths), X_raw.shape[1])
    X_raw = X_raw.iloc[:, :n]
    wavelengths = wavelengths[:n]
    X_raw.columns = wavelengths

    sample_ids = X_raw.index.to_numpy()
    X = X_raw.values.astype(float)
    return sample_ids, wavelengths, X


def load_properties(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Load properties from diesel_prop.csv format.

    Returns (sample_ids, prop_names, Y).
    """
    path = Path(path)
    Y_raw = pd.read_csv(path, header=None, skiprows=9)
    Y_raw = Y_raw.dropna(how="all")
    Y_raw = Y_raw.set_index(1)
    Y_raw = Y_raw.drop(columns=[0])
    Y_raw.index = Y_raw.index.astype(str)

    prop_row = pd.read_csv(path, header=None, skiprows=8, nrows=1)
    prop_names = prop_row.iloc[0, 2:].dropna().tolist()
    Y_raw = Y_raw.iloc[:, : len(prop_names)]
    Y_raw.columns = prop_names
    Y_raw = Y_raw.apply(pd.to_numeric, errors="coerce")
    Y_raw = Y_raw.dropna(how="all")

    sample_ids = Y_raw.index.to_numpy()
    Y = Y_raw.values.astype(float)
    return sample_ids, prop_names, Y


def load_aligned_dataset(
    spec_path: str | Path,
    prop_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load and align spectra and properties on common sample IDs."""
    spec_ids, wavelengths, X = load_spectra(spec_path)
    prop_ids, prop_names, Y = load_properties(prop_path)

    spec_df = pd.DataFrame(X, index=spec_ids)
    prop_df = pd.DataFrame(Y, index=prop_ids, columns=prop_names)

    common = spec_df.index.intersection(prop_df.index)
    spec_df = spec_df.loc[common]
    prop_df = prop_df.loc[common]

    return (
        common.to_numpy(),
        wavelengths,
        spec_df.values.astype(float),
        prop_names,
        prop_df.values.astype(float),
    )
