"""Spectral preprocessing functions ported from nir_diesel_pls.ipynb Cell A."""

import numpy as np

WL_CUTOFF = 1450

PREPROC_NAMES = (
    "Raw (MC)",
    "SNV",
    "SG 1st deriv",
    "SNV + SG",
    "SNV+SG ≤1450",
)


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate — row-wise mean/std normalization."""
    mu = X.mean(axis=1, keepdims=True)
    sig = X.std(axis=1, keepdims=True)
    return (X - mu) / sig


def sg_derivative(X: np.ndarray, window: int = 11) -> np.ndarray:
    """Savitzky-Golay first derivative (polynomial order 2)."""
    half = window // 2
    j = np.arange(-half, half + 1, dtype=float)
    coeffs = j / np.sum(j ** 2)
    return np.apply_along_axis(
        lambda row: np.convolve(row, coeffs[::-1], mode="same"),
        axis=1,
        arr=X,
    )


def snv_sg(X: np.ndarray, window: int = 11) -> np.ndarray:
    """SNV followed by SG first derivative."""
    return sg_derivative(snv(X), window=window)


def mean_center(X: np.ndarray, means: np.ndarray | None = None) -> np.ndarray:
    """Column-wise mean centering. Uses provided means for inference."""
    if means is None:
        means = X.mean(axis=0, keepdims=True)
    else:
        means = np.asarray(means, dtype=float).reshape(1, -1)
    return X - means


def _transform(name: str, X: np.ndarray, wl_mask: np.ndarray | None = None) -> np.ndarray:
    """Apply preprocessing transform without mean centering."""
    if name == "Raw (MC)":
        return X
    if name == "SNV":
        return snv(X)
    if name == "SG 1st deriv":
        return sg_derivative(X)
    if name == "SNV + SG":
        return snv_sg(X)
    if name == "SNV+SG ≤1450":
        if wl_mask is None:
            raise ValueError("wl_mask required for SNV+SG ≤1450 preprocessing")
        return snv_sg(X[:, wl_mask])
    raise ValueError(f"Unknown preprocessing method: {name!r}")


def apply_preproc(
    name: str,
    X: np.ndarray,
    *,
    col_means: np.ndarray | None = None,
    wl_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply full preprocessing (transform + mean center).

    When col_means is None, centers using means from X (calibration build).
    When col_means is provided, subtracts stored calibration means (inference).
    """
    transformed = _transform(name, X, wl_mask=wl_mask)
    return mean_center(transformed, means=col_means)


def calibration_col_means(
    name: str,
    X: np.ndarray,
    wl_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Column means of cohort-preprocessed spectra for manifest storage."""
    return apply_preproc(name, X, wl_mask=wl_mask).mean(axis=0)


def n_wavelengths_after_preproc(
    name: str,
    n_input: int,
    wl_mask: np.ndarray | None = None,
) -> int:
    """Number of features after preprocessing."""
    if name == "SNV+SG ≤1450":
        if wl_mask is None:
            raise ValueError("wl_mask required for SNV+SG ≤1450")
        return int(wl_mask.sum())
    return n_input
