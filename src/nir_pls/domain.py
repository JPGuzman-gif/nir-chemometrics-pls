"""Hotelling T² and Q residual applicability-domain checks."""

import numpy as np
from scipy import stats


def compute_ad_limits(
    X: np.ndarray,
    *,
    n_components: int = 10,
    alpha: float = 0.05,
) -> dict:
    """
    Fit PCA-based applicability domain on raw calibration spectra.

    Replicates notebook Cell 2 preamble (A=10 LVs, 95% limits).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    pca = PCA(n_components=20, random_state=42)
    T = pca.fit_transform(X_sc)
    X_rec = pca.inverse_transform(T)

    A = n_components
    T_A = T[:, :A]
    cov_inv = np.linalg.inv(np.cov(T_A.T))
    T2 = np.einsum("ni,ij,nj->n", T_A, cov_inv, T_A)
    T2_lim = stats.chi2.ppf(1 - alpha, df=A)

    Q = np.sum((X_sc - X_rec) ** 2, axis=1)
    ev = pca.explained_variance_[A:]
    th1, th2, th3 = ev.sum(), (ev ** 2).sum(), (ev ** 3).sum()
    h0 = 1 - (2 * th1 * th3) / (3 * th2 ** 2)
    Q_lim = th1 * (
        stats.norm.ppf(1 - alpha) * np.sqrt(2 * th2 * h0 ** 2) / th1
        + 1
        + th2 * h0 * (h0 - 1) / th1 ** 2
    ) ** (1 / h0)

    outliers_T2 = np.where(T2 > T2_lim)[0]
    outliers_Q = np.where(Q > Q_lim)[0]
    outliers = np.union1d(outliers_T2, outliers_Q)

    return {
        "n_components": A,
        "alpha": alpha,
        "pca_scaler_mean": scaler.mean_.tolist(),
        "pca_scaler_scale": scaler.scale_.tolist(),
        "pca_components": pca.components_[:A].tolist(),
        "pca_components_full": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "cov_inv": cov_inv.tolist(),
        "t2_limit": float(T2_lim),
        "q_limit": float(Q_lim),
        "outlier_indices": outliers.astype(int).tolist(),
    }


def check_applicability(
    spectrum: np.ndarray,
    ad: dict,
) -> dict:
    """
    Compute T² and Q for a single raw spectrum (shape n_wavelengths,).

    in_domain is True when both T² <= limit and Q <= limit.
    """
    mean = np.asarray(ad["pca_scaler_mean"], dtype=float)
    scale = np.asarray(ad["pca_scaler_scale"], dtype=float)
    components_a = np.asarray(ad["pca_components"], dtype=float)
    components_full = np.asarray(ad["pca_components_full"], dtype=float)
    pca_mean = np.asarray(ad["pca_mean"], dtype=float)
    cov_inv = np.asarray(ad["cov_inv"], dtype=float)

    x = np.asarray(spectrum, dtype=float).reshape(1, -1)
    x_sc = (x - mean) / scale
    t_full = (x_sc - pca_mean) @ components_full.T
    t_a = t_full[:, : components_a.shape[0]]

    t2 = float((t_a @ cov_inv @ t_a.T).item())
    x_rec = t_full @ components_full + pca_mean
    q = float(np.sum((x_sc - x_rec) ** 2))

    t2_limit = ad["t2_limit"]
    q_limit = ad["q_limit"]
    in_domain = t2 <= t2_limit and q <= q_limit

    return {
        "t2": t2,
        "t2_limit": t2_limit,
        "q": q,
        "q_limit": q_limit,
        "in_domain": in_domain,
    }
