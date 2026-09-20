"""PCA dimensionality reduction to 2D for visualization (PRD 9.5).

Backs the PCA scatter plots (Figs 3, 4, 6). The paper uses p=2 components.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def pca_2d(X: np.ndarray, n_components: int = 2, *, seed: int = 42):
    """Project X onto its top ``n_components`` principal components.

    Returns ``(Z, model)`` where Z has shape (n_samples, n_components).
    """
    X = np.asarray(X, dtype="float64")
    n_components = min(n_components, X.shape[1], max(1, X.shape[0]))
    model = PCA(n_components=n_components, random_state=seed)
    Z = model.fit_transform(X)
    return Z, model
