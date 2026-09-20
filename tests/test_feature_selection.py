"""Tests for labeling/feature_selection.py: Stage 0 unsupervised ranking (PRD 4.1)."""
import numpy as np

from clusterfed.labeling import feature_selection as fs


def _structured_data(seed=0, n=600, d=6):
    """3 informative features that separate two manifolds + 3 pure-noise features."""
    rng = np.random.default_rng(seed)
    n_half = n // 2
    t = np.concatenate([rng.normal(-2.0, 0.3, n_half), rng.normal(2.0, 0.3, n_half)])
    informative = np.stack([t, t * 0.5 + rng.normal(0, 0.05, n), -t + rng.normal(0, 0.05, n)], axis=1)
    noise = rng.normal(0.0, 1.0, size=(n, d - 3))
    X = np.hstack([informative, noise])
    names = [f"informative_{i}" for i in range(3)] + [f"noise_{i}" for i in range(d - 3)]
    y = (t > 0).astype(int)
    return X, y, names


def test_laplacian_score_ranks_informative_features_first():
    X, _, names = _structured_data()
    cfg = {"laplacian_knn": 8, "laplacian_subsample": None}
    ranking = fs.laplacian_score(X, cfg, names, seed=0)
    top3 = set(ranking.index[:3])
    assert top3 == {"informative_0", "informative_1", "informative_2"}


def test_laplacian_score_no_labels_needed():
    """Signature never takes y -- verifies Stage 0 fixes R2.1 (no ground-truth labels)."""
    import inspect

    sig = inspect.signature(fs.laplacian_score)
    assert "y" not in sig.parameters


def test_variance_ranking_orders_descending():
    X = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 5.0], [1.0, -5.0]])
    ranking = fs.variance_ranking(X, ["const", "varies"])
    assert list(ranking.index) == ["varies", "const"]


def test_select_features_stage0_laplacian_default(monkeypatch):
    X, y, names = _structured_data()
    cfg = {"method": "laplacian", "expected_selected": 3, "laplacian_knn": 8, "laplacian_subsample": None}
    result = fs.select_features_stage0(X, y, cfg, names, seed=0)
    assert result["method"] == "laplacian"
    assert len(result["selected"]) == 3
    assert len(result["selected_idx"]) == 3
    assert set(result["selected"]) == {"informative_0", "informative_1", "informative_2"}


def test_hybrid_keeps_spiky_discriminator_laplacian_drops():
    """The FTP-Patator fix: a near-categorical 'port-like' feature (constant for
    one mode, spread for the other) that Laplacian down-ranks must survive
    hybrid selection via the variance ranking."""
    rng = np.random.default_rng(5)
    n = 600
    t = np.concatenate([rng.normal(-2.0, 0.3, n // 2), rng.normal(2.0, 0.3, n // 2)])
    smooth = np.stack([t, t * 0.5 + rng.normal(0, 0.05, n)], axis=1)
    # Port-like: half the rows constant 0.003 (port 21 normalized), half uniform.
    port = np.concatenate([np.full(n // 2, 0.003), rng.uniform(0, 1, n // 2)])[:, None]
    low_var_noise = rng.normal(0.0, 0.01, size=(n, 2))
    X = np.hstack([smooth, port, low_var_noise])
    names = ["smooth_0", "smooth_1", "port_like", "tiny_noise_0", "tiny_noise_1"]

    cfg = {"method": "hybrid", "expected_selected": 3, "laplacian_knn": 8, "laplacian_subsample": None}
    result = fs.select_features_stage0(X, None, cfg, names, seed=5)
    assert "port_like" in result["selected"]


def test_rf_gini_legacy_arm_still_works():
    X, y, names = _structured_data()
    cfg = {"rf_n_estimators": 50, "rf_subsample": None, "importance_threshold_T": 0.8, "expected_selected": 3}
    result = fs.select_features_stage0(X, y, cfg | {"method": "rf_gini"}, names, seed=0)
    assert result["method"] == "rf_gini"
    assert len(result["selected"]) >= 1
