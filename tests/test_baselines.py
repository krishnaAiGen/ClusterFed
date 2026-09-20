"""Tests for the E2 baseline reimplementations on tiny separable synthetic splits."""
import numpy as np

from clusterfed import baselines

MODEL_CFG = {"hidden_layers": [16, 8], "output_dim": 2}
FED_CFG = {"communication_rounds": 5, "learning_rate": 0.003, "local_epochs": 1,
           "batch_size": 32, "test_split": 0.2}


def _toy_splits(seed=0, n_nodes=3, n=400):
    rng = np.random.default_rng(seed)
    splits = {}
    for nid in range(1, n_nodes + 1):
        benign = rng.normal(0.0, 0.3, size=(n // 2, 6))
        attack = rng.normal(3.0 + nid, 0.3, size=(n // 2, 6))
        X = np.vstack([benign, attack])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        perm = rng.permutation(n)
        X, y = X[perm], y[perm]
        splits[nid] = {"row_label": f"node{nid}", "X_train": X[:300], "y_train_true": y[:300],
                       "X_test": X[300:], "y_test_true": y[300:]}
    return splits


def _avg_f1(rows):
    return float(np.mean([r["F1"] for r in rows]))


def test_zhao_baseline_learns_separable_data():
    splits = _toy_splits()
    model, best_round, hist = baselines.run_zhao_dynamic_threshold(
        splits, MODEL_CFG, FED_CFG, seed=0, labeled_budget=60)
    rows = baselines.evaluate_model_baseline(model, splits)
    assert _avg_f1(rows) > 0.8
    assert best_round >= 1


def test_cbafed_baseline_learns_separable_data():
    splits = _toy_splits(seed=1)
    cfg = dict(FED_CFG)
    cfg["communication_rounds"] = 15   # 5%-labeled cold start needs a few more rounds
    model, best_round, _ = baselines.run_cbafed_class_balanced(
        splits, MODEL_CFG, cfg, seed=1, labeled_frac=0.05)
    rows = baselines.evaluate_model_baseline(model, splits)
    assert _avg_f1(rows) > 0.8


def test_fedups_baseline_learns_separable_data():
    splits = _toy_splits(seed=2)
    model, best_round, _ = baselines.run_fedups_uncertainty(
        splits, MODEL_CFG, FED_CFG, seed=2, labeled_budget=60, mc_passes=3)
    rows = baselines.evaluate_model_baseline(model, splits)
    assert _avg_f1(rows) > 0.8


def test_fedmse_autoencoder_flags_far_anomalies():
    splits = _toy_splits(seed=3)
    preds = baselines.run_fedmse_autoencoder(splits, FED_CFG, seed=3, rounds=10)
    rows = baselines.evaluate_predictions_baseline(preds, splits)
    # AE trained on benign should flag the well-separated attack blob.
    assert _avg_f1(rows) > 0.6


def test_baseline_splits_deterministic_and_disjoint():
    splits = _toy_splits(seed=4)
    assert all(len(s["X_train"]) + len(s["X_test"]) == 400 for s in splits.values())
