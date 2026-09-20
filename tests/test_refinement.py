"""Tests for labeling/refinement.py: Stage 4 MC-Dropout co-agreement self-training (PRD 4.5)."""
import numpy as np

from clusterfed.labeling.refinement import refine_node_labels


def _imbalanced_blobs(seed=0, n_benign=300, n_attack=30, d=4):
    """A clearly separable but imbalanced binary problem (minority-attack stress test)."""
    rng = np.random.default_rng(seed)
    benign = rng.normal(0.0, 0.4, size=(n_benign, d))
    attack = rng.normal(4.0, 0.4, size=(n_attack, d))
    X = np.vstack([benign, attack])
    names = np.array(["BENIGN"] * n_benign + ["Bot"] * n_attack)
    return X, names


MODEL_CFG = {"hidden_layers": [16, 8], "output_dim": 2}
CLUSTERING_CFG = {"n_clusters": 2, "n_init": 10, "known_benign_n": 50}
THRESHOLD_CFG = {"policy": "per_class_adaptive", "min_quantile": 0.2, "max_quantile": 0.8}
IMBALANCE_CFG = {"loss": "focal", "focal_gamma": 2.0}
REFINEMENT_CFG = {
    "iterations": 2, "local_epochs_per_iteration": 3, "batch_size": 32,
    "learning_rate": 0.01, "dropout_p": 0.2, "mc_dropout_passes": 5,
    "uncertainty_threshold_tau_u": 1.0, "relabel_confidence_threshold": 0.8,
    "max_promote_fraction_per_iteration": 0.5,
}


def test_refine_node_labels_produces_valid_binary_output():
    X, names = _imbalanced_blobs()
    result = refine_node_labels(
        X, names, MODEL_CFG, CLUSTERING_CFG, THRESHOLD_CFG, IMBALANCE_CFG, REFINEMENT_CFG, seed=0,
    )
    assert result.y_refined.shape == (len(X),)
    assert set(np.unique(result.y_refined)).issubset({0, 1})
    assert result.y_cluster_raw.shape == (len(X),)


def test_uncertainty_diagnostics_reported_per_iteration():
    """Diagnostic added to distinguish 'tau_u miscalibrated' from 'MC-Dropout
    uncertainty just isn't informative here' without more blind threshold
    guessing (see EXPERIMENT_LOG.md Entry 4)."""
    X, names = _imbalanced_blobs()
    result = refine_node_labels(
        X, names, MODEL_CFG, CLUSTERING_CFG, THRESHOLD_CFG, IMBALANCE_CFG, REFINEMENT_CFG, seed=0,
    )
    assert len(result.uncertainty_diagnostics) == len(result.seed_set_size_r)
    for diag in result.uncertainty_diagnostics:
        assert {"n_agree", "n_disagree", "agree_std_median", "disagree_std_median",
                "auroc_std_predicts_disagreement"}.issubset(diag.keys())


def test_seed_set_only_grows_across_iterations():
    X, names = _imbalanced_blobs()
    result = refine_node_labels(
        X, names, MODEL_CFG, CLUSTERING_CFG, THRESHOLD_CFG, IMBALANCE_CFG, REFINEMENT_CFG, seed=0,
    )
    sizes = result.seed_set_size_r
    assert all(b >= a for a, b in zip(sizes, sizes[1:]))


def test_refinement_recovers_separable_minority_class():
    """On a well-separated (if imbalanced) problem, refinement should match or
    beat the raw Stage-2 cluster bootstrap against ground truth."""
    X, names = _imbalanced_blobs(seed=3, n_attack=15)
    y_true = (names != "BENIGN").astype(int)
    result = refine_node_labels(
        X, names, MODEL_CFG, CLUSTERING_CFG, THRESHOLD_CFG, IMBALANCE_CFG, REFINEMENT_CFG, seed=3,
    )
    raw_acc = (result.y_cluster_raw == y_true).mean()
    refined_acc = (result.y_refined == y_true).mean()
    assert refined_acc >= raw_acc - 0.05  # allow small stochastic slack, no large regression


def test_single_iteration_never_dumps_the_whole_held_out_set():
    """Regression test for the bug this fix addresses: with a deliberately loose
    tau_u (as would happen if uncertainty were measured on saturated sigmoid
    probabilities instead of logits), a single iteration must not promote
    ~all of the held-out set at once -- refinement has to stay genuinely
    iterative across R rounds, not collapse into a one-shot dump."""
    X, names = _imbalanced_blobs(seed=1, n_benign=500, n_attack=50)
    loose_cfg = dict(REFINEMENT_CFG)
    loose_cfg["uncertainty_threshold_tau_u"] = 1e6  # everything "passes" the uncertainty check
    loose_cfg["max_promote_fraction_per_iteration"] = 0.3
    result = refine_node_labels(
        X, names, MODEL_CFG, CLUSTERING_CFG, THRESHOLD_CFG, IMBALANCE_CFG, loose_cfg, seed=1,
    )
    sizes = result.seed_set_size_r
    assert len(sizes) >= 1
    first_iter_growth = sizes[0]  # seed-set size after iteration 1 (starts from the Stage-3 seed set)
    # Well below the ~99.7%-in-one-shot behavior the old probability-space std produced.
    assert first_iter_growth < 0.6 * len(X)


def test_degenerate_seed_set_stops_early_without_crashing():
    """An almost-entirely-one-class node (near-empty minority) should not crash
    even if the seed set becomes degenerate."""
    X, names = _imbalanced_blobs(n_benign=100, n_attack=1)
    result = refine_node_labels(
        X, names, MODEL_CFG, CLUSTERING_CFG, THRESHOLD_CFG, IMBALANCE_CFG, REFINEMENT_CFG, seed=0,
    )
    assert result.y_refined.shape == (len(X),)
