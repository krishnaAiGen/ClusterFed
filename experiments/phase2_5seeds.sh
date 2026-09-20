#!/bin/bash
# Phase 2: E3 ablation grid, one factor at a time from the v2.1 reference.
# Reference arm = E1/v2.1 (already run in Phase 1; reused, not repeated).
# 3 seeds initially (extend to 5 by rerunning with more --seeds; resume skips done triples).
# Labeling-only arms use --skip-federated (fast); arms touching federated training run full.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python}"
SEEDS="42 43 44 45 46"

echo "=== A1: feature selection (labeling-only) ==="
$PY experiments/run_experiment.py --experiment E3 --arm A1_laplacian --seeds $SEEDS --skip-federated \
    -o feature_selection.method=laplacian
$PY experiments/run_experiment.py --experiment E3 --arm A1_variance --seeds $SEEDS --skip-federated \
    -o feature_selection.method=variance
$PY experiments/run_experiment.py --experiment E3 --arm A1_rf_gini_v1legacy --seeds $SEEDS --skip-federated \
    -o feature_selection.method=rf_gini
$PY experiments/run_experiment.py --experiment E3 --arm A1_all_features --seeds $SEEDS --skip-federated \
    -o feature_selection.method=variance -o feature_selection.expected_selected=70

echo "=== A6: clustering space (labeling-only) ==="
$PY experiments/run_experiment.py --experiment E3 --arm A6_pca2d_v1legacy --seeds $SEEDS --skip-federated \
    -o clustering.clustering_space=pca2d

echo "=== A9: bootstrap cluster count k (labeling-only) ==="
for K in 2 4 8 16 20; do
  $PY experiments/run_experiment.py --experiment E3 --arm A9_k$K --seeds $SEEDS --skip-federated \
      -o clustering.n_clusters=$K
done

echo "=== A8: refinement iterations R (full) ==="
for R in 0 1 5; do
  $PY experiments/run_experiment.py --experiment E3 --arm A8_R$R --seeds $SEEDS \
      -o refinement.iterations=$R
done

echo "=== A3: refinement design (full) ==="
$PY experiments/run_experiment.py --experiment E3 --arm A3_no_refinement_no_curation --seeds $SEEDS \
    -o refinement.iterations=0 -o federated.train_on=all
$PY experiments/run_experiment.py --experiment E3 --arm A3_naive_selftraining --seeds $SEEDS \
    -o refinement.relabel_confidence_threshold=0.8 \
    -o refinement.relabel_cluster_conf_quantile=1.0 \
    -o refinement.relabel_max_fraction_per_iteration=0.34

echo "=== A4: threshold policy (full) ==="
$PY experiments/run_experiment.py --experiment E3 --arm A4_global_threshold --seeds $SEEDS \
    -o threshold.policy=global

echo "=== A5: imbalance loss (full) ==="
$PY experiments/run_experiment.py --experiment E3 --arm A5_bce --seeds $SEEDS \
    -o imbalance.loss=bce
$PY experiments/run_experiment.py --experiment E3 --arm A5_logit_adjusted --seeds $SEEDS \
    -o imbalance.loss=logit_adjusted

echo "=== A10: federated training rows (full) ==="
$PY experiments/run_experiment.py --experiment E3 --arm A10_train_on_all --seeds $SEEDS \
    -o federated.train_on=all

echo "=== A7: aggregator / server-side knobs (full) ==="
$PY experiments/run_experiment.py --experiment E3 --arm A7_fedavg_mu0 --seeds $SEEDS \
    -o federated.fedprox_mu=0
$PY experiments/run_experiment.py --experiment E3 --arm A7_fedprox_mu01 --seeds $SEEDS \
    -o federated.fedprox_mu=0.1
$PY experiments/run_experiment.py --experiment E3 --arm A7_no_ema --seeds $SEEDS \
    -o federated.server_ema_decay=0
$PY experiments/run_experiment.py --experiment E3 --arm A7_selection_last --seeds $SEEDS \
    -o federated.model_selection=last
$PY experiments/run_experiment.py --experiment E3 --arm A7_lr1e3_v1legacy --seeds $SEEDS \
    -o federated.learning_rate=0.001

echo "=== Phase 2 complete ==="
