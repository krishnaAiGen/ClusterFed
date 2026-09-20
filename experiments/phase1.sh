#!/bin/bash
# Phase 1: E1 main results x 5 seeds + U/L bounds (R1.6).
# Resumable: rerunning skips completed (experiment, arm, seed) triples.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python}"

echo "=== Phase 1 arm 1/3: v2.1 main results ==="
$PY experiments/run_experiment.py --experiment E1 --arm v2.1 --seeds 42 43 44 45 46

echo "=== Phase 1 arm 2/3: supervised upper bound (U) ==="
$PY experiments/run_experiment.py --experiment E1 --arm upper_bound_supervised \
    --seeds 42 43 44 45 46 --label-source true

echo "=== Phase 1 arm 3/3: v1-legacy lower bound (L) ==="
$PY experiments/run_experiment.py --experiment E1 --arm lower_bound_v1 \
    --seeds 42 43 44 45 46 \
    -o feature_selection.method=rf_gini -o clustering.n_clusters=2 \
    -o clustering.clustering_space=pca2d -o refinement.iterations=0 \
    -o federated.train_on=all -o federated.server_ema_decay=0 \
    -o federated.model_selection=last -o federated.learning_rate=0.001 \
    -o federated.fedprox_mu=0

echo "=== Phase 1 complete ==="
