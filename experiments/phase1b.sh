#!/bin/bash
# Phase 1b: finish Phase 1 after the R=0 crash fix.
# - lower_bound_v1 (L): the arm that crashed
# - upper_bound_supervised_curated: NEW control -- true labels on the SAME curated
#   seed rows v2.1 trains on. Distinguishes "label quality" from "hard-row
#   optimization" as the explanation for the all-rows supervised bound scoring
#   0.52 (Entry 18). The already-stored upper_bound_supervised arm = true labels
#   on ALL rows (canonical PRD U bound).
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python}"

echo "=== Phase 1b arm 1/2: v1-legacy lower bound (L) ==="
$PY experiments/run_experiment.py --experiment E1 --arm lower_bound_v1 \
    --seeds 42 43 44 45 46 \
    -o feature_selection.method=rf_gini -o clustering.n_clusters=2 \
    -o clustering.clustering_space=pca2d -o refinement.iterations=0 \
    -o federated.train_on=all -o federated.server_ema_decay=0 \
    -o federated.model_selection=last -o federated.learning_rate=0.001 \
    -o federated.fedprox_mu=0

echo "=== Phase 1b arm 2/2: supervised-on-curated-rows control ==="
$PY experiments/run_experiment.py --experiment E1 --arm upper_bound_supervised_curated \
    --seeds 42 43 44 45 46 --label-source true

echo "=== Phase 1b complete ==="
