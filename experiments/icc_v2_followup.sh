#!/usr/bin/env bash
# The weighted supervised arm is the paper's headline reference after E16, so it
# needs the same 10 seeds as ClusterFed for the comparison to be fair.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
FINAL="-o refinement.iterations=0 -o federated.fedprox_mu=0.1 -o clustering.n_clusters=16"
echo "=== E17b: weighted supervised at 10 seeds ==="
$PY experiments/run_experiment.py --config config.yaml \
  --experiment E17 --arm supervised_balanced --seeds 42 43 44 45 46 47 48 49 50 51 \
  --label-source true $FINAL -o federated.loss_weighting=balanced \
  || echo "[warn] E17 supervised_balanced"
echo "=== E17b COMPLETE ==="
