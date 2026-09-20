#!/usr/bin/env bash
# E23: the same size-matched control, but with PSEUDO labels.
#
# E22 showed that with TRUE labels a random subset matches the curated seed set,
# so curation's benefit there is subsetting, not selection. With pseudo labels
# curation does a second thing: it removes mislabeled rows. This arm tells the
# two apart, and decides whether the ICC paper's claim that "row curation is the
# one component that matters in both label regimes" is correct as written.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
FINAL="-o refinement.iterations=0 -o federated.fedprox_mu=0.1 -o clustering.n_clusters=16"
$PY experiments/run_experiment.py --config config.yaml \
  --experiment E23 --arm pseudo_random --seeds 42 43 44 45 46 $FINAL \
  -o federated.train_on=random || echo "[warn] pseudo_random"
$PY experiments/run_experiment.py --config config.yaml \
  --experiment E23 --arm pseudo_all --seeds 42 43 44 45 46 $FINAL \
  -o federated.train_on=all || echo "[warn] pseudo_all"
echo "=== E23 COMPLETE ==="
