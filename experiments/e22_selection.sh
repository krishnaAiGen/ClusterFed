#!/usr/bin/env bash
# E22: does client-side ROW SELECTION or merely the SIZE reduction drive the
# curation effect? All arms use TRUE labels, so label correctness is held fixed
# and only the training subset changes.
#
#   curated : confidence-selected seed set (~28% of rows)
#   random  : uniformly random subset, size-matched per client
#   randstrat: random but matched to the seed set's class balance too
#   all     : every row
#
# If random ~= curated, the effect is size. If random ~= all, it is selection.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
R="$PY experiments/run_experiment.py --config config.yaml"
FINAL="-o refinement.iterations=0 -o federated.fedprox_mu=0.1 -o clustering.n_clusters=16"
S10="42 43 44 45 46 47 48 49 50 51"

echo "=== E22: row selection vs size, true labels, 10 seeds ==="
$R --experiment E22 --arm curated --seeds $S10 --label-source true $FINAL \
   -o federated.train_on=seed              || echo "[warn] curated"
$R --experiment E22 --arm random --seeds $S10 --label-source true $FINAL \
   -o federated.train_on=random            || echo "[warn] random"
$R --experiment E22 --arm randstrat --seeds $S10 --label-source true $FINAL \
   -o federated.train_on=random_stratified || echo "[warn] randstrat"
$R --experiment E22 --arm all --seeds $S10 --label-source true $FINAL \
   -o federated.train_on=all               || echo "[warn] all"
echo "=== E22 COMPLETE ==="
date
