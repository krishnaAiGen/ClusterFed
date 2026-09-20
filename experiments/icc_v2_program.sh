#!/usr/bin/env bash
# ICC v2 revision program -- addresses the convergent reviewer criticisms.
#
#   E16  tuned supervised references (the #1 criticism: the supervised arm had
#        no imbalance handling and inherited ClusterFed's hyperparameters)
#   E17  headline arms extended 5 -> 10 seeds (attainment cost had 80% spread)
#   E18  top-k WITH error feedback (the "broken compressor" strawman)
#   E19  external baselines re-run with per-round curves, so they can be priced
#        in attainment cost (no stored E2 run has a val_acc trajectory)
set -u

cd "$(dirname "$0")/.."
PY="${PY:-python}"
R="$PY experiments/run_experiment.py --config config.yaml"
FINAL="-o refinement.iterations=0 -o federated.fedprox_mu=0.1 -o clustering.n_clusters=16"
S5="42 43 44 45 46"
S10="42 43 44 45 46 47 48 49 50 51"

echo "=== E16: tuned supervised references (5 seeds) ==="
# Balanced loss, at our mu and at plain FedAvg.
$R --experiment E16 --arm sup_balanced_mu10 --seeds $S5 --label-source true $FINAL \
   -o federated.loss_weighting=balanced || echo "[warn] sup_balanced_mu10"
$R --experiment E16 --arm sup_balanced_mu0 --seeds $S5 --label-source true $FINAL \
   -o federated.loss_weighting=balanced -o federated.fedprox_mu=0.0 || echo "[warn] sup_balanced_mu0"
# Balanced loss on ALL rows (not just the curated seed set).
$R --experiment E16 --arm sup_balanced_allrows --seeds $S5 --label-source true $FINAL \
   -o federated.loss_weighting=balanced -o federated.train_on=all || echo "[warn] sup_balanced_allrows"
# Learning-rate sensitivity, in case 3e-4 simply suits ClusterFed.
$R --experiment E16 --arm sup_balanced_lr1e3 --seeds $S5 --label-source true $FINAL \
   -o federated.loss_weighting=balanced -o federated.learning_rate=0.001 || echo "[warn] sup_balanced_lr1e3"
# Unweighted-but-all-rows control, to separate weighting from row selection.
$R --experiment E16 --arm sup_unweighted_allrows --seeds $S5 --label-source true $FINAL \
   -o federated.train_on=all || echo "[warn] sup_unweighted_allrows"

echo "=== E17: headline arms at 10 seeds ==="
$R --experiment E17 --arm clusterfed --seeds $S10 $FINAL || echo "[warn] E17 clusterfed"
$R --experiment E17 --arm supervised --seeds $S10 --label-source true $FINAL || echo "[warn] E17 supervised"
$R --experiment E17 --arm legacy --seeds $S10 \
   -o feature_selection.method=rf_gini -o clustering.n_clusters=2 \
   -o clustering.clustering_space=pca2d -o refinement.iterations=0 \
   -o federated.train_on=all -o federated.server_ema_decay=0 \
   -o federated.model_selection=last -o federated.learning_rate=0.001 \
   -o federated.fedprox_mu=0 || echo "[warn] E17 legacy"

echo "=== E18: top-k with error feedback (3 seeds) ==="
for K in 0.1 0.01; do
  $R --experiment E18 --arm "topk_${K}_ef" --seeds 42 43 44 $FINAL \
     -o federated.compression=topk -o "federated.topk_fraction=${K}" \
     -o federated.error_feedback=true || echo "[warn] E18 topk_${K}_ef"
done

echo "=== E19: external baselines with per-round curves (5 seeds) ==="
$PY experiments/run_e2_baselines.py --experiment E19 --seeds $S5 || echo "[warn] E19 baselines"

echo "=== ICC v2 PROGRAM COMPLETE ==="
date
