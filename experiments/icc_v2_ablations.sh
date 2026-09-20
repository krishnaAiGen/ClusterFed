#!/usr/bin/env bash
# E20: ablations measured as one-factor deviations from the DEPLOYED config.
#
# The stored E3 grid varies one factor from the *base* config (k=12, R=3,
# mu=0.01), whose own reference never attains eps=0.85 -- so those arms cannot
# be tabulated against the deployed configuration. These re-runs can.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
R="$PY experiments/run_experiment.py --config config.yaml"
FINAL="-o refinement.iterations=0 -o federated.fedprox_mu=0.1 -o clustering.n_clusters=16"
S5="42 43 44 45 46"

echo "=== E20: ablations from the deployed configuration (5 seeds) ==="
$R --experiment E20 --arm global_threshold --seeds $S5 $FINAL \
   -o threshold.policy=global            || echo "[warn] global_threshold"
$R --experiment E20 --arm no_ema --seeds $S5 $FINAL \
   -o federated.server_ema_decay=0       || echo "[warn] no_ema"
$R --experiment E20 --arm selection_last --seeds $S5 $FINAL \
   -o federated.model_selection=last     || echo "[warn] selection_last"
$R --experiment E20 --arm train_on_all --seeds $S5 $FINAL \
   -o federated.train_on=all             || echo "[warn] train_on_all"
$R --experiment E20 --arm refine_R3 --seeds $S5 $FINAL \
   -o refinement.iterations=3            || echo "[warn] refine_R3"
$R --experiment E20 --arm k2_binary --seeds $S5 $FINAL \
   -o clustering.n_clusters=2            || echo "[warn] k2_binary"
$R --experiment E20 --arm fedavg_mu0 --seeds $S5 $FINAL \
   -o federated.fedprox_mu=0             || echo "[warn] fedavg_mu0"
echo "=== E20 COMPLETE ==="
date

echo "=== E21: do the levers stack? sampling + sparsification + fp16 ==="
$R --experiment E21 --arm c02_topk_ef --seeds $S5 $FINAL \
   -o federated.client_fraction=0.2 -o federated.compression=topk \
   -o federated.topk_fraction=0.1 -o federated.error_feedback=true \
   || echo "[warn] c02_topk_ef"
$R --experiment E21 --arm c05_topk_ef --seeds $S5 $FINAL \
   -o federated.client_fraction=0.5 -o federated.compression=topk \
   -o federated.topk_fraction=0.1 -o federated.error_feedback=true \
   || echo "[warn] c05_topk_ef"
echo "=== E21 COMPLETE ==="
date
