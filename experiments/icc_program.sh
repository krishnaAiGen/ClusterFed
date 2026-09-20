#!/usr/bin/env bash
# ICC 2027 experiment program: Tier B (rho-sweep) + Tier C (bytes-per-round axes).
# Resumable -- run_experiment.py skips (experiment, arm, seed) triples already in
# outputs/results/index.csv, so re-running after an interruption is safe.
set -u

cd "$(dirname "$0")/.."
PY="${PY:-python}"
R="$PY experiments/run_experiment.py --config config.yaml"

# The deployed configuration (E1/final_r0_mu10_k16).
FINAL="-o refinement.iterations=0 -o federated.fedprox_mu=0.1 -o clustering.n_clusters=16"

SEEDS5="42 43 44 45 46"
SEEDS3="42 43 44"

echo "=== E11: Tier B -- controlled pseudo-label noise (rho sweep), 5 seeds ==="
for RHO in 0.00 0.05 0.10 0.15 0.20 0.30; do
  echo "--- E11 rho_${RHO} ---"
  $R --experiment E11 --arm "rho_${RHO}" --seeds $SEEDS5 \
     $FINAL -o "labeling.inject_noise_rate=${RHO}" || echo "[warn] E11 rho_${RHO} failed"
done

echo "=== E12: Tier C -- partial participation, 3 seeds ==="
for C in 0.2 0.5; do
  echo "--- E12 frac_${C} ---"
  $R --experiment E12 --arm "frac_${C}" --seeds $SEEDS3 \
     $FINAL -o "federated.client_fraction=${C}" || echo "[warn] E12 frac_${C} failed"
done

echo "=== E13: Tier C -- local epochs (compute for bandwidth), 3 seeds ==="
for E in 2 5; do
  echo "--- E13 epochs_${E} ---"
  $R --experiment E13 --arm "epochs_${E}" --seeds $SEEDS3 \
     $FINAL -o "federated.local_epochs=${E}" || echo "[warn] E13 epochs_${E} failed"
done

echo "=== E14: Tier C -- update compression, 3 seeds ==="
$R --experiment E14 --arm "fp16" --seeds $SEEDS3 \
   $FINAL -o federated.compression=fp16 || echo "[warn] E14 fp16 failed"
for K in 0.1 0.01; do
  echo "--- E14 topk_${K} ---"
  $R --experiment E14 --arm "topk_${K}" --seeds $SEEDS3 \
     $FINAL -o federated.compression=topk -o "federated.topk_fraction=${K}" \
     || echo "[warn] E14 topk_${K} failed"
done

echo "=== ICC PROGRAM COMPLETE ==="
date
