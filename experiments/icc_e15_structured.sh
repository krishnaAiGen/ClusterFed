#!/usr/bin/env bash
# E15: does the STRUCTURE of pseudo-label noise matter, at matched noise rate?
#
# E11 showed that symmetric noise up to rho ~ 0.30 leaves detection quality
# essentially untouched -- so the rate alone does not set the attainment floor.
# This sweep holds the rate fixed and changes only which class the flips are
# drawn from. attack2benign is the realistic failure (an attack family read as
# benign, e.g. a mis-oriented cluster); benign2attack is its mirror image.
set -u

cd "$(dirname "$0")/.."
PY="${PY:-python}"
R="$PY experiments/run_experiment.py --config config.yaml"
FINAL="-o refinement.iterations=0 -o federated.fedprox_mu=0.1 -o clustering.n_clusters=16"
SEEDS5="42 43 44 45 46"

echo "=== E15: structured (class-conditional) label noise ==="
for RATE in 0.05 0.10 0.20; do
  echo "--- E15 a2b_${RATE} ---"
  $R --experiment E15 --arm "a2b_${RATE}" --seeds $SEEDS5 $FINAL \
     -o "labeling.inject_noise_rate=${RATE}" \
     -o labeling.inject_noise_mode=attack2benign || echo "[warn] E15 a2b_${RATE} failed"
done

echo "--- E15 b2a_0.10 (mirror-image control) ---"
$R --experiment E15 --arm "b2a_0.10" --seeds $SEEDS5 $FINAL \
   -o labeling.inject_noise_rate=0.10 \
   -o labeling.inject_noise_mode=benign2attack || echo "[warn] E15 b2a_0.10 failed"

echo "=== E15 COMPLETE ==="
date
