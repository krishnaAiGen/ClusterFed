#!/bin/bash
# Overnight autonomous run (user directive 2026-09-10): everything remaining.
# All runs are resumable via the results index; rerunning this script skips
# completed (experiment, arm, seed) triples.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
FINAL="-o refinement.iterations=0 -o federated.fedprox_mu=0.1 -o clustering.n_clusters=16"

echo "=========== [$(date +%H:%M)] STAGE 1: LOFO zero-day protocol (2017, all families, seed 42) ==========="
$PY experiments/run_lofo.py --seeds 42 $FINAL

echo "=========== [$(date +%H:%M)] STAGE 2: E8 robustness ==========="
for frac in 0.1 0.2 0.3; do
  $PY experiments/run_experiment.py --experiment E8 --arm poison${frac}_fedavg --seeds 42 43 44 \
      $FINAL -o federated.poison_fraction=$frac
  $PY experiments/run_experiment.py --experiment E8 --arm poison${frac}_median --seeds 42 43 44 \
      $FINAL -o federated.poison_fraction=$frac -o federated.aggregation_method=coordinate_median
done
for sig in 0.01 0.1; do
  $PY experiments/run_experiment.py --experiment E8 --arm dp_sigma$sig --seeds 42 43 44 \
      $FINAL -o federated.dp_sigma=$sig
done

echo "=========== [$(date +%H:%M)] STAGE 3: E9 Dirichlet non-IID ==========="
for alpha in 0.1 0.5 1.0; do
  $PY experiments/run_experiment.py --experiment E9 --arm dirichlet_a$alpha --seeds 42 43 44 \
      $FINAL -o partition.scheme=dirichlet -o partition.dirichlet_alpha=$alpha
done

echo "=========== [$(date +%H:%M)] STAGE 4: E5 scalability ==========="
for n in 5 20 50; do
  $PY experiments/run_experiment.py --experiment E5 --arm clients$n --seeds 42 43 44 \
      $FINAL -o partition.num_nodes=$n
done

echo "=========== [$(date +%H:%M)] STAGE 5: E6 zeta measurement (1 run) ==========="
$PY experiments/run_experiment.py --experiment E6 --arm zeta_final --seeds 42 \
    $FINAL -o federated.measure_zeta=true

echo "=========== [$(date +%H:%M)] STAGE 6: extend E3 ablations to 5 seeds (resume-aware) ==========="
sed 's/^SEEDS="42 43 44"$/SEEDS="42 43 44 45 46"/' experiments/phase2.sh > /tmp/phase2_5seeds.sh
bash /tmp/phase2_5seeds.sh || echo "[warn] ablation extension had a failure; continuing"

echo "=========== [$(date +%H:%M)] STAGE 7: wait for 2018 download ==========="
until [ -f "data/raw_2018/.download_complete" ]; do sleep 60; done
echo "[$(date +%H:%M)] download complete"

echo "=========== [$(date +%H:%M)] STAGE 8: 2018 preprocess ==========="
$PY experiments/run_preprocess.py --config config_2018.yaml || exit 1

echo "=========== [$(date +%H:%M)] STAGE 9: E1-2018 main results (3 seeds) ==========="
$PY experiments/run_experiment.py --config config_2018.yaml --experiment E1_2018 --arm final --seeds 42 43 44

echo "=========== [$(date +%H:%M)] STAGE 10: LOFO Bot on 2018 (Fed-DTCN-comparable) + full family sweep seed 42 ==========="
$PY experiments/run_lofo.py --config config_2018.yaml --experiment E_LOFO_2018 --seeds 42 43 44 --families Bot
$PY experiments/run_lofo.py --config config_2018.yaml --experiment E_LOFO_2018 --seeds 42

echo "=========== [$(date +%H:%M)] OVERNIGHT COMPLETE ==========="
