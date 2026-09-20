# ClusterFed

Label-free federated pseudo-labeling for zero-day attack detection in consumer IoT networks.

Each client turns its own unlabeled flow records into training labels, with no labeled
clients, no server-side labeled set and no pre-trained model. The only supervision is an
anchor sample of at most 100 flows per client that are known to be benign, used solely to
decide which clusters are benign.

This repository contains the implementation and the complete experimental record for the
manuscript currently under review at IEEE Transactions on Consumer Electronics.

## Results

On CICIDS2017 and CSE-CIC-IDS2018, 5 seeds each, zero training labels:

| | CICIDS2017 | CSE-CIC-IDS2018 |
|---|---|---|
| Pseudo-labeling F1 | 0.934 ± 0.007 | 0.950 ± 0.004 |
| Federated detection F1 | 0.865 ± 0.013 | 0.878 ± 0.029 |
| Detection accuracy | 0.844 ± 0.013 | 0.881 ± 0.017 |

A fully supervised reference trained on the same rows with ground-truth labels reaches
0.870 ± 0.008 (p = 0.48, not significant), so the label-free pipeline matches full
supervision at zero labeling cost on this partition.

## Method

Five stages run at each client, none of which reads a ground-truth label.

| Stage | What it does |
|---|---|
| 0 | Label-free feature selection: interleaved Laplacian-score and variance rankings, 78 → 32 features |
| 1 | Min–max scaling; clustering happens in the full 32-D space (PCA is for plots only) |
| 2 | `K`-means with `K = 16`, oriented benign/malicious by a benign-density likelihood-ratio test against the anchor sample; each point gets a centroid-margin confidence |
| 3 | Class-balanced adaptive thresholds select a curated *seed set*; ambiguous points are withheld rather than mislabeled |
| 4 | *(optional)* Self-training with a cluster–model co-agreement filter and MC-Dropout uncertainty |

Federated training then runs FedAvg + FedProx (`μ = 0.1`) on the curated seed rows, with a
server-side EMA and an **unsupervised** best-round model selection rule that scores rounds
against the clients' own pseudo-labels. That selection rule is the single most load-bearing
component: replacing it with a last-round readout costs 45.9 detection F1 points.

## Install

```bash
pip install -r requirements.txt
```

Python 3.11, PyTorch 2.2.1, scikit-learn 1.9.0, scipy 1.17.1, numpy 1.26.4, pandas 2.2.2.

## Data

Datasets are **not** included in this repository.

* **CICIDS2017** — download from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
  and point `paths.raw_dir` in `config.yaml` at the directory of CSVs.
* **CSE-CIC-IDS2018** — `experiments/download_2018.sh` pulls it from the public S3 bucket
  over anonymous HTTPS (~6.9 GB, resumable). `config_2018.yaml` expects it in `data/raw_2018`.

Preprocessing writes a cache to `data/cache`, which is gitignored.

## Run

```bash
# one end-to-end run on CICIDS2017
python experiments/run_all.py --config config.yaml

# or stage by stage
python experiments/run_preprocess.py --config config.yaml
python experiments/run_labeling.py  --config config.yaml   # Stages 0-4  -> Table III (labeling)
python experiments/run_federated.py --config config.yaml   # FedAvg      -> Table III (detection)
```

The generic multi-seed runner used for every reported experiment:

```bash
python experiments/run_experiment.py --experiment E1 --arm final_r0_mu10_k16 \
       --seeds 42 43 44 45 46 \
       -o refinement.iterations=0 -o federated.fedprox_mu=0.1 -o clustering.n_clusters=16
```

Results land in `outputs/results/<experiment>/<arm>/seed<N>/` and are indexed in
`outputs/results/index.csv`, which doubles as a resume ledger: re-running a completed
(experiment, arm, seed) triple skips it.

### Reproducing the paper's headline row

```bash
python experiments/run_experiment.py --experiment REPRO --arm check --seeds 42 --force \
       -o refinement.iterations=0 -o federated.fedprox_mu=0.1 -o clustering.n_clusters=16
# expect: det_f1 0.8659, acc 0.8464, lab_f1 0.9306, selected_round 115
```

### Experiment scripts

| Script | Covers |
|---|---|
| `phase1.sh`, `phase1b.sh` | Main results, 5 seeds, plus supervised and legacy reference arms |
| `phase2.sh`, `phase2_5seeds.sh` | The 24-arm ablation grid |
| `run_e2_baselines.py` | Four reimplemented federated semi-supervised baselines |
| `run_lofo.py` | Leave-one-family-out zero-day protocol |
| `run_e4_minority.py` | Per-attack-type minority and rare-class analysis |
| `run_e7_cost.py` | Communication and compute cost |
| `overnight.sh` | Robustness, privacy, Dirichlet non-IID and scalability chain |

Every script is resumable. All of them accept `PY=/path/to/python` to override the interpreter.

## Stored results

`outputs/results/` contains the complete experimental record behind the paper: **253 runs**,
with per-run metrics (including a full config snapshot), per-node tables, per-round training
curves and refinement traces. Every number in the manuscript can be recomputed from
`index.csv` without retraining anything.

## Tests

```bash
pytest tests/ -q
```

## Repository layout

```
src/clusterfed/
  data/         loading, preprocessing, partitioning (attack-family and Dirichlet schemes)
  labeling/     Stages 0-4: feature selection, clustering, thresholding, refinement
  federated/    FedAvg / FedProx server and client, EMA, unsupervised round selection
  theory/       empirical pseudo-label noise estimation
  baselines.py  reimplemented comparison methods
  results.py    run store and resume ledger
  stats.py      mean ± std and Wilcoxon signed-rank
experiments/    entry points and the shell chains that produced the paper's tables
tests/          unit and regression tests
outputs/        stored results (tracked) and figures/models (ignored)
```

## Citation

Manuscript under review. Citation details will be added on acceptance.

## License

No license file is included yet. Until one is added, default copyright applies and others
may not reuse this code; add a `LICENSE` before making the repository public if you intend
to permit reuse.
