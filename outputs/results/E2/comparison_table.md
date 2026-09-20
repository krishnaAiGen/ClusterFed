| method | detection_F1 | accuracy | wilcoxon_p_vs_ours | n_pairs | ours_better |
|---|---|---|---|---|---|
| ClusterFed v2 (ours, zero labels) | 0.865 ± 0.013 | 0.844 ± 0.013 | -- | -- | -- |
| B1 Zhao et al. 2022 (2k labeled @server) | 0.643 ± 0.258 | 0.711 ± 0.135 | 8.78e-07 | 50 | yes |
| B3 CBAFed-style (5% labeled/client) | 0.650 ± 0.151 | 0.719 ± 0.068 | 5.68e-06 | 50 | yes |
| B4 FedMSE-style AE (benign traffic) | 0.427 ± 0.005 | 0.618 ± 0.006 | 1.29e-10 | 50 | yes |
| B5 FedUPS-style (2k labeled @server) | 0.847 ± 0.015 | 0.827 ± 0.015 | 6.88e-01 | 50 | yes |
| ClusterFed v1 pipeline (L bound) | 0.721 ± 0.017 | 0.705 ± 0.012 | 1.86e-06 | 50 | yes |
| Fully supervised FedAvg (U bound) | 0.526 ± 0.024 | 0.695 ± 0.007 | 1.76e-08 | 50 | yes |
