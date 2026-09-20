"""Tests for the Phase-0 experiment infrastructure: results store, stats, overrides."""
import os
import sys

import numpy as np
import pandas as pd

from clusterfed import results, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from run_experiment import apply_overrides, parse_override  # noqa: E402

from clusterfed.config import Config  # noqa: E402


class _FakeHistory:
    rounds = [1, 2, 3]
    train_acc = [0.5, 0.6, 0.7]
    val_acc = [0.4, 0.5, 0.6]
    selected_round = 3


def test_results_roundtrip_and_resume(tmp_path):
    out = str(tmp_path)
    table = pd.DataFrame({"category": ["Bot", "Average"], "F1": [0.7, 0.7]})
    assert not results.already_done(out, "E1", "v2.1", 42)
    d = results.save_run(
        out, "E1", "v2.1", 42,
        config_snapshot={"overrides": {}},
        metrics={"labeling_raw_avg_f1": 0.9, "federated_detection_avg_f1": 0.8},
        tables={"table1": table}, histories={"pseudo": _FakeHistory()},
        wall_clock_s=12.3,
    )
    assert os.path.exists(os.path.join(d, "metrics.json"))
    assert os.path.exists(os.path.join(d, "table1.csv"))
    assert os.path.exists(os.path.join(d, "history_pseudo.csv"))
    assert results.already_done(out, "E1", "v2.1", 42)          # resume ledger works
    assert not results.already_done(out, "E1", "v2.1", 43)      # other seeds untouched

    hist = pd.read_csv(os.path.join(d, "history_pseudo.csv"))
    assert list(hist.columns) == ["round", "train_acc", "val_acc"]
    assert len(hist) == 3   # per-round curves persisted for plotting

    loaded = results.load_metrics(out, "E1", "v2.1", 42)
    assert loaded["metrics"]["labeling_raw_avg_f1"] == 0.9


def test_mean_std_and_format():
    m, s = stats.mean_std([0.8, 0.9, 1.0])
    assert abs(m - 0.9) < 1e-9
    assert s > 0
    assert "±" in stats.format_mean_std([0.8, 0.9])


def test_wilcoxon_detects_consistent_difference():
    a = [0.90, 0.91, 0.89, 0.92, 0.90, 0.91, 0.93, 0.90, 0.89, 0.92]
    b = [0.80, 0.82, 0.79, 0.81, 0.80, 0.83, 0.82, 0.81, 0.79, 0.80]
    out = stats.wilcoxon_signed_rank(a, b)
    assert out["p_value"] < 0.05
    same = stats.wilcoxon_signed_rank(a, a)
    assert same["p_value"] == 1.0


def test_override_parsing_and_application():
    assert parse_override("clustering.n_clusters=2") == ("clustering.n_clusters", 2)
    assert parse_override("federated.server_ema_decay=0.95") == ("federated.server_ema_decay", 0.95)
    assert parse_override("feature_selection.method=variance") == ("feature_selection.method", "variance")
    assert parse_override("partition.max_attack_per_node=null") == ("partition.max_attack_per_node", None)

    cfg = Config({"clustering": {"n_clusters": 12}, "federated": {"lr": 0.001}})
    new = apply_overrides(cfg, [("clustering.n_clusters", 2), ("federated.lr", 0.0003)])
    assert new["clustering"]["n_clusters"] == 2
    assert new["federated"]["lr"] == 0.0003
    assert cfg["clustering"]["n_clusters"] == 12   # original untouched (deep copy)
