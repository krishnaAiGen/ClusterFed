"""Tests for data/partition.py: the 10-node zero-day split."""
import numpy as np

from clusterfed.data import loader, partition, preprocess


def _nodes(raw_frame, cfg, seed=42):
    df = loader.normalize_labels(raw_frame)
    clean = preprocess.clean(df, cfg.preprocess)
    feats = loader.feature_columns(clean, "Label")
    return partition.build_nodes(clean, cfg.partition, seed=seed, feature_names=feats), feats


def test_ten_nodes(raw_frame, cfg):
    nodes, _ = _nodes(raw_frame, cfg)
    assert len(nodes) == 10
    assert sorted(nodes) == list(range(1, 11))


def test_each_node_only_its_attack(raw_frame, cfg):
    nodes, _ = _nodes(raw_frame, cfg)
    for nid, nd in nodes.items():
        present_attacks = set(nd.y_true_name[nd.y_binary == 1])
        assert present_attacks.issubset(set(partition.NODE_ATTACKS[nid]))
        # Benign present too (zero-day setup: own attack + benign).
        assert (nd.y_binary == 0).any()


def test_deterministic_under_seed(raw_frame, cfg):
    n1, _ = _nodes(raw_frame, cfg, seed=42)
    n2, _ = _nodes(raw_frame, cfg, seed=42)
    for nid in n1:
        assert np.array_equal(n1[nid].X, n2[nid].X)
        assert np.array_equal(n1[nid].y_binary, n2[nid].y_binary)


def _cfg_override(cfg, **kw):
    from clusterfed.config import Config
    c = dict(cfg.partition)
    c.update(kw)
    return Config(c)


def test_generalized_client_counts(raw_frame, cfg):
    """E5: partition must work for 5, 20, 50 clients with every client
    holding some attack rows (or at worst a benign-only remainder)."""
    df = loader.normalize_labels(raw_frame)
    clean = preprocess.clean(df, cfg.preprocess)
    feats = loader.feature_columns(clean, "Label")
    for n in (5, 20, 50):
        pcfg = _cfg_override(cfg, num_nodes=n)
        nodes = partition.build_nodes(clean, pcfg, seed=42, feature_names=feats)
        assert len(nodes) == n
        with_attack = sum(1 for nd in nodes.values() if nd.n_attack > 0)
        assert with_attack >= min(n, 14) - 1  # nearly every client got attack rows


def test_dirichlet_partition_spreads_families(raw_frame, cfg):
    """E9: with alpha=1.0 most clients should hold rows from multiple families."""
    df = loader.normalize_labels(raw_frame)
    clean = preprocess.clean(df, cfg.preprocess)
    feats = loader.feature_columns(clean, "Label")
    pcfg = _cfg_override(cfg, scheme="dirichlet", dirichlet_alpha=1.0)
    nodes = partition.build_nodes(clean, pcfg, seed=42, feature_names=feats)
    assert len(nodes) == 10
    multi = sum(1 for nd in nodes.values()
                if len(set(nd.y_true_name[nd.y_binary == 1])) > 1)
    assert multi >= 5


def test_coordinate_median_resists_outlier_client():
    import torch
    from clusterfed.federated import server
    good = {"w": torch.tensor([1.0, 1.0])}
    good2 = {"w": torch.tensor([1.2, 0.8])}
    poisoned = {"w": torch.tensor([100.0, -100.0])}
    med = server.coordinate_median([(good, 10), (good2, 10), (poisoned, 10)])
    assert float(med["w"][0]) < 2.0 and float(med["w"][1]) > -2.0   # outlier rejected
    avg = server.fedavg([(good, 10), (good2, 10), (poisoned, 10)])
    assert abs(float(avg["w"][0])) > 30                              # fedavg is dragged
