"""Tests for federated/: model shape (incl. MC-Dropout), FedAvg weighting, and a
short FL run using v2's refined-labels-in / pseudo_labels-dict signature."""
import numpy as np
import torch

from clusterfed.federated import server
from clusterfed.federated.model import ClusterFedNet, build_model, enable_mc_dropout


def test_model_forward_shape():
    model = ClusterFedNet(input_dim=10, hidden_layers=(200, 100, 100), output_dim=2)
    out = model(torch.randn(8, 10))
    assert out.shape == (8, 2)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_dropout_p_zero_matches_v1_shape_exactly():
    """Federated model default (dropout_p=0.0) must be architecture-identical to v1."""
    model = build_model(10, {"hidden_layers": [200, 100, 100], "output_dim": 2})
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())


def test_mc_dropout_enabled_model_gives_stochastic_outputs():
    model = build_model(10, {"hidden_layers": [16, 8], "output_dim": 2}, dropout_p=0.5)
    enable_mc_dropout(model)
    x = torch.randn(4, 10)
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2)  # dropout still active despite eval() elsewhere


def test_fedavg_weighted_average():
    a = {"w": torch.tensor([0.0, 0.0])}
    b = {"w": torch.tensor([10.0, 20.0])}
    avg = server.fedavg([(a, 1), (b, 3)])
    assert torch.allclose(avg["w"], torch.tensor([7.5, 15.0]))


def test_fedavg_equal_weight():
    a = {"w": torch.tensor([2.0])}
    b = {"w": torch.tensor([4.0])}
    avg = server.fedavg([(a, 5), (b, 5)])
    assert torch.allclose(avg["w"], torch.tensor([3.0]))


def test_short_federated_run_learns(cfg):
    """A few rounds on separable data (true labels, the upper bound) should push
    detection accuracy up."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from synthetic import make_frame

    from clusterfed.config import Config
    from clusterfed.data import loader, partition, preprocess
    from clusterfed.federated.train import run_federated

    c = dict(cfg)
    c["federated"] = dict(cfg.federated)
    c["federated"]["communication_rounds"] = 5
    c["federated"]["server_ema_decay"] = 0.0   # 5 rounds is far below the EMA horizon
    c = Config(c)

    df = loader.normalize_labels(make_frame(seed=1))
    clean = preprocess.clean(df, c.preprocess)
    feats = loader.feature_columns(clean, "Label")
    Xs, _ = preprocess.scale(clean[feats].to_numpy("float64"), c.preprocess)
    clean[feats] = Xs
    nodes = partition.build_nodes(clean, c.partition, seed=42, feature_names=feats)

    history, model, splits = run_federated(nodes, None, "true", None, c, seed=42)
    assert history.val_acc[-1] > 0.8
