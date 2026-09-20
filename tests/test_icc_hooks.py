"""Tests for the ICC Tier B/C hooks: noise injection, compression, sampling."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clusterfed.federated.server import compress_delta, uplink_bytes  # noqa: E402
from clusterfed.labeling.noise import inject, measure_rho  # noqa: E402


# ---------------------------------------------------------------- compression
def _delta(n=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(n // 2, generator=g), "b": torch.randn(n // 2, generator=g)}


def test_compress_none_is_identity():
    d = _delta()
    out = compress_delta(d, "none")
    for k in d:
        assert torch.equal(out[k], d[k])


def test_fp16_is_lossy_but_close():
    d = _delta()
    out = compress_delta(d, "fp16")
    for k in d:
        assert out[k].dtype == d[k].dtype           # cast back
        assert not torch.equal(out[k], d[k])         # but information was lost
        assert torch.allclose(out[k], d[k], atol=1e-2)


def test_topk_keeps_exactly_the_largest_fraction():
    d = _delta(n=1000)
    out = compress_delta(d, "topk", 0.1)
    kept = sum(int((v != 0).sum()) for v in out.values())
    assert 95 <= kept <= 105          # ~10% of 1000, ties permitting
    # every surviving coordinate must be at least as large as every dropped one
    surv = torch.cat([v[v != 0].abs() for v in out.values()])
    dropped = torch.cat([d[k].flatten()[out[k].flatten() == 0].abs() for k in d])
    assert surv.min() >= dropped.max()


def test_topk_fraction_one_is_lossless():
    d = _delta()
    out = compress_delta(d, "topk", 1.0)
    for k in d:
        assert torch.equal(out[k], d[k])


def test_compress_rejects_unknown_method():
    with pytest.raises(ValueError):
        compress_delta(_delta(), "bogus")


def test_uplink_bytes_accounting():
    P = 37_002
    assert uplink_bytes(P, "none") == 4 * P
    assert uplink_bytes(P, "fp16") == 2 * P
    # value + index for each retained coordinate
    assert uplink_bytes(P, "topk", 0.1) == round(0.1 * P) * 8
    assert uplink_bytes(P, "topk", 0.1) < uplink_bytes(P, "none")


# ------------------------------------------------------------------ injection
def _nodes_and_results(n=400, rho0=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < 0.5).astype(int)
    y_pseudo = y_true.copy()
    wrong = rng.choice(n, size=int(rho0 * n), replace=False)
    y_pseudo[wrong] = 1 - y_pseudo[wrong]
    mask = np.ones(n, dtype=bool)
    nodes = {0: SimpleNamespace(y_binary=y_true, row_label="n0")}
    results = {0: SimpleNamespace(y_refined=y_pseudo, seed_mask=mask)}
    return nodes, results, y_true, y_pseudo


def test_measure_rho_matches_construction():
    _, _, y_true, y_pseudo = _nodes_and_results(rho0=0.05)
    assert measure_rho(y_pseudo, y_true, None) == pytest.approx(0.05, abs=1e-9)


def test_inject_zero_rate_is_a_noop():
    nodes, results, _, y_pseudo = _nodes_and_results()
    out, diag = inject(nodes, results, 0.0, seed=1)
    assert np.array_equal(out[0], y_pseudo)
    assert diag["rho_after"] == pytest.approx(diag["rho_before"])


def test_inject_raises_rho_predictably():
    """Flipping a uniformly chosen f of rows takes rho_0 -> rho_0 + f(1-2*rho_0),
    because a flip on an already-wrong row *fixes* it."""
    nodes, results, _, _ = _nodes_and_results(n=4000, rho0=0.05)
    for f in (0.05, 0.10, 0.20):
        _, diag = inject(nodes, results, f, seed=3)
        expected = 0.05 + f * (1 - 2 * 0.05)
        assert diag["rho_after"] == pytest.approx(expected, abs=0.02)
        assert diag["rho_after"] > diag["rho_before"]


def test_inject_does_not_mutate_the_source_labels():
    nodes, results, _, _ = _nodes_and_results()
    original = results[0].y_refined.copy()
    inject(nodes, results, 0.3, seed=5)
    assert np.array_equal(results[0].y_refined, original)


def test_inject_only_touches_seed_rows():
    nodes, results, _, _ = _nodes_and_results(n=1000)
    mask = np.zeros(1000, dtype=bool)
    mask[:200] = True                       # only the first 200 rows are seed rows
    results[0].seed_mask = mask
    before = results[0].y_refined.copy()
    out, _ = inject(nodes, results, 0.5, seed=7)
    assert np.array_equal(out[0][200:], before[200:])   # non-seed rows untouched
    assert not np.array_equal(out[0][:200], before[:200])


def test_asymmetric_modes_flip_only_one_class():
    nodes, results, _, _ = _nodes_and_results(n=2000, rho0=0.0, seed=11)
    before = results[0].y_refined.copy()
    for mode, src in (("attack2benign", 1), ("benign2attack", 0)):
        out, diag = inject(nodes, results, 0.2, seed=13, mode=mode)
        changed = np.flatnonzero(out[0] != before)
        assert len(changed) > 0
        # every changed row started in the source class and left it
        assert np.all(before[changed] == src)
        assert np.all(out[0][changed] == 1 - src)
        assert diag["inject_noise_mode"] == mode


def test_asymmetric_reaches_higher_rho_than_symmetric_at_equal_rate():
    """Symmetric flips partly land on already-wrong rows and fix them; one-sided
    flips on a clean label set do not, so rho rises faster."""
    nodes, results, _, _ = _nodes_and_results(n=4000, rho0=0.10, seed=17)
    _, sym = inject(nodes, results, 0.2, seed=19, mode="symmetric")
    _, asym = inject(nodes, results, 0.2, seed=19, mode="attack2benign")
    assert asym["rho_after"] > sym["rho_after"]


def test_unknown_noise_mode_rejected():
    nodes, results, _, _ = _nodes_and_results()
    with pytest.raises(ValueError):
        inject(nodes, results, 0.1, seed=1, mode="bogus")
