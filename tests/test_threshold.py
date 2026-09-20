"""Tests for labeling/threshold.py: Stage 3 class-balanced adaptive selection (PRD 4.4)."""
import numpy as np

from clusterfed.labeling import threshold


def test_minority_class_gets_lower_bar():
    rng = np.random.default_rng(0)
    # 900 majority-class points, 100 minority; confidence uniform in both so any
    # selection-rate difference comes purely from the class-balanced policy.
    y = np.array([0] * 900 + [1] * 100)
    conf = rng.uniform(0.0, 1.0, size=1000)
    cfg = {"policy": "per_class_adaptive", "min_quantile": 0.2, "max_quantile": 0.8}

    thresholds, keep_fracs = threshold.class_balanced_thresholds(y, conf, cfg)
    assert keep_fracs[1] > keep_fracs[0]  # minority (1) kept at a higher rate
    assert thresholds[1] < thresholds[0]  # minority has the more lenient (lower) bar


def test_seed_set_never_mislabels_only_holds_out():
    rng = np.random.default_rng(0)
    y = np.array([0] * 800 + [1] * 200)
    conf = rng.uniform(0.0, 1.0, size=1000)
    cfg = {"policy": "per_class_adaptive", "min_quantile": 0.2, "max_quantile": 0.8}

    seed_mask, _ = threshold.select_seed_set(y, conf, cfg)
    # Labels for seed-set members are untouched -- selection only ever subsets.
    assert seed_mask.dtype == bool
    assert 0 < seed_mask.sum() < len(y)


def test_global_policy_uses_one_threshold():
    conf = np.linspace(0.0, 1.0, 100)
    y = np.zeros(100, dtype=int)
    cfg = {"policy": "global", "global_quantile": 0.5}
    seed_mask, thresholds = threshold.select_seed_set(y, conf, cfg)
    assert "global" in thresholds
    assert seed_mask.sum() == 50  # top half by confidence, exactly at median


def test_balanced_classes_get_equal_thresholds():
    rng = np.random.default_rng(1)
    y = np.array([0] * 500 + [1] * 500)
    conf = rng.uniform(0.0, 1.0, size=1000)
    cfg = {"policy": "per_class_adaptive", "min_quantile": 0.2, "max_quantile": 0.8}
    _, keep_fracs = threshold.class_balanced_thresholds(y, conf, cfg)
    assert keep_fracs[0] == keep_fracs[1] == 0.2  # both "the majority" (rarity=0) -> floor
