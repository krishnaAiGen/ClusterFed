"""Tests for data/preprocess.py: cleaning quirks and MinMax scaling."""
import numpy as np

from clusterfed.data import loader, preprocess


def test_clean_removes_inf_nan_dups_zerovar(raw_frame, cfg):
    df = loader.normalize_labels(raw_frame)
    clean = preprocess.clean(df, cfg.preprocess)
    feats = loader.feature_columns(clean, "Label")
    arr = clean[feats].to_numpy("float64")
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()
    # Zero-variance column dropped.
    assert " Constant Col " not in clean.columns
    # No duplicate rows remain.
    assert not clean.duplicated().any()


def test_scale_in_unit_interval(raw_frame, cfg):
    df = loader.normalize_labels(raw_frame)
    clean = preprocess.clean(df, cfg.preprocess)
    feats = loader.feature_columns(clean, "Label")
    X_scaled, scaler = preprocess.scale(clean[feats].to_numpy("float64"), cfg.preprocess)
    assert X_scaled.min() >= 0.0 - 1e-9
    assert X_scaled.max() <= 1.0 + 1e-9
    # Reusing the fitted scaler clips to [0,1].
    X2, _ = preprocess.scale(clean[feats].to_numpy("float64"), cfg.preprocess, scaler=scaler)
    assert X2.min() >= 0.0 and X2.max() <= 1.0


def test_binary_labels():
    names = np.array(["BENIGN", "Bot", "BENIGN", "DDoS"])
    y = preprocess.binary_labels(names)
    assert list(y) == [0, 1, 0, 1]
