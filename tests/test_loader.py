"""Tests for data/loader.py: label normalization, column stripping, encodings."""
import os

import pandas as pd

from clusterfed.data import loader


def test_normalize_labels_handles_mojibake(raw_frame):
    df = loader.normalize_labels(raw_frame)
    labels = set(df["Label"].unique())
    # The cp1252 en-dash web-attack labels must become canonical forms.
    assert "Web Attack - Brute Force" in labels
    assert "Web Attack - XSS" in labels
    assert "Web Attack - Sql Injection" in labels
    # No raw mojibake byte survives.
    assert not any("\x96" in lab for lab in labels)


def test_all_labels_in_canonical_set(raw_frame):
    df = loader.normalize_labels(raw_frame)
    assert set(df["Label"].unique()).issubset(set(loader.CANONICAL_LABELS))


def test_load_raw_strips_columns_and_falls_back_encoding(tmp_path, raw_frame):
    # Write a latin-1 encoded CSV with whitespace-padded columns + mojibake labels.
    path = tmp_path / "Wednesday-workingHours.pcap_ISCX.csv"
    raw_frame.to_csv(path, index=False, encoding="latin-1")
    merged = loader.load_raw(str(tmp_path))
    assert "source_file" in merged.columns
    # Column names are stripped of surrounding whitespace.
    assert all(col == col.strip() for col in merged.columns)
    assert merged["Label"].nunique() >= 10


def test_load_raw_errors_when_empty(tmp_path):
    try:
        loader.load_raw(str(tmp_path))
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError on empty raw dir")
