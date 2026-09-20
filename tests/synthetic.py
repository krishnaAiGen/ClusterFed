"""Synthetic CICIDS-2017-shaped data so the suite runs without the real dataset.

Generates a frame with the right *shape* (78 numeric features + Label), the
documented quirks (whitespace column names, inf/NaN, mojibake web-attack labels,
duplicates, a zero-variance column), and class-separable structure so clustering
and the FL model produce non-trivial metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_FEATURES = 78

# A representative subset of the canonical attacks plus the mojibake web labels.
_ATTACKS = [
    "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest",
    "DoS slowloris", "FTP-Patator", "SSH-Patator", "Heartbleed",
    "Infiltration", "PortScan",
    "Web Attack \x96 Brute Force",   # cp1252 en-dash mojibake
    "Web Attack \x96 XSS",
    "Web Attack \x96 Sql Injection",
]


def make_frame(n_benign: int = 4000, n_per_attack: int = 300, *, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic merged frame with CICIDS quirks baked in."""
    rng = np.random.default_rng(seed)

    def block(center: float, n: int) -> np.ndarray:
        return rng.normal(loc=center, scale=1.0, size=(n, N_FEATURES))

    blocks = [block(0.0, n_benign)]
    labels = ["BENIGN"] * n_benign
    for i, name in enumerate(_ATTACKS):
        # Each attack class sits at a distinct, separable center.
        blocks.append(block(3.0 + 0.6 * i, n_per_attack))
        labels.extend([name] * n_per_attack)

    X = np.vstack(blocks)
    cols = [f" Feature {i} " for i in range(N_FEATURES)]  # leading/trailing whitespace
    df = pd.DataFrame(X, columns=cols)
    df["Label"] = labels

    # Quirk: Flow Bytes/s & Flow Packets/s with inf/NaN. Reuse two feature cols.
    df = df.rename(columns={cols[0]: "Flow Bytes/s", cols[1]: " Flow Packets/s "})
    inf_idx = rng.choice(len(df), size=max(1, len(df) // 200), replace=False)
    df.loc[inf_idx[: len(inf_idx) // 2], "Flow Bytes/s"] = np.inf
    df.loc[inf_idx[len(inf_idx) // 2:], " Flow Packets/s "] = np.nan

    # Quirk: a zero-variance column.
    df[" Constant Col "] = 0.0

    # Quirk: duplicate rows.
    df = pd.concat([df, df.iloc[:50]], ignore_index=True)

    # Shuffle.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df
