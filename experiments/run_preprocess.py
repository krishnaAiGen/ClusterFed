#!/usr/bin/env python3
"""Preprocess + partition (cached). Reads CICIDS2017 CSVs from ../ClusterFed v1/data
(read-only; see config.yaml paths.raw_dir) and writes v2's own cache.

    python experiments/run_preprocess.py --config config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clusterfed import pipeline
from clusterfed.config import init_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = init_run(args.config)
    bundle = pipeline.preprocess_and_cache(cfg)
    nodes = pipeline.build_scaled_nodes(bundle, cfg, seed=int(cfg["seed"]))

    df = bundle["df"]
    print(f"\nRows: {len(df):,}  Features: {len(bundle['features'])}  "
          f"Labels: {df[cfg.preprocess.get('label_column', 'Label')].nunique()}")
    print(f"Nodes: {len(nodes)}")
    for nid in sorted(nodes):
        nd = nodes[nid]
        print(f"  Node {nid:2d} [{nd.row_label:26s}] "
              f"benign={nd.n_benign:6d} attack={nd.n_attack:6d}")


if __name__ == "__main__":
    main()
