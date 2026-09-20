#!/usr/bin/env python3
"""E7: communication + compute cost table (PRD Section 8; answers R1.7).

Measures on the final configuration:
  - model size (parameters, bytes)
  - communication per round (down: broadcast, up: client update) and total
  - per-client local-training wall-clock per round (measured on the largest
    and smallest node)
  - global-model inference latency per flow (batch and single, measured)
Writes outputs/results/E7/cost.json + a Markdown table for the manuscript.

    python experiments/run_e7_cost.py [--config config.yaml] [overrides...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from clusterfed import pipeline
from clusterfed.config import Config, configure_logging, load_config, set_global_seed
from clusterfed.federated.client import local_train
from clusterfed.federated.model import build_model
from run_experiment import apply_overrides, parse_override


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()

    configure_logging()
    cfg = apply_overrides(load_config(args.config), [parse_override(kv) for kv in args.override])
    seed = int(cfg["seed"])
    set_global_seed(seed)

    bundle = pipeline.load_cache(cfg)
    sel = pipeline.select_features(bundle, cfg, seed=seed)
    nodes = pipeline.build_scaled_nodes(bundle, cfg, seed=seed)
    input_dim = len(sel["selected_idx"])
    model = build_model(input_dim, cfg.model)

    n_params = sum(p.numel() for p in model.parameters())
    bytes_fp32 = n_params * 4
    rounds = int(cfg.federated.get("communication_rounds", 200))
    n_clients = len(nodes)
    # Per round: server broadcasts weights to each client (down) + each client
    # uploads its update (up) -> 2 * model_size * n_clients.
    bytes_per_round = 2 * bytes_fp32 * n_clients
    total_traffic = bytes_per_round * rounds

    # Measured local-training time per round, largest vs smallest node.
    sizes = {nid: nodes[nid].n_samples for nid in nodes}
    big, small = max(sizes, key=sizes.get), min(sizes, key=sizes.get)
    timings = {}
    for tag, nid in (("largest_node", big), ("smallest_node", small)):
        X = nodes[nid].X[:, sel["selected_idx"]]
        y = nodes[nid].y_binary
        t0 = time.time()
        local_train(model, X, y, cfg.federated, seed=seed)
        timings[tag] = {"node": nodes[nid].row_label, "rows": int(len(X)),
                        "local_train_s_per_round": round(time.time() - t0, 3)}

    # Inference latency (measured, CPU).
    Xb = torch.randn(4096, input_dim)
    model.eval()
    with torch.no_grad():
        model(Xb[:16])  # warmup
        t0 = time.time(); model(Xb); batch_s = time.time() - t0
        x1 = Xb[:1]
        t0 = time.time()
        for _ in range(200):
            model(x1)
        single_ms = (time.time() - t0) / 200 * 1000

    cost = {
        "model": {"parameters": int(n_params), "size_kb_fp32": round(bytes_fp32 / 1024, 1),
                   "architecture": f"{input_dim}-200-100-100-2"},
        "communication": {
            "clients": n_clients, "rounds": rounds,
            "per_round_total_mb": round(bytes_per_round / 1e6, 3),
            "per_client_per_round_kb": round(2 * bytes_fp32 / 1024, 1),
            "total_traffic_mb": round(total_traffic / 1e6, 1),
        },
        "compute": timings,
        "inference": {"batch4096_ms": round(batch_s * 1000, 2),
                       "per_flow_batch_us": round(batch_s / 4096 * 1e6, 2),
                       "single_flow_ms": round(single_ms, 3)},
        "complexity_O": {
            "stage0_feature_selection": "O(n k d) kNN graph + O(n d) scoring (subsampled n)",
            "stage2_bootstrap": "O(n K d I) k-means + O(n K d) orientation/confidence",
            "stage3_selection": "O(n log n) per-class quantiles",
            "stage4_refinement_per_iter": "O(E n_seed d_mlp) train + O(T n_held d_mlp) MC-dropout",
            "federated_per_round": "O(E n_k d_mlp) per client + O(P) aggregation/EMA",
        },
    }

    out_dir = os.path.join(cfg.paths["outputs_dir"], "results", "E7")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cost.json"), "w") as fh:
        json.dump(cost, fh, indent=2)

    md = [
        "| Quantity | Value |", "|---|---|",
        f"| Model parameters | {n_params:,} |",
        f"| Model size (fp32) | {cost['model']['size_kb_fp32']} KB |",
        f"| Comm. per client per round | {cost['communication']['per_client_per_round_kb']} KB |",
        f"| Comm. per round (all {n_clients} clients) | {cost['communication']['per_round_total_mb']} MB |",
        f"| Total traffic ({rounds} rounds) | {cost['communication']['total_traffic_mb']} MB |",
        f"| Local train/round, largest node ({timings['largest_node']['rows']} rows) | {timings['largest_node']['local_train_s_per_round']} s |",
        f"| Local train/round, smallest node ({timings['smallest_node']['rows']} rows) | {timings['smallest_node']['local_train_s_per_round']} s |",
        f"| Inference per flow (batched) | {cost['inference']['per_flow_batch_us']} µs |",
        f"| Inference single flow | {cost['inference']['single_flow_ms']} ms |",
    ]
    with open(os.path.join(out_dir, "cost_table.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(json.dumps(cost, indent=2))


if __name__ == "__main__":
    main()
