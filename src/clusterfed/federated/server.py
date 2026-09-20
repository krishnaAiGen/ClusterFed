"""FedAvg weighted aggregation (PRD 9.11).

W_avg = sum_k (n_k * w_k) / sum_k n_k -- the standard FedAvg weighting by each
client's sample count. ~30 lines, no FL framework (PRD §12).
"""
from __future__ import annotations

import torch


def fedavg(states: list[tuple[dict, int]]) -> dict:
    """Weighted-average a list of ``(state_dict, n_samples)`` into one state_dict."""
    if not states:
        raise ValueError("fedavg received no client states")
    total = sum(n for _, n in states)
    if total == 0:
        raise ValueError("fedavg total sample count is zero")

    keys = states[0][0].keys()
    avg: dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = None
        for state, n in states:
            contrib = state[key].to(torch.float64) * (n / total)
            stacked = contrib if stacked is None else stacked + contrib
        # Cast back to the original dtype of this parameter.
        avg[key] = stacked.to(states[0][0][key].dtype)
    return avg


def coordinate_median(states: list[tuple[dict, int]]) -> dict:
    """Byzantine-robust aggregation (E8): coordinate-wise median across clients
    (unweighted -- weighting by n_k would let a data-rich poisoned client
    dominate, defeating the purpose)."""
    if not states:
        raise ValueError("coordinate_median received no client states")
    keys = states[0][0].keys()
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = torch.stack([state[key].to(torch.float64) for state, _ in states], dim=0)
        out[key] = stacked.median(dim=0).values.to(states[0][0][key].dtype)
    return out


def compress_delta(delta: dict, method: str, topk_fraction: float = 0.1) -> dict:
    """Lossy compression of a client update (ICC Tier C).

    ``fp16`` halves the payload by casting to half precision. ``topk`` keeps the
    ``topk_fraction`` largest-magnitude coordinates *globally* across all tensors
    (as in deep gradient compression) and zeroes the rest; the zeros are what a
    real deployment would not transmit. Non-float buffers pass through untouched.
    """
    if method in ("none", "", None):
        return delta
    floats = {k: v for k, v in delta.items() if v.dtype.is_floating_point}
    if method == "fp16":
        out = dict(delta)
        out.update({k: v.to(torch.float16).to(v.dtype) for k, v in floats.items()})
        return out
    if method == "topk":
        flat = torch.cat([v.flatten().abs() for v in floats.values()])
        n = flat.numel()
        k = max(1, int(round(float(topk_fraction) * n)))
        if k >= n:
            return delta
        thresh = flat.kthvalue(n - k + 1).values
        out = dict(delta)
        out.update({k_: torch.where(v.abs() >= thresh, v, torch.zeros_like(v))
                    for k_, v in floats.items()})
        return out
    raise ValueError(f"unknown compression method {method!r}")


def uplink_bytes(n_params: int, method: str = "none", topk_fraction: float = 0.1) -> int:
    """Bytes a client uploads per round. Downlink is always a dense fp32
    broadcast (4*P), so it is accounted separately by the caller."""
    if method == "fp16":
        return 2 * n_params
    if method == "topk":
        # value (fp32) + coordinate index (int32) for each retained entry
        return int(round(float(topk_fraction) * n_params)) * 8
    return 4 * n_params


def aggregate(states: list[tuple[dict, int]], method: str = "fedavg") -> dict:
    if method == "coordinate_median":
        return coordinate_median(states)
    return fedavg(states)
