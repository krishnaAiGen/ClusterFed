"""Controlled pseudo-label noise injection (ICC Tier B / experiment E11).

The convergence bound's asymptotic floor is 2*delta^2 with delta <= rho*G, where
rho is the mislabeling rate of the labels a client actually trains on. Every
other arm in this project varies rho only *indirectly* (by changing the labeling
pipeline), which confounds rho with everything else the pipeline changed. This
module varies rho directly: it flips a controlled fraction of the curated seed
set's labels and leaves the rest of the pipeline untouched, so the resulting
attainment cost can be attributed to rho alone.

Flips are applied ONLY to the labels handed to federated training. The Stage-0-3
labeling metrics (Table I) are computed from the unperturbed result, so the
labeling-quality report continues to describe the real pipeline.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("clusterfed.noise")


def measure_rho(y_pseudo: np.ndarray, y_true: np.ndarray, mask: np.ndarray | None) -> float:
    """Mislabeling rate of ``y_pseudo`` against ground truth, over ``mask`` rows."""
    y_pseudo = np.asarray(y_pseudo)
    y_true = np.asarray(y_true)
    if mask is None:
        sel = np.ones(len(y_pseudo), dtype=bool)
    else:
        sel = np.asarray(mask, dtype=bool)
    if sel.sum() == 0:
        return float("nan")
    return float((y_pseudo[sel] != y_true[sel]).mean())


def _choose(sel_idx, y, mode: str, n_flip: int, rng):
    """Pick which seed rows to flip.

    ``symmetric``  -- uniformly at random, so the per-class flip rates are equal.
                      Classical symmetric noise, which provably leaves the
                      Bayes-optimal classifier unchanged.
    ``attack2benign`` / ``benign2attack`` -- flips drawn only from one pseudo-class,
                      i.e. class-conditional (asymmetric) noise, which *does*
                      move the optimum. This is the structure a failed cluster
                      orientation produces: a whole attack family read as benign.
    """
    if mode == "symmetric":
        pool = sel_idx
    elif mode == "attack2benign":
        pool = sel_idx[y[sel_idx] == 1]
    elif mode == "benign2attack":
        pool = sel_idx[y[sel_idx] == 0]
    else:
        raise ValueError(f"unknown noise mode {mode!r}")
    if len(pool) == 0:
        return np.array([], dtype=int)
    return rng.choice(pool, size=min(n_flip, len(pool)), replace=False)


def inject(nodes, refinement_results, rate: float, *, seed: int, mode: str = "symmetric"):
    """Flip ``rate`` of each node's seed-set labels. Returns
    ``(pseudo_labels, diagnostics)`` where diagnostics records the realized
    per-node and weighted mean rho before and after injection.

    Flipping is applied to the seed set only, because that is the population the
    clients train on (``federated.train_on == "seed"``). Rows are chosen
    uniformly at random within the seed set, independent of correctness, so the
    realized rho rises to approximately rho_0 + rate*(1 - 2*rho_0) rather than
    exactly rho_0 + rate -- we therefore measure it rather than assume it.
    """
    pseudo_labels = {}
    before, after, weights = [], [], []

    for node_id in sorted(nodes):
        res = refinement_results[node_id]
        y = np.array(res.y_refined, copy=True)
        y_true = np.asarray(nodes[node_id].y_binary)
        mask = res.seed_mask
        sel_idx = np.flatnonzero(np.asarray(mask, dtype=bool)) if mask is not None \
            else np.arange(len(y))

        rho0 = measure_rho(y, y_true, mask)
        if rate > 0 and len(sel_idx) > 0:
            rng = np.random.default_rng(seed * 7919 + node_id)
            n_flip = int(round(rate * len(sel_idx)))
            if n_flip > 0:
                flip = _choose(sel_idx, y, mode, n_flip, rng)
                if len(flip):
                    y[flip] = 1 - y[flip]
        rho1 = measure_rho(y, y_true, mask)

        pseudo_labels[node_id] = y
        before.append(rho0)
        after.append(rho1)
        weights.append(len(sel_idx))
        logger.info("[E11] node %d rho %.4f -> %.4f (seed rows %d)",
                    node_id, rho0, rho1, len(sel_idx))

    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else w
    diagnostics = {
        "inject_noise_rate": float(rate),
        "inject_noise_mode": mode,
        "rho_before": round(float(np.dot(w, before)), 5),
        "rho_after": round(float(np.dot(w, after)), 5),
        "rho_per_node_after": [round(float(x), 5) for x in after],
    }
    logger.info("[E11] weighted rho %.4f -> %.4f at injection rate %.3f",
                diagnostics["rho_before"], diagnostics["rho_after"], rate)
    return pseudo_labels, diagnostics
