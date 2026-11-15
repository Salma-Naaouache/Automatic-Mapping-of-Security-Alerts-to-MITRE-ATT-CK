"""Weighted score-level ensemble with weights searched on the validation set."""
from __future__ import annotations

import itertools

import numpy as np

from ..evaluate import ranking_metrics


def normalise(scores: np.ndarray) -> np.ndarray:
    """Row-normalise heterogeneous model outputs (sigmoid / OvR / softmax) to distributions."""
    s = np.clip(scores, 1e-9, None)
    return s / s.sum(1, keepdims=True)


def combine(score_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    return sum(w * normalise(s) for w, s in zip(weights, score_list))


def search_weights(y: np.ndarray, score_list: list[np.ndarray], step: float = 0.1,
                   metric: str = "MRR") -> tuple[list[float], float]:
    """Exhaustive simplex grid search (fine for <= 6 models at step 0.1)."""
    n = len(score_list)
    norm = [normalise(s) for s in score_list]
    ticks = int(round(1 / step))
    best_w, best = None, -1.0
    for combo in itertools.product(range(ticks + 1), repeat=n - 1):
        if sum(combo) > ticks:
            continue
        w = [c / ticks for c in combo] + [(ticks - sum(combo)) / ticks]
        m = ranking_metrics(y, sum(wi * s for wi, s in zip(w, norm)))[metric]
        if m > best:
            best, best_w = m, w
    return best_w, best
