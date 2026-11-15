"""Metrics for ranked multi-label ATT&CK mapping.

An analyst sees a ranked list, so ranking metrics come first:

* **P@1**      top-ranked technique is correct
* **Hit@k**    at least one correct technique in the top-k
* **R@5**      fraction of the gold techniques recovered in the top-5
* **MRR**      mean reciprocal rank of the first correct technique
* **micro / macro F1**  of the predicted *set* (top-1 + every label above a threshold)
* **Tactic@1** the top-1 technique shares a tactic with the gold labels
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def multi_hot(label_lists: list[list[str]], labels: list[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    y = np.zeros((len(label_lists), len(labels)), dtype=np.float32)
    for r, ls in enumerate(label_lists):
        for l in ls:
            if l in idx:
                y[r, idx[l]] = 1.0
    return y


def predict_sets(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Always keep the arg-max, plus every label scoring >= threshold."""
    pred = (scores >= threshold).astype(np.float32)
    pred[np.arange(len(scores)), scores.argmax(1)] = 1.0
    return pred


def tune_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    grid = np.unique(np.quantile(scores, np.linspace(0.90, 0.9995, 60)))
    best = max(grid, key=lambda t: f1_score(y, predict_sets(scores, t), average="micro", zero_division=0))
    return float(best)


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, axis=1)
    hits = np.take_along_axis(y, order, axis=1)  # gold indicator in rank order
    first = np.where(hits.any(1), hits.argmax(1), scores.shape[1])
    n_gold = np.maximum(y.sum(1), 1)
    return {
        "P@1": float(hits[:, 0].mean()),
        "Hit@3": float(hits[:, :3].any(1).mean()),
        "Hit@5": float(hits[:, :5].any(1).mean()),
        "R@5": float((hits[:, :5].sum(1) / n_gold).mean()),
        "MRR": float(np.mean(1.0 / (first + 1))),
    }


def tactic_accuracy(
    y: np.ndarray, scores: np.ndarray, labels: list[str], tactics_of
) -> float:
    top = scores.argmax(1)
    ok = 0
    for i, j in enumerate(top):
        gold = {t for k in np.flatnonzero(y[i]) for t in tactics_of(labels[k])}
        ok += bool(gold & set(tactics_of(labels[j])))
    return ok / max(len(top), 1)


def evaluate(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    labels: list[str] | None = None,
    tactics_of=None,
) -> dict[str, float]:
    out = ranking_metrics(y, scores)
    pred = predict_sets(scores, threshold)
    out["micro_F1"] = float(f1_score(y, pred, average="micro", zero_division=0))
    out["macro_F1"] = float(f1_score(y, pred, average="macro", zero_division=0))
    if labels is not None and tactics_of is not None:
        out["Tactic@1"] = tactic_accuracy(y, scores, labels, tactics_of)
    return out


def per_label_f1(y: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    return f1_score(y, predict_sets(scores, threshold), average=None, zero_division=0)
