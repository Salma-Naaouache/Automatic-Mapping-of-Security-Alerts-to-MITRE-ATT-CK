import numpy as np
import pytest

from attackmap.evaluate import evaluate, multi_hot, predict_sets, ranking_metrics, tactic_accuracy, tune_threshold

L = ["A", "B", "C", "D"]
Y = multi_hot([["A"], ["B", "C"], ["D"]], L)
S = np.array([
    [0.9, 0.05, 0.03, 0.02],  # gold A ranked 1
    [0.1, 0.2, 0.6, 0.1],     # gold B,C: C ranked 1, B ranked 2
    [0.4, 0.3, 0.2, 0.1],     # gold D ranked 4
], dtype=np.float32)


def test_multi_hot_ignores_unknown():
    y = multi_hot([["A", "Z"]], L)
    assert y.tolist() == [[1, 0, 0, 0]]


def test_ranking_metrics_hand_computed():
    m = ranking_metrics(Y, S)
    assert m["P@1"] == pytest.approx(2 / 3)
    assert m["Hit@3"] == pytest.approx(2 / 3)
    assert m["Hit@5"] == pytest.approx(1.0)
    assert m["MRR"] == pytest.approx((1 + 1 + 0.25) / 3)
    assert m["R@5"] == pytest.approx(1.0)


def test_predict_sets_always_keeps_argmax():
    p = predict_sets(S, threshold=0.95)
    assert p.sum(1).tolist() == [1, 1, 1]
    p = predict_sets(S, threshold=0.2)
    assert p[1].tolist() == [0, 1, 1, 0]


def test_threshold_and_f1():
    t = tune_threshold(Y, S)
    m = evaluate(Y, S, t)
    assert 0 <= m["micro_F1"] <= 1 and 0 <= m["macro_F1"] <= 1


def test_tactic_accuracy():
    tac = {"A": ["x"], "B": ["y"], "C": ["y"], "D": ["z"]}.get
    assert tactic_accuracy(Y, S, L, tac) == pytest.approx(2 / 3)
