from collections import Counter

import pytest

from attackmap.config import PROCESSED_DIR
from attackmap.data.build import load_corpus, load_labels, normalise, select_labels, split


def test_normalise_merges_rolls_up_and_filters(kb):
    rows = [
        {"text": "APT used PowerShell to execute the payload", "labels": ["T1059.001"], "source": "a"},
        {"text": "APT used  PowerShell to execute the payload!", "labels": ["T1027"], "source": "b"},  # dup
        {"text": "too short", "labels": ["T1059"], "source": "a"},
        {"text": "a long enough sentence with a bogus label", "labels": ["T9999", "nonsense"], "source": "a"},
    ]
    out = normalise(rows, kb)
    assert len(out) == 1
    assert out[0]["labels"] == ["T1027", "T1059"]
    assert out[0]["labels_full"] == ["T1027", "T1059.001"]
    assert "execution" in out[0]["tactics"]


def test_select_labels():
    rows = [{"labels": ["A"]}] * 5 + [{"labels": ["B"]}] * 2
    assert select_labels(rows, min_count=3) == ["A"]


def test_split_covers_every_label_in_every_split():
    rows = [{"text": f"{l}{i}", "labels": [l]} for l in "ABCDE" for i in range(40)]
    out = split(rows, seed=1)
    c = Counter(r["split"] for r in out)
    assert abs(c["train"] / len(out) - 0.8) < 0.03
    for l in "ABCDE":
        assert {r["split"] for r in out if r["labels"] == [l]} == {"train", "val", "test"}
    assert split(rows, seed=1) == out  # deterministic


@pytest.mark.skipif(not (PROCESSED_DIR / "corpus.jsonl").exists(), reason="corpus not built")
def test_real_corpus_integrity():
    rows, labels = load_corpus(), set(load_labels())
    assert len(rows) > 20_000 and len(labels) >= 100
    assert {r["source"] for r in rows} == {"attack_procedure", "tram_cti", "sigma_siem"}
    seen = {}
    for r in rows:
        assert r["labels"] and set(r["labels"]) <= labels
        assert seen.setdefault(r["text"], r["split"]) == r["split"], "same text in two splits (leakage)"
    test_labels = Counter(l for r in rows if r["split"] == "test" for l in r["labels"])
    assert set(test_labels) == labels  # every label is evaluated
