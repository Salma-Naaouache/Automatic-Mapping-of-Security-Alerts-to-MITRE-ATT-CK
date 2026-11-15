"""Build the unified, de-duplicated, split corpus.

Output: ``data/processed/corpus.jsonl`` with fields
``text, labels (parent ids), labels_full (as annotated), tactics, source, origin, split``
and ``data/processed/labels.json`` (the classifier label space).
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from ..config import MIN_EXAMPLES_PER_LABEL, PROCESSED_DIR, SEED, SPLIT
from ..text import clean, parent_technique
from .attack_kb import KnowledgeBase, load_stix, parse_procedures
from .sources import load_sigma, load_tram

CORPUS_PATH = PROCESSED_DIR / "corpus.jsonl"
LABELS_PATH = PROCESSED_DIR / "labels.json"


def _key(text: str) -> str:
    norm = "".join(ch for ch in text.lower() if ch.isalnum())
    return hashlib.md5(norm.encode()).hexdigest()


def normalise(rows: list[dict], kb: KnowledgeBase) -> list[dict]:
    """Clean text, roll labels up to parent techniques, drop unknown ids, merge duplicates."""
    merged: dict[str, dict] = {}
    for r in rows:
        text = clean(r["text"])
        if len(text.split()) < 4:
            continue
        full, parents = set(), set()
        for lab in r["labels"]:
            try:
                p = parent_technique(lab)
            except ValueError:
                continue
            if p in kb:  # drops revoked / deprecated ids
                parents.add(p)
                full.add(lab.upper())
        if not parents:
            continue
        k = _key(text)
        if k in merged:
            merged[k]["labels"] = sorted(set(merged[k]["labels"]) | parents)
            merged[k]["labels_full"] = sorted(set(merged[k]["labels_full"]) | full)
        else:
            merged[k] = {
                "text": text,
                "labels": sorted(parents),
                "labels_full": sorted(full),
                "source": r["source"],
                "origin": r.get("origin", ""),
            }
    for r in merged.values():
        r["tactics"] = sorted({t for lab in r["labels"] for t in kb.tactics_of(lab)})
    return list(merged.values())


def select_labels(rows: list[dict], min_count: int = MIN_EXAMPLES_PER_LABEL) -> list[str]:
    counts = Counter(l for r in rows for l in r["labels"])
    return sorted(l for l, c in counts.items() if c >= min_count)


def split(rows: list[dict], seed: int = SEED, fractions=SPLIT) -> list[dict]:
    """Iterative, label-aware split: rows are assigned rarest-label-first so every
    label gets examples in val and test (approximate multi-label stratification)."""
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    counts = Counter(l for r in rows for l in r["labels"])
    rows.sort(key=lambda r: min(counts[l] for l in r["labels"]))
    names = ["train", "val", "test"]
    target = {l: [c * f for f in fractions] for l, c in counts.items()}
    have = {l: [0, 0, 0] for l in counts}
    for r in rows:
        rare = min(r["labels"], key=lambda l: counts[l])
        deficit = [target[rare][i] - have[rare][i] for i in range(3)]
        i = max(range(3), key=lambda j: (deficit[j], -j))
        r["split"] = names[i]
        for l in r["labels"]:
            have[l][i] += 1
    return rows


def build(write: bool = True) -> tuple[list[dict], list[str]]:
    bundle = load_stix()
    kb = KnowledgeBase.from_bundle(bundle)
    kb.save()
    raw = parse_procedures(bundle) + load_tram() + load_sigma()
    rows = normalise(raw, kb)
    labels = select_labels(rows)
    keep = set(labels)
    for r in rows:
        r["labels"] = [l for l in r["labels"] if l in keep]
    rows = [r for r in rows if r["labels"]]
    rows = split(rows)
    if write:
        with open(CORPUS_PATH, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        LABELS_PATH.write_text(
            json.dumps([{"id": l, "name": kb.name_of(l), "tactics": kb.tactics_of(l)} for l in labels], indent=1)
        )
    return rows, labels


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def load_labels(path: Path = LABELS_PATH) -> list[str]:
    return [d["id"] for d in json.loads(path.read_text())]
