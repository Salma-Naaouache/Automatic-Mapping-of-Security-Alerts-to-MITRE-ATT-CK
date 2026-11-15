"""Common interface implemented by every mapper."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Mapper(ABC):
    """A model that scores every ATT&CK label for each input text.

    ``predict_proba`` returns an ``(n_texts, n_labels)`` array of independent
    per-label probabilities in [0, 1] (multi-label: they need not sum to 1).
    """

    name: str = "mapper"

    def __init__(self, labels: list[str]):
        self.labels = list(labels)
        self.threshold = 0.5

    @abstractmethod
    def fit(self, texts: list[str], y: np.ndarray, val: tuple[list[str], np.ndarray] | None = None):
        ...

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Mapper":
        ...

    # helpers ---------------------------------------------------------------
    def _write_meta(self, path: Path, **extra) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(
            json.dumps({"name": self.name, "labels": self.labels, "threshold": self.threshold, **extra}, indent=1)
        )

    @staticmethod
    def _read_meta(path: Path) -> dict:
        return json.loads((Path(path) / "meta.json").read_text())
