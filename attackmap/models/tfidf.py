"""Strong classical baseline: word + character TF-IDF -> one-vs-rest logistic regression.

Also serves as the *explainer*: its linear weights give faithful per-term evidence.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from .base import Mapper


class TfidfMapper(Mapper):
    name = "tfidf"

    def __init__(self, labels: list[str], C: float = 8.0):
        super().__init__(labels)
        self.C = C
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=150_000,
                                    token_pattern=r"(?u)<\w+>|\b[\w\-\.\\]{2,}\b", lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                    max_features=150_000)
        self.clf = OneVsRestClassifier(
            LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=2000), n_jobs=-1
        )

    def _x(self, texts, fit=False):
        if fit:
            x = hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        else:
            x = hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        x.sort_indices()  # joblib memmaps workers' copies read-only; pre-sort avoids in-place sort
        return x

    def fit(self, texts, y, val=None):
        self.clf.fit(self._x(texts, fit=True), y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._x(texts)).astype(np.float32)

    def explain(self, text: str, label_idx: int, k: int = 8) -> list[tuple[str, float]]:
        """Top word n-grams pushing ``text`` towards label ``label_idx``."""
        xw = self.word.transform([text])
        coef = self.clf.estimators_[label_idx].coef_.ravel()[: xw.shape[1]]
        contrib = xw.multiply(coef).tocsr()
        vocab = self.word.get_feature_names_out()
        pairs = [(vocab[j], float(v)) for j, v in zip(contrib.indices, contrib.data) if v > 0]
        return sorted(pairs, key=lambda p: -p[1])[:k]

    def save(self, path: Path):
        self._write_meta(Path(path), C=self.C)
        joblib.dump({"word": self.word, "char": self.char, "clf": self.clf}, Path(path) / "model.joblib",
                    compress=3)

    @classmethod
    def load(cls, path: Path):
        meta = cls._read_meta(path)
        m = cls(meta["labels"], C=meta.get("C", 8.0))
        m.threshold = meta["threshold"]
        parts = joblib.load(Path(path) / "model.joblib")
        m.word, m.char, m.clf = parts["word"], parts["char"], parts["clf"]
        return m
