"""Sentence-Transformer semantic mapper.

Two modes share one class:

* **zero-shot** (``finetune=False``): embed each technique's name + description from the
  ATT&CK KB and rank by cosine similarity.  Needs *no labelled data* and covers every
  technique and sub-technique in ATT&CK, including those absent from training.
* **fine-tuned** (``finetune=True``): contrastive training (MultipleNegativesRankingLoss)
  pulls each alert towards its technique description and away from in-batch negatives.
  Prediction then blends cosine-to-prototype (description + class centroid) with a
  similarity-weighted k-NN vote over training examples.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.attack_kb import KnowledgeBase
from .base import Mapper


def _softmax(x: np.ndarray, t: float) -> np.ndarray:
    z = (x - x.max(1, keepdims=True)) / t
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class SBertMapper(Mapper):
    name = "sbert"

    def __init__(self, labels, model_name="sentence-transformers/all-MiniLM-L6-v2", finetune=True,
                 epochs: int = 1, batch_size: int = 64, temperature: float = 0.05, k: int = 15,
                 display_name: str | None = None, kb: KnowledgeBase | None = None):
        super().__init__(labels)
        self.model_name, self.finetune, self.epochs = model_name, finetune, epochs
        self.batch_size, self.temperature, self.k = batch_size, temperature, k
        self.name = display_name or ("sbert" if finetune else "sbert_zeroshot")
        self.kb = kb or KnowledgeBase.load()
        self.encoder = None
        self.label_emb: np.ndarray | None = None
        self.train_emb: np.ndarray | None = None
        self.train_y: np.ndarray | None = None

    # ------------------------------------------------------------------ utils
    def _load_encoder(self, source: str):
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(source, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(list(texts), batch_size=128, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def label_doc(self, tid: str) -> str:
        t = self.kb.get(tid)
        return t.as_document() if t else tid

    # ------------------------------------------------------------- training
    def fit(self, texts, y, val=None, log=print):
        self._load_encoder(self.model_name)
        if self.finetune:
            self._contrastive(texts, y, log)
        self.label_emb = self.encode([self.label_doc(l) for l in self.labels])
        if self.finetune:
            self.train_emb = self.encode(texts)
            self.train_y = y.astype(np.float32)
            # prototype = technique description + centroid of its training examples
            cent = (self.train_y.T @ self.train_emb) / np.maximum(self.train_y.sum(0)[:, None], 1)
            proto = self.label_emb + cent
            self.label_emb = proto / np.linalg.norm(proto, axis=1, keepdims=True)
        return self

    def _contrastive(self, texts, y, log):
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers

        anchors, positives = [], []
        for t, row in zip(texts, y):
            for j in np.flatnonzero(row):
                anchors.append(t)
                positives.append(self.label_doc(self.labels[j]))
        ds = Dataset.from_dict({"anchor": anchors, "positive": positives}).shuffle(seed=42)
        args = SentenceTransformerTrainingArguments(
            output_dir="/tmp/attackmap-sbert",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=5e-5,
            warmup_steps=0.1,
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # avoid same technique twice in a batch
            logging_steps=50,
            save_strategy="no",
            report_to=[],
            seed=42,
        )
        log(f"[{self.name}] contrastive fine-tuning on {len(ds)} pairs")
        SentenceTransformerTrainer(model=self.encoder, args=args, train_dataset=ds,
                                   loss=MultipleNegativesRankingLoss(self.encoder)).train()

    # ------------------------------------------------------------- inference
    def predict_proba(self, texts):
        e = self.encode(texts)
        p = _softmax(e @ self.label_emb.T, self.temperature)
        if self.train_emb is not None:
            sims = e @ self.train_emb.T
            top = np.argpartition(-sims, self.k, axis=1)[:, : self.k]
            w = np.exp(np.take_along_axis(sims, top, 1) / self.temperature)
            w /= w.sum(1, keepdims=True)
            knn = np.einsum("nk,nkl->nl", w, self.train_y[top])
            p = 0.5 * p + 0.5 * knn
        return np.clip(p, 0.0, 1.0).astype(np.float32)

    def retrieve(self, text: str, candidates: list[str], top: int = 5) -> list[tuple[str, float]]:
        """Rank arbitrary ATT&CK ids (e.g. sub-techniques, long-tail techniques) for ``text``."""
        cache = self.__dict__.setdefault("_doc_cache", {})
        missing = [c for c in candidates if c not in cache]
        if missing:
            cache.update(zip(missing, self.encode([self.label_doc(c) for c in missing])))
        sims = self.encode([text])[0] @ np.stack([cache[c] for c in candidates]).T
        order = np.argsort(-sims)[:top]
        return [(candidates[i], float(sims[i])) for i in order]

    # ----------------------------------------------------------- persistence
    def save(self, path: Path):
        path = Path(path)
        self._write_meta(path, model_name=self.model_name, finetune=self.finetune,
                         temperature=self.temperature, k=self.k, display_name=self.name)
        self.encoder.save(str(path / "encoder"))
        np.save(path / "label_emb.npy", self.label_emb)
        if self.train_emb is not None:
            np.save(path / "train_emb.npy", self.train_emb.astype(np.float16))
            np.save(path / "train_y.npy", self.train_y.astype(np.uint8))

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        meta = cls._read_meta(path)
        m = cls(meta["labels"], meta["model_name"], finetune=meta["finetune"], temperature=meta["temperature"],
                k=meta["k"], display_name=meta.get("display_name"))
        m.threshold = meta["threshold"]
        m._load_encoder(str(path / "encoder"))
        m.label_emb = np.load(path / "label_emb.npy")
        if (path / "train_emb.npy").exists():
            m.train_emb = np.load(path / "train_emb.npy").astype(np.float32)
            m.train_y = np.load(path / "train_y.npy").astype(np.float32)
        return m
