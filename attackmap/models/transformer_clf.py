"""Fine-tuned BERT / RoBERTa multi-label classifier.

Pretrained encoder + linear head, trained end-to-end with BCE.  CPU-friendly:
length-bucketed batches with dynamic padding, linear warm-up/decay, gradient clipping,
best-epoch checkpointing on validation MRR.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from .base import Mapper


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TransformerMapper(Mapper):
    name = "transformer"

    def __init__(self, labels, model_name="distilroberta-base", max_len=128, batch_size=32, epochs=3,
                 lr=5e-5, seed=42, display_name: str | None = None):
        super().__init__(labels)
        self.model_name, self.max_len, self.batch_size = model_name, max_len, batch_size
        self.epochs, self.lr, self.seed = epochs, lr, seed
        self.name = display_name or model_name.split("/")[-1]
        self.tokenizer = None
        self.model = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------ utils
    def _init(self, source: str):
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source,
            num_labels=len(self.labels),
            problem_type="multi_label_classification",
            id2label=dict(enumerate(self.labels)),
            label2id={l: i for i, l in enumerate(self.labels)},
            ignore_mismatched_sizes=True,  # checkpoints that ship a differently-sized head
        )

    def _batches(self, texts, y=None, shuffle=False):
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_len)["input_ids"]
        idx = sorted(range(len(texts)), key=lambda i: len(enc[i]))
        chunks = [idx[i : i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
        if shuffle:
            random.shuffle(chunks)
        pad = self.tokenizer.pad_token_id
        for ch in chunks:
            L = max(len(enc[i]) for i in ch)
            ids = torch.full((len(ch), L), pad, dtype=torch.long)
            mask = torch.zeros((len(ch), L), dtype=torch.long)
            for r, i in enumerate(ch):
                ids[r, : len(enc[i])] = torch.tensor(enc[i])
                mask[r, : len(enc[i])] = 1
            yb = torch.tensor(y[ch]) if y is not None else None
            yield ch, ids, mask, yb

    # ------------------------------------------------------------- training
    def fit(self, texts, y, val=None, log=print):
        _set_seed(self.seed)
        if self.model is None:
            self._init(self.model_name)
        # The randomly initialised head gets a larger LR than the pretrained encoder:
        # with 135 sparse BCE targets this markedly speeds up convergence on CPU budgets.
        head = [p for n, p in self.model.named_parameters() if n.startswith("classifier")]
        body = [p for n, p in self.model.named_parameters() if not n.startswith("classifier")]
        opt = torch.optim.AdamW(
            [{"params": body, "lr": self.lr}, {"params": head, "lr": self.lr * 20}], weight_decay=0.01
        )
        steps = math.ceil(len(texts) / self.batch_size) * self.epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        loss_fn = nn.BCEWithLogitsLoss()
        best = max((h.get("MRR", -1.0) for h in self.history), default=-1.0)  # resume-aware
        offset = len(self.history)
        # When resuming, the checkpoint we started from is the incumbent best.
        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()} if offset else None
        for ep in range(offset, offset + self.epochs):
            self.model.train()
            t0, tot, n = time.time(), 0.0, 0
            for step, (_, ids, mask, yb) in enumerate(self._batches(texts, y, shuffle=True)):
                logits = self.model(input_ids=ids, attention_mask=mask).logits
                loss = loss_fn(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                tot, n = tot + loss.item(), n + 1
                if step % 100 == 0:
                    log(f"[{self.name}] epoch {ep + 1} step {step} loss {tot / n:.4f} ({time.time() - t0:.0f}s)")
            rec = {"epoch": ep + 1, "train_loss": tot / n, "seconds": time.time() - t0}
            if val is not None:
                from ..evaluate import ranking_metrics

                rec.update(ranking_metrics(val[1], self.predict_proba(val[0])))
                if rec["MRR"] > best:
                    best = rec["MRR"]
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.history.append(rec)
            log(f"[{self.name}] {rec}")
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.inference_mode()
    def predict_proba(self, texts):
        self.model.eval()
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        for ch, ids, mask, _ in self._batches(list(texts)):
            out[ch] = torch.sigmoid(self.model(input_ids=ids, attention_mask=mask).logits).numpy()
        return out

    # ----------------------------------------------------------- persistence
    def save(self, path: Path):
        path = Path(path)
        self._write_meta(path, model_name=self.model_name, max_len=self.max_len, display_name=self.name,
                         history=self.history)
        self.model.save_pretrained(path / "hf")
        self.tokenizer.save_pretrained(path / "hf")

    @classmethod
    def load(cls, path: Path):
        meta = cls._read_meta(path)
        m = cls(meta["labels"], model_name=meta["model_name"], max_len=meta["max_len"],
                display_name=meta.get("display_name"))
        m.threshold, m.history = meta["threshold"], meta.get("history", [])
        m._init(str(Path(path) / "hf"))
        return m
