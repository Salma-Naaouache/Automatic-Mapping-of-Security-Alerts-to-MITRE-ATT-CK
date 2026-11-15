"""BiLSTM with additive attention pooling, trained from scratch.

    tokens -> embedding(200) -> 2-layer BiLSTM(2x192) -> [attention-pool ; max-pool] -> dropout -> linear

The attention distribution over tokens is exposed via ``attention()`` so the UI can
highlight which words drove a prediction.
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import Mapper

_TOKEN = re.compile(r"<[a-z0-9]+>|[a-z0-9_]+(?:[.\-][a-z0-9_]+)*|[\\/:$%@=]")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; IOC placeholders (``<IP>``, ``<SHA256>``...) stay atomic."""
    return [t.upper() if t.startswith("<") else t for t in _TOKEN.findall(text.lower())]


class _Net(nn.Module):
    def __init__(self, vocab: int, n_labels: int, emb: int = 200, hidden: int = 192, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.emb_drop = nn.Dropout(0.2)
        self.lstm = nn.LSTM(emb, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.att = nn.Sequential(nn.Linear(2 * hidden, 128), nn.Tanh(), nn.Linear(128, 1, bias=False))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(4 * hidden, n_labels)

    def forward(self, ids, return_att=False):
        mask = ids != 0
        lengths = mask.sum(1).clamp(min=1).cpu()
        # Packing makes the backward LSTM start at each sequence's real end, so outputs
        # never depend on how much padding the batch happens to contain.
        packed = nn.utils.rnn.pack_padded_sequence(self.emb_drop(self.emb(ids)), lengths, batch_first=True,
                                                   enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=ids.shape[1])
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e9).softmax(-1)
        att_pool = (a.unsqueeze(-1) * h).sum(1)
        max_pool = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        logits = self.out(self.drop(torch.cat([att_pool, max_pool], -1)))
        return (logits, a) if return_att else logits


class BiLSTMMapper(Mapper):
    """``predict_proba`` returns a softmax distribution over labels (trained with soft-target CE)."""

    name = "bilstm"

    def __init__(self, labels, epochs: int = 12, batch_size: int = 64, lr: float = 2e-3, max_len: int = 120,
                 min_freq: int = 2, max_vocab: int = 40_000, seed: int = 42):
        super().__init__(labels)
        self.epochs, self.batch_size, self.lr, self.max_len = epochs, batch_size, lr, max_len
        self.min_freq, self.max_vocab, self.seed = min_freq, max_vocab, seed
        self.vocab: dict[str, int] = {}
        self.net: _Net | None = None
        self.history: list[dict] = []

    def _encode(self, text: str) -> list[int]:
        ids = [self.vocab.get(t, 1) for t in tokenize(text)[: self.max_len]]
        return ids or [1]

    def _batches(self, texts, y=None, shuffle=False):
        enc = [self._encode(t) for t in texts]
        idx = sorted(range(len(enc)), key=lambda i: len(enc[i]))
        chunks = [idx[i : i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
        if shuffle:
            random.shuffle(chunks)
        for ch in chunks:
            L = max(len(enc[i]) for i in ch)
            ids = torch.zeros((len(ch), L), dtype=torch.long)
            for r, i in enumerate(ch):
                ids[r, : len(enc[i])] = torch.tensor(enc[i])
            yield ch, ids, (torch.tensor(y[ch]) if y is not None else None)

    def fit(self, texts, y, val=None, log=print):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        counts = Counter(t for x in texts for t in tokenize(x))
        words = [w for w, c in counts.most_common(self.max_vocab) if c >= self.min_freq]
        self.vocab = {"<pad>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate(words)}}
        self.net = _Net(len(self.vocab), len(self.labels))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=self.epochs * (len(texts) // self.batch_size + 1)
        )
        # Soft-target softmax cross-entropy (gold labels share the mass). From scratch,
        # per-label BCE over 135 mostly-negative outputs stalls at the label prior.
        def loss_fn(logits, yb):
            target = yb / yb.sum(1, keepdim=True).clamp(min=1)
            return -(target * logits.log_softmax(-1)).sum(1).mean()

        best, best_state, patience = -1.0, None, 0
        for ep in range(self.epochs):
            self.net.train()
            t0, tot, n = time.time(), 0.0, 0
            for _, ids, yb in self._batches(texts, y, shuffle=True):
                loss = loss_fn(self.net(ids), yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
                tot, n = tot + loss.item(), n + 1
            rec = {"epoch": ep + 1, "train_loss": tot / n, "seconds": time.time() - t0}
            if val is not None:
                from ..evaluate import ranking_metrics

                rec.update(ranking_metrics(val[1], self.predict_proba(val[0])))
                if rec["MRR"] > best:
                    best, patience = rec["MRR"], 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    patience += 1
            self.history.append(rec)
            log(f"[bilstm] {rec}")
            if patience >= 3:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.inference_mode()
    def predict_proba(self, texts):
        self.net.eval()
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        for ch, ids, _ in self._batches(list(texts)):
            out[ch] = self.net(ids).softmax(-1).numpy()
        return out

    @torch.inference_mode()
    def attention(self, text: str) -> list[tuple[str, float]]:
        self.net.eval()
        toks = tokenize(text)[: self.max_len] or ["<unk>"]
        ids = torch.tensor([self._encode(text)])
        _, a = self.net(ids, return_att=True)
        return list(zip(toks, a[0].tolist()))

    def save(self, path: Path):
        path = Path(path)
        self._write_meta(path, max_len=self.max_len, history=self.history)
        (path / "vocab.json").write_text(json.dumps(self.vocab))
        torch.save(self.net.state_dict(), path / "bilstm.pt")

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        meta = cls._read_meta(path)
        m = cls(meta["labels"], max_len=meta["max_len"])
        m.threshold, m.history = meta["threshold"], meta.get("history", [])
        m.vocab = json.loads((path / "vocab.json").read_text())
        m.net = _Net(len(m.vocab), len(m.labels))
        m.net.load_state_dict(torch.load(path / "bilstm.pt", map_location="cpu"))
        m.net.eval()
        return m
