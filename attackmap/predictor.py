"""Production inference: ensemble of trained mappers -> analyst-ready ATT&CK mapping."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import ARTIFACTS_DIR
from .data.attack_kb import KnowledgeBase
from .models import registry
from .models.ensemble import combine
from .text import clean, extract_iocs

ENSEMBLE_PATH = ARTIFACTS_DIR / "ensemble.json"


def confidence_band(score: float) -> str:
    return "high" if score >= 0.6 else "medium" if score >= 0.3 else "low"


class AttackMapper:
    """Load whichever trained models exist and serve ensemble predictions.

    ``models``: optional explicit subset, e.g. ``["tfidf"]`` for a lightweight deployment.
    """

    def __init__(self, models: list[str] | None = None, root: Path = ARTIFACTS_DIR):
        self.root = Path(root)
        self.kb = KnowledgeBase.load()
        cfg = json.loads((self.root / "ensemble.json").read_text()) if (self.root / "ensemble.json").exists() else {}
        weights = cfg.get("weights", {})
        names = models or list(weights) or [m for m in registry.MODEL_NAMES if (self.root / m).exists()]
        self.models = {n: registry.load(n, self.root) for n in names if (self.root / n / "meta.json").exists()}
        if not self.models:
            raise FileNotFoundError(f"no trained models found under {self.root}")
        self.weights = {n: weights.get(n, 1.0) for n in self.models}
        if not any(self.weights.values()):
            self.weights = {n: 1.0 for n in self.models}
        self.labels = next(iter(self.models.values())).labels
        self.threshold = cfg.get("threshold", 0.15) if (models is None or set(models) == set(weights)) else 0.15
        self.retriever = self.models.get("sbert") or self.models.get("sbert_zeroshot")
        self._children = {}
        for tid, t in self.kb.techniques.items():
            if t.is_subtechnique:
                self._children.setdefault(t.parent, []).append(tid)
        self._long_tail = [t for t, v in self.kb.parents.items() if t not in set(self.labels)]

    # ------------------------------------------------------------------ core
    def scores(self, texts: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cleaned = [clean(t) for t in texts]
        active = [n for n in self.models if self.weights[n] > 0]
        per = {n: self.models[n].predict_proba(cleaned) for n in active}
        total = sum(self.weights[n] for n in active)
        ens = combine([per[n] for n in active], [self.weights[n] / total for n in active])
        return ens, per

    def predict(self, text: str, top_k: int = 5, explain: bool = True) -> dict:
        return self.predict_batch([text], top_k=top_k, explain=explain)[0]

    def predict_batch(self, texts: list[str], top_k: int = 5, explain: bool = False) -> list[dict]:
        t0 = time.perf_counter()
        ens, per = self.scores(texts)
        results = []
        for i, text in enumerate(texts):
            order = np.argsort(-ens[i])[:top_k]
            techs = []
            for rank, j in enumerate(order):
                tid = self.labels[j]
                t = self.kb.get(tid)
                techs.append({
                    "rank": rank + 1,
                    "id": tid,
                    "name": t.name if t else tid,
                    "tactics": t.tactics if t else [],
                    "score": round(float(ens[i, j]), 4),
                    "confidence": confidence_band(float(ens[i, j])),
                    "selected": bool(rank == 0 or ens[i, j] >= self.threshold),
                    "per_model": {n: round(float(p[i, j]), 4) for n, p in per.items()},
                    "url": t.url if t else "",
                })
            results.append({
                "text": text,
                "techniques": techs,
                "tactics": self._tactic_profile(ens[i]),
                "iocs": extract_iocs(text),
            })
        if explain:
            for r in results:
                self._explain(r)
        dt = (time.perf_counter() - t0) * 1000
        for r in results:
            r["latency_ms"] = round(dt / len(texts), 1)
        return results

    # --------------------------------------------------------------- extras
    def _tactic_profile(self, row: np.ndarray) -> list[dict]:
        agg: dict[str, float] = {}
        for j, s in enumerate(row):
            for tac in self.kb.tactics_of(self.labels[j]):
                agg[tac] = agg.get(tac, 0.0) + float(s)
        tot = sum(agg.values()) or 1.0
        out = []
        for tac in self.kb.tactics:
            if tac.shortname in agg:
                out.append({"id": tac.id, "shortname": tac.shortname, "name": tac.name,
                            "score": round(agg[tac.shortname] / tot, 4)})
        return sorted(out, key=lambda d: -d["score"])

    def _explain(self, r: dict) -> None:
        text, top = r["text"], r["techniques"][0]
        cleaned = clean(text)
        ev: dict = {}
        if "tfidf" in self.models:
            ev["terms"] = self.models["tfidf"].explain(cleaned, self.labels.index(top["id"]))
        if "bilstm" in self.models:
            att = self.models["bilstm"].attention(cleaned)
            ev["attention"] = [(tok, round(a, 4)) for tok, a in att]
        r["evidence"] = ev
        if self.retriever is not None:
            for t in r["techniques"][:3]:
                kids = self._children.get(t["id"], [])
                if kids:
                    t["subtechniques"] = [
                        {"id": k, "name": self.kb.name_of(k), "similarity": round(s, 3)}
                        for k, s in self.retriever.retrieve(cleaned, kids, top=3)
                    ]
            r["long_tail"] = [
                {"id": k, "name": self.kb.name_of(k), "tactics": self.kb.tactics_of(k), "similarity": round(s, 3)}
                for k, s in self.retriever.retrieve(cleaned, self._long_tail, top=3)
            ]

    # ------------------------------------------------------------ navigator
    def navigator_layer(self, results: list[dict], name: str = "attackmap detections") -> dict:
        """ATT&CK Navigator (v4.5 layer format) heat-map of selected techniques."""
        agg: dict[str, float] = {}
        count: dict[str, int] = {}
        for r in results:
            for t in r["techniques"]:
                if t["selected"]:
                    agg[t["id"]] = max(agg.get(t["id"], 0.0), t["score"])
                    count[t["id"]] = count.get(t["id"], 0) + 1
        return {
            "name": name,
            "versions": {"layer": "4.5", "navigator": "5.1.0"},
            "domain": "enterprise-attack",
            "description": f"Generated by attackmap from {len(results)} alert(s)/report(s)",
            "gradient": {"colors": ["#fff3e0", "#ff9800", "#d84315"], "minValue": 0, "maxValue": max(count.values(), default=1)},
            "techniques": [
                {"techniqueID": tid, "score": count[tid], "comment": f"max confidence {agg[tid]:.2f}",
                 "enabled": True, "showSubtechniques": False}
                for tid in sorted(agg)
            ],
        }
