"""REST API + web console.

    uvicorn attackmap.api:app --port 8000
    ATTACKMAP_MODELS=tfidf,sbert  uvicorn attackmap.api:app   # lighter deployment
"""
from __future__ import annotations

import json
import os
import random
import re
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import PROCESSED_DIR, REPORTS_DIR, ROOT
from .predictor import AttackMapper

WEB = ROOT / "web"
MAX_CHARS = 20_000

app = FastAPI(
    title="ATT&CK Mapper",
    version=__version__,
    description="Automatic mapping of security alerts and CTI reports to MITRE ATT&CK tactics & techniques.",
)


@lru_cache(maxsize=1)
def mapper() -> AttackMapper:
    sel = os.environ.get("ATTACKMAP_MODELS")
    return AttackMapper(models=sel.split(",") if sel else None)


# ------------------------------------------------------------------ schemas
class MapRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=MAX_CHARS, examples=["powershell.exe -nop -w hidden -enc SQBFAFgA..."])
    top_k: int = Field(5, ge=1, le=20)
    explain: bool = True


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)
    top_k: int = Field(3, ge=1, le=10)


class ReportRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=200_000)
    min_score: float = Field(0.30, ge=0, le=1)


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for chunk in text.split("\n") for p in _SENT.split(chunk)]
    return [p for p in parts if len(p.split()) >= 5]


# ---------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    m = mapper()
    return {
        "status": "ok",
        "version": __version__,
        "models": {n: round(w, 3) for n, w in m.weights.items()},
        "labels": len(m.labels),
        "attack_techniques": len(m.kb),
        "tactics": [t.name for t in m.kb.tactics],
    }


@app.post("/api/map")
def map_alert(req: MapRequest):
    return mapper().predict(req.text, top_k=req.top_k, explain=req.explain)


@app.post("/api/map/batch")
def map_batch(req: BatchRequest):
    texts = [t for t in req.texts if t and t.strip()]
    if not texts:
        raise HTTPException(422, "no non-empty texts")
    if any(len(t) > MAX_CHARS for t in texts):
        raise HTTPException(413, f"each text must be <= {MAX_CHARS} chars")
    m = mapper()
    results = m.predict_batch(texts, top_k=req.top_k)
    return {"results": results, "navigator_layer": m.navigator_layer(results)}


@app.post("/api/report")
def map_report(req: ReportRequest):
    """CTI report mode: sentence-level mapping, aggregated into a technique summary."""
    m = mapper()
    sents = split_sentences(req.text)[:400]
    if not sents:
        raise HTTPException(422, "no sentences of 5+ words found")
    results = m.predict_batch(sents, top_k=3)
    hits = [r for r in results if r["techniques"][0]["score"] >= req.min_score]
    summary: dict[str, dict] = {}
    for idx, r in enumerate(results):
        t = r["techniques"][0]
        if t["score"] < req.min_score:
            continue
        s = summary.setdefault(t["id"], {"id": t["id"], "name": t["name"], "tactics": t["tactics"],
                                         "max_score": 0.0, "sentences": []})
        s["max_score"] = max(s["max_score"], t["score"])
        s["sentences"].append({"index": idx, "text": r["text"], "score": t["score"]})
    techniques = sorted(summary.values(), key=lambda d: (-len(d["sentences"]), -d["max_score"]))
    return {
        "n_sentences": len(sents),
        "n_mapped": len(hits),
        "techniques": techniques,
        "sentences": [{"text": r["text"], "top": r["techniques"][0]} for r in results],
        "navigator_layer": m.navigator_layer(hits, name="CTI report"),
    }


@app.get("/api/techniques/{tid}")
def technique(tid: str):
    t = mapper().kb.get(tid)
    if not t:
        raise HTTPException(404, f"unknown technique {tid}")
    return t.__dict__


@app.get("/api/matrix")
def matrix():
    """Enterprise matrix restricted to parent techniques, ordered by tactic."""
    m = mapper()
    in_model = set(m.labels)
    cols = []
    for tac in m.kb.tactics:
        techs = sorted(
            ({"id": t.id, "name": t.name, "modelled": t.id in in_model}
             for t in m.kb.parents.values() if tac.shortname in t.tactics),
            key=lambda d: d["name"],
        )
        cols.append({"id": tac.id, "shortname": tac.shortname, "name": tac.name, "techniques": techs})
    return cols


@app.get("/api/metrics")
def metrics():
    p = REPORTS_DIR / "results.json"
    if not p.exists():
        raise HTTPException(404, "run scripts/ensemble.py first")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/examples")
def examples(n: int = 6, source: str | None = None):
    path = PROCESSED_DIR / "corpus.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "test" and (source is None or r["source"] == source)]
    rng = random.Random()
    return [{"text": r["text"], "labels": r["labels"], "source": r["source"]} for r in rng.sample(rows, min(n, len(rows)))]
