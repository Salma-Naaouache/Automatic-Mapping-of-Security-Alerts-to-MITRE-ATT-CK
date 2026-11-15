"""Loaders for the three labelled corpora.

* ATT&CK procedure examples  -> CTI prose ("APT29 used PowerShell to ...")
* TRAM (MITRE CTID)         -> sentences from real CTI reports, analyst-annotated
* SigmaHQ rules             -> SIEM detections rendered as alert text, tagged ``attack.tXXXX``
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from ..config import RAW_DIR, SIGMA_DIRS, SIGMA_FILE_URL, SIGMA_LIST_URL, TRAM_URLS
from .download import fetch, fetch_many

_TECH_TAG = re.compile(r"^attack\.(t\d{4}(?:\.\d{3})?)$", re.I)


# --------------------------------------------------------------------------- TRAM
def load_tram() -> list[dict]:
    rows: dict[str, dict] = {}
    single = json.loads(fetch(TRAM_URLS["single"], RAW_DIR / "tram_single.json").read_text())
    for r in single:
        if r.get("label"):
            rows.setdefault(r["text"].strip(), {"labels": set(), "doc": r.get("doc_title", "")})[
                "labels"
            ].add(r["label"])
    multi = json.loads(fetch(TRAM_URLS["multi"], RAW_DIR / "tram_multi.json").read_text())
    for r in multi:
        if r.get("labels"):
            rows.setdefault(r["sentence"].strip(), {"labels": set(), "doc": r.get("doc_title", "")})[
                "labels"
            ].update(r["labels"])
    return [
        {"text": t, "labels": sorted(v["labels"]), "source": "tram_cti", "origin": v["doc"]}
        for t, v in rows.items()
        if len(t) > 15
    ]


# -------------------------------------------------------------------------- Sigma
def _flatten_detection(det: dict, limit: int = 12) -> list[str]:
    """Turn a Sigma ``detection`` block into short 'field op value' indicator strings."""
    out: list[str] = []

    def walk(node, field=None):
        if len(out) >= limit:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("condition", "timeframe"):
                    continue
                walk(v, k if "|" in k or field is None or not k.startswith("selection") else field)
        elif isinstance(node, list):
            for v in node[:4]:
                walk(v, field)
        elif node is not None and field is not None:
            name, *mods = str(field).split("|")
            op = mods[0] if mods else "is"
            val = str(node)
            if len(val) < 120:
                out.append(f"{name} {op} {val}")

    for key, val in det.items():
        if key not in ("condition", "timeframe"):
            walk(val, None if isinstance(val, dict) else key)
    return out


def sigma_rule_to_alert(rule: dict) -> str | None:
    """Render a Sigma rule like the alert a SIEM would raise when it fires."""
    title = rule.get("title")
    if not title:
        return None
    ls = rule.get("logsource", {}) or {}
    src = "/".join(str(ls[k]) for k in ("product", "service", "category") if ls.get(k))
    parts = [f"Alert: {title}."]
    if rule.get("description"):
        desc = " ".join(str(rule["description"]).split())
        parts.append(desc if desc.endswith(".") else desc + ".")
    if src:
        parts.append(f"Log source: {src}.")
    ind = _flatten_detection(rule.get("detection", {}) or {})
    if ind:
        parts.append("Matched: " + "; ".join(ind) + ".")
    return " ".join(parts)


def sigma_labels(rule: dict) -> list[str]:
    tags = rule.get("tags") or []
    return sorted({m.group(1).upper() for t in tags if (m := _TECH_TAG.match(str(t)))})


def load_sigma() -> list[dict]:
    listing = json.loads(fetch(SIGMA_LIST_URL, RAW_DIR / "sigma_listing.json").read_text())
    paths = [
        f["name"]
        for f in listing["files"]
        if f["name"].endswith(".yml") and f["name"].startswith(SIGMA_DIRS)
    ]
    pairs = [(SIGMA_FILE_URL.format(path=p), RAW_DIR / "sigma" / p.lstrip("/")) for p in paths]
    files = fetch_many(pairs)
    rows = []
    for p, path in zip(paths, files):
        if path is None:
            continue
        try:
            for rule in yaml.safe_load_all(Path(path).read_text(encoding="utf-8")):
                if not isinstance(rule, dict):
                    continue
                labels = sigma_labels(rule)
                text = sigma_rule_to_alert(rule)
                if labels and text:
                    rows.append({"text": text, "labels": labels, "source": "sigma_siem", "origin": p})
        except yaml.YAMLError:
            continue
    return rows
