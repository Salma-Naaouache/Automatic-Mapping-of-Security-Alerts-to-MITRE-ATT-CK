"""MITRE ATT&CK Enterprise knowledge base built from the official STIX 2.1 bundle."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import ATTACK_STIX_URL, PROCESSED_DIR, RAW_DIR
from ..text import clean, parent_technique
from .download import fetch



@dataclass
class Tactic:
    shortname: str
    id: str
    name: str
    order: int


def parse_tactics(bundle: dict) -> list[Tactic]:
    """Tactics in Enterprise-matrix order, read from the bundle (so new tactics such as
    'Stealth' / 'Defense Impairment' in recent ATT&CK releases are handled automatically)."""
    tac = {
        o["id"]: o
        for o in bundle["objects"]
        if o.get("type") == "x-mitre-tactic" and not o.get("revoked") and not o.get("x_mitre_deprecated")
    }
    matrix = next(o for o in bundle["objects"] if o.get("type") == "x-mitre-matrix")
    out = []
    for i, ref in enumerate(r for r in matrix["tactic_refs"] if r in tac):
        o = tac[ref]
        out.append(Tactic(o["x_mitre_shortname"], _external_id(o)[0], o["name"], i))
    return out


@dataclass
class Technique:
    id: str
    name: str
    tactics: list[str]
    description: str
    platforms: list[str] = field(default_factory=list)
    detection: str = ""
    is_subtechnique: bool = False
    url: str = ""

    @property
    def parent(self) -> str:
        return parent_technique(self.id)

    def as_document(self) -> str:
        """Text used to represent a technique for retrieval (zero-shot)."""
        first = self.description.split(". ")
        return f"{self.name}. " + ". ".join(first[:4])


def _external_id(obj: dict) -> tuple[str, str]:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", ""), ref.get("url", "")
    return "", ""


def load_stix(path: Path | None = None) -> dict:
    path = path or fetch(ATTACK_STIX_URL, RAW_DIR / "enterprise-attack.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_techniques(bundle: dict) -> dict[str, Technique]:
    order = {t.shortname: t.order for t in parse_tactics(bundle)}
    out: dict[str, Technique] = {}
    for obj in bundle["objects"]:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        tid, url = _external_id(obj)
        if not tid:
            continue
        tactics = [
            p["phase_name"]
            for p in obj.get("kill_chain_phases", [])
            if p.get("kill_chain_name") == "mitre-attack" and p["phase_name"] in order
        ]
        out[tid] = Technique(
            id=tid,
            name=obj.get("name", ""),
            tactics=sorted(set(tactics), key=order.__getitem__),
            description=clean(obj.get("description", ""), mask=False),
            platforms=obj.get("x_mitre_platforms", []),
            detection=clean(obj.get("x_mitre_detection", ""), mask=False),
            is_subtechnique=bool(obj.get("x_mitre_is_subtechnique")),
            url=url,
        )
    return out


def parse_procedures(bundle: dict) -> list[dict]:
    """ATT&CK *procedure examples*: 'uses' relationships whose description says how a
    group / malware / tool / campaign used a technique.  Real CTI prose, expert-labelled."""
    by_id = {o["id"]: o for o in bundle["objects"] if "id" in o}
    rows = []
    for obj in bundle["objects"]:
        if obj.get("type") != "relationship" or obj.get("relationship_type") != "uses":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        target = by_id.get(obj.get("target_ref", ""))
        if not target or target.get("type") != "attack-pattern" or target.get("revoked"):
            continue
        tid, _ = _external_id(target)
        desc = obj.get("description", "")
        if not tid or not desc:
            continue
        source = by_id.get(obj.get("source_ref", ""), {})
        rows.append(
            {
                "text": desc,
                "labels": [tid],
                "source": "attack_procedure",
                "origin": f"{source.get('type', '?')}:{source.get('name', '?')}",
            }
        )
    return rows


class KnowledgeBase:
    """Lookup helper over techniques, with parent roll-up and tactic metadata."""

    def __init__(self, techniques: dict[str, Technique], tactics: list[Tactic]):
        self.techniques = techniques
        self.tactics = tactics
        self.tactic_by_short = {t.shortname: t for t in tactics}
        self.parents = {k: v for k, v in techniques.items() if not v.is_subtechnique}

    # ---- persistence -------------------------------------------------------
    def save(self, path: Path = PROCESSED_DIR / "attack_kb.json") -> Path:
        path.write_text(
            json.dumps(
                {
                    "tactics": [asdict(t) for t in self.tactics],
                    "techniques": {k: asdict(v) for k, v in self.techniques.items()},
                },
                indent=1,
            )
        )
        return path

    @classmethod
    def load(cls, path: Path = PROCESSED_DIR / "attack_kb.json") -> "KnowledgeBase":
        if not path.exists():
            kb = cls.from_bundle(load_stix())
            kb.save(path)
            return kb
        data = json.loads(path.read_text())
        return cls(
            {k: Technique(**v) for k, v in data["techniques"].items()},
            [Tactic(**t) for t in data["tactics"]],
        )

    @classmethod
    def from_bundle(cls, bundle: dict) -> "KnowledgeBase":
        return cls(parse_techniques(bundle), parse_tactics(bundle))

    # ---- queries -----------------------------------------------------------
    def get(self, tid: str) -> Technique | None:
        return self.techniques.get(tid.upper())

    def tactics_of(self, tid: str) -> list[str]:
        t = self.get(tid) or self.get(parent_technique(tid))
        return t.tactics if t else []

    def name_of(self, tid: str) -> str:
        t = self.get(tid)
        return t.name if t else tid

    def __len__(self) -> int:
        return len(self.techniques)

    def __contains__(self, tid: str) -> bool:
        return tid.upper() in self.techniques
