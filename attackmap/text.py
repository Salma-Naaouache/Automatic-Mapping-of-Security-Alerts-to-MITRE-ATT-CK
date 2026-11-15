"""Text normalisation for CTI prose and SIEM alerts.

Security text is full of high-entropy tokens (IPs, hashes, GUIDs, paths) that are
noise for a classifier but carry *type* information.  We replace each one with a
typed placeholder so ``185.220.101.4`` and ``10.0.0.7`` both become ``<IP>``.
"""
from __future__ import annotations

import re

_CITATION = re.compile(r"\(Citation:[^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_PLACEHOLDERS = ("URL", "SHA256", "SHA1", "MD5", "GUID", "EMAIL", "IP")
# HTML-ish tags, but never our own IOC placeholders (keeps clean() idempotent).
_HTML = re.compile(r"<(?!(?:%s)>)[^>]{1,40}>" % "|".join(_PLACEHOLDERS))

# Order matters: longest / most specific patterns first.
_IOC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("<URL>", re.compile(r"\b(?:https?|ftp|hxxps?)://[^\s\"'<>]+", re.I)),
    ("<SHA256>", re.compile(r"\b[a-f0-9]{64}\b", re.I)),
    ("<SHA1>", re.compile(r"\b[a-f0-9]{40}\b", re.I)),
    ("<MD5>", re.compile(r"\b[a-f0-9]{32}\b", re.I)),
    ("<GUID>", re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)),
    ("<EMAIL>", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("<IP>", re.compile(r"\b(?:\d{1,3}(?:\.|\[\.\])){3}\d{1,3}(?::\d+)?\b")),
]

_WS = re.compile(r"\s+")


def strip_markup(text: str) -> str:
    """Remove ATT&CK citations, markdown links (keep anchor text) and stray HTML tags."""
    text = _CITATION.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _HTML.sub(" ", text)
    return text


def mask_iocs(text: str) -> str:
    for token, pattern in _IOC_PATTERNS:
        text = pattern.sub(f" {token} ", text)
    return text


def clean(text: str, mask: bool = True) -> str:
    """Full normalisation used everywhere (training *and* inference)."""
    if not text:
        return ""
    text = strip_markup(str(text))
    if mask:
        text = mask_iocs(text)
    return _WS.sub(" ", text).strip()


def parent_technique(tid: str) -> str:
    """``T1059.001`` -> ``T1059``; also upper-cases and validates the id."""
    tid = tid.strip().upper()
    if not re.fullmatch(r"T\d{4}(\.\d{3})?", tid):
        raise ValueError(f"not an ATT&CK technique id: {tid!r}")
    return tid.split(".")[0]


_IOC_EXTRACT = {
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "url": re.compile(r"\b(?:https?|hxxps?)://[^\s\"'<>]+", re.I),
    "sha256": re.compile(r"\b[a-f0-9]{64}\b", re.I),
    "sha1": re.compile(r"\b[a-f0-9]{40}\b", re.I),
    "md5": re.compile(r"\b[a-f0-9]{32}\b", re.I),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "cve": re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I),
    "file": re.compile(r"\b[\w\-]+\.(?:exe|dll|ps1|bat|vbs|js|hta|scr|lnk|sys|msi|docm|xlsm|iso|zip|rar|sh|py)\b", re.I),
    "registry": re.compile(r"\b(?:HKLM|HKCU|HKEY_[A-Z_]+)\\[^\s;\"']+", re.I),
}


def extract_iocs(text: str) -> dict[str, list[str]]:
    """Indicators of compromise found in the raw (unmasked) text, de-duplicated in order."""
    out: dict[str, list[str]] = {}
    taken: set[str] = set()
    for kind, pat in _IOC_EXTRACT.items():
        vals = []
        for m in pat.findall(text):
            if m not in taken and not (kind in ("sha1", "md5") and any(m in t for t in taken)):
                vals.append(m)
                taken.add(m)
        if vals:
            out[kind] = vals[:20]
    return out
