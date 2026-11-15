"""Cached HTTP downloads."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "attackmap-research/1.0"


def fetch(url: str, dest: Path, retries: int = 4, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` unless it is already cached."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=120)
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            return dest
        except requests.RequestException as exc:  # pragma: no cover - network
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"download failed: {url}: {last}")


def fetch_many(pairs: list[tuple[str, Path]], workers: int = 16) -> list[Path | None]:
    def _one(p: tuple[str, Path]) -> Path | None:
        try:
            return fetch(*p)
        except RuntimeError:
            return None

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(_one, pairs))
