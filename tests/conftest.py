import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from attackmap.config import ARTIFACTS_DIR, PROCESSED_DIR  # noqa: E402

LABELS = ["T1003", "T1059", "T1566", "T1486"]
_TEMPLATES = {
    "T1003": ["mimikatz dumped credentials from lsass memory on {h}", "procdump accessed lsass.exe to steal passwords on {h}",
              "sekurlsa logonpasswords executed on {h}", "comsvcs minidump of lsass written on {h}"],
    "T1059": ["powershell.exe -enc payload executed on {h}", "cmd.exe /c whoami spawned by winword on {h}",
              "bash -c curl | sh executed on {h}", "encoded powershell command line observed on {h}"],
    "T1566": ["spearphishing email with malicious attachment sent to {h}", "phishing link clicked by user on {h}",
              "malicious macro document delivered by email to {h}", "phishing campaign targeted finance staff {h}"],
    "T1486": ["ransomware encrypted files with .lockbit extension on {h}", "files encrypted and ransom note dropped on {h}",
              "mass file encryption detected on share {h}", "ransomware payload encrypted the disk of {h}"],
}


def make_dataset(n_per_template=6, seed=0):
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for lab, temps in _TEMPLATES.items():
        for t in temps:
            for _ in range(n_per_template):
                texts.append(t.format(h=f"host{rng.integers(1000)}"))
                labels.append([lab])
    return texts, labels


@pytest.fixture(scope="session")
def tiny():
    from attackmap.evaluate import multi_hot
    texts, labs = make_dataset()
    return texts, multi_hot(labs, LABELS), LABELS


@pytest.fixture(scope="session")
def kb():
    from attackmap.data.attack_kb import KnowledgeBase
    if not (PROCESSED_DIR / "attack_kb.json").exists():
        pytest.skip("knowledge base not built (run scripts/build_dataset.py)")
    return KnowledgeBase.load()


def has_artifacts(*names):
    return all((ARTIFACTS_DIR / n / "meta.json").exists() for n in names)
