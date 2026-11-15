"""Central paths and hyper-parameters."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ATTACKMAP_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.environ.get("ATTACKMAP_ARTIFACTS", ROOT / "artifacts"))
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

ATTACK_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
)
TRAM_URLS = {
    "single": "https://raw.githubusercontent.com/center-for-threat-informed-defense/tram/main/data/tram2-data/single_label.json",
    "multi": "https://raw.githubusercontent.com/center-for-threat-informed-defense/tram/main/data/tram2-data/multi_label.json",
}
SIGMA_LIST_URL = "https://data.jsdelivr.com/v1/packages/gh/SigmaHQ/sigma@master?structure=flat"
SIGMA_FILE_URL = "https://cdn.jsdelivr.net/gh/SigmaHQ/sigma@master{path}"
SIGMA_DIRS = ("/rules/", "/rules-emerging-threats/", "/rules-threat-hunting/")

# Label space: techniques are rolled up to their parent (T1059.001 -> T1059) and a
# technique must have at least this many examples to become a classifier label.
MIN_EXAMPLES_PER_LABEL = 25
SEED = 42
SPLIT = (0.8, 0.1, 0.1)

MODELS = {
    "bert": "bert-base-uncased",
    "roberta": "distilroberta-base",
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
}

for _d in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
