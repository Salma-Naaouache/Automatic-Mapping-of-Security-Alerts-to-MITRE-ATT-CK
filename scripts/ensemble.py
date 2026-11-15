"""Step 3 — search ensemble weights on validation, report everything on test."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attackmap.config import ARTIFACTS_DIR, REPORTS_DIR  # noqa: E402
from attackmap.data.attack_kb import KnowledgeBase  # noqa: E402
from attackmap.data.build import load_corpus, load_labels  # noqa: E402
from attackmap.evaluate import evaluate, multi_hot, tune_threshold  # noqa: E402
from attackmap.models.ensemble import combine, search_weights  # noqa: E402
from attackmap.models.registry import MODEL_NAMES  # noqa: E402

rows, labels, kb = load_corpus(), load_labels(), KnowledgeBase.load()
Y = {s: multi_hot([r["labels"] for r in rows if r["split"] == s], labels) for s in ("val", "test")}
names = [n for n in MODEL_NAMES if (ARTIFACTS_DIR / n / "scores_val.npy").exists()]
S = {s: [np.load(ARTIFACTS_DIR / n / f"scores_{s}.npy") for n in names] for s in ("val", "test")}
print("models:", names)

w, best = search_weights(Y["val"], S["val"])
ens_val, ens_test = combine(S["val"], w), combine(S["test"], w)
thr = tune_threshold(Y["val"], ens_val)
cfg = {"weights": dict(zip(names, w)), "threshold": thr, "val_MRR": best}
(ARTIFACTS_DIR / "ensemble.json").write_text(json.dumps(cfg, indent=1))
np.save(ARTIFACTS_DIR / "ensemble_scores_test.npy", ens_test)
print("weights:", cfg["weights"])

table = {}
for n in names:
    table[n] = json.loads((ARTIFACTS_DIR / n / "metrics.json").read_text())
table["ensemble"] = {
    "model": "ensemble", "threshold": thr,
    "val": evaluate(Y["val"], ens_val, thr, labels, kb.tactics_of),
    "test": evaluate(Y["test"], ens_test, thr, labels, kb.tactics_of),
}
# per-source breakdown on test
src = np.array([r["source"] for r in rows if r["split"] == "test"])
for n, s in list(zip(names, S["test"])) + [("ensemble", ens_test)]:
    thr_n = table[n]["threshold"]
    table[n]["test_by_source"] = {
        k: evaluate(Y["test"][src == k], s[src == k], thr_n, labels, kb.tactics_of) for k in sorted(set(src))
    }
(REPORTS_DIR / "results.json").write_text(json.dumps(table, indent=1))
hdr = ["P@1", "Hit@3", "Hit@5", "MRR", "micro_F1", "macro_F1", "Tactic@1"]
print(f"{'model':16s}" + "".join(f"{h:>10s}" for h in hdr))
for n, r in table.items():
    print(f"{n:16s}" + "".join(f"{r['test'][h]:10.3f}" for h in hdr))
