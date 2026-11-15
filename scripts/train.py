"""Step 2 — train one model family, tune its threshold on val, score val + test.

    python scripts/train.py tfidf
    python scripts/train.py roberta --epochs 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attackmap.config import ARTIFACTS_DIR  # noqa: E402
from attackmap.data.attack_kb import KnowledgeBase  # noqa: E402
from attackmap.data.build import load_corpus, load_labels  # noqa: E402
from attackmap.evaluate import evaluate, multi_hot, tune_threshold  # noqa: E402
from attackmap.models import registry  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=registry.MODEL_NAMES)
ap.add_argument("--epochs", type=int)
ap.add_argument("--limit", type=int, help="subsample train set (smoke tests)")
args = ap.parse_args()

rows, labels, kb = load_corpus(), load_labels(), KnowledgeBase.load()
part = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
if args.limit:
    part = {s: v[: args.limit] for s, v in part.items()}
X = {s: [r["text"] for r in v] for s, v in part.items()}
Y = {s: multi_hot([r["labels"] for r in v], labels) for s, v in part.items()}

kw = {"epochs": args.epochs} if args.epochs else {}
model = registry.build(args.model, labels, **kw)
t0 = time.time()
model.fit(X["train"], Y["train"], val=(X["val"], Y["val"]))
train_s = time.time() - t0

out_dir = ARTIFACTS_DIR / args.model
out_dir.mkdir(parents=True, exist_ok=True)
s_val = model.predict_proba(X["val"])
model.threshold = tune_threshold(Y["val"], s_val)
t1 = time.time()
s_test = model.predict_proba(X["test"])
infer_ms = (time.time() - t1) / len(X["test"]) * 1000
model.save(out_dir)
np.save(out_dir / "scores_val.npy", s_val)
np.save(out_dir / "scores_test.npy", s_test)

res = {
    "model": args.model,
    "train_seconds": round(train_s, 1),
    "infer_ms_per_text": round(infer_ms, 2),
    "threshold": model.threshold,
    "val": evaluate(Y["val"], s_val, model.threshold, labels, kb.tactics_of),
    "test": evaluate(Y["test"], s_test, model.threshold, labels, kb.tactics_of),
}
(out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
