"""Step 5 — out-of-distribution check on hand-written SOC alerts (never seen in training)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attackmap.config import REPORTS_DIR, ROOT  # noqa: E402
from attackmap.predictor import AttackMapper  # noqa: E402

rows = [json.loads(l) for l in open(ROOT / "data/challenge/soc_alerts.jsonl")]
m = AttackMapper()
res = m.predict_batch([r["text"] for r in rows], top_k=5)
p1 = h3 = h5 = tac = 0
lines = []
for r, out in zip(rows, res):
    ids = [t["id"] for t in out["techniques"]]
    gold = set(r["labels"])
    ok1, ok3, ok5 = ids[0] in gold, bool(gold & set(ids[:3])), bool(gold & set(ids))
    gold_tac = {t for g in gold for t in m.kb.tactics_of(g)}
    okt = bool(gold_tac & set(out["techniques"][0]["tactics"]))
    p1, h3, h5, tac = p1 + ok1, h3 + ok3, h5 + ok5, tac + okt
    lines.append({"text": r["text"], "gold": sorted(gold), "top5": ids, "top1_ok": ok1, "hit5": ok5})
    print(f"{'✔' if ok1 else ('~' if ok5 else '✘')} gold={','.join(sorted(gold)):18s} pred={ids[0]} {out['techniques'][0]['name'][:38]:38s} | {r['text'][:60]}")
n = len(rows)
summary = {"n": n, "P@1": p1 / n, "Hit@3": h3 / n, "Hit@5": h5 / n, "Tactic@1": tac / n}
print(json.dumps(summary, indent=1))
(REPORTS_DIR / "challenge.json").write_text(json.dumps({"summary": summary, "items": lines}, indent=1))
