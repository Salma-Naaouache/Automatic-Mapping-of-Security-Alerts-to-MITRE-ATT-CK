"""Fill README result tables from reports/*.json (keeps numbers in sync with artifacts)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = {"tfidf": "TF-IDF + LogReg (baseline)", "bilstm": "BiLSTM + attention", "sbert_zeroshot": "Sentence-Transformer, zero-shot",
         "sbert": "Sentence-Transformer, fine-tuned", "roberta": "RoBERTa (distilroberta-base)", "bert": "BERT (bert-base-uncased)",
         "ensemble": "**Ensemble**"}
res = json.loads((ROOT / "reports/results.json").read_text())
cols = ["P@1", "Hit@3", "Hit@5", "MRR", "micro_F1", "macro_F1", "Tactic@1"]
best = {c: max(r["test"][c] for r in res.values()) for c in cols}
lines = ["| Model | " + " | ".join(cols) + " | train | ms/text |", "|---" * (len(cols) + 3) + "|"]
for n in NAMES:
    if n not in res:
        continue
    r = res[n]
    cells = [(f"**{r['test'][c]:.3f}**" if abs(r["test"][c] - best[c]) < 1e-9 else f"{r['test'][c]:.3f}") for c in cols]
    tr = f"{r['train_seconds'] / 60:.0f} min" if "train_seconds" in r else "—"
    lines.append(f"| {NAMES[n]} | " + " | ".join(cells) + f" | {tr} | {r.get('infer_ms_per_text', '—')} |")
w = json.loads((ROOT / "artifacts/ensemble.json").read_text())["weights"]
lines.append("")
lines.append("Ensemble weights (searched on validation): " + ", ".join(f"{k} {v:.1f}" for k, v in w.items() if v > 0) + ".")
readme = (ROOT / "README.md").read_text()


def fill(tag, body):
    global readme
    readme = re.sub(rf"<!-- {tag} -->.*?<!-- /{tag} -->|<!-- {tag} -->", f"<!-- {tag} -->\n{body}\n<!-- /{tag} -->", readme,
                    flags=re.S)


fill("RESULTS_TABLE", "\n".join(lines))
ch = ROOT / "reports/challenge.json"
if ch.exists():
    s = json.loads(ch.read_text())["summary"]
    fill("CHALLENGE", f"**Out-of-distribution check.** {s['n']} hand-written SOC alerts in formats never seen in training "
         f"(EDR, firewall, Zeek, AWS GuardDuty, Azure AD; `data/challenge/`): **P@1 {s['P@1']:.0%} · Hit@3 {s['Hit@3']:.0%} · "
         f"Hit@5 {s['Hit@5']:.0%} · Tactic@1 {s['Tactic@1']:.0%}**. Per-alert results are in `reports/challenge.json`.")
if len(sys.argv) > 1:
    fill("TESTS", f"Latest run: `{sys.argv[1]}`")
(ROOT / "README.md").write_text(readme)
print("README updated")
