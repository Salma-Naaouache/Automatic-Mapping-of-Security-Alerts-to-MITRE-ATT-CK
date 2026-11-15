"""Step 4 — publication figures in reports/figures/ (run after scripts/ensemble.py)."""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attackmap.config import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR  # noqa: E402
from attackmap.data.attack_kb import KnowledgeBase  # noqa: E402
from attackmap.data.build import load_corpus, load_labels  # noqa: E402
from attackmap.evaluate import multi_hot  # noqa: E402

# Validated reference palette (light mode) — categorical slots in fixed order.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("seq", ["#f2f6fc", "#a9c8ef", "#2a78d6", "#0d3c78"])
NAMES = {"tfidf": "TF-IDF + LogReg", "bilstm": "BiLSTM + attention", "sbert_zeroshot": "SBERT zero-shot",
         "sbert": "SBERT fine-tuned", "roberta": "RoBERTa", "bert": "BERT", "ensemble": "Ensemble"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 12.5,
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=200)
    plt.close(fig)
    print("wrote", FIGURES_DIR / name)


res = json.loads((REPORTS_DIR / "results.json").read_text())
rows, labels, kb = load_corpus(), load_labels(), KnowledgeBase.load()
test = [r for r in rows if r["split"] == "test"]
Y = multi_hot([r["labels"] for r in test], labels)
ens = np.load(ARTIFACTS_DIR / "ensemble_scores_test.npy")
order = [n for n in NAMES if n in res]

# 1 — model comparison --------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(order) + 1.4))
y = np.arange(len(order))[::-1]
h = 0.26
for k, (metric, col) in enumerate([("P@1", S1), ("Hit@5", S2), ("MRR", S3)]):
    vals = [res[n]["test"][metric] for n in order]
    ax.barh(y + (1 - k) * (h + 0.02), vals, height=h, color=col, label=metric)
    if metric == "P@1":
        for yi, v in zip(y, vals):
            ax.text(v + 0.008, yi + h + 0.02, f"{v:.1%}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, [NAMES[n] for n in order])
for t in ax.get_yticklabels():
    if t.get_text() == "Ensemble":
        t.set_fontweight("bold")
ax.set_xlim(0, 1.05)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
ax.set_title("Held-out test set: ranking quality by model", pad=26)
save(fig, "model_comparison.png")

# 2 — per data source ---------------------------------------------------------
srcs = {"attack_procedure": "ATT&CK procedure\nexamples (CTI)", "tram_cti": "TRAM CTI\nreport sentences",
        "sigma_siem": "Sigma SIEM\nalerts"}
fig, ax = plt.subplots(figsize=(7.5, 3.6))
models_src = [n for n in ("tfidf", "roberta", "ensemble") if n in res]
w = 0.26
for k, (n, col) in enumerate(zip(models_src, (S1, S2, S3))):
    vals = [res[n]["test_by_source"][s]["P@1"] for s in srcs]
    xs = np.arange(len(srcs)) + (k - (len(models_src) - 1) / 2) * (w + 0.02)
    ax.bar(xs, vals, width=w, color=col, label=NAMES[n])
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.01, f"{v:.0%}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(len(srcs)), srcs.values())
ax.set_ylim(0, 1.08)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.13))
ax.set_title("P@1 by input type", pad=24)
save(fig, "per_source.png")

# 3 — per-tactic F1 (top-1 tactic assignment) --------------------------------
top = ens.argmax(1)
tac_rows = []
for tac in kb.tactics:
    gold = np.array([any(tac.shortname in kb.tactics_of(l) for l in r["labels"]) for r in test])
    pred = np.array([tac.shortname in kb.tactics_of(labels[j]) for j in top])
    if gold.sum() < 5:
        continue
    tp = (gold & pred).sum()
    f1 = 2 * tp / max(gold.sum() + pred.sum(), 1)
    tac_rows.append((tac.name, f1, int(gold.sum())))
tac_rows.sort(key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(tac_rows) + 1.2))
ax.barh([t[0] for t in tac_rows], [t[1] for t in tac_rows], color=S1, height=0.62)
for i, (_, f1, n) in enumerate(tac_rows):
    ax.text(f1 + 0.01, i, f"{f1:.2f}  (n={n})", va="center", fontsize=8.5, color=INK2)
ax.set_xlim(0, 1.15)
ax.grid(axis="y", visible=False)
ax.set_title("Ensemble F1 per ATT&CK tactic (test)")
save(fig, "tactic_f1.png")

# 4 — training curves ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.6))
drawn = 0
for n, col in zip(("bert", "roberta", "bilstm"), (S1, S2, S3)):
    meta = ARTIFACTS_DIR / n / "meta.json"
    if not meta.exists():
        continue
    hist = json.loads(meta.read_text()).get("history", [])
    if not hist:
        continue
    ep, mrr = [h["epoch"] for h in hist], [h["MRR"] for h in hist]
    ax.plot(ep, mrr, color=col, lw=2, marker="o", ms=6, mec=SURF, mew=2, label=NAMES[n])
    ax.annotate(NAMES[n], (ep[-1], mrr[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9, color=INK2)
    drawn += 1
if drawn:
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MRR")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(loc="lower right")
    ax.set_title("Learning curves")
    save(fig, "training_curves.png")
else:
    plt.close(fig)

# 5 — label distribution ------------------------------------------------------
cnt = Counter(l for r in rows for l in r["labels"])
vals = sorted(cnt.values(), reverse=True)
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.bar(range(len(vals)), vals, color=S1, width=0.8)
ax.set_yscale("log")
ax.set_xlabel(f"technique rank (of {len(vals)} labels)")
ax.set_ylabel("examples (log)")
for i in range(3):
    tid = cnt.most_common(3)[i][0]
    ax.annotate(f"{tid} {kb.name_of(tid)}", (i, vals[i]), xytext=(18, -6 - 13 * i), textcoords="offset points",
                fontsize=8.5, color=INK2, arrowprops={"arrowstyle": "-", "color": GRID})
ax.grid(axis="x", visible=False)
ax.set_title(f"Long-tailed label space: {len(rows):,} texts, {len(vals)} techniques")
save(fig, "label_distribution.png")

# 6 — tactic confusion (primary tactic of gold vs prediction) -----------------
tnames = [t.shortname for t in kb.tactics]
M = np.zeros((len(tnames), len(tnames)))
for r, j in zip(test, top):
    g = kb.tactics_of(r["labels"][0])
    p = kb.tactics_of(labels[j])
    if g and p:
        M[tnames.index(g[0]), tnames.index(p[0])] += 1
keep = M.sum(1) > 0
M, tn = M[keep][:, keep], [t.name for t, k in zip(kb.tactics, keep) if k]
Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(Mn, cmap=BLUES, vmin=0, vmax=1)
ax.set_xticks(range(len(tn)), tn, rotation=55, ha="right", fontsize=8.5)
ax.set_yticks(range(len(tn)), tn, fontsize=8.5)
ax.grid(False)
for i in range(len(tn)):
    for j in range(len(tn)):
        if Mn[i, j] >= 0.05:
            ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if Mn[i, j] > 0.55 else INK)
ax.set_xlabel("predicted (primary tactic of top-1 technique)")
ax.set_ylabel("gold (primary tactic)")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="row share")
ax.set_title("Tactic-level confusion (row-normalised, test)")
save(fig, "tactic_confusion.png")
