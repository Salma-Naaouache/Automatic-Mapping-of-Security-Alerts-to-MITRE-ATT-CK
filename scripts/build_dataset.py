"""Step 1 — download sources and build the labelled corpus."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attackmap.data.build import build  # noqa: E402

rows, labels = build()
print(f"rows={len(rows)} labels={len(labels)}")
print("by split :", dict(Counter(r["split"] for r in rows)))
print("by source:", dict(Counter(r["source"] for r in rows)))
print("multi-label rows:", sum(len(r["labels"]) > 1 for r in rows))
