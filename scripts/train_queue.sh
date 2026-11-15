#!/usr/bin/env bash
# Train the heavy models one after another (CPU-only box: never run two at once).
set -u
cd "$(dirname "$0")/.."
WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done; fi
for job in "bert --epochs 3" "sbert" "bilstm"; do
  name=${job%% *}
  echo "$(date +%T) start $name"
  python -u scripts/train.py $job > artifacts/logs/$name.log 2>&1
  echo "$(date +%T) done $name rc=$?"
done
