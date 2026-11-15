#!/usr/bin/env bash
# Second training round + reporting. Waits for train_queue.sh, then:
# extra transformer epochs from the saved checkpoints, ensemble, figures, OOD check.
set -u
cd "$(dirname "$0")/.."
while pgrep -f train_queue.sh >/dev/null; do sleep 20; done
log() { echo "$(date +%T) $*"; }
log "bilstm";         python -u scripts/train.py bilstm > artifacts/logs/bilstm.log 2>&1;                log "rc=$?"
log "sbert";          python -u scripts/train.py sbert > artifacts/logs/sbert.log 2>&1;                  log "rc=$?"
log "resume bert";    python -u scripts/train.py bert --resume --epochs 3    > artifacts/logs/bert_resume.log 2>&1;    log "rc=$?"
log "resume roberta"; python -u scripts/train.py roberta --resume --epochs 2 > artifacts/logs/roberta_resume.log 2>&1; log "rc=$?"
log "ensemble";       python scripts/ensemble.py     > artifacts/logs/ensemble.log 2>&1;  log "rc=$?"
log "figures";        python scripts/make_figures.py > artifacts/logs/figures.log 2>&1;   log "rc=$?"
log "challenge";      python scripts/challenge.py    > artifacts/logs/challenge.log 2>&1; log "rc=$?"
log "ALL DONE"
