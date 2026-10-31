#!/bin/bash
# The original twins: lookup or communication (compute.py), the translator (translator.py) on the original twins and
# on later checkpoints (does readability drift?), then where "linked twins are happier" comes from (linkedness.py). Waits for run_next.sh. Resumable (.DONE files). Run in tmux.
# Usage (from the repo root): bash detection/run_twins.sh INIT_STEP_DIR
set -u
INIT=$1
R=/workspace/runs/twins
CHATS=/workspace/runs/where_read.json
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/.hf_home NL_DEVICE=cuda PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $R
note() { echo "$(date '+%F %T') $1" | tee -a $R/chain.log; }
stage() { local name=$1; shift; [ -f $R/$name.DONE ] && return 0
    note "$name start"; "$@" > $R/$name.log 2>&1 && touch $R/$name.DONE; local s=$?; note "$name exit $s"; return $s; }
until grep -q "all done" /workspace/runs/next/chain.log 2>/dev/null; do sleep 60; done
stage compute $PY detection/compute.py --run $INIT --chats /workspace/runs/where_read.json --out $R/compute.json
# the translator: the original twins first (the others are also read with its translator), then the drift without a
# monitor (the lam 0 runs: 300 steps on the original chat data; 300/600/900 steps on the merged data), then the runs
# trained against a monitor
T=$R/translator
mkdir -p $T
stage tr_start $PY detection/translator.py --run $INIT --chats $CHATS --out $T/start.json
for RUN in sweep_lam0_300:/workspace/runs/hide/lam0/step_300 late_lam0_300:/workspace/runs/hide_late/lam0_s0/step_300 \
           late_lam0_600:/workspace/runs/hide_late/lam0_s0/step_600 late_lam0_900:/workspace/runs/hide_late/lam0_s0/step_900 \
           sweep_lam1_300:/workspace/runs/hide/lam1/step_300 late_pay10_900:/workspace/runs/hide_late/lam1_pay10/step_900 \
           late_lam1_s0_900:/workspace/runs/hide_late/lam1_s0/step_900; do
    stage tr_${RUN%%:*} $PY detection/translator.py --run ${RUN#*:} --chats $CHATS --out $T/${RUN%%:*}.json \
        --transfer_from $T/start.json.writes.pt
done
stage linkedness $PY detection/linkedness.py --run $INIT --out $R/linkedness.json --n 80
note "all done"
