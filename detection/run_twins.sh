#!/bin/bash
# The original twins: lookup or communication (compute.py), then where "linked twins are happier" comes from
# (linkedness.py). Waits for run_next.sh. Resumable (.DONE files). Run in tmux.
# Usage (from the repo root): bash detection/run_twins.sh INIT_STEP_DIR
set -u
INIT=$1
R=/workspace/runs/twins
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/.hf_home NL_DEVICE=cuda PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $R
note() { echo "$(date '+%F %T') $1" | tee -a $R/chain.log; }
stage() { local name=$1; shift; [ -f $R/$name.DONE ] && return 0
    note "$name start"; "$@" > $R/$name.log 2>&1 && touch $R/$name.DONE; local s=$?; note "$name exit $s"; return $s; }
until grep -q "all done" /workspace/runs/next/chain.log 2>/dev/null; do sleep 60; done
stage compute $PY detection/compute.py --run $INIT --chats /workspace/runs/where_read.json --out $R/compute.json
stage linkedness $PY detection/linkedness.py --run $INIT --out $R/linkedness.json --n 80
note "all done"
