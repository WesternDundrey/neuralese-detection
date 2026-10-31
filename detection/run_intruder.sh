#!/bin/bash
# intruder.py (can a twin tell something foreign is on its link, and does it say so?) on the starting twins, the
# lam 0 control, the first sweep's lam 1 and the information-weighted hiding run. Resumable (.DONE files). Run in tmux.
# Usage (from the repo root): bash detection/run_intruder.sh INIT_STEP_DIR
set -u
INIT=$1
R=/workspace/runs/intruder
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/.hf_home NL_DEVICE=cuda PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $R
note() { echo "$(date '+%F %T') $1" | tee -a $R/chain.log; }
stage() { local name=$1; shift; [ -f $R/$name.DONE ] && return 0
    note "$name start"; "$@" > $R/$name.log 2>&1 && touch $R/$name.DONE; local s=$?; note "$name exit $s"; return $s; }
for RUN in start:$INIT late_lam1_pay10:/workspace/runs/hide_late/lam1_pay10/step_900 \
           late_lam0_s0:/workspace/runs/hide_late/lam0_s0/step_900 sweep_lam1_300:/workspace/runs/hide/lam1/step_300; do
    NAME=${RUN%%:*}; DIR=${RUN#*:}
    stage $NAME $PY detection/intruder.py --run $DIR --chat_data /workspace/runs/data/chat.json \
        --chats /workspace/runs/where_read.json --out $R/$NAME.json
done
note "all done"
