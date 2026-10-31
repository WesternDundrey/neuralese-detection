#!/bin/bash
# Is the information-weighted twin's "yes, something else is on my link" a belief about the link or a general Yes
# bias? intruder.py --report_only with the link question and three yes/no controls (sky, model, watched) on the
# starting twins, the lam 0 control, the information-weighted run and a collapsed lam 1 run. Waits for run_intruder.sh.
# Usage (from the repo root): bash detection/run_intruder_controls.sh INIT_STEP_DIR
set -u
INIT=$1
R=/workspace/runs/intruder
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/.hf_home NL_DEVICE=cuda PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date '+%F %T') $1" | tee -a $R/chain.log; }
stage() { local name=$1; shift; [ -f $R/$name.DONE ] && return 0
    note "$name start"; "$@" > $R/$name.log 2>&1 && touch $R/$name.DONE; local s=$?; note "$name exit $s"; return $s; }
until grep -q "all done" $R/chain.log; do sleep 60; done
for RUN in late_lam1_pay10:/workspace/runs/hide_late/lam1_pay10/step_900 start:$INIT \
           late_lam0_s0:/workspace/runs/hide_late/lam0_s0/step_900 late_lam1_s0:/workspace/runs/hide_late/lam1_s0/step_900; do
    NAME=${RUN%%:*}; DIR=${RUN#*:}
    stage controls_$NAME $PY detection/intruder.py --run $DIR --chat_data /workspace/runs/data/chat.json \
        --chats /workspace/runs/where_read.json --report_only --questions tap,sky,model,watched \
        --out $R/controls_$NAME.json
done
note "controls done"
