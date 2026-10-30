#!/bin/bash
# The hiding-game sweep, one GPU: evaluates the starting twins, then trains and evaluates one run per lam (0 is the
# control: the same chat training without the monitor). Each finished stage leaves a .DONE file, so rerunning the
# script resumes where it stopped. Run it in tmux; progress goes to $R/chain.log.
# Usage (from the repo root): bash detection/run_hide.sh INIT_STEP_DIR [LAMS...]   (default lams: 0 0.25 1 4)
set -u
INIT=$1; shift
LAMS=${*:-0 0.25 1 4}
R=/workspace/runs/hide
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/.hf_home NL_DEVICE=cuda PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=/workspace/runs/data/chat.json CHATS=/workspace/runs/where_read.json
mkdir -p $R/eval
note() { echo "$(date '+%F %T') $1" | tee -a $R/chain.log; }
stage() { local name=$1; shift; [ -f $R/$name.DONE ] && return 0
    note "$name start"; "$@" > $R/$name.log 2>&1 && touch $R/$name.DONE; local s=$?; note "$name exit $s"; return $s; }

stage eval_start $PY detection/hide_eval.py --run $INIT --chat_data $DATA --chats $CHATS --out $R/eval/start.json
for L in $LAMS; do
    stage train_lam$L $PY detection/hide.py --init $INIT --chat_data $DATA --out $R/lam$L --lam $L || continue
    for S in 100 200 300; do
        stage eval_lam${L}_$S $PY detection/hide_eval.py --run $R/lam$L/step_$S --chat_data $DATA --chats $CHATS \
            --out $R/eval/lam${L}_$S.json
    done
done
note "all done"
