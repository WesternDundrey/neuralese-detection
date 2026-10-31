#!/bin/bash
# The late-layer hiding test: can the twins hide link activity at link layer 27 and the final layer while the city
# still passes? Longer runs (900 steps) on the merged chat data, an MLP monitor at those two depths only, the penalty
# on each chat's mean score, three seeds at lam 1 and one control at lam 0. Every checkpoint gets hide_eval.py (fresh
# probes at every depth, information, link statistics); the last one also source_control.py (knowledge from the prompt
# and a stranger partner). Each finished stage leaves a .DONE file, so rerunning resumes. Run it in tmux.
# Usage (from the repo root): bash detection/run_hide_late.sh INIT_STEP_DIR
set -u
INIT=$1
R=/workspace/runs/hide_late
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/.hf_home NL_DEVICE=cuda PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TEST=/workspace/runs/data/chat.json DATA=/workspace/runs/data/chat_merged.json CHATS=/workspace/runs/where_read.json
mkdir -p $R/eval $R/source
note() { echo "$(date '+%F %T') $1" | tee -a $R/chain.log; }
stage() { local name=$1; shift; [ -f $R/$name.DONE ] && return 0
    note "$name start"; "$@" > $R/$name.log 2>&1 && touch $R/$name.DONE; local s=$?; note "$name exit $s"; return $s; }

[ -s $DATA ] || $PY -c "
import json, sys
parts = [json.load(open(p)) for p in sys.argv[2:]]
json.dump({'train': [c for p in parts for c in p['train']], 'test': parts[0]['test']}, open(sys.argv[1], 'w'))
print({'train': sum(len(p['train']) for p in parts)})" $DATA $TEST /workspace/runs/data/chat_s1.json \
    /workspace/runs/data/chat_s2.json | tee -a $R/chain.log

for RUN in lam1_s0:1:0 lam0_s0:0:0 lam1_s1:1:1 lam1_s2:1:2; do
    IFS=: read NAME LAM SEED <<< "$RUN"
    stage train_$NAME $PY detection/hide.py --init $INIT --chat_data $DATA --out $R/$NAME --lam $LAM --seed $SEED \
        --steps 900 --save_every 300 --depths 27,final --probe mlp --chat_level || continue
    for S in 300 600 900; do
        stage eval_${NAME}_$S $PY detection/hide_eval.py --run $R/$NAME/step_$S --chat_data $TEST --chats $CHATS \
            --out $R/eval/${NAME}_$S.json
    done
    stage source_$NAME $PY detection/source_control.py --run $R/$NAME/step_900 --chat_data $TEST \
        --out $R/source/$NAME.json
done
note "all done"
