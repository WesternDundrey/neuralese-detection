#!/bin/bash
# After the late-layer hiding test (run_hide_late.sh, seed 2 skipped): the planted-signal test (plant.py) on the
# starting twins and the hiding checkpoints, then the information-weighted hiding run (the payload cross-entropy x10, so
# quitting the channel costs more than hiding it), its evaluations and its planted-signal test. Waits for run_hide_late
# to finish; each finished stage leaves a .DONE file, so rerunning resumes. Run it in tmux.
# Usage (from the repo root): bash detection/run_after_late.sh INIT_STEP_DIR
set -u
INIT=$1
R=/workspace/runs/hide_late
P=/workspace/runs/plant
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/.hf_home NL_DEVICE=cuda PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TEST=/workspace/runs/data/chat.json DATA=/workspace/runs/data/chat_merged.json CHATS=/workspace/runs/where_read.json
mkdir -p $P
note() { echo "$(date '+%F %T') $1" | tee -a $R/chain.log; }
stage() { local name=$1; shift; [ -f $R/$name.DONE ] && return 0
    note "$name start"; "$@" > $R/$name.log 2>&1 && touch $R/$name.DONE; local s=$?; note "$name exit $s"; return $s; }
until grep -q "all done" $R/chain.log; do sleep 60; done

for RUN in start:$INIT sweep_lam1_300:/workspace/runs/hide/lam1/step_300 late_lam0_s0:$R/lam0_s0/step_900 \
           late_lam1_s0:$R/lam1_s0/step_900 late_lam1_s1:$R/lam1_s1/step_900; do
    NAME=${RUN%%:*}; DIR=${RUN#*:}
    stage plant_$NAME $PY detection/plant.py --run $DIR --chats $CHATS --out $P/$NAME.json
done

NAME=lam1_pay10
stage train_$NAME $PY detection/hide.py --init $INIT --chat_data $DATA --out $R/$NAME --lam 1 --seed 0 --pay_weight 10 \
    --steps 900 --save_every 300 --depths 27,final --probe mlp --chat_level || exit 1
for S in 300 600 900; do
    stage eval_${NAME}_$S $PY detection/hide_eval.py --run $R/$NAME/step_$S --chat_data $TEST --chats $CHATS \
        --out $R/eval/${NAME}_$S.json
done
stage source_$NAME $PY detection/source_control.py --run $R/$NAME/step_900 --chat_data $TEST --out $R/source/$NAME.json
stage plant_late_$NAME $PY detection/plant.py --run $R/$NAME/step_900 --chats $CHATS --out $P/late_$NAME.json
note "after done"
