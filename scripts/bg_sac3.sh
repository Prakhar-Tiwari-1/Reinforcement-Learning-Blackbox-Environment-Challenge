#!/usr/bin/env bash
# Phase 3 of the SAC run, TRAINING server only: 100k more steps (run "sac3") from the best training-world checkpoint over
# sac and sac2, with a replay buffer that keeps all experience so far (phase 1's buffer + phase 2's own rows, ~1.8M,
# capacity 3M so nothing is dropped). Waits for the running evaluation (PID $1) to end first: one program per server.
cd /Data/prakhar.tiwari/Reinforcement_Learning_Challenge
set -a; . ./.env; set +a
LOG=results/jobs/sac.log
while [ -n "$1" ] && kill -0 "$1" 2>/dev/null; do sleep 10; done
BEST=$(.venv/bin/python -c "
import torch
c = [(s, f'runs/{r}/{n}') for r in ('sac', 'sac2') for s, n in torch.load(f'runs/{r}/latest.pt', weights_only=False)['trainer']['best']]
print(max(c)[1])")
echo "$(date '+%F %T') phase 3: continuing from $BEST (best training-world score over sac and sac2), 100k steps" >> $LOG
[ -f runs/sac_merged_buffer.npz ] || [ -f runs/sac3/latest.pt ] || \
  .venv/bin/python src/train/merge_buffers.py runs/sac_merged_buffer.npz runs/sac/buffer.npz runs/sac2/buffer.npz >> $LOG 2>&1
until [ -f runs/sac3/DONE ]; do
  .venv/bin/python src/train/sac_discrete.py --name sac3 --init "$BEST" --prefill false --buffer-from runs/sac_merged_buffer.npz \
    --buffer 3000000 --total-steps 100000 >> $LOG 2>&1
  [ -f runs/sac3/DONE ] || { echo "$(date '+%F %T') sac3 exited without DONE; restarting in 60 s" >> $LOG; sleep 60; }
done
