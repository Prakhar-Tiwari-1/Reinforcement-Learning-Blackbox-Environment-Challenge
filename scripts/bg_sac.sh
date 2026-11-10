#!/usr/bin/env bash
# SAC-Discrete on the TRAINING server only (src/train/sac_discrete.py), detached (linger is disabled on this machine).
# Phase 1: 200k steps from scratch (run "sac"). Phase 2: 200k more steps (run "sac2") starting from the phase-1
# checkpoint with the best training-world score (top of the trainer's best list), reusing phase 1's replay buffer.
# Each phase restarts after a crash and resumes from runs/<name>/latest.pt.
cd /Data/prakhar.tiwari/Reinforcement_Learning_Challenge
set -a; . ./.env; set +a
LOG=results/jobs/sac.log
until [ -f runs/sac/DONE ]; do
  .venv/bin/python src/train/sac_discrete.py --name sac --total-steps 200000 >> $LOG 2>&1
  [ -f runs/sac/DONE ] || { echo "$(date '+%F %T') sac exited without DONE; restarting in 60 s" >> $LOG; sleep 60; }
done
BEST=$(.venv/bin/python -c "import torch; b=torch.load('runs/sac/latest.pt', weights_only=False)['trainer']['best']; print('runs/sac/' + b[0][1])")
echo "$(date '+%F %T') phase 2: continuing from $BEST (best training-world score in run sac)" >> $LOG
until [ -f runs/sac2/DONE ]; do
  .venv/bin/python src/train/sac_discrete.py --name sac2 --init "$BEST" --prefill false --buffer-from runs/sac/buffer.npz \
    --total-steps 200000 >> $LOG 2>&1
  [ -f runs/sac2/DONE ] || { echo "$(date '+%F %T') sac2 exited without DONE; restarting in 60 s" >> $LOG; sleep 60; }
done
