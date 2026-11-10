#!/usr/bin/env bash
# Phased co-player DDQN run (20 Oct), as a detached process (linger is disabled on this machine, so systemd --user jobs
# stop at logout; processes started with setsid survive it). Restarts after a crash and resumes from runs/coplay2/latest.pt.
cd /Data/prakhar.tiwari/Reinforcement_Learning_Challenge
set -a; . ./.env; set +a
until [ -f runs/coplay2/DONE ]; do
  .venv/bin/python src/train/ddqn.py --name coplay2 --init data/cand/coplay_500k.pt --prefill false \
    --buffer-from runs/coplay/buffer.npz --lr 5e-5 --eps-start 0.02 --eps-end 0.01 --eps-steps 50000 \
    --total-steps 500000 --ckpt-every 20480 --best-window 20480 --keep-last 30 --n-learners 2 --coplayer-switch 5000 \
    --phase-steps 100000 --pool-schedule runs/coplay2_pools.json --coplayers self \
    --died-weight 2.0 --food-weight 1.2 --w-danger 1.0 >> results/jobs/coplay2.log 2>&1
  [ -f runs/coplay2/DONE ] || { echo "$(date '+%F %T') ddqn exited without DONE; restarting in 60 s" >> results/jobs/coplay2.log; sleep 60; }
done
