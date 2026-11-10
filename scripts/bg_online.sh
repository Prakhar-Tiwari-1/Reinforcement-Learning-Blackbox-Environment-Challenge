#!/usr/bin/env bash
# Online fine-tuning IN THE CONTEST WORLD (src/train/ddqn_contest_online.py), detached (linger is disabled here).
# Holds results/.contest.lock (one program per token); restarts after a crash and resumes from runs/online143k/state.pt.
cd /Data/prakhar.tiwari/Reinforcement_Learning_Challenge
set -a; . ./.env; set +a
while true; do
  flock -n results/.contest.lock .venv/bin/python src/train/ddqn_contest_online.py --name online143k \
    --init data/cand/coplay_143k.pt --old runs/coplay/buffer.npz --revert-below 32 >> results/jobs/online143k.log 2>&1
  code=$?; [ $code -eq 1 ] && grep -q "AuthenticationError\|PermissionDenied" results/jobs/online143k.log && break
  echo "$(date '+%F %T') online fine-tuning exited ($code); restarting in 60 s" >> results/jobs/online143k.log; sleep 60
done
