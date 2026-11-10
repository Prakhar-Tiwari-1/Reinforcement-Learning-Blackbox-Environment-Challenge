#!/usr/bin/env bash
# Play the best policy in the contest world for a fixed time, then stop. Detached-safe (setsid), one program per
# token (results/.contest.lock), restarts after a crash until the deadline. Usage: scripts/bg_play_for.sh SECONDS
cd /Data/prakhar.tiwari/Reinforcement_Learning_Challenge
set -a; . ./.env; set +a
deadline=$(( $(date +%s) + ${1:?seconds} ))
echo "$(date '+%F %T') play-for: co-player 143k + anti-stall until $(date -d @$deadline '+%F %T')" >> results/contest_live.log
while [ "$(date +%s)" -lt "$deadline" ]; do
  left=$(( deadline - $(date +%s) ))
  timeout -s INT -k 120 "$left" flock -n results/.contest.lock .venv/bin/python src/contest_record.py --name play143k_as10 \
    --load data/cand/coplay_143k.pt --anti-stall 10 --log-every 5000 >> results/contest_live.log 2>&1
  [ "$(date +%s)" -lt "$deadline" ] && sleep 30
done
echo "$(date '+%F %T') play-for: deadline reached, stopped" >> results/contest_live.log
