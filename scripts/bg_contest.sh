#!/usr/bin/env bash
# Contest-world job as a detached process (linger is disabled on this machine). Holds results/.contest.lock for its
# whole life, so a second contest program cannot start by accident (one program per token). Edit the A/B line to
# change what plays. live_ab.sh is resumable (finished runs are skipped) and then keeps playing $AB_PLAY_LOAD.
cd /Data/prakhar.tiwari/Reinforcement_Learning_Challenge
set -a; . ./.env; set +a
export AB_LOG=results/contest_ab_live8.log AB_PLAY_LOAD=data/cand/coplay_143k.pt
exec flock -n results/.contest.lock src/train/live_ab.sh 10000 3 data/cand/coplay2_500k.pt data/cand/coplay_143k.pt:as10 data/cand/coplay_143k.pt data/cand/coplay2_500k.pt:as10 >> results/contest_ab_live8.log 2>&1
