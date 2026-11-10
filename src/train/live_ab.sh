#!/usr/bin/env bash
# Live A/B test in the CONTEST world with recording: each candidate plays STEPS steps per round (order reversed every
# other round), recorded to data/live/ab_<tag>_*.npz. Output lines 'round R | ckpt | final: ...' (src/train/pool_compare.py).
# After the rounds, keeps playing the submission policy with recording (data/live/play_*.npz).
# Resumable: a run already reported in LOG ('round R | ckpt | final:') is skipped, so after a crash or a restart of
# the unit it continues where it stopped (run it with Restart=always).
# Usage: src/train/live_ab.sh STEPS ROUNDS ckpt1 ckpt2 ...      (run as the only contest program)
cd /Data/prakhar.tiwari/Reinforcement_Learning_Challenge
LOG=${AB_LOG:-results/contest_ab_live.log}
steps=$1; rounds=$2; shift 2; cands=("$@")
for ((r=1; r<=rounds; r++)); do
  if (( r % 2 == 0 )); then order=($(printf '%s\n' "${cands[@]}" | tac)); else order=("${cands[@]}"); fi
  for c in "${order[@]}"; do
    ckpt=${c%%:*}; opt=""; tag=$(basename "$ckpt" .pt)
    if [[ "$c" == *:as* ]]; then k=${c##*:as}; opt="--anti-stall $k"; tag="${tag}_as$k"; fi   # 'ckpt.pt:as10' = anti-stall 10
    if grep -qF "round $r | $c | final:" "$LOG" 2>/dev/null; then continue; fi
    for attempt in 1 2 3; do       # contest_record.py already waits out server problems; this covers crashes
      line=$(.venv/bin/python src/contest_record.py --name "ab_$tag" --load "$ckpt" $opt --steps "$steps" --log-every "$steps" 2>&1 | grep final:)
      [ -n "$line" ] && break
      sleep 30
    done
    [ -n "$line" ] && echo "round $r | $c | $line"
  done
done
echo "### DONE"
# then keep playing (recorded): $AB_PLAY_LOAD if set (a checkpoint), else the policy embedded in prakhar.tiwari.py
if [ -n "$AB_PLAY_LOAD" ]; then
  exec .venv/bin/python src/contest_record.py --name "play_$(basename "$AB_PLAY_LOAD" .pt)" --load "$AB_PLAY_LOAD" --log-every 5000
fi
exec .venv/bin/python src/contest_record.py --name play --log-every 5000
