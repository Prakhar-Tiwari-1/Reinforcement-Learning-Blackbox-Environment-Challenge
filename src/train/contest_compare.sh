#!/usr/bin/env bash
# Head-to-head comparison of checkpoints in the CONTEST world (frozen, greedy), alternating order per round.
# Usage: src/train/contest_compare.sh STEPS ROUNDS ckpt1 ckpt2 ...   -> results/contest_compare.log
cd /Data/prakhar.tiwari/Reinforcement_Learning_Challenge
steps=$1; rounds=$2; shift 2; cands=("$@")
for ((r=1; r<=rounds; r++)); do
  if (( r % 2 == 0 )); then order=($(printf '%s\n' "${cands[@]}" | tac)); else order=("${cands[@]}"); fi
  for c in "${order[@]}"; do
    line=$(.venv/bin/python prakhar.tiwari.py --no-learn --steps "$steps" --log-every "$steps" --load "$c" 2>&1 | grep final:)
    echo "round $r | $c | $line"
  done
done
echo "### DONE"
