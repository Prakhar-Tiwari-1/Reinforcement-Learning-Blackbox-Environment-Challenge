#!/usr/bin/env bash
# One command to run everything in the background, safely.
#
# Jobs run as systemd *user services*: they keep running when you close the terminal, VS Code, or SSH,
# or lose the network (your account has "linger" enabled, so they even run when you are fully logged out).
# Training jobs restart automatically after a crash and resume from their last checkpoint.
#
#   scripts/rl.sh train NAME [train.py options]   start/resume a training run (training server)
#   scripts/rl.sh job NAME SCRIPT [args]          run any python script (training server) as a safe background job
#   scripts/rl.sh contest [prakhar.tiwari.py options]  play in the contest world (default: --no-learn)
#   scripts/rl.sh select NAME [choose_checkpoint.py options] evaluate a run's best checkpoints in the contest world
#   scripts/rl.sh status                          what is running + last progress line of every run
#   scripts/rl.sh logs NAME                       progress table of a run (reward, food, deaths, damage)
#   scripts/rl.sh follow NAME|contest             live output (Ctrl-C to stop watching; the job goes on)
#   scripts/rl.sh stop NAME|contest|select-NAME   stop a job cleanly (it saves before exiting)
#   scripts/rl.sh prune [--yes]                   free disk space (dry run without --yes)
#   scripts/rl.sh scores [N]                      last N contest-world scores found in results/ (default 10)
#
# Examples:
#   scripts/rl.sh train v2 --init runs/v2_init.pt --total-steps 1500000
#   scripts/rl.sh logs v2
#   scripts/rl.sh stop v2 && scripts/rl.sh select v2 --embed
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
cmd="${1:-status}"; shift || true

unit_active() { systemctl --user is-active "$1" >/dev/null 2>&1; }
# Jobs survive logout / SSH disconnects only if "linger" is on for this user: check (and fix) before anything else.
ensure_linger() {
  if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
    loginctl enable-linger "$USER" 2>/dev/null
    if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = "yes" ]; then
      echo "note: linger was off; enabled it (jobs now survive logout)"
    else
      echo "WARNING: linger is OFF and could not be enabled: jobs will stop when you log out (ask an admin)"
    fi
  fi
}
case "${1:-status}" in train|job|contest|record|select|status) ensure_linger ;; esac

case "$cmd" in
  train)
    name="${1:?usage: rl.sh train NAME [options]}"; shift
    for u in $(systemctl --user list-units 'rl-train-*' 'rl-job-*' --state=active --no-legend --plain | awk '{print $1}'); do
      echo "another training-server job is running ($u): only one program per token on the training server."; exit 1
    done
    mkdir -p "runs/$name"
    # --no-block: return immediately; Restart=on-failure: crash -> restart in 60 s -> resumes from latest.pt
    systemd-run --user --unit="rl-train-$name" --working-directory="$ROOT" \
      -p Restart=on-failure -p RestartSec=60 -p StartLimitIntervalSec=0 \
      /bin/bash -c "exec '$PY' src/train/train.py --name '$name' $* >> 'runs/$name/stdout.log' 2>&1"
    echo "started rl-train-$name; progress: scripts/rl.sh logs $name"
    ;;
  job)
    name="${1:?usage: rl.sh job NAME SCRIPT [args]}"; script="${2:?}"; shift 2
    for u in $(systemctl --user list-units 'rl-train-*' 'rl-job-*' --state=active --no-legend --plain | awk '{print $1}'); do
      echo "another training-server job is running ($u): only one program per token."; exit 1
    done
    mkdir -p results/jobs
    systemd-run --user --unit="rl-job-$name" --working-directory="$ROOT" \
      -p Restart=on-failure -p RestartSec=60 -p StartLimitIntervalSec=0 \
      /bin/bash -c "exec '$PY' '$script' $* >> 'results/jobs/$name.log' 2>&1"
    echo "started rl-job-$name; output: results/jobs/$name.log"
    ;;
  contest)
    args="${*:---no-learn}"
    systemd-run --user --unit=rl-contest --working-directory="$ROOT" \
      -p Restart=always -p RestartSec=30 -p StartLimitIntervalSec=0 \
      /bin/bash -c "exec '$PY' prakhar.tiwari.py $args --log-every 500 >> results/contest_live.log 2>&1"
    echo "started rl-contest ($args); output: results/contest_live.log  (scripts/rl.sh follow contest)"
    ;;
  select)
    name="${1:?usage: rl.sh select NAME [options]}"; shift
    if unit_active rl-contest; then echo "stop the contest player first: scripts/rl.sh stop contest"; exit 1; fi
    systemd-run --user --unit="rl-select-$name" --working-directory="$ROOT" \
      /bin/bash -c "exec '$PY' src/train/choose_checkpoint.py --run '$name' $* >> 'runs/$name/selection.log' 2>&1"
    echo "started rl-select-$name; output: runs/$name/selection.log"
    ;;
  status)
    echo "== linger (jobs survive logout): $(loginctl show-user "$USER" -p Linger --value)"
    echo "== background jobs"
    systemctl --user list-units 'rl-*' --all --no-legend --plain | awk '{printf "  %-28s %s %s\n", $1, $3, $4}' || true
    echo "== runs (last logged window)"
    for d in runs/*/; do
      n=$(basename "$d"); [ -f "$d/log.jsonl" ] || continue
      last=$(tail -1 "$d/log.jsonl")
      printf "  %-12s %s%s\n" "$n" "$last" "$( [ -f "$d/DONE" ] && echo '  [DONE]')"
    done
    echo "== disk used by runs/ and data/"; du -sh runs data 2>/dev/null | sed 's/^/  /'
    ;;
  logs)
    name="${1:?usage: rl.sh logs NAME}"
    "$PY" - "$name" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(f"runs/{sys.argv[1]}/log.jsonl")]
print(f"{'step':>9} {'reward/1000':>11} {'food':>6} {'deaths':>6} {'damage':>6} {'lr':>9} {'ent':>7} {'sps':>5}  time")
for r in rows[-40:]:
    print(f"{r['step']:9d} {r['reward_per_1000']:+11.1f} {r['food_per_1000']:6.2f} {r['deaths_per_1000']:6.2f} "
          f"{r['damage_per_1000']:6.1f} {r['lr']:9.2e} {r['ent']:7.4f} {r['steps_per_s']:5.1f}  {r['time']}")
if len(rows) >= 10:
    import statistics as st
    print(f"mean reward of the last 10 windows: {st.mean(r['reward_per_1000'] for r in rows[-10:]):+.1f}")
EOF
    ;;
  follow)
    name="${1:?usage: rl.sh follow NAME|contest}"
    if [ "$name" = contest ]; then tail -f results/contest_live.log; else tail -f "runs/$name/stdout.log"; fi
    ;;
  stop)
    name="${1:?usage: rl.sh stop NAME|contest|select-NAME}"
    case "$name" in contest) u=rl-contest ;; record) u=rl-record ;; select-*|job-*) u="rl-$name" ;; *) u="rl-train-$name" ;; esac
    systemctl --user stop "$u" && echo "stopped $u (it saved its state)"
    systemctl --user reset-failed "$u" 2>/dev/null || true
    ;;
  prune)
    "$PY" src/train/prune.py "$@"
    ;;
  record)
    # contest player that ALSO records every step (same frozen policy as the submission); replaces 'contest'
    name="${1:?usage: rl.sh record NAME [contest_record.py options]}"; shift
    if unit_active rl-contest; then echo "stop the contest player first: scripts/rl.sh stop contest"; exit 1; fi
    systemd-run --user --unit=rl-record --working-directory="$ROOT" \
      -p Restart=always -p RestartSec=30 -p StartLimitIntervalSec=0 \
      /bin/bash -c "exec '$PY' src/contest_record.py --name '$name' $* >> results/contest_live.log 2>&1"
    echo "started rl-record ($name); output: results/contest_live.log, data: data/live/${name}_*.npz"
    ;;
  scores)
    "$PY" src/contest_scores.py "${1:-10}"
    ;;
  *)
    sed -n '2,25p' "$0"; exit 1 ;;
esac
