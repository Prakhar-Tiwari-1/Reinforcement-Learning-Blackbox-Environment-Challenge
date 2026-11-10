"""Pick the checkpoint to submit by playing it in the CONTEST world (frozen), alternating candidates.

Candidates: the run's best_*.pt and latest.pt, plus the weights currently embedded in prakhar.tiwari.py
("embedded"). Each candidate plays --reps runs of --steps steps; the order alternates to be fair if the
contest world changes during the evaluation. Results: runs/<name>/selection.json.

With --embed, the winner is embedded into prakhar.tiwari.py if it beats the currently embedded weights
by more than --margin (reward per 1,000 steps). The previous file is backed up in runs/<name>/.

Do NOT run this while scripts/rl.sh contest is playing (one program per token on the contest server:
a second one would replace the playing agent). From Monday 19 Oct, contest runs count for real.

Usage: uv run python src/train/choose_checkpoint.py --run v2 --steps 6000 --reps 2 [--embed]
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "bin" / "python")
PAT = re.compile(r"final: (\d+) steps .*reward/1000 ([+-][\d.]+) \(\+:(\d+) -:(\d+)\) \| missed (\d+)")


def contest_player_running():
    r = subprocess.run(["systemctl", "--user", "is-active", "rl-contest"], capture_output=True, text=True)
    return r.stdout.strip() == "active"


def play(ckpt, steps):
    cmd = [PY, str(ROOT / "prakhar.tiwari.py"), "--no-learn", "--steps", str(steps), "--log-every", str(steps)]
    if ckpt != "embedded":
        cmd += ["--load", str(ckpt)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
    m = PAT.search(out)
    if not m:
        return None
    st, rw, f, d, missed = m.groups()
    return dict(steps=int(st), reward_per_1000=float(rw), food=int(f), deaths=int(d), missed=int(missed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--embed", action="store_true")
    p.add_argument("--margin", type=float, default=3.0)
    args = p.parse_args()
    if contest_player_running():
        sys.exit("the contest player (rl-contest) is running: stop it first (scripts/rl.sh stop contest)")
    run = ROOT / "runs" / args.run
    cands = ["embedded"] + sorted(str(x) for x in run.glob("best_*.pt"))
    if (run / "latest.pt").exists():
        snap = run / "latest_eval.pt"
        shutil.copy(run / "latest.pt", snap)          # frozen copy: the trainer may overwrite latest.pt
        cands.append(str(snap))
    results = {c: [] for c in cands}
    for rep in range(args.reps):
        order = cands if rep % 2 == 0 else cands[::-1]
        for c in order:
            r = play(c, args.steps)
            results[c].append(r)
            print(f"{time.strftime('%H:%M')} rep {rep + 1} {Path(c).name:22s} {r}", flush=True)
    table = []
    for c, rs in results.items():
        rs = [r for r in rs if r]
        if rs:
            table.append(dict(candidate=Path(c).name, path=c, runs=[r["reward_per_1000"] for r in rs],
                              mean=float(np.mean([r["reward_per_1000"] for r in rs])),
                              missed=sum(r["missed"] for r in rs)))
    table.sort(key=lambda t: -t["mean"])
    print("\nranking (contest world, frozen):")
    for t in table:
        print(f"  {t['candidate']:22s} mean {t['mean']:+6.1f}   runs {t['runs']}   missed {t['missed']}")
    json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M"), steps=args.steps, reps=args.reps, ranking=table),
              open(run / "selection.json", "w"), indent=1)
    if args.embed and table:
        best = table[0]
        emb = next((t for t in table if t["candidate"] == "embedded"), None)
        if best["candidate"] != "embedded" and (emb is None or best["mean"] > emb["mean"] + args.margin):
            backup = run / f"submission_backup_{time.strftime('%Y%m%d_%H%M')}.py"
            shutil.copy(ROOT / "prakhar.tiwari.py", backup)
            subprocess.run([PY, str(ROOT / "src" / "build_submission.py"), best["path"]], cwd=ROOT, check=True)
            print(f"EMBEDDED {best['candidate']} ({best['mean']:+.1f} vs embedded {emb['mean'] if emb else float('nan'):+.1f}); "
                  f"previous file saved as {backup.name}")
        else:
            print("kept the currently embedded weights (no candidate is clearly better)")


if __name__ == "__main__":
    main()
