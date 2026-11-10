"""List the latest contest-world scores found in results/ (newest first).

Reads every results/contest_*.log:
  - contest_live.log (written by `scripts/rl.sh contest`): one row per session (a session starts at each
    'agent ...' line, e.g. after an automatic restart); a session without a 'final:' line is still running
    or was stopped, and its last progress line is shown.
  - contest_compare*.log / contest_test*.log (checkpoint tests): one row per finished run.
Columns: reward per 1,000 steps (sum of r), number of steps with r = +5, number of `died`, missed actions.
The time is the log file's last modification (the logs have no per-line times).

Usage: .venv/bin/python src/contest_scores.py [N]        (default N = 10)   or   scripts/rl.sh scores [N]
"""

import glob
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINE = re.compile(r"(\d+) steps .*reward/1000 ([+-][\d.]+) \(\+:(\d+) -:(\d+)\) \| missed (\d+)")


def runs_in(path):
    """(policy, steps, reward/1000, r=+5 count, died count, missed, status) for each run in one log file."""
    name, out = os.path.basename(path), []
    if name == "contest_live.log":
        session = None
        for line in open(path, errors="replace"):
            if line.startswith("agent "):
                if session:
                    out.append(session)
                session = ["submission (live)", 0, 0.0, 0, 0, 0, "running/stopped"]
            m = LINE.search(line)
            if m and session:
                session[1:6] = [int(m[1]), float(m[2]), int(m[3]), int(m[4]), int(m[5])]
                if line.startswith("final:"):
                    session[6] = "finished"
        if session:
            out.append(session)
        return out
    policy = "?"
    for line in open(path, errors="replace"):
        if line.startswith("### "):                       # contest_test*.log: '### <checkpoint>' then runs
            policy = line[4:].strip()
            continue
        if "final:" not in line:
            continue
        m = LINE.search(line)
        if m:
            p = line.split("|")[1].strip() if line.startswith("round") else policy
            out.append([p, int(m[1]), float(m[2]), int(m[3]), int(m[4]), int(m[5]), "finished"])
    return out


def main(n=10):
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "contest_*.log")), key=os.path.getmtime):
        when = time.strftime("%a %d %b %H:%M", time.localtime(os.path.getmtime(path)))
        rows += [[when, os.path.basename(path)] + r for r in runs_in(path)]
    if not rows:
        print("no contest results in results/ yet")
        return
    print(f"{'log modified':16} {'policy':32} {'steps':>7} {'reward/1000':>11} {'r=+5':>5} {'died':>4} {'missed':>6}  status   (log file)")
    for when, f, pol, st, rw, pos, died, missed, status in rows[::-1][:n]:
        pol = os.path.basename(pol).replace(".pt", "")
        print(f"{when:16} {pol:32} {st:7d} {rw:+11.1f} {pos:5d} {died:4d} {missed:6d}  {status:8} ({f})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
