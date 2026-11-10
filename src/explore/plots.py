"""Figures for the reports (docs/img/).

1. learning_curve.png   reward per 1,000 agent-steps along the whole training history
                        (baseline -> continued -> long run with the submission code -> fine-tuning)
2. policy_comparison.png  evaluation results of every approach, training and contest worlds

Usage: uv run python src/explore/plots.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "img"
PAT = re.compile(r"^(\d+) steps .*\(\+:(\d+) -:(\d+)\)")


def ppo_log(path):
    L = [e for e in json.load(open(path)) if e["step"] % 2048 == 0]
    return np.array([e["step"] for e in L]), np.array([e["reward_per_1000"] for e in L])


def sub_log(path, n_agents=4):
    xs, ys, prev = [], [], (0, 0, 0)
    if not Path(path).exists():
        return np.array([]), np.array([])
    for line in open(path):
        m = PAT.match(line)
        if not m:
            continue
        s, f, d = map(int, m.groups())
        if s <= prev[0]:
            break  # only the first training run of the log
        ds, df, dd = s - prev[0], f - prev[1], d - prev[2]
        xs.append(s)
        ys.append(5 * (df - dd) / (ds * n_agents) * 1000)
        prev = (s, f, d)
    return np.array(xs), np.array(ys)


def smooth(y, k=5):
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def learning_curve():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    offset = 0
    stages = [("1. PPO from scratch", *ppo_log(ROOT / "data/ppo_baseline_log.json")),
              ("2. continued", *ppo_log(ROOT / "data/ppo_cont_log.json")),
              ("3. long run (submission code)", *sub_log(ROOT / "results/overnight.log")),
              ("4. fine-tuning (lr 1e-4, less entropy)", *sub_log(ROOT / "results/overnight2.log"))]
    for name, x, y in stages:
        if len(x) == 0:
            continue
        ax.plot(offset + x, y, alpha=0.25)
        ax.plot(offset + x, smooth(y), lw=2, label=name)
        offset += x[-1]
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(-17.8, color="grey", ls="--", lw=1)
    ax.text(offset * 0.01, -16, "random agent (≈ −18)", color="grey")
    ax.set_xlabel("training steps (each step = 4 agents)")
    ax.set_ylabel("reward per 1,000 agent-steps")
    ax.set_title("Training history (training server; thin = raw windows, thick = smoothed)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=130)


def policy_comparison():
    rows = [json.loads(l) for l in open(ROOT / "results" / "eval_summary.jsonl")]
    g = defaultdict(list)
    for r in rows:
        g[(r["mode"], r["policy"])].append(r)
    names = {"random": "random", "heur:away_from_left=A": "rules v1", "heur2": "rules v2 (healing)",
             "q:data/q_mc.pt": "offline Q (MC)", "q:data/q_fqi.pt": "offline Q (FQI)",
             "mpc:data/world_model.pt": "world model + MPC", "ppo": "PPO baseline",
             "torch:data/ppo_seed1.pt": "PPO seed 1", "torch:data/ppo_cont.pt": "PPO continued",
             "torch:data/ppo_cont_snapshot80k.pt": "PPO continued", "torch:data/sub_long.pt": "long run",
             "torch:data/sub_fine.pt": "fine-tuned"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, mode in zip(axes, ["training", "contest"]):
        agg = defaultdict(lambda: [0, 0, 0])
        for (m, pol), rs in g.items():
            if m != mode or pol not in names:
                continue
            a = agg[names[pol]]
            a[0] += sum(r["food"] for r in rs)
            a[1] += sum(r["deaths"] for r in rs)
            a[2] += sum(r["steps"] for r in rs)
        if mode == "contest":  # runs of the submission file itself (logged as "final:" lines)
            for label, logs in [("stage-2 weights in submission, frozen", ["contest_tests.log", "contest_overnight.log"]),
                                ("stage-2 weights, online lr 1e-4", ["contest_tests.log"]),
                                ("stage-2 weights, online conservative", ["contest_overnight.log"])]:
                for lg in logs:
                    path = ROOT / "results" / lg
                    if not path.exists():
                        continue
                    cur = None
                    for line in open(path):
                        if "--no-learn" in line or line.startswith("### frozen"):
                            cur = "stage-2 weights in submission, frozen"
                        elif line.startswith("### conservative"):
                            cur = "stage-2 weights, online conservative"
                        elif "prakhar.tiwari.py" in line and "--no-learn" not in line:
                            cur = "stage-2 weights, online lr 1e-4"
                        m = re.search(r"final: (\d+) steps .*\(\+:(\d+) -:(\d+)\)", line)
                        if m and cur == label:
                            st, f, d = map(int, m.groups())
                            a = agg[label]
                            a[0] += f
                            a[1] += d
                            a[2] += st
        if mode == "contest":  # checkpoint-selection runs: "### <checkpoint>" followed by a "final:" line
            def lab(name):
                if "sub_fine_snap196608" in name or re.search(r"sub_fine\s*$", name):
                    return "fine-tuned (EMBEDDED)"
                if "sub_fine_snap" in name:
                    return "fine-tuning, earlier snapshots"
                if "snap190k" in name:
                    return "long run, 190k snapshot"
                if "final300k" in name:
                    return "long run, 300k final"
                return None
            for lg in ["contest_select.log", "contest_select2.log", "contest_snapshots.log", "contest_sublong190k.log",
                       "contest_sublong300k.log", "contest_submission_final_fine.log"]:
                path = ROOT / "results" / lg
                if not path.exists():
                    continue
                cur = {"contest_sublong190k.log": "long run, 190k snapshot", "contest_sublong300k.log": "long run, 300k final",
                       "contest_submission_final_fine.log": "fine-tuned (EMBEDDED)"}.get(lg)
                for line in open(path):
                    if line.startswith("###"):
                        cur = lab(line)
                    m = re.search(r"final: (\d+) steps .*\(\+:(\d+) -:(\d+)\)", line)
                    if m and cur:
                        st, f, d = map(int, m.groups())
                        a = agg[cur]
                        a[0] += f
                        a[1] += d
                        a[2] += st
        items = sorted(agg.items(), key=lambda kv: 5 * (kv[1][0] - kv[1][1]) / kv[1][2])
        vals = [5 * (f - d) / s * 1000 for _, (f, d, s) in items]
        ax.barh([k for k, _ in items], vals, color=["tab:red" if v < 0 else "tab:green" for v in vals])
        for i, v in enumerate(vals):
            ax.text(v + (1 if v >= 0 else -1), i, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{mode} world")
        ax.set_xlabel("reward per 1,000 agent-steps (pooled over runs)")
    fig.tight_layout()
    fig.savefig(OUT / "policy_comparison.png", dpi=130)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    learning_curve()
    policy_comparison()
    print("saved", sorted(p.name for p in OUT.glob("*.png")))
