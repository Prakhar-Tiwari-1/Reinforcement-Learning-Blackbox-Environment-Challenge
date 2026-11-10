"""Run one policy per agent on the training server and record everything (same format as collect.py).

Each of the 4 training agents can run a different policy, so one run compares up to 4 variants
in the same world. Results: data/eval_<name>.npz + a summary line per agent in results/eval_summary.jsonl.

Usage (from the project root):
    uv run python src/explore/evaluate.py --name probe1 --steps 5000 \\
        --policies random ppo heur "heur:away_from_left=A"
    uv run python src/explore/evaluate.py --mode contest --policies heur --steps 120 --name contest1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_utils import make_env  # noqa: E402
from policies import make  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def summarise(R, D, Applied=None):
    lives = D.sum()
    out = dict(steps=int(len(R)), reward_per_1000=float(R.mean() * 1000), food_per_1000=float((R > 0).mean() * 1000),
               deaths_per_1000=float(D.mean() * 1000), food=int((R > 0).sum()), deaths=int(lives),
               food_per_life=float((R > 0).sum() / max(lives, 1)))
    if Applied is not None:
        out["action_applied_rate"] = float(Applied.mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", default=["random"])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["training", "contest"], default="training")
    p.add_argument("--name", required=True)
    args = p.parse_args()

    env = make_env(args.mode)
    agents = env.possible_agents
    n = len(agents)
    specs = (args.policies * n)[:n] if args.mode == "training" else args.policies[:1]
    pols = [make(s, seed=args.seed * 100 + i) for i, s in enumerate(specs)]
    k = env.observation_space(agents[0]).shape[0]

    O = np.zeros((args.steps + 1, n, k), np.float32)
    A = np.zeros((args.steps, n, 2), np.int8)
    R = np.zeros((args.steps, n), np.float32)
    D = np.zeros((args.steps, n), bool)
    Applied = np.ones((args.steps, n), bool)
    think = []
    t_done = 0
    t0 = time.monotonic()
    try:
        obs, _ = env.reset(seed=args.seed) if args.mode == "training" else env.reset()
        O[0] = [obs[a] for a in agents]
        for t in range(args.steps):
            ts = time.monotonic()
            act = np.array([pols[i].act(O[t, i]) for i in range(n)])
            think.append(time.monotonic() - ts)
            obs, rew, _, _, infos = env.step({a: act[i] for i, a in enumerate(agents)})
            A[t], O[t + 1] = act, [obs[a] for a in agents]
            R[t] = [rew[a] for a in agents]
            D[t] = [infos[a].get("died", False) for a in agents]
            if args.mode == "contest":
                Applied[t] = [infos[a].get("shared_world", {}).get("action_applied", True) for a in agents]
            t_done = t + 1
            if (t + 1) % 1000 == 0 or (args.mode == "contest" and (t + 1) % 20 == 0):
                print(f"{t + 1}/{args.steps} {time.monotonic() - t0:.0f}s reward {R[:t + 1].sum(0)} deaths {D[:t + 1].sum(0)}",
                      flush=True)
    finally:
        env.close()
        T = t_done
        O, A, R, D, Applied = O[:T + 1], A[:T], R[:T], D[:T], Applied[:T]
        (ROOT / "data").mkdir(exist_ok=True)
        np.savez_compressed(ROOT / "data" / f"eval_{args.name}.npz", obs=O, actions=A, rewards=R, died=D,
                            applied=Applied, policies=np.array(specs), agents=np.array(agents))
        (ROOT / "results").mkdir(exist_ok=True)
        with open(ROOT / "results" / "eval_summary.jsonl", "a") as f:
            for i in range(n):
                s = dict(run=args.name, mode=args.mode, seed=args.seed, agent=agents[i], policy=specs[i],
                         think_ms=float(np.mean(think) * 1000) if think else 0.0,
                         **summarise(R[:, i], D[:, i], Applied[:, i] if args.mode == "contest" else None))
                f.write(json.dumps(s) + "\n")
                print(json.dumps(s))


if __name__ == "__main__":
    main()
