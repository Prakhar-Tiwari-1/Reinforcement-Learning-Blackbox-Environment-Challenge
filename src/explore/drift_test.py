"""Drift test in the training world: does online learning help when the other agents change?

agent_0 = our agent, learning online (lr 1e-4, as in the contest)
agent_1 = our agent, frozen (same starting weights)
agent_2, agent_3 = "the others", whose behaviour changes every phase:
    phase 1: strong (frozen PPO)   phase 2: random   phase 3: reckless (always [1,1])   phase 4: strong again

Usage: uv run python src/explore/drift_test.py --phase-steps 5000 --load data/ppo_cont.pt
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "explore"))
from env_utils import make_env  # noqa: E402
from policies import make  # noqa: E402

spec = importlib.util.spec_from_file_location("sub", ROOT / "prakhar.tiwari.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

PHASES = [("strong (PPO)", "ppo"), ("random", "random"), ("reckless [1,1]", "const:1,1"), ("strong again", "ppo")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase-steps", type=int, default=5000)
    p.add_argument("--load", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--name", default="drift")
    args = p.parse_args()

    env = make_env("training")
    ag = env.possible_agents
    o_space, a_space = env.observation_space(ag[0]), env.action_space(ag[0])
    online = sub.Agent(o_space, a_space, learn=True, lr=1e-4, seed=args.seed)
    frozen = sub.Agent(o_space, a_space, learn=False, seed=args.seed + 1)
    if args.load:
        online.load(ROOT / args.load)
        frozen.load(ROOT / args.load)
    results = []
    try:
        obs, _ = env.reset(seed=args.seed)
        for phase, spec_ in PHASES:
            others = [make(spec_, seed=args.seed + k) for k in (2, 3)]
            R = np.zeros((args.phase_steps, 4))
            D = np.zeros((args.phase_steps, 4), bool)
            for t in range(args.phase_steps):
                acts = {}
                acts.update(online.act({ag[0]: obs[ag[0]]}))
                acts.update(frozen.act({ag[1]: obs[ag[1]]}))
                acts[ag[2]], acts[ag[3]] = others[0].act(obs[ag[2]]), others[1].act(obs[ag[3]])
                obs, rew, _, _, infos = env.step(acts)
                online.observe({ag[0]: rew[ag[0]]})
                R[t] = [rew[a] for a in ag]
                D[t] = [infos[a].get("died", False) for a in ag]
            row = dict(phase=phase, online=float(R[:, 0].mean() * 1000), frozen=float(R[:, 1].mean() * 1000),
                       others=float(R[:, 2:].mean() * 1000), online_deaths=int(D[:, 0].sum()),
                       frozen_deaths=int(D[:, 1].sum()), online_updates=online.n_updates)
            results.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(results, open(ROOT / "results" / f"{args.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
