"""Collect experience on the training server and save it to data/<name>.npz.

Policies:
    random          uniform random actions for every agent
    const           each agent holds a fixed action: agent_i plays action i ([0,0],[0,1],[1,0],[1,1])
    sticky          random actions held for a random 1-20 steps (more coherent movement than i.i.d.)

Usage (from the project root):
    uv run python src/explore/collect.py --policy random --steps 20000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env_utils import make_env  # noqa: E402

ACTIONS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
DATA = Path(__file__).resolve().parents[2] / "data"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["random", "const", "sticky"], default="random")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--state-every", type=int, default=10, help="save env.state() every k steps (analysis only)")
    p.add_argument("--name", default=None)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    env = make_env("training")
    agents = env.possible_agents
    n = len(agents)
    obs_dim = env.observation_space(agents[0]).shape[0]

    O = np.zeros((args.steps + 1, n, obs_dim), np.float32)   # O[t] = observation before action t
    A = np.zeros((args.steps, n, 2), np.int8)
    R = np.zeros((args.steps, n), np.float32)
    D = np.zeros((args.steps, n), bool)                      # infos[agent]["died"]
    states, state_t = [], []

    sticky_left = np.zeros(n, int)
    sticky_act = np.zeros((n, 2), np.int8)
    try:
        obs, _ = env.reset(seed=args.seed)
        O[0] = [obs[a] for a in agents]
        t0 = time.monotonic()
        for t in range(args.steps):
            if args.policy == "random":
                act = ACTIONS[rng.integers(4, size=n)]
            elif args.policy == "const":
                act = ACTIONS[np.arange(n) % 4]
            else:
                for i in range(n):
                    if sticky_left[i] == 0:
                        sticky_act[i] = ACTIONS[rng.integers(4)]
                        sticky_left[i] = rng.integers(1, 21)
                    sticky_left[i] -= 1
                act = sticky_act.copy()
            obs, rew, _, _, infos = env.step({a: act[i] for i, a in enumerate(agents)})
            A[t] = act
            O[t + 1] = [obs[a] for a in agents]
            R[t] = [rew[a] for a in agents]
            D[t] = [infos[a].get("died", False) for a in agents]
            if args.state_every and t % args.state_every == 0:
                states.append(env.state().astype(np.float32))
                state_t.append(t)
            if (t + 1) % 2000 == 0:
                print(f"{t + 1}/{args.steps} steps, {time.monotonic() - t0:.0f}s, "
                      f"reward so far {R[: t + 1].sum(0)}, deaths {D[: t + 1].sum(0)}", flush=True)
    finally:
        env.close()

    DATA.mkdir(exist_ok=True)
    out = DATA / f"{args.name or args.policy}.npz"
    np.savez_compressed(out, obs=O, actions=A, rewards=R, died=D,
                        states=np.array(states), state_t=np.array(state_t), agents=np.array(agents))
    print("saved", out)


if __name__ == "__main__":
    main()
