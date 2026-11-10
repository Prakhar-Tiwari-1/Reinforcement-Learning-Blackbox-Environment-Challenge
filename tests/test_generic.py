"""Generality test: the *same* agent class (prakhar.tiwari.py) learns from scratch on local Gymnasium
environments with different observation / action spaces, knowing nothing but the spaces.

    CartPole-v1      Box obs        -> Discrete(2)
    FrozenLake-v1    Discrete(16)   -> Discrete(4)       (deterministic 4x4 map, sparse reward)
    Pendulum-v1      Box obs        -> Box(1) action
    BitMatch (toy)   Box(4)         -> MultiBinary(3)    (reward = bits matching the sign of the obs)
    BitMatchDisc     Discrete(8)    -> MultiBinary(3)    (reward = bits matching the binary code of the obs)

For each: average return of the learning agent over its last 20% of training episodes vs a random agent.

Usage: uv run python tests/test_generic.py [steps_scale]
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sub", ROOT / "prakhar.tiwari.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)


class BitMatch(gym.Env):
    """Contextual bandit-like toy task with MultiBinary actions (episodes of 50 steps)."""

    def __init__(self, discrete_obs=False):
        self.discrete = discrete_obs
        self.observation_space = gym.spaces.Discrete(8) if discrete_obs else gym.spaces.Box(-1, 1, (4,), np.float32)
        self.action_space = gym.spaces.MultiBinary(3)
        self.rng = np.random.default_rng(0)

    def _obs(self):
        if self.discrete:
            self.code = int(self.rng.integers(8))
            self.target = np.array([(self.code >> b) & 1 for b in range(3)])
            return self.code
        o = self.rng.uniform(-1, 1, 4).astype(np.float32)
        self.target = (o[:3] > 0).astype(int)
        return o

    def reset(self, seed=None, options=None):
        self.t = 0
        return self._obs(), {}

    def step(self, a):
        r = float((np.asarray(a) == self.target).sum())
        self.t += 1
        return self._obs(), r, False, self.t >= 50, {}


ENVS = {
    "CartPole-v1 (Box -> Discrete)": (lambda: gym.make("CartPole-v1"), 60_000),
    "FrozenLake-v1 4x4 (Discrete -> Discrete)": (lambda: gym.make("FrozenLake-v1", is_slippery=False), 30_000),
    "Pendulum-v1 (Box -> Box)": (lambda: gym.make("Pendulum-v1"), 100_000),
    "BitMatch (Box -> MultiBinary)": (lambda: BitMatch(False), 20_000),
    "BitMatch (Discrete -> MultiBinary)": (lambda: BitMatch(True), 20_000),
}


def run(make, steps, learner, seed=0):
    env = make()
    obs, _ = env.reset(seed=seed)
    returns, ep_ret = [], 0.0
    for _ in range(steps):
        if learner is None:
            a = env.action_space.sample()
        else:
            a = learner.act({"me": obs})["me"]
        obs, r, term, trunc, _ = env.step(a)
        ep_ret += r
        if learner is not None:
            # time-limit truncation is treated as an episode end (simple, slightly biased; keeps the data)
            learner.observe({"me": r}, {"me": term or trunc})
        if term or trunc:
            returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
    return returns


def main(scale=1.0):
    results = {}
    for name, (make, steps) in ENVS.items():
        steps = int(steps * scale)
        env = make()
        t0 = time.time()
        agent = sub.Agent(env.observation_space, env.action_space, learn=True, lr=3e-4, pretrained=None)
        learned = run(make, steps, agent)
        rand = run(make, min(steps, 20_000), None)
        tail = learned[int(len(learned) * 0.8):] or learned
        res = dict(steps=steps, episodes=len(learned), random_mean=float(np.mean(rand)),
                   learned_first10pct=float(np.mean(learned[: max(1, len(learned) // 10)])),
                   learned_last20pct=float(np.mean(tail)), seconds=round(time.time() - t0))
        results[name] = res
        print(f"{name:42s} random {res['random_mean']:9.2f} | learning: first 10% {res['learned_first10pct']:9.2f}"
              f" -> last 20% {res['learned_last20pct']:9.2f}   ({steps} steps, {res['seconds']}s)", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(results, open(ROOT / "results" / "generic_tests.json", "w"), indent=1)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
