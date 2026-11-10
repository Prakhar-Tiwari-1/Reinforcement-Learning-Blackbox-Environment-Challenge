"""Smoke test: connect to the training server and play random actions for a few steps."""

import sys

import numpy as np

from env_utils import make_env

n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50

env = make_env("training")
agents = env.possible_agents
print("agents:", agents)
print("observation space:", env.observation_space(agents[0]))
print("action space:", env.action_space(agents[0]))

try:
    observations, infos = env.reset(seed=0)
    returns = {agent: 0.0 for agent in agents}
    for _ in range(n_steps):  # the episode is eternal: bound the loop
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        for agent, reward in rewards.items():
            returns[agent] += reward
    print(f"mean reward per step over {n_steps} random steps:")
    for agent in agents:
        print(f"  {agent}: {returns[agent] / n_steps:+.4f}")
    print("last observation of", agents[0], np.round(observations[agents[0]], 3))
finally:
    env.close()  # discard the world
