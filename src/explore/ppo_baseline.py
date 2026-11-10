"""First learning run: a small shared-parameter PPO agent on the training server.

Goal: see what a plain learner can reach vs the random baseline, not to be the final agent.

Design (justified in docs/):
- Actor and critic MLPs with one hidden layer of 64 units each (64 + 64 = 128 units in total).
- LayerNorm on the hidden layer, running mean/std normalisation of the observations
  (the declared Box bounds are not trusted: dim 9 goes below 0).
- Input = current observation + previous action (cheap memory of what we just did).
- MultiBinary(2) action = 2 independent Bernoulli heads (works for any MultiBinary(n)).
- The 4 training agents share the same network: 4x more data per step (parameter sharing).
- The episode never ends; deaths are not treated as terminations (the -5 already teaches).

Usage: uv run python src/explore/ppo_baseline.py --steps 30000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env_utils import make_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class RunningNorm:
    def __init__(self, dim):
        self.mean, self.var, self.count = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -5, 5)


def head(i, o, hidden=64):
    return nn.Sequential(nn.Linear(i, hidden), nn.LayerNorm(hidden), nn.Tanh(), nn.Linear(hidden, o))


class ActorCritic(nn.Module):
    def __init__(self, in_dim, n_bits):
        super().__init__()
        self.actor, self.critic = head(in_dim, n_bits), head(in_dim, 1)

    def dist(self, x):
        return torch.distributions.Bernoulli(logits=self.actor(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None, help="checkpoint to continue from, e.g. data/ppo_baseline.pt")
    p.add_argument("--out", default="ppo_baseline", help="output name: data/<out>.pt and data/<out>_log.json")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    env = make_env("training")
    agents = env.possible_agents
    n = len(agents)
    k = env.observation_space(agents[0]).shape[0]
    nb = env.action_space(agents[0]).n
    in_dim = k + nb
    net = ActorCritic(in_dim, nb)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    norm = RunningNorm(k)
    if args.init:
        ck = torch.load(ROOT / args.init, weights_only=False)
        net.load_state_dict(ck["net"])
        norm.mean, norm.var, norm.count = ck["norm_mean"], ck["norm_var"], 1e5
    gamma, lam, clip, ent_coef, reward_scale = 0.99, 0.95, 0.2, 0.01, 0.2

    log, window_r, window_d = [], [], []
    prev_act = np.zeros((n, nb), np.float32)
    t0 = time.monotonic()
    try:
        obs, _ = env.reset(seed=args.seed)
        o = np.array([obs[a] for a in agents])
        for it in range(args.steps // args.rollout):
            B_x, B_a, B_lp, B_v, B_r = [], [], [], [], []
            for _ in range(args.rollout):
                norm.update(o)
                x = torch.tensor(np.concatenate([norm(o), prev_act], 1), dtype=torch.float32)
                with torch.no_grad():
                    d = net.dist(x)
                    a = d.sample()
                    lp, v = d.log_prob(a).sum(1), net.critic(x).squeeze(1)
                act = a.numpy().astype(np.int8)
                obs, rew, _, _, infos = env.step({ag: act[i] for i, ag in enumerate(agents)})
                r = np.array([rew[ag] for ag in agents], np.float32)
                died = np.array([infos[ag].get("died", False) for ag in agents])
                B_x.append(x), B_a.append(a), B_lp.append(lp), B_v.append(v), B_r.append(torch.tensor(r))
                window_r.append(r), window_d.append(died)
                o = np.array([obs[ag] for ag in agents])
                prev_act = act.astype(np.float32)

            # GAE over the rollout (per agent), bootstrapping from the value of the last observation
            with torch.no_grad():
                x_last = torch.tensor(np.concatenate([norm(o), prev_act], 1), dtype=torch.float32)
                next_v = net.critic(x_last).squeeze(1)
            X, Aa, LP, V = torch.stack(B_x), torch.stack(B_a), torch.stack(B_lp), torch.stack(B_v)
            Rr = torch.stack(B_r) * reward_scale
            adv = torch.zeros_like(Rr)
            gae = torch.zeros(n)
            for t in reversed(range(args.rollout)):
                nv = next_v if t == args.rollout - 1 else V[t + 1]
                delta = Rr[t] + gamma * nv - V[t]
                gae = delta + gamma * lam * gae
                adv[t] = gae
            ret = adv + V
            X, Aa, LP, adv, ret = X.flatten(0, 1), Aa.flatten(0, 1), LP.flatten(0, 1), adv.flatten(), ret.flatten()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            for _ in range(4):
                for mb in torch.randperm(len(X)).split(256):
                    d = net.dist(X[mb])
                    ratio = torch.exp(d.log_prob(Aa[mb]).sum(1) - LP[mb])
                    pg = -torch.min(ratio * adv[mb], ratio.clamp(1 - clip, 1 + clip) * adv[mb]).mean()
                    vl = ((net.critic(X[mb]).squeeze(1) - ret[mb]) ** 2).mean()
                    loss = pg + 0.5 * vl - ent_coef * d.entropy().sum(1).mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()

            step = (it + 1) * args.rollout
            if step % 2048 == 0 or step >= args.steps - args.rollout:
                wr, wd = np.array(window_r), np.array(window_d)
                with torch.no_grad():
                    probs = torch.sigmoid(net.actor(X[-256:])).mean(0).numpy()
                entry = dict(step=step, reward_per_1000=float(wr.mean() * 1000),
                             pos_per_1000=float((wr > 0).mean() * 1000), deaths_per_1000=float(wd.mean() * 1000),
                             p_bits=probs.round(3).tolist(), seconds=round(time.monotonic() - t0))
                log.append(entry)
                print(json.dumps(entry), flush=True)
                window_r, window_d = [], []
                save(log, net, norm, args.out)  # checkpoint at every log line, so an interrupted run keeps its results
    finally:
        env.close()
    save(log, net, norm, args.out)


def save(log, net, norm, out="ppo_baseline"):
    (ROOT / "data").mkdir(exist_ok=True)
    json.dump(log, open(ROOT / "data" / f"{out}_log.json", "w"), indent=1)
    torch.save({"net": net.state_dict(), "norm_mean": norm.mean, "norm_var": norm.var}, ROOT / "data" / f"{out}.pt")


if __name__ == "__main__":
    main()
