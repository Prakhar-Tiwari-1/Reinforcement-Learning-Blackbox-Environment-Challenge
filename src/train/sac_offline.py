"""Offline SAC-Discrete: learn only from a saved replay buffer (no play during training), then evaluate the actor.

Same networks as src/train/sac_discrete.py (actor 64+64 = the policy, twin critics 256+256, automatic alpha).
--cql W adds the conservative Q-learning penalty (Kumar et al. 2020) to each critic:
    W * mean( logsumexp_a Q(s, a) - Q(s, a_data) )
which pushes down the Q-values of actions that are rare in the buffer (plain offline Q-learning over-values them).

Train (GPU only, no server):
    .venv/bin/python src/train/sac_offline.py train --name sac_off --buffer runs/ddqn/buffer.npz --updates 400000
Evaluate a saved actor in the TRAINING world (4 agents play it, frozen; needs the training server free):
    .venv/bin/python src/train/sac_offline.py eval runs/sac_off/final.pt --steps 6000 --mode greedy
Outputs: runs/<name>/{config.json, log.jsonl, final.pt}; evaluations are appended to results/eval_summary.jsonl.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ddqn import ACTIONS, DEV, ROOT, Features, Replay, sub  # noqa: E402
from sac_discrete import Actor, Critic  # noqa: E402


def train(args):
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(run / "config.json", "w"), indent=1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    norm = torch.load(ROOT / "models" / "FINAL_ddqn_500k.pt", weights_only=False)   # the buffer's normalisation
    d_in = 11 * 4 + 2
    buf = Replay(1_000_000, d_in)
    buf.load(ROOT / args.buffer)
    print(f"loaded {buf.n} transitions from {args.buffer}", flush=True)

    actor = Actor(d_in).to(DEV)
    critics = nn.ModuleList([Critic(d_in, 256) for _ in range(2)]).to(DEV)
    targets = nn.ModuleList([Critic(d_in, 256) for _ in range(2)]).to(DEV)
    targets.load_state_dict(critics.state_dict())
    log_alpha = torch.tensor(math.log(0.1), device=DEV, requires_grad=True)
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-4)
    opt_c = torch.optim.Adam(critics.parameters(), lr=3e-4)
    opt_al = torch.optim.Adam([log_alpha], lr=3e-4)
    target_ent = args.target_entropy * math.log(4)
    # fixed held-out sample to measure how often the actor's top action equals the action stored in the buffer
    s_chk, a_chk, *_ = buf.sample(20_000)

    acc = dict(lc=0.0, cql=0.0, q=0.0, ent=0.0, alpha=0.0, n=0)
    t0 = time.time()
    for u in range(1, args.updates + 1):
        s, a, r, s2, g = buf.sample(256)
        alpha = log_alpha.exp().detach()
        with torch.no_grad():
            pi2, logpi2 = actor(s2)
            q2 = torch.min(targets[0](s2), targets[1](s2))
            y = r + g * (pi2 * (q2 - alpha * logpi2)).sum(1)
        qall = [c(s) for c in critics]
        qs = [q.gather(1, a[:, None]).squeeze(1) for q in qall]
        loss_c = sum(F.smooth_l1_loss(q, y) for q in qs)
        cql = sum((torch.logsumexp(qa, 1) - q).mean() for qa, q in zip(qall, qs))
        opt_c.zero_grad()
        (loss_c + args.cql * cql).backward()
        nn.utils.clip_grad_norm_(critics.parameters(), 10.0)
        opt_c.step()

        pi, logpi = actor(s)
        qmin = torch.min(qall[0], qall[1]).detach()
        loss_a = (pi * (alpha * logpi - qmin)).sum(1).mean()
        opt_a.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        opt_a.step()

        ent = -(pi.detach() * logpi.detach()).sum(1).mean()
        loss_al = log_alpha * (ent - target_ent)
        opt_al.zero_grad()
        loss_al.backward()
        opt_al.step()
        with torch.no_grad():
            for c, t in zip(critics, targets):
                for pc, pt in zip(c.parameters(), t.parameters()):
                    pt.mul_(1 - 0.005).add_(0.005 * pc)
        acc["lc"] += float(loss_c.detach()) / 2
        acc["cql"] += float(cql.detach()) / 2
        acc["q"] += float(qs[0].detach().mean())
        acc["ent"] += float(ent)
        acc["alpha"] += float(alpha)
        acc["n"] += 1
        if u % args.log_every == 0:
            with torch.no_grad():
                agree = float((actor(s_chk)[0].argmax(1) == a_chk).float().mean())
            n = acc["n"]
            e = dict(update=u, loss=round(acc["lc"] / n, 5), cql_gap=round(acc["cql"] / n, 4), mean_q=round(acc["q"] / n, 3),
                     ent=round(acc["ent"] / n, 3), alpha=round(acc["alpha"] / n, 4), agree_with_buffer=round(agree, 3),
                     updates_per_s=round(u / (time.time() - t0), 1))
            print(json.dumps(e), flush=True)
            with open(run / "log.jsonl", "a") as f:
                f.write(json.dumps(e) + "\n")
            acc = dict(lc=0.0, cql=0.0, q=0.0, ent=0.0, alpha=0.0, n=0)
    torch.save({"algo": "sac", "actor": actor.state_dict(), "critics": critics.state_dict(), "log_alpha": log_alpha.item(),
                "norm_mean": norm["norm_mean"], "norm_std": norm["norm_std"], "hist": 4}, run / "final.pt")
    print(f"saved {run / 'final.pt'}", flush=True)


def evaluate(args):
    sub.load_dotenv_if_any()
    url = os.environ.get("PZ_TRAIN_URL", "https://csc53439ep.jdhp.org/training/api")
    if "/training/" not in url:              # evaluation must never play in the contest world
        raise SystemExit(f"refusing to play on {url}: not the training server")
    ck = torch.load(ROOT / args.ckpt, weights_only=False, map_location=DEV)
    actor = Actor(11 * ck["hist"] + 2).to(DEV)
    actor.load_state_dict(ck["actor"])
    feat = Features(ck["norm_mean"], ck["norm_std"], ck["hist"])
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = sub.make_env(url)
    ag = env.possible_agents
    obs, _ = env.reset(seed=args.seed)
    S = {a: feat(a, obs[a]) for a in ag}
    tot = dict(r=0.0, pos=0, died=0)
    counts = np.zeros(4, int)
    try:
        for _ in range(args.steps):
            with torch.no_grad():
                pi, _ = actor(torch.tensor(np.array([S[a] for a in ag]), device=DEV))
                idx = (pi.argmax(1) if args.mode == "greedy" else torch.multinomial(pi, 1).squeeze(1)).cpu().numpy()
            acts = {a: ACTIONS[idx[k]] for k, a in enumerate(ag)}
            obs, rew, _, _, infos = env.step(acts)
            for k, a in enumerate(ag):
                feat.prev[a] = acts[a].astype(np.float32)
                S[a] = feat(a, obs[a])
                tot["r"] += rew[a]
                tot["pos"] += rew[a] > 0
                tot["died"] += bool(infos[a].get("died", False))
                counts[idx[k]] += 1
    finally:
        env.close()
    n = args.steps * len(ag)
    e = dict(run=f"{Path(args.ckpt).parent.name}/{Path(args.ckpt).stem}_{args.mode}", world="training", steps=args.steps,
             reward_per_1000=round(tot["r"] / n * 1000, 2), food_per_1000=round(tot["pos"] / n * 1000, 2),
             deaths_per_1000=round(tot["died"] / n * 1000, 2),
             action_share={str(ACTIONS[i].tolist()): round(c / n, 3) for i, c in enumerate(counts)},
             time=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps(e), flush=True)
    with open(ROOT / "results" / "eval_summary.jsonl", "a") as f:
        f.write(json.dumps(e) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--buffer", default="runs/ddqn/buffer.npz")
    t.add_argument("--updates", type=int, default=400_000)
    t.add_argument("--cql", type=float, default=0.0)
    t.add_argument("--target-entropy", type=float, default=0.35)
    t.add_argument("--log-every", type=int, default=20_000)
    t.add_argument("--seed", type=int, default=0)
    e = sp.add_parser("eval")
    e.add_argument("ckpt")
    e.add_argument("--steps", type=int, default=6000)
    e.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    e.add_argument("--seed", type=int, default=71)
    args = p.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
