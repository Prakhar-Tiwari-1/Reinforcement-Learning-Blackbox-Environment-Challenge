"""Option 1: fine-tune a DDQN policy OFFLINE on recorded live-contest steps (data/live/*.npz), on the GPU.

Why: in the live contest world (other participants) 500k gets r=+5 half as often as in the training world
(docs/EXPERIMENTS_17OCT.md, results/live_vs_training_now.txt). DDQN is off-policy, so every recorded live step
(whatever policy played it) can be reused. Old training data is mixed in so the network does not forget.

Data:
  live  : every data/live/<pattern>.npz file = one continuous stretch of one agent (obs T+1, actions, rewards, died),
          turned into n-step transitions exactly like src/train/ddqn.py (same features, shaping, reward scale 0.2).
          The newest --holdout fraction of live transitions is kept apart to measure the TD error on unseen live data.
  old   : the replay buffer of the 500k run (runs/ddqn/buffer.npz, same features / n-step).
  Each minibatch = --live-frac live + the rest old.
Learning: Double DQN, Huber loss, Adam(--lr), target network refreshed every --target-every updates, from --init.
    --anchor A adds A * mean_a (Q(s,a) - Q_init(s,a))^2 on the live states of each batch: a conservative update that
    keeps the policy close to --init where the narrow live data says little (offline RL constraint, cf. CQL / BRAC).
Output: runs/<name>/upd_<k>.pt (policy checkpoints, loadable by prakhar.tiwari.py --load) + runs/<name>/log.jsonl.

Usage: .venv/bin/python src/train/ddqn_live_ft.py --name liveft1 [--updates 30000 --live-frac 0.5 --lr 3e-5]
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "train"))
sys.argv, _argv = sys.argv[:1], sys.argv          # ddqn.py has no import-time argument parsing; keep ours intact
import ddqn  # noqa: E402

sys.argv = _argv
DEV = ddqn.DEV


def live_transitions(files, mean, std, n_step, gamma, shaping):
    """n-step transitions (s, a, R, s_n, gamma^n) from single-agent live recordings, in file order (oldest first)."""
    out = []
    for f in files:
        d = np.load(f)
        O, A, R, D = d["obs"], d["actions"], d["rewards"], d["died"]
        if len(R) < n_step + 1:
            continue
        feat, ns = ddqn.Features(mean, std, 4), ddqn.NStep(n_step, gamma)
        s = feat(0, O[0])
        for t in range(len(R)):
            a = int(A[t][0]) * 2 + int(A[t][1])
            r = float(R[t]) + (shaping(O[t], O[t + 1], bool(D[t])) if shaping else 0.0)
            feat.prev[0] = A[t].astype(np.float32)
            s2 = feat(0, O[t + 1])
            out += ns.push(0, s, a, 0.2 * r, s2)
            s = s2
    return out


def to_tensors(trs):
    s, a, r, s2, g = zip(*trs)
    f = lambda x, dt: torch.tensor(np.array(x), dtype=dt, device=DEV)  # noqa: E731
    return f(s, torch.float32), f(a, torch.long), f(r, torch.float32), f(s2, torch.float32), f(g, torch.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--init", default="models/FINAL_ddqn_500k.pt")
    p.add_argument("--live", default="data/live/*.npz", help="glob of live recordings")
    p.add_argument("--old", default="runs/ddqn/buffer.npz")
    p.add_argument("--live-frac", type=float, default=0.5)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--updates", type=int, default=30000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--target-every", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anchor", type=float, default=0.0,
                   help="weight of a penalty mean_a (Q(s,a) - Q_init(s,a))^2 on live states: keeps the policy near --init")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ROOT / args.init, weights_only=False, map_location="cpu")
    mean, std = ck["norm_mean"], ck["norm_std"]
    online, target = ddqn.QNet(46).to(DEV), ddqn.QNet(46).to(DEV)
    online.load_state_dict({k: v.to(DEV) for k, v in ck["net"].items()})
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=args.lr)
    shaping = ddqn.sub.ContestShaping(0.99)

    files = sorted(glob.glob(str(ROOT / args.live)), key=lambda f: Path(f).stat().st_mtime)
    trs = live_transitions(files, mean, std, 5, 0.99, shaping)
    n_hold = int(len(trs) * args.holdout)
    train_live, hold = to_tensors(trs[:len(trs) - n_hold]), to_tensors(trs[len(trs) - n_hold:])
    b = np.load(ROOT / args.old)
    n_old = int(b["n"])
    old = tuple(torch.tensor(b[k][:n_old], device=DEV) for k in ("s", "a", "r", "s2", "g"))
    old = (old[0].float(), old[1].long(), old[2].float(), old[3].float(), old[4].float())
    n_live = len(train_live[2])
    print(f"live files {len(files)} -> {n_live} train + {n_hold} held-out live transitions; old {n_old}; "
          f"init {args.init}; device {DEV}", flush=True)

    def td(batch, net=online):
        s, a, r, s2, g = batch
        with torch.no_grad():
            a2 = net(s2).argmax(1, keepdim=True)
            y = r + g * target(s2).gather(1, a2).squeeze(1)
        q = net(s).gather(1, a[:, None]).squeeze(1)
        return nn.functional.smooth_l1_loss(q, y), q

    def evaluate():
        with torch.no_grad():
            l, q = td(hold)
            s = hold[0]
            greedy = online(s).argmax(1)
            agree = (greedy == ref(s).argmax(1)).float().mean()
        return float(l), float(q.mean()), float(agree)

    ref = ddqn.QNet(46).to(DEV)                       # the starting policy, to measure how much the policy moved
    ref.load_state_dict(online.state_dict())
    l0, q0, _ = evaluate()
    print(json.dumps(dict(update=0, holdout_loss=round(l0, 5), holdout_q=round(q0, 3))), flush=True)
    nl = int(args.batch * args.live_frac)
    t0, acc = time.time(), 0.0
    for u in range(1, args.updates + 1):
        il = torch.as_tensor(rng.integers(0, n_live, nl), device=DEV)
        io = torch.as_tensor(rng.integers(0, n_old, args.batch - nl), device=DEV)
        batch = tuple(torch.cat([x[il], y[io]]) for x, y in zip(train_live, old))
        loss, _ = td(batch)
        if args.anchor:
            s_live = batch[0][:nl]
            with torch.no_grad():
                q_ref = ref(s_live)
            loss = loss + args.anchor * ((online(s_live) - q_ref) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        acc += float(loss)
        if u % args.target_every == 0:
            target.load_state_dict(online.state_dict())
        if u % args.ckpt_every == 0:
            hl, hq, agree = evaluate()
            e = dict(update=u, train_loss=round(acc / args.ckpt_every, 5), holdout_loss=round(hl, 5), holdout_q=round(hq, 3),
                     same_action_as_init=round(agree, 3), seconds=round(time.time() - t0))
            acc = 0.0
            print(json.dumps(e), flush=True)
            with open(run / "log.jsonl", "a") as f:
                f.write(json.dumps(e) + "\n")
            torch.save({"algo": "ddqn", "net": {k: v.cpu() for k, v in online.state_dict().items()}, "norm_mean": mean,
                        "norm_std": std, "hist": 4, "source": f"{args.name} update {u} from {args.init}"},
                       run / f"upd_{u:06d}.pt")


if __name__ == "__main__":
    main()
