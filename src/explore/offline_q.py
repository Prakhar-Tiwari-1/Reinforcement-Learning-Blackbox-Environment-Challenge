"""Offline Q-learning on everything collected so far (GPU), to get a data-driven starting policy.

Two estimators, both with a Q-network  (obs 11 + previous action 2) -> Q value of each of the 4 actions,
two hidden layers of 64 units (128 in total), LayerNorm (as in PQN, Gallici et al. 2024):

  mc   Monte-Carlo targets  G_t = sum_{h<H} gamma^h r_{t+h}: the value of the *data-collecting* policies.
       Acting greedily on it = one step of policy improvement over them (policy iteration).
  fqi  Fitted Q-iteration: targets r_t + gamma * max_a' Q_old(s', a'), repeated. Estimates the value of the
       best policy the data supports (offline DQN). The random data covers all 4 actions everywhere, which
       limits the usual offline extrapolation problem.

The episode never ends and deaths are not terminal (the agent respawns), so no terminal masking.

Usage: uv run python src/explore/offline_q.py --method fqi --out data/q_fqi.pt
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def qnet():
    return nn.Sequential(nn.Linear(13, 64), nn.LayerNorm(64), nn.ReLU(),
                         nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 4))


def load(paths, gamma, horizon):
    S, PA, A, R, S2, A2prev, G, SRC = [], [], [], [], [], [], [], []
    for p in paths:
        d = np.load(p)
        O, Ac, Rw = d["obs"], d["actions"].astype(np.float32), d["rewards"]
        T, n = Ac.shape[:2]
        disc = gamma ** np.arange(horizon)
        for a in range(n):
            prev = np.vstack([np.zeros((1, 2), np.float32), Ac[:-1, a]])
            r = Rw[:, a]
            # Monte-Carlo truncated returns (only where the full horizon is available)
            g = np.convolve(r[::-1], disc[::-1], mode="full")[::-1][horizon - 1:horizon - 1 + T] if False else None
            g = np.array([np.dot(r[t:t + horizon], disc[:min(horizon, T - t)]) for t in range(T)], np.float32)
            valid = np.arange(T) < T - 1
            S.append(O[:-1, a][valid])
            PA.append(prev[valid])
            A.append((Ac[:, a, 0] * 2 + Ac[:, a, 1])[valid])
            R.append(r[valid])
            S2.append(O[1:, a][valid])
            A2prev.append(Ac[:, a][valid])
            G.append(g[valid])
            SRC.append(np.full(valid.sum(), (np.arange(T) < T - horizon)[valid].mean()))
    cat = lambda x: np.concatenate(x)  # noqa: E731
    return dict(s=cat(S), pa=cat(PA), a=cat(A).astype(np.int64), r=cat(R), s2=cat(S2), pa2=cat(A2prev), g=cat(G))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["mc", "fqi"], default="fqi")
    p.add_argument("--data", nargs="+", default=None)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--iters", type=int, default=30, help="FQI iterations")
    p.add_argument("--epochs", type=int, default=3, help="passes over the data per FQI iteration")
    p.add_argument("--out", default="data/q_fqi.pt")
    args = p.parse_args()
    torch.manual_seed(0)
    paths = args.data or sorted(glob.glob(str(ROOT / "data" / "*.npz")))
    t0 = time.time()
    D = load(paths, args.gamma, args.horizon)
    N = len(D["a"])
    mean, var = D["s"].mean(0), D["s"].var(0)
    norm = lambda x: np.clip((x - mean) / np.sqrt(var + 1e-8), -5, 5).astype(np.float32)  # noqa: E731
    X = torch.tensor(np.hstack([norm(D["s"]), D["pa"]]), device=DEV)
    X2 = torch.tensor(np.hstack([norm(D["s2"]), D["pa2"]]), device=DEV)
    Aa = torch.tensor(D["a"], device=DEV)
    Rr = torch.tensor(D["r"] / 5.0, device=DEV)          # rewards in units of "one event"
    Gg = torch.tensor(D["g"] / 5.0, device=DEV)
    print(f"{N} transitions from {len(paths)} files on {DEV}, loaded in {time.time() - t0:.0f}s")

    net = qnet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = 4096
    hist = []

    def fit(target, epochs):
        for _ in range(epochs):
            for idx in torch.randperm(N, device=DEV).split(bs):
                q = net(X[idx]).gather(1, Aa[idx, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q, target[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return float(loss)

    if args.method == "mc":
        loss = fit(Gg, 20)
        hist.append(dict(loss=loss))
    else:
        for it in range(args.iters):
            with torch.no_grad():
                target = torch.cat([Rr[i] + args.gamma * net(X2[i]).max(1).values
                                    for i in torch.arange(N, device=DEV).split(65536)])
            loss = fit(target, args.epochs)
            with torch.no_grad():
                q = torch.cat([net(X[i]) for i in torch.arange(N, device=DEV).split(65536)])
            greedy = torch.bincount(q.argmax(1), minlength=4).float() / N
            hist.append(dict(iter=it, loss=loss, mean_q=float(q.mean()), greedy_action_share=greedy.tolist()))
            print(json.dumps(hist[-1]), flush=True)

    with torch.no_grad():
        q = torch.cat([net(X[i]) for i in torch.arange(N, device=DEV).split(65536)]).cpu().numpy()
    share = np.bincount(q.argmax(1), minlength=4) / N
    print(f"done in {time.time() - t0:.0f}s; greedy action share [0,0],[0,1],[1,0],[1,1] = {share.round(3)}")
    torch.save({"net": {k: v.cpu() for k, v in net.state_dict().items()}, "norm_mean": mean, "norm_var": var,
                "args": vars(args), "data": [str(Path(x).name) for x in paths], "history": hist}, ROOT / args.out)


if __name__ == "__main__":
    main()
