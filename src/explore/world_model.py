"""Architecture B prototype: a learned world model (GPU) used by a short-horizon MPC planner.

Model (one network, 2 x 64 = 128 hidden units):
    input  = normalised obs (11) + previous action (2) + action one-hot (4)
    output = predicted change of the observation (11)  +  logits of [nothing, food, death] for this step

Planner (policies.MPCPolicy): for each of 16 plans "action a1 for 2 steps, then a2 for H-2 steps",
roll the model forward and score  sum_t gamma^t (P(food) - P(death)); play the first action of the best plan.

Usage: uv run python src/explore/world_model.py --out data/world_model.pt
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model():
    return nn.Sequential(nn.Linear(17, 64), nn.LayerNorm(64), nn.ReLU(),
                         nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 11 + 3))


def load(paths):
    X, Y, E, TT = [], [], [], []
    for p in paths:
        d = np.load(p)
        O, A, R, D = d["obs"], d["actions"].astype(np.float32), d["rewards"], d["died"]
        T, n = A.shape[:2]
        for a in range(n):
            prev = np.vstack([np.zeros((1, 2), np.float32), A[:-1, a]])
            onehot = np.eye(4, dtype=np.float32)[(A[:, a, 0] * 2 + A[:, a, 1]).astype(int)]
            X.append(np.hstack([O[:-1, a], prev, onehot]))
            Y.append(O[1:, a] - O[:-1, a])
            ev = np.zeros(T, np.int64)
            ev[R[:, a] > 0], ev[D[:, a]] = 1, 2
            E.append(ev)
            TT.append(np.arange(T) / T)
    return np.vstack(X), np.vstack(Y), np.concatenate(E), np.concatenate(TT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/world_model.pt")
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()
    torch.manual_seed(0)
    paths = sorted(glob.glob(str(ROOT / "data" / "*.npz")))
    X, Y, E, TT = load(paths)
    mean, var = X[:, :11].mean(0), X[:, :11].var(0)
    dscale = Y.std(0) + 1e-6
    Xn = X.copy()
    Xn[:, :11] = np.clip((X[:, :11] - mean) / np.sqrt(var + 1e-8), -5, 5)
    tr, te = TT < 0.8, TT >= 0.8
    t = lambda a: torch.tensor(a, device=DEV)  # noqa: E731
    Xt, Yt, Et = t(Xn.astype(np.float32)), t((Y / dscale).astype(np.float32)), t(E)
    print(f"{len(X)} transitions, {int((E == 1).sum())} food, {int((E == 2).sum())} deaths, from {len(paths)} files on {DEV}")

    net = model().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    w = torch.tensor([1.0, 10.0, 10.0], device=DEV)
    idx_tr = torch.tensor(np.flatnonzero(tr), device=DEV)
    for ep in range(args.epochs):
        for b in idx_tr[torch.randperm(len(idx_tr), device=DEV)].split(4096):
            out = net(Xt[b])
            loss = nn.functional.mse_loss(out[:, :11], Yt[b]) + nn.functional.cross_entropy(out[:, 11:], Et[b], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        out = net(Xt[torch.tensor(te, device=DEV)])
        mse = ((out[:, :11] - Yt[torch.tensor(te, device=DEV)]) ** 2).mean(0).cpu().numpy()
        pr = torch.softmax(out[:, 11:], 1).cpu().numpy()
    e = E[te]

    def auc(s, l):
        o = np.argsort(np.concatenate([s[l], s[~l]]))
        r = np.empty(len(o))
        r[o] = np.arange(1, len(o) + 1)
        npos = l.sum()
        return (r[:npos].sum() - npos * (npos + 1) / 2) / (npos * (~l).sum())

    print(f"test: normalised delta-MSE per dim {mse.round(3)}")
    print(f"test: AUC food {auc(pr[:, 1], e == 1):.3f}   AUC death {auc(pr[:, 2], e == 2):.3f}")
    torch.save({"net": {k: v.cpu() for k, v in net.state_dict().items()}, "norm_mean": mean, "norm_var": var,
                "dscale": dscale, "data": [Path(x).name for x in paths]}, ROOT / args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
