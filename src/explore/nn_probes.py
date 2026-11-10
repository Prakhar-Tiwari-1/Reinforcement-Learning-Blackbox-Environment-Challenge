"""Neural-network probes on collected data, to understand the world (not to play it).

1. Reward model     (obs, action) -> P(reward > 0), P(reward < 0)
   - with vs without the action input: do our actions influence rewards?
   - permutation importance: which observation dims announce a reward?
2. Dynamics model   (obs, action) -> next obs
   - vs a "copy the last observation" baseline and vs a model without the action:
     how predictable is the world, and how much do our actions steer what we see?
3. Memory test      same models with the 3 previous observations stacked in:
   does history help (i.e. is one observation not Markov enough)?

All splits are chronological (first 80% train, last 20% test), so no leakage between
consecutive, highly correlated steps.

Usage: uv run python src/explore/nn_probes.py data/random.npz [data/sticky.npz ...]
"""

import sys

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
HIDDEN = 64


def load(paths, history=0):
    """Return X_obs (N, (history+1)*k), X_act (N, 4 one-hot), next obs, reward, chronological index."""
    Xo, Xa, Y, R, Tix = [], [], [], [], []
    for p in paths:
        d = np.load(p)
        O, A, Rw = d["obs"], d["actions"], d["rewards"]
        T, n = A.shape[:2]
        for a in range(n):
            for t in range(history, T):
                Xo.append(np.concatenate([O[t - h, a] for h in range(history + 1)]))
                Xa.append(np.eye(4)[A[t, a, 0] * 2 + A[t, a, 1]])
                Y.append(O[t + 1, a])
                R.append(Rw[t, a])
                Tix.append(t / T)
    f = lambda x: torch.tensor(np.array(x), dtype=torch.float32)  # noqa: E731
    return f(Xo), f(Xa), f(Y), f(R), np.array(Tix)


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, o))


def fit(model, X, Y, loss_fn, epochs=30, bs=512):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(X[b]), Y[b]).backward()
            opt.step()
    return model


def auc(score, label):
    """Area under the ROC curve (probability a positive is ranked above a negative)."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def reward_probe(Xo, Xa, R, train, test):
    # classes: 0 = no reward, 1 = positive, 2 = negative
    y = torch.zeros(len(R), dtype=torch.long)
    y[R > 0], y[R < 0] = 1, 2
    print(f"  test set: {int((y[test] == 1).sum())} positive, {int((y[test] == 2).sum())} negative rewards"
          f" out of {int(test.sum())}")
    weights = torch.tensor([1.0, 20.0, 20.0])  # rewards are rare: up-weight them
    loss = nn.CrossEntropyLoss(weight=weights)
    results = {}
    for name, X in [("obs+action", torch.cat([Xo, Xa], 1)), ("obs only", Xo), ("action only", Xa)]:
        m = fit(mlp(X.shape[1], 3), X[train], y[train], loss)
        with torch.no_grad():
            p = torch.softmax(m(X[test]), 1).numpy()
        yt = y[test].numpy()
        results[name] = (m, X)
        print(f"  {name:12s}  AUC(r>0) {auc(p[:, 1], yt == 1):.3f}   AUC(r<0) {auc(p[:, 2], yt == 2):.3f}")

    m, X = results["obs+action"]
    k = Xo.shape[1]
    with torch.no_grad():
        p = torch.softmax(m(X[test]), 1).numpy()
    yt = y[test].numpy()
    base_pos, base_neg = auc(p[:, 1], yt == 1), auc(p[:, 2], yt == 2)
    print("  permutation importance (AUC drop when a dim is shuffled), obs dims of the current step:")
    rng = np.random.default_rng(0)
    for i in range(min(k, 11)):
        Xp = X[test].clone()
        Xp[:, i] = Xp[rng.permutation(len(Xp)), i]
        with torch.no_grad():
            pp = torch.softmax(m(Xp), 1).numpy()
        print(f"    dim {i:2d}: r>0 {base_pos - auc(pp[:, 1], yt == 1):+.3f}   r<0 {base_neg - auc(pp[:, 2], yt == 2):+.3f}")


def dynamics_probe(Xo, Xa, Y, train, test, k):
    mse = nn.MSELoss()
    cur = Xo[:, :k]
    copy = ((cur[test] - Y[test]) ** 2).mean(0)
    out = {"copy last obs": copy}
    for name, X in [("obs+action", torch.cat([Xo, Xa], 1)), ("obs only", Xo)]:
        m = fit(mlp(X.shape[1], k), X[train], Y[train], mse)
        with torch.no_grad():
            out[name] = ((m(X[test]) - Y[test]) ** 2).mean(0)
    var = Y[test].var(0)
    print("  test MSE per next-obs dim (lower is better); 'R2' = 1 - MSE/variance")
    print("  dim   variance  copy_last   obs_only  obs+action   R2(obs+action)  action_gain")
    for i in range(k):
        gain = (out["obs only"][i] - out["obs+action"][i]) / max(out["obs only"][i], 1e-8)
        print(f"  {i:3d}   {var[i]:.4f}    {out['copy last obs'][i]:.4f}    {out['obs only'][i]:.4f}"
              f"    {out['obs+action'][i]:.4f}      {1 - out['obs+action'][i] / max(var[i], 1e-8):+.3f}"
              f"        {gain:+.1%}")
    tot = {n_: float(v.mean()) for n_, v in out.items()}
    print(f"  overall: {tot}")
    return tot


def main(paths):
    print(f"data: {paths}")
    for history in [0, 3]:
        Xo, Xa, Y, R, tix = load(paths, history)
        k = Y.shape[1]
        train, test = torch.tensor(tix < 0.8), torch.tensor(tix >= 0.8)
        print(f"\n========== history = {history} previous observations (input dim {Xo.shape[1]}) ==========")
        print("\n# 1. Reward model")
        reward_probe(Xo, Xa, R, train, test)
        print("\n# 2. Dynamics model")
        dynamics_probe(Xo, Xa, Y, train, test, k)


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/random.npz"])
