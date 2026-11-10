"""How many steps ahead can the agent see food and damage coming? (GPU)

Events:
    food    reward > 0 within the next k steps
    damage  health (obs 8) drops by more than 0.01 in one step, or death, within the next k steps

For k in (1, 3, 5, 10, 20, 40):
  - single-signal AUC of each observation dim (and of its change over the last 3 steps)
  - AUC of a small MLP on the current observation vs on the last 4 observations stacked
And the mean of every dim as a function of "steps until the event".

Usage: uv run python src/explore/lead_time.py  -> results/lead_time.txt
"""

import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KS = (1, 3, 5, 10, 20, 40)


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ranks = (first + (counts + 1) / 2.0)[inv]  # average rank for ties (binary dims are mostly ties)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def within(ev, k):
    """within[t] = any event in ev[t : t+k]  (t = index of the observation the agent acts on)."""
    c = np.concatenate([[0], np.cumsum(ev)])
    T = len(ev)
    hi = np.minimum(np.arange(T) + k, T)
    return (c[hi] - c[np.arange(T)]) > 0


def load():
    O, H, F, Dm, until_f, until_d = [], [], [], [], [], []
    for p in sorted(glob.glob(str(ROOT / "data" / "*.npz"))):
        d = np.load(p)
        if "obs" not in d or "rewards" not in d:
            continue
        obs, R, D = d["obs"], d["rewards"], d["died"]
        T = len(R)
        for a in range(R.shape[1]):
            o = obs[:T, a]
            h8 = obs[:, a, 8]
            dmg = ((h8[1:T + 1] - h8[:T]) < -0.01) & ~D[:, a] | D[:, a]
            food = R[:, a] > 0
            hist = np.concatenate([np.vstack([np.repeat(o[:1], j, 0), o[:T - j]]) for j in range(4)], 1)
            O.append(o), H.append(hist), F.append(food), Dm.append(dmg)
            for ev, out in ((food, until_f), (dmg, until_d)):
                nxt = np.full(T, 10**6)
                last = 10**6
                for t in range(T - 1, -1, -1):
                    last = 0 if ev[t] else (last + 1 if last < 10**6 else last)
                    nxt[t] = last
                out.append(nxt)
    return (np.vstack(O), np.vstack(H), np.concatenate(F), np.concatenate(Dm),
            np.concatenate(until_f), np.concatenate(until_d))


def mlp_auc(X, y, train):
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    Xt = (Xt - Xt[train].mean(0)) / (Xt[train].std(0) + 1e-6)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    pw = torch.tensor((1 - y[train].mean()) / max(y[train].mean(), 1e-6), device=DEV)
    idx = torch.tensor(np.flatnonzero(train), device=DEV)
    for _ in range(6):
        for b in idx[torch.randperm(len(idx), device=DEV)].split(8192):
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[b]).squeeze(1), yt[b], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        s = torch.cat([net(Xt[i]).squeeze(1) for i in torch.arange(len(Xt), device=DEV).split(65536)]).cpu().numpy()
    return auc(s[~train], y[~train])


def main():
    torch.manual_seed(0)
    O, H, F, Dm, uf, ud = load()
    N = len(O)
    rng = np.random.default_rng(0)
    train = rng.random(N) < 0.8  # random split is fine here: labels are about the future, features are the present
    out = [f"{N} agent-steps, {F.sum()} food events, {Dm.sum()} damage events (health drop > 0.01 or death)\n"]

    out.append("## Mean of each dim vs steps until the next event (only steps whose next event is within 40)")
    for name, u in (("FOOD", uf), ("DAMAGE", ud)):
        out.append(f"\n### {name}\n  steps_until   " + " ".join(f"  d{i:<4d}" for i in range(11)))
        for k in (40, 20, 10, 5, 3, 2, 1, 0):
            m = u == k
            if m.sum():
                out.append(f"  {k:5d}  n={m.sum():6d} " + " ".join(f"{v:7.3f}" for v in O[m].mean(0)))
        far = u > 100
        out.append("  far (>100)     " + " ".join(f"{v:7.3f}" for v in O[far].mean(0)))

    delta = O - H[:, 11 * 3:11 * 4]  # change over the last 3 steps
    for name, ev in (("FOOD", F), ("DAMAGE", Dm)):
        out.append(f"\n## {name}: single-signal AUC (value of a dim, or its 3-step change) for 'event within k steps'")
        out.append("  (AUC > 0.5: higher value = event more likely; < 0.5: lower value = event more likely)")
        out.append("  k     " + " ".join(f"  d{i:<3d}" for i in range(11)) + "   | best 3-step changes")
        for k in KS:
            y = within(ev, k)
            vals = [auc(O[:, i], y) for i in range(11)]
            dvals = [auc(delta[:, i], y) for i in range(11)]
            top = sorted(range(11), key=lambda i: -abs(dvals[i] - 0.5))[:3]
            out.append(f"  {k:3d}  " + " ".join(f"{v:6.3f}" for v in vals) + "   | " +
                       ", ".join(f"Δd{i} {dvals[i]:.3f}" for i in top))
        out.append(f"\n## {name}: MLP AUC, current observation vs last 4 observations")
        for k in KS:
            y = within(ev, k)
            out.append(f"  k={k:3d}   now: {mlp_auc(O, y, train):.3f}   history(4): {mlp_auc(H, y, train):.3f}   "
                       f"(base rate {y.mean():.3f})")
    text = "\n".join(out)
    print(text)
    (ROOT / "results" / "lead_time.txt").write_text(text)


if __name__ == "__main__":
    main()
