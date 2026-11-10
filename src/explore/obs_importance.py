"""How many of the 11 observation dims are really useful? (GPU, offline)

1. Predictive value: MLP predicting "food within 10 steps" and "damage within 10 steps"
   - with all 11 dims, leaving each dim out (AUC drop), and with only the best k dims (greedy forward selection).
2. Use by the trained policy (data/sub_fine.pt): shuffle one dim across the recorded observations and measure
   how much the action probabilities change (mean |delta p|), i.e. how much the agent relies on that input.

Usage: uv run python src/explore/obs_importance.py  -> results/obs_importance.txt
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "explore"))
from lead_time import auc, load, within  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_auc(X, y, train, epochs=5):
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    Xt = (Xt - Xt[train].mean(0)) / (Xt[train].std(0) + 1e-6)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    pw = torch.tensor((1 - y[train].mean()) / max(y[train].mean(), 1e-6), device=DEV)
    idx = torch.tensor(np.flatnonzero(train), device=DEV)
    for _ in range(epochs):
        for b in idx[torch.randperm(len(idx), device=DEV)].split(16384):
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[b]).squeeze(1), yt[b], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        s = net(Xt[torch.tensor(~train, device=DEV)]).squeeze(1).cpu().numpy()
    return auc(s, y[~train])


def main():
    torch.manual_seed(0)
    O, H, F, Dm, _, _ = load()
    N = len(O)
    train = np.random.default_rng(0).random(N) < 0.8
    out = [f"{N} agent-steps\n"]
    targets = {"food<=10": within(F, 10), "damage<=10": within(Dm, 10)}
    dims = list(range(11))
    # (H holds the last 4 observations; use current obs + its 3-step change, so "trend" info is included)
    trend = O - H[:, 33:44]
    feat = lambda ds: np.hstack([O[:, ds], trend[:, ds]])  # noqa: E731

    out.append("## 1a. Leave-one-out: AUC with all dims, and AUC drop when one dim (value + trend) is removed")
    out.append(f"  {'target':11s} {'all 11':>7s}  " + " ".join(f"  -d{i:<3d}" for i in dims))
    for name, y in targets.items():
        full = fit_auc(feat(dims), y, train)
        drops = [full - fit_auc(feat([d for d in dims if d != i]), y, train) for i in dims]
        out.append(f"  {name:11s} {full:7.3f}  " + " ".join(f"{v:+7.3f}" for v in drops))
        print(out[-1], flush=True)

    out.append("\n## 1b. Greedy forward selection: best k dims (value + trend) and the AUC they reach")
    for name, y in targets.items():
        chosen, line = [], []
        for _ in range(6):
            best = max((d for d in dims if d not in chosen), key=lambda d: fit_auc(feat(chosen + [d]), y, train, 3))
            chosen.append(best)
            line.append(f"+d{best} -> {fit_auc(feat(chosen), y, train):.3f}")
        out.append(f"  {name:11s} " + "   ".join(line))
        print(out[-1], flush=True)

    out.append("\n## 2. How much the trained policy (data/sub_fine.pt) uses each input")
    spec = importlib.util.spec_from_file_location("sub", ROOT / "prakhar.tiwari.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    from gymnasium.spaces import Box, MultiBinary

    ag = sub.Agent(Box(0, 1, (11,), np.float32), MultiBinary(2), learn=False, pretrained=None)
    ag.load(ROOT / "data" / "sub_fine.pt")
    rng = np.random.default_rng(1)
    S = O[rng.choice(N, 200_000, replace=False)]
    prev = rng.integers(0, 2, (len(S), 2)).astype(np.float32)

    def probs(x):
        z = np.clip((x - ag.enc.mean) / np.sqrt(ag.enc.var + 1e-8), -5, 5).astype(np.float32)
        with torch.no_grad():
            return torch.sigmoid(ag.net.actor(torch.tensor(np.hstack([z, prev])))).numpy()

    base = probs(S)
    eff = []
    for i in dims:
        Sp = S.copy()
        Sp[:, i] = Sp[rng.permutation(len(Sp)), i]
        eff.append(np.abs(probs(Sp) - base).mean())
    tot = sum(eff)
    for i in sorted(dims, key=lambda i: -eff[i]):
        out.append(f"  d{i:<2d}  mean |change of action probability| {eff[i]:.4f}   share {eff[i] / tot:5.1%}")
    text = "\n".join(out)
    print(text)
    (ROOT / "results" / "obs_importance.txt").write_text(text)


if __name__ == "__main__":
    main()
