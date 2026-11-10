"""Statistical analysis of a dataset collected by collect.py.

Usage (from the project root):
    uv run python src/explore/analyze.py data/random.npz
"""

import sys

import numpy as np

ACTION_NAMES = ["[0,0]", "[0,1]", "[1,0]", "[1,1]"]


def action_index(a):
    return a[..., 0] * 2 + a[..., 1]


def main(path):
    d = np.load(path)
    O, A, R, D = d["obs"], d["actions"], d["rewards"], d["died"]
    T, n, k = A.shape[0], A.shape[1], O.shape[2]
    obs, nxt = O[:-1], O[1:]           # obs[t] seen before action t, nxt[t] seen after it
    ai = action_index(A)
    print(f"dataset {path}: {T} steps x {n} agents, obs dim {k}")

    # ---- rewards ----------------------------------------------------------------------------
    print("\n## Rewards")
    vals, counts = np.unique(R, return_counts=True)
    print("  distinct reward values:", dict(zip(vals.tolist(), counts.tolist())))
    print(f"  mean reward per agent-step: {R.mean():+.4f}   per agent: {np.round(R.mean(0), 4)}")
    print(f"  deaths: {D.sum()} total ({D.sum() / (T * n) * 1000:.2f} per 1000 agent-steps)")
    print(f"  reward on death steps: {np.unique(R[D], return_counts=True)}")
    print(f"  reward on non-death steps (nonzero only): {np.unique(R[~D & (R != 0)], return_counts=True)}")

    # ---- per-action effects -------------------------------------------------------------------
    print("\n## Effect of each action (random-policy data)")
    print("  action   mean_r    P(r>0)   P(r<0)   P(died)   mean obs change per dim")
    for j in range(4):
        m = ai == j
        delta = (nxt - obs)[m].mean(0)
        print(f"  {ACTION_NAMES[j]}  {R[m].mean():+.4f}  {np.mean(R[m] > 0):.4f}  {np.mean(R[m] < 0):.4f}"
              f"  {D[m].mean():.4f}   {np.round(delta, 3)}")

    # ---- observations -----------------------------------------------------------------------
    print("\n## Observation dims")
    flat = obs.reshape(-1, k)
    print("  dim   mean   std    min    max   frac0  n_unique  lag1_autocorr")
    for i in range(k):
        x = obs[:, :, i]
        ac = np.mean([np.corrcoef(x[:-1, a], x[1:, a])[0, 1] for a in range(n) if x[:, a].std() > 0])
        print(f"  {i:3d}  {flat[:, i].mean():.3f}  {flat[:, i].std():.3f}  {flat[:, i].min():+.3f}"
              f"  {flat[:, i].max():.3f}  {np.mean(flat[:, i] == 0):.2f}  {len(np.unique(np.round(flat[:, i], 4))):7d}"
              f"   {ac:.3f}")

    print("\n  correlation between dims (|r| > 0.3):")
    C = np.corrcoef(flat.T)
    for i in range(k):
        for j in range(i + 1, k):
            if abs(C[i, j]) > 0.3:
                print(f"    dim {i} ~ dim {j}: {C[i, j]:+.2f}")

    # ---- what precedes rewards and deaths -------------------------------------------------------
    print("\n## Observation right before events vs baseline (mean of each dim)")
    base = flat.mean(0)
    for label, mask in [("r>0", R > 0), ("r<0", R < 0), ("died", D)]:
        if mask.sum():
            print(f"  {label:5s} (n={mask.sum():4d}) before: {np.round(obs[mask].mean(0) - base, 3)}")
            print(f"  {'':5s}            after : {np.round(nxt[mask].mean(0) - base, 3)}")

    # ---- lifetimes ----------------------------------------------------------------------------
    print("\n## Lifetimes (steps between deaths)")
    lifetimes = []
    for a in range(n):
        idx = np.flatnonzero(D[:, a])
        lifetimes += np.diff(np.concatenate([[-1], idx])).tolist()
    if lifetimes:
        lt = np.array(lifetimes)
        print(f"  n={len(lt)}  mean {lt.mean():.0f}  median {np.median(lt):.0f}  min {lt.min()}  max {lt.max()}")

    # ---- slow trends: does a dim drift over a life? (e.g. energy) --------------------------------
    print("\n## Mean of each dim vs steps since last death (life phase)")
    since = np.zeros((T, n), int)
    for a in range(n):
        c = 0
        for t in range(T):
            since[t, a] = c
            c = 0 if D[t, a] else c + 1
    for lo, hi in [(0, 10), (10, 50), (50, 150), (150, 400), (400, 10**9)]:
        m = (since >= lo) & (since < hi)
        if m.sum():
            print(f"  {lo:4d}-{hi if hi < 10**9 else 'inf':>4}: n={m.sum():6d}  {np.round(obs[m].mean(0), 3)}")

    # ---- global state (for our understanding only; not available to the agent in the contest) --
    if "states" in d and len(d["states"]):
        S = d["states"]
        print(f"\n## Global state snapshots: {S.shape}")
        rows_used = (np.abs(S).sum(2) > 0).sum(1)
        print(f"  non-empty rows per snapshot: min {rows_used.min()} max {rows_used.max()} mean {rows_used.mean():.1f}")
        last = S[-1][np.abs(S[-1]).sum(1) > 0]
        types, cnt = np.unique(last[:, 0], return_counts=True)
        print(f"  column 0 values in last snapshot (entity type?): {dict(zip(types.tolist(), cnt.tolist()))}")
        print("  non-zero columns:", np.flatnonzero(np.abs(S.reshape(-1, S.shape[2])).sum(0) > 0).tolist())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/random.npz")
