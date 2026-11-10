"""Is the training world stationary? Food and death rates per time window over a long run.

Also counts entities per type in the global state snapshots (analysis only: state() is forbidden
in the contest and never used by the agent).

Usage: uv run python src/explore/stationarity.py data/long_random.npz [window]
"""

import sys

import numpy as np


def main(path, window=3000):
    d = np.load(path)
    R, D = d["rewards"], d["died"]
    T = len(R)
    print(f"{path}: {T} steps x {R.shape[1]} agents, window {window}")
    print("  window          food/1000  deaths/1000  reward/1000   mean dim9   mean dim1")
    rates = []
    for s in range(0, T - window + 1, window):
        r, dd = R[s:s + window], D[s:s + window]
        o = d["obs"][s:s + window]
        f, de = (r > 0).mean() * 1000, dd.mean() * 1000
        rates.append((f, de))
        print(f"  {s:6d}-{s + window:6d}   {f:8.2f}   {de:9.2f}   {r.mean() * 1000:+10.1f}   {o[..., 9].mean():9.3f}   {o[..., 1].mean():9.3f}")
    rates = np.array(rates)
    half = len(rates) // 2
    print(f"  first half: food {rates[:half, 0].mean():.2f} deaths {rates[:half, 1].mean():.2f}   "
          f"second half: food {rates[half:, 0].mean():.2f} deaths {rates[half:, 1].mean():.2f}")
    if "states" in d and len(d["states"]):
        S, st = d["states"], d["state_t"]
        print("\n  entity counts by type (column 0 of state, rows with any non-zero), over time:")
        for i in np.linspace(0, len(S) - 1, 6).astype(int):
            rows = S[i][np.abs(S[i]).sum(1) > 0]
            types, cnt = np.unique(rows[:, 0], return_counts=True)
            print(f"    step {st[i]:6d}: total {len(rows):3d}  " + "  ".join(f"type {t:+.0f}: {c}" for t, c in zip(types, cnt)))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
