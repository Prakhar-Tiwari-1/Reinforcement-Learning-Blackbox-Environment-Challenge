"""What changed in the contest world? Compare live recordings (data/live/*.npz, submission policy) with references.

References:
  - TRAINING world, same DDQN policy family: the last 1M transitions of runs/ddqn (frame t of each stored state,
    un-normalised) -> observation statistics only (rewards in the buffer are shaped n-step sums).
  - CONTEST world last week (15-16 Oct): data/eval_contest_*.npz (random and PPO policies) -> observation statistics.
Everything is reported as measured quantities: r, `died`, o0..o10.

Usage: .venv/bin/python src/explore/live_shift.py [glob of live files]  -> prints, and writes results/live_shift.txt
"""

import glob
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_live(pattern):
    O, A, R, D, W, T = [], [], [], [], [], []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        d = np.load(f)
        O.append(d["obs"][:-1])
        A.append(d["actions"])
        R.append(d["rewards"])
        D.append(d["died"])
        W.append(d["world_step"])
        T.append(d["wall"])
    return [np.concatenate(x) for x in (O, A, R, D, W, T)]


def obs_stats(O):
    return dict(mean=O.mean(0), o1_gt_09=(O[:, 1] > 0.9).mean(), o4_or_o7=((O[:, 4] > 0.5) | (O[:, 7] > 0.5)).mean(),
                o9_gt_075=(O[:, 9] > 0.75).mean(), o9_lt_02=(O[:, 9] < 0.2).mean(), o3_or_o6=((O[:, 3] > 0.5) | (O[:, 6] > 0.5)).mean())


def main(pattern="data/live/*.npz"):
    out = []
    pr = lambda s="": (print(s), out.append(s))  # noqa: E731
    O, A, R, D, W, T = load_live(pattern)
    n = len(R)
    pr(f"LIVE contest (submission policy): {n} steps, {len(glob.glob(str(ROOT / pattern)))} files")
    pr(f"  reward/1000 {R.sum() / n * 1000:+.1f} | r=+5 per 1000 {(R > 0).mean() * 1000:.1f} | died per 1000 {D.mean() * 1000:.2f}")
    gaps = np.diff(W)
    pr(f"  world step increments between our steps: mean {gaps.mean():.3f}, >1 in {(gaps > 1).mean():.1%} of steps; "
       f"wall time per step {np.median(np.diff(T)) * 1000:.0f} ms (median)")
    acts = A[:, 0] * 2 + A[:, 1]
    pr("  action use: " + ", ".join(f"{['[0,0]', '[0,1]', '[1,0]', '[1,1]'][j]} {(acts == j).mean():.1%}" for j in range(4)))

    # references
    ck = torch.load(ROOT / "models/FINAL_ddqn_500k.pt", weights_only=False, map_location="cpu")
    b = np.load(ROOT / "runs/ddqn/buffer.npz")
    Otr = b["s"][:, :11].astype(np.float32) * np.asarray(ck["norm_std"], np.float32) + np.asarray(ck["norm_mean"], np.float32)
    Oc = np.concatenate([np.load(f)["obs"][:-1, 0] for f in sorted(glob.glob(str(ROOT / "data/eval_contest_*.npz")))])
    refs = {"LIVE now (DDQN)": obs_stats(O), "training, DDQN": obs_stats(Otr), "contest 15-16 Oct (rand/PPO)": obs_stats(Oc)}
    pr("\nObservation statistics (fraction of steps / means)")
    pr(f"{'':30} {'o1>0.9':>7} {'o4|o7=1':>8} {'o3|o6=1':>8} {'o9>0.75':>8} {'o9<0.2':>7}")
    for k, s in refs.items():
        pr(f"{k:30} {s['o1_gt_09']:7.3f} {s['o4_or_o7']:8.3f} {s['o3_or_o6']:8.3f} {s['o9_gt_075']:8.3f} {s['o9_lt_02']:7.3f}")
    pr("\nMean of o0..o10")
    for k, s in refs.items():
        pr(f"{k:30} " + " ".join(f"{v:5.2f}" for v in s["mean"]))

    # conversion: how often a high o9 is followed by r=+5
    def conv(O, R, h=10):
        idx = np.where(O[:-h, 9] > 0.85)[0]
        hit = np.array([R[i:i + h].max() > 0 for i in idx]) if len(idx) else np.array([0.0])
        return len(idx), hit.mean()
    nl, cl = conv(O, R)
    pr(f"\nP(r=+5 within 10 steps | o9 > 0.85): LIVE {cl:.3f} over {nl} steps")
    Rc = np.concatenate([np.load(f)["rewards"][:, 0] for f in sorted(glob.glob(str(ROOT / "data/eval_contest_*.npz")))])
    nc, cc = conv(Oc, Rc)
    pr(f"                                       contest 15-16 Oct (rand/PPO) {cc:.3f} over {nc} steps")
    # steps before died
    di = np.where(D)[0]
    di = di[di >= 20]
    if len(di):
        pr(f"\nBefore `died` (LIVE, {len(di)} cases): mean o1 at t-20/-10/-5/-1: "
           + " / ".join(f"{O[di - k, 1].mean():.2f}" for k in (20, 10, 5, 1))
           + "; o8 at t-20/-10/-1: " + " / ".join(f"{O[di - k, 8].mean():.2f}" for k in (20, 10, 1))
           + f"; o4|o7=1 at t-1: {((O[di - 1, 4] > .5) | (O[di - 1, 7] > .5)).mean():.2f}")
    (ROOT / "results" / "live_shift.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
