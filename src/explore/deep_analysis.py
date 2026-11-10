"""Event-centred analysis: energy, starvation, causes of death, what each action does.

Works on any dataset(s) produced by collect.py (or evaluate.py). Pure numpy, offline.

Usage: uv run python src/explore/deep_analysis.py data/random.npz [data/sticky.npz ...]
"""

import sys

import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=150)
ACTION_NAMES = ["[0,0]", "[0,1]", "[1,0]", "[1,1]"]


def load(paths):
    """Concatenate per-agent trajectories: list of dicts (obs (T+1,k), act (T,), rew (T,), died (T,))."""
    trajs = []
    for p in paths:
        p, _, only = p.partition("#")          # "file.npz#0" = only agent 0 of that file
        d = np.load(p)
        O, A, R, D = d["obs"], d["actions"], d["rewards"], d["died"]
        for a in range(A.shape[1]):
            if only and a != int(only):
                continue
            trajs.append(dict(obs=O[:, a], act=A[:, a, 0] * 2 + A[:, a, 1], rew=R[:, a], died=D[:, a], src=p))
    return trajs


def event_window(trajs, mask_key, before, after, dims):
    """Mean of the given obs dims from `before` steps before to `after` steps after each event.

    obs index t+1 is the observation *after* action t, so offset 0 = observation right after the event step."""
    rows = []
    for tr in trajs:
        ev = np.flatnonzero(mask_key(tr))
        O = tr["obs"]
        for t in ev:
            lo, hi = t + 1 - before, t + 1 + after
            if lo >= 0 and hi < len(O):
                rows.append(O[lo:hi + 1][:, dims])
    return np.array(rows)


def lives(tr):
    """Split a trajectory into lives: list of (start, end) indices of action steps; end = death step."""
    deaths = np.flatnonzero(tr["died"])
    out, s = [], 0
    for e in deaths:
        out.append((s, e, True))
        s = e + 1
    out.append((s, len(tr["died"]) - 1, False))  # last life is censored (no death observed)
    return out


def main(paths):
    trajs = load(paths)
    T = sum(len(t["act"]) for t in trajs)
    print(f"data: {paths}  ({len(trajs)} agent trajectories, {T} agent-steps)")

    # ---------------------------------------------------------------- 1. dim 8 around food / death
    print("\n## 1. Observation dims around events (offset 0 = obs right after the event step)")
    for name, key in [("food (+5)", lambda tr: tr["rew"] > 0), ("death", lambda tr: tr["died"])]:
        W = event_window(trajs, key, 30, 10, list(range(11)))
        print(f"\n### {name}: n = {len(W)}")
        print("  offset " + " ".join(f"   d{i:<3d}" for i in range(11)))
        for off in [-30, -20, -10, -5, -3, -2, -1, 0, 1, 2, 5, 10]:
            print(f"  {off:+5d}  " + " ".join(f"{v:7.3f}" for v in W[:, off + 30].mean(0)))

    # ---------------------------------------------------------------- 2. lives: food per life, life length
    print("\n## 2. Lives")
    L = []
    for tr in trajs:
        for s, e, dead in lives(tr):
            if e < s:
                continue
            seg = slice(s, e + 1)
            L.append(dict(length=e - s + 1, food=int((tr["rew"][seg] > 0).sum()), dead=dead,
                          d8_start=tr["obs"][s + 1, 8] if s + 1 < len(tr["obs"]) else np.nan,
                          d8_end=tr["obs"][e, 8], moving=np.mean(tr["act"][seg] != 0)))
    full = [x for x in L if x["dead"]]
    lengths = np.array([x["length"] for x in full])
    food = np.array([x["food"] for x in full])
    print(f"  completed lives: {len(full)}   mean length {lengths.mean():.0f}  median {np.median(lengths):.0f}")
    print(f"  food per life: mean {food.mean():.2f}; distribution {dict(zip(*np.unique(food, return_counts=True)))}")
    for f in sorted(set(food.tolist()))[:6]:
        m = food == f
        print(f"    lives with {f} food: n={m.sum():4d}  mean length {lengths[m].mean():6.0f}  median {np.median(lengths[m]):6.0f}")
    print(f"  correlation(food in life, life length) = {np.corrcoef(food, lengths)[0, 1]:+.2f}")

    # ---------------------------------------------------------------- 3. dim 8 dynamics: drain rate, gain per food
    print("\n## 3. Dim 8 (energy?) dynamics")
    dd, acts, foodstep, deathstep = [], [], [], []
    for tr in trajs:
        O = tr["obs"]
        dd.append(O[1:, 8] - O[:-1, 8])
        acts.append(tr["act"])
        foodstep.append(tr["rew"] > 0)
        deathstep.append(tr["died"])
    dd, acts, foodstep, deathstep = map(np.concatenate, (dd, acts, foodstep, deathstep))
    normal = ~foodstep & ~deathstep
    print(f"  mean change per ordinary step: {dd[normal].mean():+.5f}  (median {np.median(dd[normal]):+.5f})")
    for j in range(4):
        m = normal & (acts == j)
        print(f"    action {ACTION_NAMES[j]}: {dd[m].mean():+.5f} per step")
    print(f"  change on food steps:  mean {dd[foodstep].mean():+.4f}  median {np.median(dd[foodstep]):+.4f}")
    print(f"  change on death steps: mean {dd[deathstep].mean():+.4f}")
    if dd[normal].mean() < 0:
        print(f"  => steps to drain from 1 to 0 at the average rate: {1 / -dd[normal].mean():.0f}")

    # ---------------------------------------------------------------- 4. causes of death
    print("\n## 4. What does the agent see just before dying? (obs before the death step)")
    pre = []
    for tr in trajs:
        for t in np.flatnonzero(tr["died"]):
            pre.append(tr["obs"][t])
    pre = np.array(pre)
    allobs = np.concatenate([tr["obs"][:-1] for tr in trajs])
    print(f"  deaths: {len(pre)}")
    print(f"  dim 8 before death: quantiles 10/50/90% = {np.percentile(pre[:, 8], [10, 50, 90])}  "
          f"(all steps: {np.percentile(allobs[:, 8], [10, 50, 90])})")
    print(f"  dim 1 before death: quantiles 10/50/90% = {np.percentile(pre[:, 1], [10, 50, 90])}  "
          f"(all steps: {np.percentile(allobs[:, 1], [10, 50, 90])})")
    low_e = pre[:, 8] < np.percentile(allobs[:, 8], 5)
    danger = (pre[:, 4] + pre[:, 7]) > 0
    print(f"  deaths with dim 8 in the lowest 5% of all steps: {low_e.mean():.1%}")
    print(f"  deaths with 'danger colour' (dim 4 or 7) on: {danger.mean():.1%}   (base rate {np.mean((allobs[:, 4] + allobs[:, 7]) > 0):.1%})")
    print(f"  deaths with neither: {np.mean(~low_e & ~danger):.1%}")
    # time since last food at death
    since = []
    for tr in trajs:
        last = -1
        for t in range(len(tr["act"])):
            if tr["rew"][t] > 0:
                last = t
            if tr["died"][t]:
                since.append(t - last if last >= 0 else np.nan)
                last = t
    since = np.array(since)
    print(f"  steps since last food (or respawn) at death: median {np.nanmedian(since):.0f}, "
          f"quantiles 10/90% {np.nanpercentile(since, [10, 90])}")

    # ---------------------------------------------------------------- 5. hazard: P(death) vs features
    print("\n## 5. Death hazard (P(die this step)) vs observation bins")
    died = np.concatenate([tr["died"] for tr in trajs])
    for dim in [1, 8, 9, 10]:
        x = allobs[:, dim]
        qs = np.unique(np.percentile(x, [0, 20, 40, 60, 80, 100]))
        line = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (x >= lo) & (x <= hi)
            line.append(f"[{lo:.2f},{hi:.2f}]: {died[m].mean() * 1000:5.1f}")
        print(f"  dim {dim:2d} (deaths per 1000): " + "  ".join(line))
    for dim in [0, 2, 3, 4, 5, 6, 7]:
        on = allobs[:, dim] > 0.5
        print(f"  dim {dim} on: {died[on].mean() * 1000:6.1f}  off: {died[~on].mean() * 1000:5.1f}  per 1000   "
              f"food on: {np.mean(np.concatenate([tr['rew'] for tr in trajs])[on] > 0) * 1000:5.1f}")

    # ---------------------------------------------------------------- 6. what actions do to vision
    print("\n## 6. Food rate vs dim 9 (P(food next step) per bin)")
    rew = np.concatenate([tr["rew"] for tr in trajs])
    x = allobs[:, 9]
    for lo, hi in [(-1, 0), (0, 0.001), (0.001, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)]:
        m = (x >= lo) & (x < hi)
        if m.sum():
            print(f"  dim 9 in [{lo:+.2f},{hi:.2f}): n={m.sum():6d}  P(food) {np.mean(rew[m] > 0) * 1000:6.1f}/1000")
    print("  P(food | dim 9 > 0.75) by action:")
    for j in range(4):
        m = (x > 0.75) & (acts == j)
        print(f"    {ACTION_NAMES[j]}: n={m.sum():5d}  {np.mean(rew[m] > 0) * 1000:6.1f}/1000")
    print("  change of dim 9 by action when 0.2 < dim 9 < 0.9 (does an action move toward food?):")
    d9 = np.concatenate([tr["obs"][1:, 9] - tr["obs"][:-1, 9] for tr in trajs])
    for j in range(4):
        m = (x > 0.2) & (x < 0.9) & (acts == j) & ~foodstep & ~deathstep
        print(f"    {ACTION_NAMES[j]}: n={m.sum():5d}  mean change {d9[m].mean():+.4f}")
    print("  vision changes by action (mean |left-right| asymmetry change), dims (2,5),(3,6),(4,7):")
    for j in range(4):
        m = (acts == j) & normal
        s = []
        for l, r in [(2, 5), (3, 6), (4, 7)]:
            dl = np.concatenate([tr["obs"][1:, l] - tr["obs"][:-1, l] for tr in trajs])
            dr = np.concatenate([tr["obs"][1:, r] - tr["obs"][:-1, r] for tr in trajs])
            s.append(f"({l},{r}): L {dl[m].mean():+.4f} R {dr[m].mean():+.4f}")
        print(f"    {ACTION_NAMES[j]}: " + "  ".join(s))


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/random.npz"])
