"""Rebuild a DDQN replay buffer with a longer history (8 frames) and longer returns (10-step), offline.

The online part of runs/ddqn/buffer.npz was written in order: at each server step, one transition per agent
(4 agents), so entry x and entry x+4 belong to the same agent one step apart. Each stored state holds frames
t, t-1, t-2, t-3 (11 numbers each) + the previous action; the stored return is the 5-step return.
    8-frame state at t    = frames t..t-3 of entry k  +  frames t-4..t-7 of entry k-16 (same agent, 4 steps earlier)
    10-step return        = R5(k) + gamma^5 * R5(k+20)            (entry k+20 = same agent, 5 steps later)
    next state (t+10)     = frames of entry k+40 + frames of entry k+24, previous action of entry k+40
An entry is kept only if every consecutive pair x -> x+4 between k-16 and k+40 is the same agent one step
later (frame t of x equals frame t-1 of x+4). This drops wrap-around points, resets and restarts.

Usage: .venv/bin/python src/train/widen_buffer.py runs/ddqn/buffer.npz runs/h8n10_buffer.npz
"""
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
GAMMA, F = 0.99, 11
d = np.load(src)
n, i0 = int(d["n"]), int(d["i"])
order = np.r_[i0:n, 0:i0] if i0 < n else np.arange(n)       # oldest -> newest
s, a, r, g = (d[x][order] for x in ("s", "a", "r", "g"))
cont = np.zeros(n, bool)                                      # cont[x]: x+4 continues x (same agent, next step)
cont[:-4] = np.all(s[4:, F:2 * F] == s[:-4, :F], axis=1) & np.any(s[:-4, :F] != s[:-4, F:2 * F], axis=1)
k = np.arange(16, n - 40)
ok = np.ones(len(k), bool)
for m in range(-16, 40, 4):
    ok &= cont[k + m]
ok &= np.isclose(g[k], GAMMA ** 5) & np.isclose(g[k + 20], GAMMA ** 5)
k = k[ok]
S = np.concatenate([s[k, :4 * F], s[k - 16, :4 * F], s[k, 4 * F:]], 1)
S2 = np.concatenate([s[k + 40, :4 * F], s[k + 24, :4 * F], s[k + 40, 4 * F:]], 1)
R = r[k] + GAMMA ** 5 * r[k + 20]
m = len(k)
np.savez(dst, s=S.astype(np.float16), s2=S2.astype(np.float16), a=a[k], r=R.astype(np.float32),
         g=np.full(m, GAMMA ** 10, np.float32), i=m, n=m)
print(f"{n} entries -> kept {m} ({m / n:.1%}) with history 8 and 10-step returns -> {dst}")
# self-check: the stored 5-step next state of k must be entry k+20 (same agent, 5 steps later)
s2 = d["s2"][order]
print("check s2[k] == s[k+20]:", float(np.mean(np.all(s2[k] == s[k + 20], axis=1))))
