"""Double DQN (with dueling head, n-step returns, LayerNorm) on the TRAINING server, crash-proof.

Network (= the policy, so <= 128 hidden units): input 46 = last 4 observations (normalised) + previous action,
    46 -> 64 -> 64 (LayerNorm + ReLU) -> dueling head: V(s) + A(s,a) - mean_a A(s,a), 4 joint actions
    ([0,0], [0,1], [1,0], [1,1]). The target network is a copy of the same size, used only to compute targets.

Learning (per environment step, 4 agents -> 4 new transitions):
    n-step Double-DQN target   y = sum_{i<n} gamma^i r_i + gamma^n * Q_target(s_n, argmax_a Q_online(s_n, a))
    Huber loss, Adam, gradient clipping, `--updates-per-step` minibatches of `--batch` on the GPU,
    target network refreshed every `--target-every` updates.
    Rewards: true reward (+ optional potential-based shaping, same as the PPO run), scaled by 0.2.
    Deaths are not terminal (the agent respawns; the episode never ends).
Exploration: temporally-extended epsilon-greedy ("ez-greedy", Dabney et al. 2020): with probability eps a random
    action is chosen and HELD for 1, 2, 4 or 8 steps. eps decays linearly from --eps-start to --eps-end.
Replay buffer: --buffer transitions; pre-filled from every recording in data/ (and data/moves/), i.e. the
    ~900k agent-steps collected so far (off-policy learning can reuse them).

Files in runs/<name>/ (same scheme as train.py): config.json, latest.pt (online + target + optimizer + counters),
    buffer.npz (replay buffer, saved with latest.pt), step_*.pt (last 3), best_*.pt (top 3 by recent true reward),
    log.jsonl. Restarting the same command resumes.

Usage:  scripts/rl.sh job ddqn src/train/ddqn.py --name ddqn --total-steps 500000
        scripts/rl.sh job ddqn_ft src/train/ddqn.py --name ddqn_ft --init models/FINAL_ddqn_500k.pt \
            --lr 3e-5 --eps-start 0.02 --eps-end 0.005 --eps-steps 200000 --total-steps 300000   (fine-tune)
"""

import argparse
import glob
import json
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("sub", ROOT / "prakhar.tiwari.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIONS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
HIST = 4
DEFAULTS = dict(init="none", total_steps=500_000, buffer=1_000_000, batch=256, updates_per_step=2, lr=1e-4, gamma=0.99, n_step=5,
                target_every=2000, eps_start=0.3, eps_end=0.02, eps_steps=300_000, shaping=True, prefill=True, hist=HIST,
                buffer_from="none", coplayers="none", n_learners=4, coplayer_switch=5000, pool_schedule="none", phase_steps=100_000, died_weight=1.0, food_weight=1.0, w_danger=0.5,
                log_every=4096, ckpt_every=20480, keep_last=3, keep_best=3, best_window=40960, seed=0)
STOP = False


def _stop(*_):
    global STOP
    STOP = True


class QNet(nn.Module):
    """Dueling Q-network: 2 hidden layers of 64 (= 128 hidden units, the policy limit)."""

    def __init__(self, d, n_actions=4):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU())
        self.v, self.a = nn.Linear(64, 1), nn.Linear(64, n_actions)

    def forward(self, x):
        h = self.body(x)
        a = self.a(h)
        return self.v(h) + a - a.mean(-1, keepdim=True)


class Features:
    """Fixed normalisation (computed once from recorded data) so that stored features stay consistent."""

    def __init__(self, mean, std, n_hist=HIST):
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.n_hist, self.hist, self.prev = n_hist, {}, {}

    def reset(self, i=None):
        for d in (self.hist, self.prev):
            d.clear() if i is None else d.pop(i, None)

    def __call__(self, i, obs):
        h = self.hist.setdefault(i, deque(maxlen=self.n_hist))
        h.appendleft(np.clip((np.asarray(obs, np.float32) - self.mean) / self.std, -5, 5))
        fr = list(h) + [h[-1]] * (self.n_hist - len(h))
        return np.concatenate(fr + [self.prev.get(i, np.zeros(2, np.float32))]).astype(np.float32)


class Replay:
    def __init__(self, cap, d):
        self.cap, self.n, self.i = cap, 0, 0
        self.s = np.zeros((cap, d), np.float16)
        self.s2 = np.zeros((cap, d), np.float16)
        self.a = np.zeros(cap, np.int8)
        self.r = np.zeros(cap, np.float32)
        self.g = np.zeros(cap, np.float32)   # discount to apply to the bootstrap: gamma^k (0 if terminal)

    def add(self, s, a, r, s2, g):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.g[self.i] = s, a, r, s2, g
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        idx = np.random.randint(0, self.n, b)
        f = lambda x: torch.tensor(x[idx], device=DEV)  # noqa: E731
        return f(self.s).float(), f(self.a).long(), f(self.r), f(self.s2).float(), f(self.g)

    def save(self, path):
        tmp = f"{path}.tmp.npz"
        np.savez(tmp, s=self.s[: self.n], s2=self.s2[: self.n], a=self.a[: self.n], r=self.r[: self.n], g=self.g[: self.n],
                 i=self.i, n=self.n)
        os.replace(tmp, path)

    def load(self, path):
        d = np.load(path)
        n = int(d["n"])
        self.s[:n], self.s2[:n], self.a[:n], self.r[:n], self.g[:n] = d["s"], d["s2"], d["a"], d["r"], d["g"]
        self.n, self.i = n, int(d["i"]) % self.cap


class NStep:
    """Per-agent n-step accumulator: emits (s0, a0, R_n, s_n, gamma^n)."""

    def __init__(self, n, gamma):
        self.n, self.gamma, self.q = n, gamma, {}

    def push(self, i, s, a, r, s2):
        q = self.q.setdefault(i, deque())
        q.append((s, a, r, s2))
        out = []
        if len(q) == self.n:
            R = sum(self.gamma ** k * q[k][2] for k in range(self.n))
            out.append((q[0][0], q[0][1], R, s2, self.gamma ** self.n))
            q.popleft()
        return out

    def clear(self, i=None):
        self.q.clear() if i is None else self.q.pop(i, None)


def recorded_transitions(feat, nstep, shaping, gamma):
    """Turn every recorded run in data/ into n-step transitions (for pre-filling the buffer)."""
    files = sorted(glob.glob(str(ROOT / "data" / "*.npz"))) + sorted(f for f in glob.glob(str(ROOT / "data" / "moves" / "chunk_*.npz")) if ".tmp" not in f)
    for p in files:
        d = np.load(p)
        if not {"obs", "actions", "rewards", "died"} <= set(d.files):
            continue
        O, A, R, D = d["obs"], d["actions"], d["rewards"], d["died"]
        T = len(R)
        for i in range(R.shape[1]):
            feat.reset()
            nstep.clear()
            s = feat(0, O[0, i])
            for t in range(T):
                a = int(A[t, i, 0]) * 2 + int(A[t, i, 1])
                r = float(R[t, i]) + (shaping(O[t, i], O[t + 1, i], bool(D[t, i])) if shaping else 0.0)
                feat.prev[0] = A[t, i].astype(np.float32)
                s2 = feat(0, O[t + 1, i])
                yield from nstep.push(0, s, a, 0.2 * r, s2)
                s = s2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes", "on"), default=None)
        else:
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)
    args = p.parse_args()
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULTS)
    if (run / "config.json").exists():
        cfg.update(json.load(open(run / "config.json")))
    for k in DEFAULTS:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    json.dump(cfg, open(run / "config.json", "w"), indent=1)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(2)
    sub.load_dotenv_if_any()
    import pzclient

    # --w-danger: weight of the shaping term -o1 * max(o4, o7) (signal before `died`; default 0.5 as before)
    shaping = sub.ContestShaping(cfg["gamma"], w_danger=cfg["w_danger"]) if cfg["shaping"] else None
    # Learning reward only (logged scores stay the true reward): `died` counts -5 * died_weight, r = +5 counts
    # +5 * food_weight. died_weight > 1 makes avoiding `died` (e.g. collisions in a crowded world) worth more.
    def learn_reward(r, died):
        return r + (-5.0 * (cfg["died_weight"] - 1.0) if died else 0.0) + (5.0 * (cfg["food_weight"] - 1.0) if r > 0 else 0.0)
    H = cfg["hist"]
    d_in = 11 * H + 2
    online, target = QNet(d_in).to(DEV), QNet(d_in).to(DEV)
    opt = torch.optim.Adam(online.parameters(), lr=cfg["lr"])
    buf = Replay(cfg["buffer"], d_in)
    state = {"step": 0, "updates": 0, "best": [], "restarts": 0}

    if (run / "latest.pt").exists():
        ck = torch.load(run / "latest.pt", weights_only=False)
        online.load_state_dict(ck["net"])
        target.load_state_dict(ck["target"])
        opt.load_state_dict(ck["opt"])
        for g in opt.param_groups:          # the configured lr wins over the one saved in the optimizer state
            g["lr"] = cfg["lr"]
        mean, std = ck["norm_mean"], ck["norm_std"]
        state.update(ck["trainer"])
        state["restarts"] += 1
        if (run / "buffer.npz").exists():
            buf.load(run / "buffer.npz")
        print(f"RESUMED {args.name} at step {state['step']} (buffer {buf.n})", flush=True)
    else:
        allobs = np.concatenate([np.load(f)["obs"].reshape(-1, 11) for f in glob.glob(str(ROOT / "data" / "*.npz"))
                                 if "obs" in np.load(f).files])
        mean, std = allobs.mean(0), allobs.std(0) + 1e-6
        if cfg["init"] != "none":             # start from a saved DDQN policy (e.g. models/FINAL_ddqn_500k.pt)
            ck0 = torch.load(ROOT / cfg["init"], weights_only=False)
            sd = ck0["net"]
            w = sd["body.0.weight"]
            if w.shape[1] < d_in:              # longer history: zero weights for the extra (older) frames,
                old = w.shape[1] - 2           # so the network starts out computing exactly the same Q-values
                sd = dict(sd)
                sd["body.0.weight"] = torch.cat([w[:, :old], torch.zeros(w.shape[0], d_in - 2 - old, dtype=w.dtype, device=w.device), w[:, old:]], 1)
                print(f"widened the input from {w.shape[1]} to {d_in} (zero weights for the older frames)", flush=True)
            online.load_state_dict(sd)
            mean, std = ck0["norm_mean"], ck0["norm_std"]   # keep the normalisation the network was trained with
            print(f"initialised from {cfg['init']}", flush=True)
        target.load_state_dict(online.state_dict())
        if cfg["prefill"]:
            t0 = time.time()
            for tr in recorded_transitions(Features(mean, std, H), NStep(cfg["n_step"], cfg["gamma"]), shaping, cfg["gamma"]):
                buf.add(*tr)
            print(f"pre-filled the buffer with {buf.n} recorded transitions in {time.time() - t0:.0f}s", flush=True)
        if cfg["buffer_from"] != "none":    # reuse another run's replay buffer (same features and n-step only)
            buf.load(ROOT / cfg["buffer_from"])
            print(f"loaded the replay buffer {cfg['buffer_from']} ({buf.n} transitions)", flush=True)
        print(f"NEW run {args.name}: Q-network {d_in}-64-64-4 (128 hidden units), dueling, device {DEV}", flush=True)
    feat = Features(mean, std, H)
    nstep = NStep(cfg["n_step"], cfg["gamma"])

    def save_all():
        tmp = run / "latest.pt.tmp"
        torch.save({"algo": "ddqn", "net": online.state_dict(), "target": target.state_dict(), "opt": opt.state_dict(),
                    "norm_mean": mean, "norm_std": std, "hist": H, "trainer": state}, tmp)
        os.replace(tmp, run / "latest.pt")
        buf.save(run / "buffer.npz")

    def save_policy(path):
        tmp = f"{path}.tmp"
        torch.save({"algo": "ddqn", "net": online.state_dict(), "norm_mean": mean, "norm_std": std, "hist": H}, tmp)
        os.replace(tmp, path)

    def learn():
        s, a, r, s2, g = buf.sample(cfg["batch"])
        with torch.no_grad():
            a2 = online(s2).argmax(1, keepdim=True)                       # Double DQN: online picks...
            y = r + g * target(s2).gather(1, a2).squeeze(1)               # ...target evaluates
        q = online(s).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        state["updates"] += 1
        if state["updates"] % cfg["target_every"] == 0:
            target.load_state_dict(online.state_dict())
        return float(loss), float(q.mean())

    url = os.environ.get("PZ_TRAIN_URL", "https://csc53439ep.jdhp.org/training/api")
    env = sub.make_env(url)
    ag = env.possible_agents
    hold, hold_a = {a: 0 for a in ag}, {a: 0 for a in ag}
    # Co-players (--coplayers spec1,spec2,...): only the first --n-learners agents use and train the Q-network; the
    # others each play a policy drawn from the pool, re-drawn every --coplayer-switch steps. A spec is "random",
    # "sticky" (random action held 1-20 steps), "self" (latest step_*.pt of this run, i.e. a lagged copy of the
    # learner) or a checkpoint path (DDQN or PPO, played frozen through prakhar.tiwari.py's Agent).
    learners = ag if cfg["coplayers"] == "none" else ag[:cfg["n_learners"]]
    pool = [] if cfg["coplayers"] == "none" else cfg["coplayers"].split(",")
    # --pool-schedule FILE (JSON list of pools): the pool changes every --phase-steps steps (phase k uses entry k, the
    # last entry is kept afterwards). At each phase boundary a frozen copy of the learner is saved as phase_<k>.pt; the
    # spec "phase_snaps" stands for all of them (the learner meets its own past versions: fictitious self-play).
    schedule = json.load(open(ROOT / cfg["pool_schedule"])) if cfg["pool_schedule"] != "none" else None
    phase = {"k": -1}

    def set_phase():
        k = state["step"] // cfg["phase_steps"]
        if schedule is None or k == phase["k"]:
            return False
        phase["k"] = k
        snaps = [str(x.relative_to(ROOT)) for x in sorted(run.glob("phase_*.pt"))]
        new = []
        for spec in schedule[min(k, len(schedule) - 1)]:
            new += (snaps or ["self"]) if spec == "phase_snaps" else [spec]
        pool[:] = new
        print(f"phase {k} (from step {k * cfg['phase_steps']}): pool {pool}", flush=True)
        return True
    co_spec, co_agent, co_hold = {a: None for a in ag if a not in learners}, {}, {a: [0, 0] for a in ag}

    def pick_coplayers():
        for a in co_spec:
            spec = pool[np.random.randint(len(pool))]
            co_spec[a], co_agent[a] = spec, None
            if spec not in ("random", "sticky"):
                path = ROOT / spec
                if spec == "self":
                    snaps = sorted(run.glob("step_*.pt"))
                    path = snaps[-1] if snaps else ROOT / cfg["init"]
                pol = sub.Agent(env.observation_space(a), env.action_space(a), learn=False, pretrained=None)
                pol.load(str(path))
                co_agent[a] = pol

    def coplayer_action(a, o):
        if co_agent[a] is not None:
            return np.asarray(co_agent[a].act({a: o})[a], np.int8).reshape(2)
        h = co_hold[a]
        if co_spec[a] == "random" or h[0] <= 0:
            h[1], h[0] = np.random.randint(4), (np.random.randint(1, 21) if co_spec[a] == "sticky" else 1)
        h[0] -= 1
        return ACTIONS[h[1]]

    if co_spec:
        set_phase()
        pick_coplayers()
        print(f"co-players: learners {learners}, pool {pool}, now {co_spec}", flush=True)
    recent = deque(maxlen=max(1, cfg["best_window"] // cfg["log_every"]))
    win = dict(r=0.0, pos=0, died=0, drop=0, n=0, t0=time.monotonic(), loss=0.0, q=0.0, nl=0, co_r=0.0)
    try:
        while state["step"] < cfg["total_steps"] and not STOP:
            try:
                obs, _ = env.reset(seed=cfg["seed"] + 1000 * state["restarts"] + state["step"] % 997)
                feat.reset()
                nstep.clear()
                for pol in co_agent.values():
                    if pol is not None:
                        pol.forget_all()
                S = {a: feat(a, obs[a]) for a in learners}
                while state["step"] < cfg["total_steps"] and not STOP:
                    frac = min(1.0, state["step"] / cfg["eps_steps"])
                    eps = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])
                    if co_spec and schedule is not None and state["step"] > 0 and state["step"] % cfg["phase_steps"] == 0 \
                            and not (run / f"phase_{state['step'] // cfg['phase_steps']:02d}.pt").exists():
                        save_policy(run / f"phase_{state['step'] // cfg['phase_steps']:02d}.pt")   # frozen learner copy
                    new_phase = bool(co_spec) and state["step"] > 0 and set_phase()
                    if co_spec and state["step"] > 0 and (new_phase or state["step"] % cfg["coplayer_switch"] == 0):
                        pick_coplayers()
                    with torch.no_grad():
                        q = online(torch.tensor(np.array([S[a] for a in learners]), device=DEV)).cpu().numpy()
                    acts = {a: coplayer_action(a, obs[a]) for a in co_spec}
                    for k, a in enumerate(learners):
                        if hold[a] > 0:
                            hold[a] -= 1
                        elif np.random.rand() < eps:
                            hold_a[a], hold[a] = np.random.randint(4), int(np.random.choice([1, 2, 4, 8])) - 1
                        else:
                            hold_a[a] = int(q[k].argmax())
                        acts[a] = ACTIONS[hold_a[a]]
                    obs2, rew, _, _, infos = env.step(acts)
                    for a in co_spec:
                        win["co_r"] += rew[a]
                    for a in learners:
                        died = bool(infos[a].get("died", False))
                        r = learn_reward(rew[a], died) + (shaping(obs[a], obs2[a], died) if shaping else 0.0)
                        feat.prev[a] = acts[a].astype(np.float32)
                        s2 = feat(a, obs2[a])
                        for tr in nstep.push(a, S[a], hold_a[a], 0.2 * r, s2):
                            buf.add(*tr)
                        S[a] = s2
                        win["r"] += rew[a]
                        win["pos"] += rew[a] > 0
                        win["died"] += died
                        win["drop"] += (not died) and (obs2[a][8] < obs[a][8] - 0.01)
                    obs = obs2
                    if buf.n >= 10_000:
                        for _ in range(cfg["updates_per_step"]):
                            l_, q_ = learn()
                            win["loss"] += l_
                            win["q"] += q_
                            win["nl"] += 1
                    state["step"] += 1
                    win["n"] += 1
                    if state["step"] % cfg["log_every"] == 0:
                        n = win["n"] * len(learners)
                        e = dict(step=state["step"], reward_per_1000=round(win["r"] / n * 1000, 2),
                                 food_per_1000=round(win["pos"] / n * 1000, 2), deaths_per_1000=round(win["died"] / n * 1000, 2),
                                 damage_per_1000=round(win["drop"] / n * 1000, 2), eps=round(eps, 4),
                                 loss=round(win["loss"] / max(win["nl"], 1), 5), mean_q=round(win["q"] / max(win["nl"], 1), 3),
                                 updates=state["updates"], buffer=buf.n,
                                 steps_per_s=round(win["n"] / (time.monotonic() - win["t0"]), 1),
                                 time=time.strftime("%Y-%m-%d %H:%M:%S"), lr=cfg["lr"], ent=0.0, shaping_per_1000=0.0)
                        if co_spec:
                            e["coplayer_reward_per_1000"] = round(win["co_r"] / (win["n"] * len(co_spec)) * 1000, 2)
                            e["coplayers"] = ",".join(Path(str(v)).stem for v in co_spec.values())
                        print(json.dumps(e), flush=True)
                        with open(run / "log.jsonl", "a") as f:
                            f.write(json.dumps(e) + "\n")
                        recent.append(e["reward_per_1000"])
                        win = dict(r=0.0, pos=0, died=0, drop=0, n=0, t0=time.monotonic(), loss=0.0, q=0.0, nl=0, co_r=0.0)
                    if state["step"] % cfg["ckpt_every"] == 0:
                        snap = run / f"step_{state['step']:08d}.pt"
                        save_policy(snap)
                        for old in sorted(run.glob("step_*.pt"))[:-cfg["keep_last"]]:
                            old.unlink()
                        if len(recent) == recent.maxlen:
                            score = float(np.mean(recent))
                            best = run / f"best_{state['step']:08d}.pt"
                            state["best"].append([score, best.name])
                            state["best"].sort(key=lambda x: -x[0])
                            if best.name in [b[1] for b in state["best"][:cfg["keep_best"]]]:
                                save_policy(best)
                            for _, nm in state["best"][cfg["keep_best"]:]:
                                (run / nm).unlink(missing_ok=True)
                            state["best"] = state["best"][:cfg["keep_best"]]
                        save_all()
            except pzclient.RemoteEnvError as err:
                transient = sub.is_transient(err)        # unreachable, disconnected, 408/429/5xx
                if not transient:
                    raise
                print(f"{time.strftime('%H:%M:%S')} server problem ({type(err).__name__}): saving, retry in 30 s", flush=True)
                save_all()
                time.sleep(30)
                try:
                    env.close()
                except Exception:
                    pass
                while not STOP:                       # retry until the server is back
                    try:
                        env = sub.make_env(url)
                        break
                    except pzclient.RemoteEnvError as e2:
                        print(f"{time.strftime('%H:%M:%S')} reconnect failed ({type(e2).__name__}); retry in 60 s", flush=True)
                        time.sleep(60)
    finally:
        save_all()
        try:
            env.close()
        except Exception:
            pass
        print(f"STOPPED at step {state['step']}; best {state['best']}", flush=True)
        if state["step"] >= cfg["total_steps"]:
            (run / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
