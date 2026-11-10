"""Discrete Soft Actor-Critic (SAC-Discrete, Christodoulou 2019) on the TRAINING server, crash-proof.

Same inputs, rewards, n-step returns and pre-filled replay buffer as src/train/ddqn.py (imported from there), so the
comparison with DDQN changes only the learning algorithm.

Networks:
    actor (= the policy, <= 128 hidden units): 46 -> 64 -> 64 (LayerNorm + ReLU) -> 4 logits over the joint actions
        ([0,0], [0,1], [1,0], [1,1]); pi(a|s) = softmax(logits).
    2 critics (only used for training, so they may be larger): 46 -> 256 -> 256 -> Q(s, a) for the 4 actions,
        each with a target copy updated by Polyak averaging (tau).
Learning (per environment step, `--updates-per-step` minibatches):
    soft value   V(s') = sum_a pi(a|s') [min_k Q_target_k(s', a) - alpha log pi(a|s')]
    critic       y = R_n + gamma^n V(s_n), Huber loss on both critics
    actor        sum_a pi(a|s) [alpha log pi(a|s) - min_k Q_k(s, a)]
    temperature  alpha tuned so that the policy entropy stays near --target-entropy (fraction of log 4)
Exploration: actions are sampled from pi (no epsilon). Deaths are not terminal (the agent respawns).

Files in runs/<name>/: config.json, latest.pt (all networks + optimizers + counters), buffer.npz, step_*.pt (last 3),
best_*.pt (top 3 by recent true reward), log.jsonl. Policy files hold {"algo": "sac", "actor", "critics", "log_alpha", ...};
`--init FILE` continues from one of them, `--buffer-from runs/X/buffer.npz` reuses a replay buffer.

Usage:  scripts/bg_sac.sh     (200k steps as run "sac", then 200k more as "sac2" from the best training-world checkpoint)
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ddqn  # noqa: E402, F401  (also loads prakhar.tiwari.py as ddqn.sub)

from ddqn import ACTIONS, DEV, ROOT, Features, NStep, Replay, recorded_transitions, sub  # noqa: E402

DEFAULTS = dict(init="none", buffer_from="none", total_steps=200_000, buffer=1_000_000, batch=256, updates_per_step=2, lr_actor=1e-4, lr_critic=3e-4,
                lr_alpha=3e-4, alpha_init=0.1, target_entropy=0.35, tau=0.005, gamma=0.99, n_step=5, critic_hidden=256,
                shaping=True, prefill=True, hist=4, w_danger=0.5,
                log_every=4096, ckpt_every=20480, keep_last=3, keep_best=3, best_window=40960, seed=0)
STOP = False


def _stop(*_):
    global STOP
    STOP = True


class Actor(nn.Module):
    """Policy: 2 hidden layers of 64 (= 128 hidden units, the policy limit), outputs 4 logits."""

    def __init__(self, d, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(),
                                 nn.Linear(64, n_actions))

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, -1), F.log_softmax(logits, -1)


class Critic(nn.Module):
    def __init__(self, d, h, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
                                 nn.Linear(h, n_actions))

    def forward(self, x):
        return self.net(x)


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

    shaping = sub.ContestShaping(cfg["gamma"], w_danger=cfg["w_danger"]) if cfg["shaping"] else None
    H = cfg["hist"]
    d_in = 11 * H + 2
    actor = Actor(d_in).to(DEV)
    critics = nn.ModuleList([Critic(d_in, cfg["critic_hidden"]) for _ in range(2)]).to(DEV)
    targets = nn.ModuleList([Critic(d_in, cfg["critic_hidden"]) for _ in range(2)]).to(DEV)
    log_alpha = torch.tensor(math.log(cfg["alpha_init"]), device=DEV, requires_grad=True)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg["lr_actor"])
    opt_c = torch.optim.Adam(critics.parameters(), lr=cfg["lr_critic"])
    opt_al = torch.optim.Adam([log_alpha], lr=cfg["lr_alpha"])
    target_ent = cfg["target_entropy"] * math.log(4)
    buf = Replay(cfg["buffer"], d_in)
    state = {"step": 0, "updates": 0, "best": [], "restarts": 0}

    if (run / "latest.pt").exists():
        ck = torch.load(run / "latest.pt", weights_only=False)
        actor.load_state_dict(ck["actor"])
        critics.load_state_dict(ck["critics"])
        targets.load_state_dict(ck["targets"])
        with torch.no_grad():
            log_alpha.copy_(torch.tensor(ck["log_alpha"]))
        opt_a.load_state_dict(ck["opt_a"])
        opt_c.load_state_dict(ck["opt_c"])
        opt_al.load_state_dict(ck["opt_al"])
        mean, std = ck["norm_mean"], ck["norm_std"]
        state.update(ck["trainer"])
        state["restarts"] += 1
        if (run / "buffer.npz").exists():
            buf.load(run / "buffer.npz")
        print(f"RESUMED {args.name} at step {state['step']} (buffer {buf.n})", flush=True)
    else:
        # same normalisation as the DDQN run, so the pre-filled features are identical
        ck0 = torch.load(ROOT / "models" / "FINAL_ddqn_500k.pt", weights_only=False)
        mean, std = ck0["norm_mean"], ck0["norm_std"]
        if cfg["init"] != "none":             # continue from a saved SAC policy file (actor + critics + alpha)
            ck0 = torch.load(ROOT / cfg["init"], weights_only=False)
            actor.load_state_dict(ck0["actor"])
            critics.load_state_dict(ck0["critics"])
            with torch.no_grad():
                log_alpha.copy_(torch.tensor(ck0["log_alpha"]))
            mean, std = ck0["norm_mean"], ck0["norm_std"]
            print(f"initialised from {cfg['init']}", flush=True)
        targets.load_state_dict(critics.state_dict())
        if cfg["prefill"]:
            t0 = time.time()
            for tr in recorded_transitions(Features(mean, std, H), NStep(cfg["n_step"], cfg["gamma"]), shaping, cfg["gamma"]):
                buf.add(*tr)
            print(f"pre-filled the buffer with {buf.n} recorded transitions in {time.time() - t0:.0f}s", flush=True)
        if cfg["buffer_from"] != "none":    # reuse another run's replay buffer (same features and n-step only)
            buf.load(ROOT / cfg["buffer_from"])
            print(f"loaded the replay buffer {cfg['buffer_from']} ({buf.n} transitions)", flush=True)
        print(f"NEW run {args.name}: actor {d_in}-64-64-4 (128 hidden units), 2 critics {d_in}-{cfg['critic_hidden']}-"
              f"{cfg['critic_hidden']}-4, device {DEV}", flush=True)
    feat = Features(mean, std, H)
    nstep = NStep(cfg["n_step"], cfg["gamma"])

    def save_all():
        tmp = run / "latest.pt.tmp"
        torch.save({"algo": "sac", "actor": actor.state_dict(), "critics": critics.state_dict(), "targets": targets.state_dict(),
                    "log_alpha": log_alpha.item(), "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict(),
                    "opt_al": opt_al.state_dict(), "norm_mean": mean, "norm_std": std, "hist": H, "trainer": state}, tmp)
        os.replace(tmp, run / "latest.pt")
        buf.save(run / "buffer.npz")

    def save_policy(path):
        tmp = f"{path}.tmp"
        torch.save({"algo": "sac", "actor": actor.state_dict(), "critics": critics.state_dict(), "log_alpha": log_alpha.item(),
                    "norm_mean": mean, "norm_std": std, "hist": H}, tmp)
        os.replace(tmp, path)

    def learn():
        s, a, r, s2, g = buf.sample(cfg["batch"])
        alpha = log_alpha.exp().detach()
        with torch.no_grad():
            pi2, logpi2 = actor(s2)
            q2 = torch.min(targets[0](s2), targets[1](s2))
            y = r + g * (pi2 * (q2 - alpha * logpi2)).sum(1)
        qs = [c(s).gather(1, a[:, None]).squeeze(1) for c in critics]
        loss_c = sum(F.smooth_l1_loss(q, y) for q in qs)
        opt_c.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(critics.parameters(), 10.0)
        opt_c.step()

        pi, logpi = actor(s)
        with torch.no_grad():
            qmin = torch.min(critics[0](s), critics[1](s))
        loss_a = (pi * (alpha * logpi - qmin)).sum(1).mean()
        opt_a.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        opt_a.step()

        ent = -(pi.detach() * logpi.detach()).sum(1).mean()
        loss_al = log_alpha * (ent - target_ent)        # alpha rises when the entropy falls below the target
        opt_al.zero_grad()
        loss_al.backward()
        opt_al.step()

        with torch.no_grad():
            for c, t in zip(critics, targets):
                for pc, pt in zip(c.parameters(), t.parameters()):
                    pt.mul_(1 - cfg["tau"]).add_(cfg["tau"] * pc)
        state["updates"] += 1
        return float(loss_c.detach()) / 2, float(qs[0].mean()), float(ent), float(alpha)

    url = os.environ.get("PZ_TRAIN_URL", "https://csc53439ep.jdhp.org/training/api")
    if "/training/" not in url:              # this trainer must never play in the contest world
        raise SystemExit(f"refusing to train on {url}: not the training server")
    env = sub.make_env(url)
    ag = env.possible_agents
    recent = deque(maxlen=max(1, cfg["best_window"] // cfg["log_every"]))
    win = dict(r=0.0, pos=0, died=0, drop=0, n=0, t0=time.monotonic(), loss=0.0, q=0.0, ent=0.0, alpha=0.0, nl=0)
    try:
        while state["step"] < cfg["total_steps"] and not STOP:
            try:
                obs, _ = env.reset(seed=cfg["seed"] + 1000 * state["restarts"] + state["step"] % 997)
                feat.reset()
                nstep.clear()
                S = {a: feat(a, obs[a]) for a in ag}
                while state["step"] < cfg["total_steps"] and not STOP:
                    with torch.no_grad():
                        pi, _ = actor(torch.tensor(np.array([S[a] for a in ag]), device=DEV))
                        idx = torch.multinomial(pi, 1).squeeze(1).cpu().numpy()
                    acts = {a: ACTIONS[idx[k]] for k, a in enumerate(ag)}
                    obs2, rew, _, _, infos = env.step(acts)
                    for k, a in enumerate(ag):
                        died = bool(infos[a].get("died", False))
                        r = rew[a] + (shaping(obs[a], obs2[a], died) if shaping else 0.0)
                        feat.prev[a] = acts[a].astype(np.float32)
                        s2 = feat(a, obs2[a])
                        for tr in nstep.push(a, S[a], int(idx[k]), 0.2 * r, s2):
                            buf.add(*tr)
                        S[a] = s2
                        win["r"] += rew[a]
                        win["pos"] += rew[a] > 0
                        win["died"] += died
                        win["drop"] += (not died) and (obs2[a][8] < obs[a][8] - 0.01)
                    obs = obs2
                    if buf.n >= 10_000:
                        for _ in range(cfg["updates_per_step"]):
                            l_, q_, e_, al_ = learn()
                            win["loss"] += l_
                            win["q"] += q_
                            win["ent"] += e_
                            win["alpha"] += al_
                            win["nl"] += 1
                    state["step"] += 1
                    win["n"] += 1
                    if state["step"] % cfg["log_every"] == 0:
                        n, nl = win["n"] * len(ag), max(win["nl"], 1)
                        e = dict(step=state["step"], reward_per_1000=round(win["r"] / n * 1000, 2),
                                 food_per_1000=round(win["pos"] / n * 1000, 2), deaths_per_1000=round(win["died"] / n * 1000, 2),
                                 damage_per_1000=round(win["drop"] / n * 1000, 2), loss=round(win["loss"] / nl, 5),
                                 mean_q=round(win["q"] / nl, 3), ent=round(win["ent"] / nl, 3), alpha=round(win["alpha"] / nl, 4),
                                 updates=state["updates"], buffer=buf.n,
                                 steps_per_s=round(win["n"] / (time.monotonic() - win["t0"]), 1),
                                 time=time.strftime("%Y-%m-%d %H:%M:%S"))
                        print(json.dumps(e), flush=True)
                        with open(run / "log.jsonl", "a") as f:
                            f.write(json.dumps(e) + "\n")
                        recent.append(e["reward_per_1000"])
                        win = dict(r=0.0, pos=0, died=0, drop=0, n=0, t0=time.monotonic(), loss=0.0, q=0.0, ent=0.0, alpha=0.0, nl=0)
                    if state["step"] % cfg["ckpt_every"] == 0:
                        save_policy(run / f"step_{state['step']:08d}.pt")
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
                if not sub.is_transient(err):          # unreachable, disconnected, 408/429/5xx are retried
                    raise
                print(f"{time.strftime('%H:%M:%S')} server problem ({type(err).__name__}): saving, retry in 30 s", flush=True)
                save_all()
                time.sleep(30)
                try:
                    env.close()
                except Exception:
                    pass
                while not STOP:
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
