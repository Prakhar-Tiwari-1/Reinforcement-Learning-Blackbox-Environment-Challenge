"""Long, crash-proof PPO training on the TRAINING server, with the submission's own Agent.

Everything lives in runs/<name>/:
    config.json        the settings of the run (fixed at creation; reused when resuming)
    latest.pt          full training state (network, optimizer, normaliser, counters), saved every --ckpt-every
    step_<N>.pt        periodic snapshots (only the last --keep-last are kept)
    best_<N>.pt        the --keep-best checkpoints with the highest TRUE reward over the last --best-window steps
    log.jsonl          one line per --log-every steps: true reward, food, deaths, damage, shaping, speed...
    stdout.log         console output (when started with scripts/rl.sh)

Robustness:
    - Restart the same command (or let systemd restart it): it resumes from latest.pt. At most --ckpt-every
      steps of experience are lost, never the model.
    - Saves are atomic (temp file + rename): a crash during a save cannot corrupt a checkpoint.
    - Server hiccups (unreachable, 502/503/504, agent not connected) -> wait, rebuild the connection, go on.
    - SIGTERM / Ctrl-C (systemctl stop, scripts/rl.sh stop) -> save latest.pt, close the world, exit.
    - Disk: old step_*.pt and worse best_*.pt are deleted automatically.

Usage (normally through scripts/rl.sh):
    uv run python src/train/train.py --name v2 --init runs/v2_init.pt --total-steps 1000000
    uv run python src/train/train.py --name v2                    # resume
    uv run python src/train/train.py --name v2 --total-steps 2000000   # resume and extend
"""

import argparse
import importlib.util
import json
import os
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sub", ROOT / "prakhar.tiwari.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

DEFAULTS = dict(init="runs/v2_init.pt", total_steps=1_000_000, rollout=512, epochs=4, minibatch=512,
                lr=3e-4, lr_end=5e-5, ent=0.01, ent_end=0.002, shaping=True, w_food=0.5, w_danger=0.5,
                w_health=1.0, damage_penalty=0.0, critic_warmup=30, log_every=4096, ckpt_every=20480,
                keep_last=3, keep_best=3, best_window=40960, seed=0)

STOP = False


def _stop(*_):
    global STOP
    STOP = True


def log(run, entry):
    line = json.dumps(entry)
    print(line, flush=True)
    with open(run / "log.jsonl", "a") as f:
        f.write(line + "\n")


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
    cfg_path = run / "config.json"
    cfg = dict(DEFAULTS)
    if cfg_path.exists():
        cfg.update(json.load(open(cfg_path)))
    for k in DEFAULTS:  # explicit command-line values win (e.g. extend --total-steps when resuming)
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    json.dump(cfg, open(cfg_path, "w"), indent=1)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    sub.load_dotenv_if_any()
    torch.set_num_threads(2)
    url = os.environ.get("PZ_TRAIN_URL", "https://csc53439ep.jdhp.org/training/api")
    import pzclient

    def connect():
        while not STOP:
            try:
                return sub.make_env(url)
            except pzclient.RemoteEnvError as e:
                print(f"connection failed ({e}); retrying in 30 s", flush=True)
                time.sleep(30)

    env = connect()
    if env is None:
        return
    ag = env.possible_agents
    agent = sub.Agent(env.observation_space(ag[0]), env.action_space(ag[0]), learn=True, lr=cfg["lr"],
                      rollout=cfg["rollout"], epochs=cfg["epochs"], minibatch=cfg["minibatch"], ent_coef=cfg["ent"],
                      pretrained=None, seed=cfg["seed"])
    shaping = (sub.ContestShaping(agent.gamma, cfg["w_food"], cfg["w_danger"], cfg["w_health"], cfg["damage_penalty"])
               if cfg["shaping"] and sub.is_contest_world(env.observation_space(ag[0]), env.action_space(ag[0])) else None)

    state = {"step": 0, "best": [], "restarts": 0}
    if (run / "latest.pt").exists():
        ck = agent.load(run / "latest.pt")
        state.update(ck.get("trainer", {}))
        state["restarts"] += 1
        print(f"RESUMED {args.name} at step {state['step']} (restart #{state['restarts']})", flush=True)
    elif cfg["init"] and cfg["init"] != "none":
        agent.load(ROOT / cfg["init"])
        agent.critic_only_updates = cfg["critic_warmup"]
        print(f"NEW run {args.name} from {cfg['init']} (critic warm-up {cfg['critic_warmup']} updates)", flush=True)
    else:
        print(f"NEW run {args.name} from scratch", flush=True)
    agent.reward_scale = agent.reward_scale or 0.2
    print(f"policy {agent.arch['actor']} = {agent.policy_hidden_units} hidden units | critic {agent.arch['critic']} | "
          f"history {agent.arch['history']} | shaping {shaping is not None} | config {cfg}", flush=True)

    def schedule():
        frac = min(1.0, state["step"] / max(cfg["total_steps"], 1))
        agent.set_lr(cfg["lr"] + frac * (cfg["lr_end"] - cfg["lr"]))
        agent.ent_coef = cfg["ent"] + frac * (cfg["ent_end"] - cfg["ent"])

    def save_all():
        agent.save(run / "latest.pt", full=True, extra={"trainer": state})

    recent = deque(maxlen=max(1, cfg["best_window"] // cfg["log_every"]))   # recent windows' true reward
    win = dict(r=0.0, food=0, deaths=0, damage=0, shaped=0.0, n=0, t0=time.monotonic())
    n_agents = len(ag)
    try:
        while state["step"] < cfg["total_steps"] and not STOP:
            try:
                obs, _ = env.reset(seed=cfg["seed"] + 1000 * state["restarts"] + state["step"] % 997)
                agent.forget_all()
                while state["step"] < cfg["total_steps"] and not STOP:
                    schedule()
                    actions = agent.act({a: obs[a] for a in env.agents})
                    obs2, rew, term, trunc, infos = env.step(actions)
                    shaped = {}
                    for a in rew:
                        died = bool(infos[a].get("died", False))
                        f = shaping(obs[a], obs2[a], died) if shaping else 0.0
                        shaped[a] = rew[a] + f
                        win["r"] += rew[a]
                        win["food"] += rew[a] > 0
                        win["deaths"] += died
                        win["damage"] += (not died) and (obs2[a][8] < obs[a][8] - 0.01)
                        win["shaped"] += f
                    agent.observe(shaped, {a: bool(term[a] or trunc[a]) for a in rew})
                    obs = obs2
                    state["step"] += 1
                    win["n"] += 1
                    if state["step"] % cfg["log_every"] == 0:
                        n = win["n"] * n_agents
                        e = dict(step=state["step"], reward_per_1000=round(win["r"] / n * 1000, 2),
                                 food_per_1000=round(win["food"] / n * 1000, 2),
                                 deaths_per_1000=round(win["deaths"] / n * 1000, 2),
                                 damage_per_1000=round(win["damage"] / n * 1000, 2),
                                 shaping_per_1000=round(win["shaped"] / n * 1000, 2),
                                 lr=round(agent.lr, 7), ent=round(agent.ent_coef, 5), updates=agent.n_updates,
                                 steps_per_s=round(win["n"] / (time.monotonic() - win["t0"]), 1),
                                 time=time.strftime("%Y-%m-%d %H:%M:%S"))
                        log(run, e)
                        recent.append(e["reward_per_1000"])
                        win = dict(r=0.0, food=0, deaths=0, damage=0, shaped=0.0, n=0, t0=time.monotonic())
                    if state["step"] % cfg["ckpt_every"] == 0:
                        save_all()
                        snap = run / f"step_{state['step']:08d}.pt"
                        agent.save(snap)
                        for old in sorted(run.glob("step_*.pt"))[:-cfg["keep_last"]]:
                            old.unlink()
                        if len(recent) == recent.maxlen:      # best-so-far by recent TRUE reward
                            score = float(np.mean(recent))
                            best = run / f"best_{state['step']:08d}.pt"
                            state["best"].append([score, best.name])
                            state["best"].sort(key=lambda x: -x[0])
                            if best.name in [b[1] for b in state["best"][:cfg["keep_best"]]]:
                                agent.save(best)
                            for _, name in state["best"][cfg["keep_best"]:]:
                                (run / name).unlink(missing_ok=True)
                            state["best"] = state["best"][:cfg["keep_best"]]
                            save_all()
            except pzclient.RemoteEnvError as err:
                transient = isinstance(err, (pzclient.ServerUnreachableError, pzclient.AgentNotConnectedError)) or \
                    getattr(err, "status_code", None) in (502, 503, 504)
                if not transient:
                    raise
                print(f"{time.strftime('%H:%M:%S')} server problem ({type(err).__name__}): saving, waiting 30 s, "
                      f"reconnecting", flush=True)
                save_all()
                time.sleep(30)
                try:
                    env.close()
                except Exception:
                    pass
                env = connect()
                if env is None:
                    break
    finally:
        save_all()
        try:
            env.close()
        except Exception:
            pass
        print(f"STOPPED at step {state['step']} ({'signal' if STOP else 'done'}); best: {state['best']}", flush=True)
        if state["step"] >= cfg["total_steps"]:
            (run / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
