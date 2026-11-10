"""Play the contest world with the SUBMISSION's policy (prakhar.tiwari.py, frozen) and record every step.

Same policy and same reconnection handling as `python prakhar.tiwari.py --no-learn`, so it can replace the
official contest player; in addition it saves, every --chunk steps, data/live/<name>_<k>.npz with:
    obs (T+1, 11), actions (T, 2), rewards (T,), died (T,), applied (T,), world_step (T,), wall (T,)
world_step = infos["shared_world"]["step"] (the world's step counter), wall = time of the step (s since epoch).
Progress lines have the same format as prakhar.tiwari.py (parsed by scripts/rl.sh scores).

Usage: scripts/rl.sh record NAME      (or .venv/bin/python src/contest_record.py --name NAME [--load X.pt])
"""

import argparse
import importlib.util
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sub", ROOT / "prakhar.tiwari.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=10_000_000)
    p.add_argument("--chunk", type=int, default=5000)
    p.add_argument("--load", default=None, help="checkpoint instead of the embedded weights")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--anti-stall", type=int, default=0, help="see Agent.stall_k in prakhar.tiwari.py (0 = off)")
    args = p.parse_args()
    sub.load_dotenv_if_any()
    import pzclient
    import torch

    torch.set_num_threads(1)
    out = ROOT / "data" / "live"
    out.mkdir(parents=True, exist_ok=True)
    k0 = len(list(out.glob(f"{args.name}_*.npz")))      # continue numbering after a restart

    env = sub.connect()
    me = env.possible_agents[0]
    agent = sub.Agent(env.observation_space(me), env.action_space(me), learn=False,
                      pretrained=None if args.load else "auto")
    if args.load:
        agent.load(args.load)
    agent.stall_k = args.anti_stall
    print(f"agent {me}: recording {args.name} | DDQN {agent.qnet is not None} | history {agent.arch['history']} | anti-stall {agent.stall_k}", flush=True)

    failures = 0

    def join():
        """(Re)join the shared world; on 3 failures in a row, rebuild the connection. Retries forever."""
        nonlocal env, failures
        while True:
            try:
                if env is None:
                    env = sub.connect()
                o, _ = env.reset()
                failures = 0
                return o
            except pzclient.RemoteEnvError as e:
                if not sub.is_transient(e):
                    raise
                failures += 1
                print(f"{time.strftime('%H:%M:%S')} join failed ({type(e).__name__}); retry in {sub.RETRY_SECONDS} s", flush=True)
                if failures >= 3:
                    sub.safe_close(env)
                    env = None
                time.sleep(sub.RETRY_SECONDS)

    rec = {k: [] for k in ("obs", "actions", "rewards", "died", "applied", "world_step", "wall")}
    stats = sub.Stats()
    k = k0

    def flush(final_obs):
        nonlocal k
        if not rec["rewards"]:
            return
        tmp = out / f"{args.name}_{k:04d}.tmp.npz"
        np.savez_compressed(tmp, obs=np.array(rec["obs"] + [final_obs], np.float32),
                            **{n: np.array(v) for n, v in rec.items() if n != "obs"})
        os.replace(tmp, out / f"{args.name}_{k:04d}.npz")
        k += 1
        for v in rec.values():
            v.clear()

    obs = join()
    try:
        for _ in range(args.steps):
            ts = time.monotonic()
            action = agent.act({me: obs[me]})
            stats.think.append(time.monotonic() - ts)
            try:
                obs2, rew, term, trunc, infos = env.step(action)
            except pzclient.RemoteEnvError as e:
                if not sub.is_transient(e):
                    raise
                flush(obs[me])
                agent.forget(me)
                failures += 1
                print(f"{time.strftime('%H:%M:%S')} step failed ({type(e).__name__}); recovering", flush=True)
                if failures >= 3:
                    sub.safe_close(env)
                    env = None
                if env is None or isinstance(e, pzclient.AgentNotConnectedError):
                    obs = join()
                else:
                    time.sleep(sub.RETRY_SECONDS)
                continue
            failures = 0
            sw = infos[me].get("shared_world", {})
            applied = sw.get("action_applied", True)
            died = bool(infos[me].get("died", False))
            rec["obs"].append(np.asarray(obs[me], np.float32))
            rec["actions"].append(np.asarray(action[me], np.int8))
            rec["rewards"].append(float(rew[me]))
            rec["died"].append(died)
            rec["applied"].append(bool(applied))
            rec["world_step"].append(int(sw.get("step", -1)))
            rec["wall"].append(time.time())
            obs = obs2
            stats.steps += 1
            stats.reward += rew[me]
            stats.food += rew[me] > 0
            stats.deaths += died
            stats.missed += not applied
            if len(rec["rewards"]) >= args.chunk:
                flush(obs[me])
            if stats.steps % args.log_every == 0:
                print(stats.line(agent), flush=True)
    finally:
        flush(obs[me])
        if env is not None:
            sub.safe_close(env)
        print("final:", stats.line(agent), flush=True)


if __name__ == "__main__":
    main()
