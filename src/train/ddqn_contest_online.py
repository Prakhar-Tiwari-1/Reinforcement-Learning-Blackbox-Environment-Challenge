"""Online fine-tuning of a DDQN policy IN THE CONTEST WORLD: it learns from every live step while it plays.

Why: the contest world (other participants) differs from the training world; offline fine-tuning on recorded live
data failed because that data came from greedy play only (no exploration). Here the agent explores a little, so it
also sees the outcome of actions it would not normally take, and it keeps learning as the crowd changes.

Acting   : epsilon-greedy on Q (random action held 1, 2, 4 or 8 steps, as in training) + anti-stall: if the
           observation has not changed for --anti-stall steps while it keeps choosing "no movement", it plays its best
           moving action for 5 steps.
Learning : Double DQN, n-step 5, same features / shaping / reward scale as src/train/ddqn.py, --updates-per-step
           updates per live step. Each minibatch = --new-frac from the CONTEST buffer (all live recordings in
           data/live/ at start, then every new live step) + the rest from the --old training buffer (no forgetting).
Safeguard: every --ckpt-every steps, the reward of the last --window live steps is compared:
           - best so far -> saved as best.pt (and online_<step>.pt is kept for later live tests);
           - below --revert-below -> weights go back to best.pt (or --init), the learning rate is halved.
Robust   : resumes from runs/<name>/state.pt after a restart; waits out server problems like the contest player;
           records every live step in data/live/online_<name>_*.npz (same format as src/contest_record.py).

Usage: .venv/bin/python src/train/ddqn_contest_online.py --name online500k      (the ONLY program on the contest server)
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "train"))
_argv, sys.argv = sys.argv, sys.argv[:1]
import ddqn  # noqa: E402
import ddqn_live_ft  # noqa: E402

sys.argv = _argv
sub, DEV, ACTIONS = ddqn.sub, ddqn.DEV, ddqn.ACTIONS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--init", default="models/FINAL_ddqn_500k.pt")
    p.add_argument("--old", default="runs/ddqn/buffer.npz")
    p.add_argument("--live", default="data/live/*.npz", help="live recordings that pre-fill the contest buffer")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--eps", type=float, default=0.03)
    p.add_argument("--anti-stall", type=int, default=10)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--new-frac", type=float, default=0.5)
    p.add_argument("--target-every", type=int, default=2000)
    p.add_argument("--window", type=int, default=20000)
    p.add_argument("--ckpt-every", type=int, default=10000)
    p.add_argument("--revert-below", type=float, default=25.0, help="reward/1000 over the window that triggers a revert")
    p.add_argument("--steps", type=int, default=10_000_000)
    p.add_argument("--log-every", type=int, default=1000)
    args = p.parse_args()
    sub.load_dotenv_if_any()
    import pzclient

    torch.set_num_threads(2)
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    rec_dir = ROOT / "data" / "live"
    rec_dir.mkdir(parents=True, exist_ok=True)

    ck0 = torch.load(ROOT / args.init, weights_only=False, map_location="cpu")
    mean, std = ck0["norm_mean"], ck0["norm_std"]
    online, target = ddqn.QNet(46).to(DEV), ddqn.QNet(46).to(DEV)
    online.load_state_dict({k: v.to(DEV) for k, v in ck0["net"].items()})
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=args.lr)
    st = dict(step=0, updates=0, lr=args.lr, best=None, reverts=0, since_revert=0)
    newbuf = ddqn.Replay(1_000_000, 46)
    if (run / "state.pt").exists():                                      # resume
        s = torch.load(run / "state.pt", weights_only=False, map_location="cpu")
        online.load_state_dict({k: v.to(DEV) for k, v in s["net"].items()})
        target.load_state_dict({k: v.to(DEV) for k, v in s["target"].items()})
        opt.load_state_dict(s["opt"])
        st.update(s["st"])
        for g in opt.param_groups:
            g["lr"] = st["lr"]
        newbuf.load(run / "new_buffer.npz")
        print(f"RESUMED {args.name} at live step {st['step']} (contest buffer {newbuf.n}, lr {st['lr']:.1e})", flush=True)
    else:
        files = sorted(glob.glob(str(ROOT / args.live)), key=os.path.getmtime)
        for tr in ddqn_live_ft.live_transitions(files, mean, std, 5, 0.99, sub.ContestShaping(0.99)):
            newbuf.add(*tr)
        print(f"NEW {args.name} from {args.init}: contest buffer pre-filled with {newbuf.n} transitions "
              f"from {len(files)} live recordings", flush=True)
    b = np.load(ROOT / args.old)
    n_old = int(b["n"])
    old = [torch.tensor(b[k][:n_old], device=DEV) for k in ("s", "a", "r", "s2", "g")]
    old = (old[0].float(), old[1].long(), old[2].float(), old[3].float(), old[4].float())
    print(f"old training buffer {n_old} | lr {st['lr']:.1e} | eps {args.eps} | anti-stall {args.anti_stall} | device {DEV}", flush=True)

    def save_policy(path, net=online):
        torch.save({"algo": "ddqn", "net": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "norm_mean": mean,
                    "norm_std": std, "hist": 4, "source": f"{args.name} live step {st['step']}"}, path)

    def save_state():
        tmp = run / "state.pt.tmp"
        torch.save({"net": online.state_dict(), "target": target.state_dict(), "opt": opt.state_dict(), "st": st}, tmp)
        os.replace(tmp, run / "state.pt")
        newbuf.save(run / "new_buffer.npz")

    def learn():
        nn_ = int(args.batch * args.new_frac)
        i_new = np.random.randint(0, newbuf.n, nn_)
        i_old = torch.as_tensor(np.random.randint(0, n_old, args.batch - nn_), device=DEV)
        f = lambda x: torch.tensor(x[i_new], device=DEV)  # noqa: E731
        s = torch.cat([f(newbuf.s).float(), old[0][i_old]])
        a = torch.cat([f(newbuf.a).long(), old[1][i_old]])
        r = torch.cat([f(newbuf.r), old[2][i_old]])
        s2 = torch.cat([f(newbuf.s2).float(), old[3][i_old]])
        g = torch.cat([f(newbuf.g), old[4][i_old]])
        with torch.no_grad():
            a2 = online(s2).argmax(1, keepdim=True)
            y = r + g * target(s2).gather(1, a2).squeeze(1)
        q = online(s).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        st["updates"] += 1
        if st["updates"] % args.target_every == 0:
            target.load_state_dict(online.state_dict())
        return float(loss.detach())

    # ---------------------------------------------------------------- live loop
    env = sub.connect()
    me = env.possible_agents[0]
    shaping = sub.ContestShaping(0.99)
    feat, nstep = ddqn.Features(mean, std, 4), ddqn.NStep(5, 0.99)
    hold = [0, 0]                      # [steps left, action index] of a held random action
    stall = dict(n=0, last=None, unstall=0)
    win = deque(maxlen=args.window)    # (r=+5, died) of the last live steps
    log = dict(r=0.0, pos=0, died=0, n=0, loss=0.0, nl=0, t0=time.monotonic())
    rec = {k: [] for k in ("obs", "actions", "rewards", "died", "applied", "world_step", "wall")}
    k_rec = len(list(rec_dir.glob(f"online_{args.name}_*.npz")))
    failures = 0

    def flush(final_obs):
        nonlocal k_rec
        if not rec["rewards"]:
            return
        tmp = rec_dir / f"online_{args.name}_{k_rec:04d}.tmp.npz"
        np.savez_compressed(tmp, obs=np.array(rec["obs"] + [final_obs], np.float32),
                            **{n: np.array(v) for n, v in rec.items() if n != "obs"})
        os.replace(tmp, rec_dir / f"online_{args.name}_{k_rec:04d}.npz")
        k_rec += 1
        for v in rec.values():
            v.clear()

    def join():
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

    def restart_trajectory(o):
        feat.reset(me)
        nstep.clear(me)
        stall.update(n=0, last=None, unstall=0)
        return feat(me, o[me])

    obs = join()
    S = restart_trajectory(obs)
    try:
        while st["step"] < args.steps:
            with torch.no_grad():
                q = online(torch.tensor(S[None], device=DEV)).cpu().numpy()[0]
            if hold[0] > 0:
                hold[0] -= 1
                j = hold[1]
            elif np.random.rand() < args.eps:
                j = np.random.randint(4)
                hold[:] = [int(np.random.choice([1, 2, 4, 8])) - 1, j]
            else:
                j = int(q.argmax())
                o = np.asarray(obs[me], np.float64).ravel()
                stall["n"] = stall["n"] + 1 if stall["last"] is not None and np.abs(o - stall["last"]).max() < 1e-3 else 0
                stall["last"] = o
                if args.anti_stall and stall["n"] >= args.anti_stall and j == 0:
                    stall["unstall"] = 5
                if stall["unstall"] > 0 and j == 0:
                    j = int(np.where(ACTIONS.any(1), q, -np.inf).argmax())
                    stall["unstall"] -= 1
            action = {me: ACTIONS[j]}
            try:
                obs2, rew, term, trunc, infos = env.step(action)
            except pzclient.RemoteEnvError as e:
                if not sub.is_transient(e):
                    raise
                flush(obs[me])
                failures += 1
                print(f"{time.strftime('%H:%M:%S')} step failed ({type(e).__name__}); recovering", flush=True)
                if failures >= 3:
                    sub.safe_close(env)
                    env = None
                if env is None or isinstance(e, pzclient.AgentNotConnectedError):
                    obs = join()
                else:
                    time.sleep(sub.RETRY_SECONDS)
                S = restart_trajectory(obs)
                continue
            failures = 0
            sw = infos[me].get("shared_world", {})
            died = bool(infos[me].get("died", False))
            r_true = float(rew[me])
            r = r_true + shaping(obs[me], obs2[me], died)
            feat.prev[me] = ACTIONS[j].astype(np.float32)
            s2 = feat(me, obs2[me])
            for tr in nstep.push(me, S, j, 0.2 * r, s2):
                newbuf.add(*tr)
            for name, v in (("obs", np.asarray(obs[me], np.float32)), ("actions", ACTIONS[j]), ("rewards", r_true),
                            ("died", died), ("applied", bool(sw.get("action_applied", True))),
                            ("world_step", int(sw.get("step", -1))), ("wall", time.time())):
                rec[name].append(v)
            S, obs = s2, obs2
            for _ in range(args.updates_per_step):
                log["loss"] += learn()
                log["nl"] += 1
            st["step"] += 1
            st["since_revert"] += 1
            win.append((r_true > 0, died))
            log["r"] += r_true
            log["pos"] += r_true > 0
            log["died"] += died
            log["n"] += 1
            if len(rec["rewards"]) >= 5000:
                flush(obs[me])
            if st["step"] % args.log_every == 0:
                n = log["n"]
                e = dict(step=st["step"], reward_per_1000=round(log["r"] / n * 1000, 1), food_per_1000=round(log["pos"] / n * 1000, 1),
                         deaths_per_1000=round(log["died"] / n * 1000, 2), loss=round(log["loss"] / max(log["nl"], 1), 5),
                         lr=st["lr"], reverts=st["reverts"], steps_per_s=round(n / (time.monotonic() - log["t0"]), 1),
                         time=time.strftime("%Y-%m-%d %H:%M:%S"))
                print(json.dumps(e), flush=True)
                with open(run / "log.jsonl", "a") as f:
                    f.write(json.dumps(e) + "\n")
                log.update(r=0.0, pos=0, died=0, n=0, loss=0.0, nl=0, t0=time.monotonic())
            if st["step"] % args.ckpt_every == 0:
                save_policy(run / f"online_{st['step']:07d}.pt")
                if len(win) == win.maxlen and st["since_revert"] >= args.window:
                    w = np.array(win)
                    score = 5 * (w[:, 0].sum() - w[:, 1].sum()) / len(w) * 1000
                    if st["best"] is None or score > st["best"][0]:
                        st["best"] = [float(score), st["step"]]
                        save_policy(run / "best.pt")
                        print(f"{time.strftime('%H:%M:%S')} new best window {score:+.1f} at live step {st['step']}", flush=True)
                    elif score < args.revert_below:
                        src = run / "best.pt" if (run / "best.pt").exists() else ROOT / args.init
                        ckb = torch.load(src, weights_only=False, map_location="cpu")
                        online.load_state_dict({k: v.to(DEV) for k, v in ckb["net"].items()})
                        target.load_state_dict(online.state_dict())
                        st["lr"] /= 2
                        opt = torch.optim.Adam(online.parameters(), lr=st["lr"])
                        st["reverts"] += 1
                        st["since_revert"] = 0
                        win.clear()
                        print(f"{time.strftime('%H:%M:%S')} REVERT: window {score:+.1f} < {args.revert_below} -> weights of "
                              f"{src.name}, lr {st['lr']:.1e}", flush=True)
                save_state()
    finally:
        flush(obs[me])
        save_state()
        if env is not None:
            sub.safe_close(env)
        print(f"STOPPED at live step {st['step']}", flush=True)


if __name__ == "__main__":
    main()
