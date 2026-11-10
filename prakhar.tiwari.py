"""Prakhar Tiwari's agent for the CSC_53439_EP multi-agent RL challenge (single-file submission).

HOW TO RUN
    python prakhar.tiwari.py                       # join the contest world, play the embedded policy for 50,000 steps
    python prakhar.tiwari.py --anti-stall 0        # same, without the anti-stall rule
    python prakhar.tiwari.py --steps 20000         # same, but stop after 20,000 steps
    python prakhar.tiwari.py --mode training --steps 100000 --save my.pt    # learn in your own training world

    Requires numpy, torch, pettingzoo and pzclient (python-dotenv is optional, to read a .env file).
    The token is read from the environment variable PETTINGZOO_TOKEN; it is never written in this file.
    Server: $PETTINGZOO_API_URL if set, otherwise the contest server https://csc53439ep.jdhp.org/api.

WHAT IT PLAYS
    * Contest world (observations Box(11), actions MultiBinary(2)): the EMBEDDED Double-DQN policy (section 7),
      played greedily and frozen, with an anti-stall rule (below).
        - Network: input = last 4 observations + previous action (46 numbers) -> 64 -> 64 -> 4 Q-values, one per
          joint action [0,0], [0,1], [1,0], [1,1]. 64 + 64 = 128 hidden units (the policy limit).
        - Weights: "co-player 143k" = a DDQN trained 500,000 steps in the training world with 3 copies of itself,
          then 143,360 more steps with 2 of the 4 agents playing OTHER policies (older checkpoints, PPO, random).
    * Any other world: a generic PPO actor-critic that learns from scratch, knowing only the spaces.
        - Observations Box / Discrete / MultiBinary; actions Discrete / MultiBinary / Box.
        - Policy 64 + 64 = 128 hidden units; critic 256 + 256 (used only for learning, not for acting).
        - Input = last 4 encoded observations + previous action (a short memory of how values change).

MEASURED PERFORMANCE (reward per 1,000 steps = sum of r, r in {-5, 0, +5}; live contest world, frozen play)
    | policy                                   | quiet world (16-17 Oct) | crowded world (20-21 Oct)       |
    | random actions                           | -14                     | -                               |
    | DDQN trained only with copies of itself  | +148 (280k steps)       | +35.1 (85k steps)               |
    | THIS FILE: co-player 143k + anti-stall   | -                       | +39.4 (124,595 steps, 21 Oct)   |

WHY IT PERFORMS THIS WAY (from recorded live steps; o0..o10 = the 11 observation values)
    1. The contest world became crowded (other participants). The rules did not change: the same policy still
       scores +135 in the training world. Live, it is near high o9 (o9 rises before r = +5) as often as before,
       but r = +5 follows within 10 steps only 20% of the time instead of 42% (others reach the source first),
       and `died` happens about twice as often. Contact situations (o1 = 1 with o4 = 1, which precede `died`)
       are 3.5x more frequent.
    2. Training against other policies (co-player training) helps: 143k gets r = +5 as often as before but
       `died` one third less often (2.8 vs 4.2 per 1,000), because it moves less in contact situations.
    3. Greedy play can FREEZE: in a contact situation the policy may prefer [0,0]; staying still keeps the
       observation identical, so it chooses [0,0] again, for up to thousands of steps (no reward meanwhile).
       In training, 1-2% random actions hide this. ANTI-STALL: after 10 identical observations while choosing
       [0,0], play the best moving action (by Q) for 5 steps. Frozen time fell from 8% to about 1% of steps.
    4. What did not help (measured live): fine-tuning offline on recorded live data (+16 to +32: the data came
       from greedy play only, so it learned to avoid the high-o9 situations), learning online during the contest
       (+34), and checkpoints chosen by training-world score (they range from +9 to +43 live).
    Details: docs/ (STATUS.md, DDQN.md, EXPERIMENTS_17OCT.md) and results/ (live_pool_20oct.txt, why_bad_checkpoints.txt).

FILE LAYOUT
    1. Spaces          encode any observation space, decode any action space
    2. Networks        PPO actor-critic and the dueling Q-network
    3. Reward shaping  optional potential-based shaping for the contest world (learning only)
    4. Agent           one shared policy for every agent id: acting, saving / loading, PPO learning
    5. Running         connection to the server, contest loop, training loop
    6. Command line
    7. Embedded weights
"""

import argparse
import base64
import io
import os
import sys
import time
import zlib
from collections import deque

import numpy as np
import torch
import torch.nn as nn

CONTEST_URL = "https://csc53439ep.jdhp.org/api"
TRAINING_URL = "https://csc53439ep.jdhp.org/training/api"
CONTEST_STEPS = 50_000                                               # default length of a contest run (--steps)
TRAINING_STEPS = 10_000_000                                          # default length of a training run (--steps)
DDQN_ANTI_STALL = 10                                                 # anti-stall threshold for DDQN play (0 = off)
POLICY_UNIT_LIMIT = 128                                              # rule: at most 128 hidden units in the policy
ARCH_V2 = {"history": 4, "actor": (64, 64), "critic": (256, 256)}   # default PPO architecture
ARCH_LEGACY = {"history": 1, "actor": (64,), "critic": (64,)}       # format of the first (v1) PPO checkpoints


# ============================================================================== 1. spaces


def space_kind(space):
    name = type(space).__name__
    if name in ("Box", "Discrete", "MultiBinary"):
        return name
    raise ValueError(f"unsupported space {space}")


def is_contest_world(obs_space, act_space):
    """True for the spaces of the contest world: Box(11) observations, MultiBinary(2) actions."""
    return (space_kind(obs_space) == "Box" and tuple(obs_space.shape) == (11,)
            and space_kind(act_space) == "MultiBinary" and int(np.prod(act_space.shape)) == 2)


class ObsEncoder:
    """Turns an observation of any flat space into a float vector.
    Box: standardised with running mean / variance (declared bounds are not trusted), clipped to +-5.
    Discrete: one-hot. MultiBinary: 0/1 floats."""

    def __init__(self, space):
        self.kind = space_kind(space)
        if self.kind == "Box":
            self.dim = int(np.prod(space.shape))
            self.mean, self.var, self.count = np.zeros(self.dim), np.ones(self.dim), 1e-4
        elif self.kind == "Discrete":
            self.n, self.start = int(space.n), int(getattr(space, "start", 0))
            self.dim = self.n
        else:
            self.dim = int(np.prod(space.shape))

    def update(self, x):
        """Running mean / variance (parallel Welford update) of raw Box observations."""
        if self.kind != "Box":
            return
        x = np.asarray(x, np.float64).reshape(-1, self.dim)
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, obs):
        if self.kind == "Box":
            x = np.asarray(obs, np.float64).ravel()
            return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -5, 5).astype(np.float32)
        if self.kind == "Discrete":
            v = np.zeros(self.n, np.float32)
            v[int(np.asarray(obs).ravel()[0]) - self.start] = 1.0
            return v
        return np.asarray(obs, np.float32).ravel()


class ActionHead:
    """Distribution over the actions of any flat action space, from the actor's raw outputs."""

    def __init__(self, space):
        self.kind = space_kind(space)
        self.space = space
        if self.kind == "Discrete":
            self.n_out = self.prev_dim = int(space.n)
            self.start = int(getattr(space, "start", 0))
        elif self.kind == "MultiBinary":
            self.n_out = self.prev_dim = int(np.prod(space.shape))
        else:  # Box: Gaussian on an unbounded variable u, squashed by tanh into [low, high]
            self.n_out = self.prev_dim = int(np.prod(space.shape))
            self.low, self.high = space.low.ravel().astype(np.float32), space.high.ravel().astype(np.float32)

    def dist(self, out, log_std=None):
        if self.kind == "Discrete":
            return torch.distributions.Categorical(logits=out)
        if self.kind == "MultiBinary":
            return torch.distributions.Independent(torch.distributions.Bernoulli(logits=out), 1)
        return torch.distributions.Independent(torch.distributions.Normal(out, log_std.exp()), 1)

    def to_env(self, sample):
        """Sampled tensor -> (action for the environment, float vector fed back as 'previous action')."""
        s = sample.numpy()
        if self.kind == "Discrete":
            prev = np.zeros(self.n_out, np.float32)
            prev[int(s)] = 1.0
            return int(s) + self.start, prev
        if self.kind == "MultiBinary":
            return s.astype(np.int8).reshape(self.space.shape), s.astype(np.float32)
        t = np.tanh(s)
        a = self.low + (t + 1) / 2 * (self.high - self.low)
        return a.astype(self.space.dtype).reshape(self.space.shape), t.astype(np.float32)


# ============================================================================== 2. networks


def mlp(i, hidden, o):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """PPO networks. Only the actor is the policy (<= 128 hidden units); the critic is used only for learning."""

    def __init__(self, in_dim, out_dim, arch, box=False):
        super().__init__()
        assert sum(arch["actor"]) <= POLICY_UNIT_LIMIT, "policy network exceeds the 128 hidden-unit limit"
        self.actor = mlp(in_dim, arch["actor"], out_dim)
        self.critic = mlp(in_dim, arch["critic"], 1)
        self.log_std = nn.Parameter(torch.full((out_dim,), -0.5)) if box else None


class QNet(nn.Module):
    """Dueling Q-network of the embedded DDQN policy (trained by src/train/ddqn.py): 64 + 64 = 128 hidden units.
    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)."""

    def __init__(self, d, n_actions):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU())
        self.v, self.a = nn.Linear(64, 1), nn.Linear(64, n_actions)

    def forward(self, x):
        h = self.body(x)
        a = self.a(h)
        return self.v(h) + a - a.mean(-1, keepdim=True)


# ============================================================================== 3. reward shaping


class ContestShaping:
    """Potential-based reward shaping for the contest world (Ng, Harada & Russell, 1999). Learning only.

    F(s, s') = gamma * Phi(s') - Phi(s) is added to the reward when learning. With such an F the best policy
    for the shaped reward is also the best policy for the true reward, but the learner gets a signal on the
    steps BEFORE r = +5 or `died`, instead of only on the step where they happen.

    Phi(o) = w9 * o9  -  w1 * o1 * max(o4, o7)  +  w8 * o8          (o0..o10 = the 11 observation values)
        o9 rises from ~0.13 to ~0.91 over the 20 steps before r = +5     (alone: AUC 0.93 for "r = +5 within 10 steps")
        o1 rises from ~0.43 to ~0.96 over the 20 steps before `died`;   o4 / o7 = 1 more often just before `died`
        o8 falls by 0.05-0.10 per step before `died`, and `died` follows when o8 reaches ~0.05-0.09
    damage_penalty (optional, NOT potential-based, makes the objective more cautious):
        - damage_penalty * (decrease of o8 on this step).
    On `died`: F = -Phi(s) (the potential after `died` counts as 0), so the reset of o8 to 0.5 is never rewarded.
    """

    def __init__(self, gamma=0.99, w_food=0.5, w_danger=0.5, w_health=1.0, damage_penalty=0.0, scale=5.0):
        self.gamma, self.wf, self.wd, self.wh, self.dp, self.scale = gamma, w_food, w_danger, w_health, damage_penalty, scale

    def phi(self, o):
        o = np.asarray(o, np.float64).ravel()
        return (self.wf * np.clip(o[9], 0, 1) - self.wd * o[1] * max(o[4], o[7]) + self.wh * o[8])

    def __call__(self, o, o2, died):
        """Shaping term in reward units (multiplied by `scale`: 1 unit of Phi = 5 reward points)."""
        if died:
            f = -self.phi(o)
        else:
            f = self.gamma * self.phi(o2) - self.phi(o)
            if self.dp:
                f -= self.dp * max(0.0, float(o[8]) - float(o2[8]))
        return self.scale * f


# ============================================================================== 4. agent


class Agent:
    """One shared policy for every agent id.

    In the contest world it loads the embedded DDQN weights and plays greedily on the Q-values (frozen).
    Elsewhere (or with --scratch) it is a PPO learner: act() collects transitions, observe() attaches the
    rewards, and an update runs every `rollout` steps per agent.
    """

    def __init__(self, obs_space, act_space, learn=True, lr=3e-4, rollout=256, epochs=4, minibatch=256,
                 gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.01, arch=None, pretrained="auto", seed=0):
        torch.manual_seed(seed)
        self.obs_space, self.act_space = obs_space, act_space
        self.enc, self.head = ObsEncoder(obs_space), ActionHead(act_space)
        self.learn, self.lr, self.rollout, self.epochs, self.minibatch = learn, lr, rollout, epochs, minibatch
        self.gamma, self.lam, self.clip, self.ent_coef = gamma, lam, clip, ent_coef
        self.critic_only_updates = 0     # > 0: next updates train only the critic (warm-up of a new critic)
        # Reward scale: fixed when pre-trained (the critic learned that scale), otherwise running std of returns.
        self.reward_scale, self.ret_stats = None, [0.0, 1.0, 1e-4]
        self.n_updates, self.pretrained = 0, False
        self.qnet, self.q_eps = None, 0.0    # set when a DDQN checkpoint is loaded; q_eps = random-action rate
        # Anti-stall (DDQN play): if the observation has not changed for stall_k steps in a row and the greedy action
        # is "all zeros" (no movement), play the best MOVING action (by Q) for stall_hold steps. A greedy deterministic
        # policy can otherwise stop in a situation where staying keeps the observation identical, and stay there forever
        # (measured live: up to 3,416 steps of [0,0] with o1 = 1). 0 = off.
        self.stall_k, self.stall_hold = 0, 5
        self._build(dict(arch or ARCH_V2))
        if pretrained == "auto":
            pretrained = PRETRAINED if (is_contest_world(obs_space, act_space) and PRETRAINED.strip()) else None
        if pretrained:
            self.load_bytes(pretrained if isinstance(pretrained, bytes) else _decode(pretrained))

    def _build(self, arch):
        self.arch = {"history": int(arch["history"]), "actor": tuple(arch["actor"]), "critic": tuple(arch["critic"])}
        self.in_dim = self.enc.dim * self.arch["history"] + self.head.prev_dim
        self.net = ActorCritic(self.in_dim, self.head.n_out, self.arch, box=self.head.kind == "Box")
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.prev, self.hist, self.buf, self.last_x, self.ret = {}, {}, {}, {}, {}
        self.stall_n, self.last_raw, self.unstall = {}, {}, {}

    @property
    def policy_hidden_units(self):
        return sum(self.arch["actor"])

    # ------------------------------------------------------------------ saving / loading
    def state(self, full=False):
        s = {"arch": dict(self.arch), "net": self.net.state_dict(), "norm_mean": getattr(self.enc, "mean", None),
             "norm_var": getattr(self.enc, "var", None), "norm_count": getattr(self.enc, "count", None),
             "reward_scale": self.reward_scale, "ret_stats": list(self.ret_stats), "n_updates": self.n_updates}
        if full:
            s["opt"] = self.opt.state_dict()
        return s

    def save(self, path, full=False, extra=None):
        """Atomic save (write to a temp file, then rename), so a crash never leaves a half-written file."""
        s = self.state(full)
        if extra:
            s.update(extra)
        tmp = f"{path}.tmp"
        torch.save(s, tmp)
        os.replace(tmp, path)

    def load(self, path):
        with open(path, "rb") as f:
            return self.load_bytes(f.read())

    def load_bytes(self, raw):
        """Load a DDQN checkpoint (frozen Q-network policy) or a PPO checkpoint (actor-critic)."""
        ck = torch.load(io.BytesIO(raw), weights_only=False, map_location="cpu")   # also on machines without a GPU
        if ck.get("algo") == "ddqn":
            return self._load_ddqn(ck)
        return self._load_ppo(ck)

    def _load_ddqn(self, ck):
        assert self.head.kind == "MultiBinary", "DDQN checkpoints are for MultiBinary action spaces"
        self._build({"history": ck.get("hist", 4), "actor": (64, 64), "critic": (64,)})
        n_bits = self.head.n_out
        self.qnet = QNet(self.in_dim, 2 ** n_bits)
        self.qnet.load_state_dict(ck["net"])
        # joint action j <-> bits, e.g. for 2 bits: 0 = [0,0], 1 = [0,1], 2 = [1,0], 3 = [1,1]
        self.q_actions = np.array([[(j >> (n_bits - 1 - b)) & 1 for b in range(n_bits)] for j in range(2 ** n_bits)],
                                  dtype=np.int8)
        # the fixed normalisation the Q-network was trained with (count = 1e9: never updated)
        self.enc.mean, self.enc.var, self.enc.count = np.asarray(ck["norm_mean"]), np.asarray(ck["norm_std"]) ** 2, 1e9
        self.learn, self.pretrained, self.reward_scale = False, True, 0.2
        self.stall_k = DDQN_ANTI_STALL                                  # anti-stall on for DDQN play (see header)
        return ck

    def _load_ppo(self, ck):
        self.qnet = None
        arch = ck.get("arch", ARCH_LEGACY)
        if {k: tuple(v) if k != "history" else v for k, v in arch.items()} != self.arch:
            self._build(arch)
        self.net.load_state_dict(ck["net"])
        if ck.get("norm_mean") is not None and self.enc.kind == "Box":
            self.enc.mean, self.enc.var = np.asarray(ck["norm_mean"]), np.asarray(ck["norm_var"])
            self.enc.count = ck.get("norm_count") or 1e5
        self.reward_scale = ck.get("reward_scale", 0.2)
        self.ret_stats = ck.get("ret_stats", self.ret_stats)
        self.n_updates = ck.get("n_updates", 0)
        if "opt" in ck:
            try:
                self.opt.load_state_dict(ck["opt"])
            except ValueError:
                pass
        self.pretrained = True
        return ck

    def set_lr(self, lr):
        self.lr = lr
        for g in self.opt.param_groups:
            g["lr"] = lr

    # ------------------------------------------------------------------ acting
    def forget(self, agent_id):
        """Forget the trajectory of this agent id (rejoin, reconnection): its memory restarts empty."""
        for d in (self.prev, self.hist, self.buf, self.last_x, self.ret, self.stall_n, self.last_raw, self.unstall):
            d.pop(agent_id, None)

    def forget_all(self):
        for i in list(self.hist) + list(self.buf):
            self.forget(i)

    def _input(self, i, obs):
        """Network input of agent i: its last `history` encoded observations (newest first) + previous action."""
        h = self.hist.get(i)
        if h is None:
            h = self.hist[i] = deque(maxlen=self.arch["history"])
        h.appendleft(obs)
        frames = list(h) + [h[-1]] * (self.arch["history"] - len(h))   # pad with the oldest frame
        prev = self.prev.get(i, np.zeros(self.head.prev_dim, np.float32))
        return np.concatenate([self.enc(f) for f in frames] + [prev])

    def act(self, observations):
        """observations: {agent_id: obs} -> {agent_id: action}. PPO: learns first if a rollout is complete."""
        ids = list(observations)
        if self.learn and self.enc.kind == "Box":
            self.enc.update(np.array([np.asarray(observations[i], np.float64).ravel() for i in ids]))
        x = torch.tensor(np.array([self._input(i, observations[i]) for i in ids]), dtype=torch.float32)
        for k, i in enumerate(ids):
            self.last_x[i] = x[k]
        if self.qnet is not None:
            return self._act_q(ids, x, observations)
        return self._act_ppo(ids, x)

    def _act_q(self, ids, x, observations=None):
        """DDQN: the action with the highest Q-value (a random one with probability q_eps, 0 by default)."""
        with torch.no_grad():
            q = self.qnet(x).numpy()
        actions = {}
        for k, i in enumerate(ids):
            j = int(np.random.randint(len(q[k]))) if np.random.rand() < self.q_eps else int(q[k].argmax())
            if self.stall_k and observations is not None:
                o = np.asarray(observations[i], np.float64).ravel()
                lo = self.last_raw.get(i)
                self.stall_n[i] = self.stall_n.get(i, 0) + 1 if lo is not None and np.abs(o - lo).max() < 1e-3 else 0
                self.last_raw[i] = o
                if self.stall_n[i] >= self.stall_k and not self.q_actions[j].any():
                    self.unstall[i] = self.stall_hold
                if self.unstall.get(i, 0) > 0 and not self.q_actions[j].any():
                    moving = self.q_actions.any(1)
                    j = int(np.where(moving, q[k], -np.inf).argmax())
                    self.unstall[i] -= 1
            actions[i] = self.q_actions[j].reshape(self.head.space.shape)
            self.prev[i] = self.q_actions[j].astype(np.float32)
        return actions

    def _act_ppo(self, ids, x):
        """PPO: sample from the policy; when learning, store the transition until its reward arrives."""
        if self.learn and self._ready(ids):
            self._update(ids)
        with torch.no_grad():
            d = self.head.dist(self.net.actor(x), self.net.log_std)
            a = d.sample()
            logp = d.log_prob(a)
            v = self.net.critic(x).squeeze(-1) if self.learn else torch.zeros(len(ids))
        actions = {}
        for k, i in enumerate(ids):
            env_a, prev = self.head.to_env(a[k])
            actions[i] = env_a
            self.prev[i] = prev
            if self.learn:
                self.buf.setdefault(i, []).append([x[k], a[k], logp[k], v[k], None, False])
        return actions

    def observe(self, rewards, dones=None, applied=None):
        """Attach the reward (and termination) of the last action of each agent id (PPO learning only).
        `applied[i] == False` (contest: our action arrived too late) drops that transition."""
        for i, r in rewards.items():
            if not self.learn or not self.buf.get(i):
                continue
            if applied is not None and not applied.get(i, True):
                self.buf[i].pop()
                continue
            done = bool(dones.get(i, False)) if dones else False
            self._scale_update(i, float(r), done)
            self.buf[i][-1][4] = float(r)
            self.buf[i][-1][5] = done
            if done:
                self.prev.pop(i, None)
                self.hist.pop(i, None)

    # ------------------------------------------------------------------ PPO learning
    def _scale_update(self, i, r, done):
        """Running variance of the discounted returns (used to scale rewards when no fixed scale is set)."""
        g = self.ret.get(i, 0.0) * self.gamma + r
        self.ret[i] = 0.0 if done else g
        m, v, c = self.ret_stats
        c2 = c + 1
        d = g - m
        m2 = m + d / c2
        self.ret_stats = [m2, (v * c + d * (g - m2)) / c2, c2]

    def _scale(self):
        return self.reward_scale if self.reward_scale else 1.0 / (np.sqrt(self.ret_stats[1]) + 1e-8)

    def _ready(self, ids):
        done = [i for i in ids if len([t for t in self.buf.get(i, []) if t[4] is not None]) >= self.rollout]
        return len(done) == len(ids) and len(ids) > 0

    def _update(self, ids):
        """One PPO update: GAE advantages per agent, then `epochs` passes of clipped-objective minibatches."""
        X, A, LP, ADV, RET = [], [], [], [], []
        scale = self._scale()
        for i in ids:
            traj = [t for t in self.buf[i] if t[4] is not None]
            self.buf[i] = [t for t in self.buf[i] if t[4] is None]
            with torch.no_grad():
                v_next = self.net.critic(self.last_x[i][None]).squeeze()
            vs = torch.stack([t[3] for t in traj])
            r = torch.tensor([t[4] for t in traj]) * scale
            dn = torch.tensor([float(t[5]) for t in traj])
            adv, gae = torch.zeros(len(traj)), torch.tensor(0.0)
            for t in reversed(range(len(traj))):
                nv = v_next if t == len(traj) - 1 else vs[t + 1]
                delta = r[t] + self.gamma * nv * (1 - dn[t]) - vs[t]
                gae = delta + self.gamma * self.lam * (1 - dn[t]) * gae
                adv[t] = gae
            X.append(torch.stack([t[0] for t in traj])), A.append(torch.stack([t[1] for t in traj]))
            LP.append(torch.stack([t[2] for t in traj])), ADV.append(adv), RET.append(adv + vs)
        X, A, LP, ADV, RET = map(torch.cat, (X, A, LP, ADV, RET))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        critic_only = self.critic_only_updates > 0
        for _ in range(self.epochs):
            for mb in torch.randperm(len(X)).split(self.minibatch):
                vl = ((self.net.critic(X[mb]).squeeze(-1) - RET[mb]) ** 2).mean()
                if critic_only:
                    loss = 0.5 * vl
                else:
                    d = self.head.dist(self.net.actor(X[mb]), self.net.log_std)
                    ratio = torch.exp(d.log_prob(A[mb]) - LP[mb])
                    pg = -torch.min(ratio * ADV[mb], ratio.clamp(1 - self.clip, 1 + self.clip) * ADV[mb]).mean()
                    loss = pg + 0.5 * vl - self.ent_coef * d.entropy().mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
        if critic_only:
            self.critic_only_updates -= 1
        self.n_updates += 1


# ============================================================================== 5. running


def load_dotenv_if_any():
    """Read a local .env file if python-dotenv is installed; default the server to the contest world."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if "PETTINGZOO_API_URL" not in os.environ:
        os.environ["PETTINGZOO_API_URL"] = CONTEST_URL


def make_env(api_url=None):
    import pettingzoo
    import pzclient  # noqa: F401  (registers the remote environments)

    kw = {"api_url": api_url} if api_url else {}
    return pettingzoo.make("parallel", "csc53439ep/csc53439ep-remote-v1", **kw)


RETRY_SECONDS = 10            # wait between two reconnection attempts
TRANSIENT_STATUS = (408, 429, 500, 502, 503, 504)


def is_transient(e):
    """Server-side or network problem that is worth waiting out (as opposed to e.g. a rejected token)."""
    import pzclient

    return (isinstance(e, (pzclient.ServerUnreachableError, pzclient.AgentNotConnectedError))
            or getattr(e, "status_code", None) in TRANSIENT_STATUS)


def connect(api_url=None):
    """make_env, retried until the server answers (it may be down or unreachable when we start)."""
    import pzclient

    while True:
        try:
            return make_env(api_url)
        except pzclient.RemoteEnvError as e:
            if not is_transient(e):
                raise
            print(f"{time.strftime('%H:%M:%S')} cannot reach the server ({type(e).__name__}); retry in {RETRY_SECONDS} s",
                  flush=True)
            time.sleep(RETRY_SECONDS)


def safe_close(env):
    try:
        env.close()
    except Exception:  # noqa: BLE001  (the server may already be gone)
        pass


class Stats:
    """Progress line. Format parsed by the evaluation scripts: 'reward/1000 X (+:N -:M) | missed K'."""

    def __init__(self):
        self.t0, self.steps, self.reward, self.food, self.deaths, self.missed = time.monotonic(), 0, 0.0, 0, 0, 0
        self.think = []

    def line(self, agent):
        n = max(self.steps, 1)
        return (f"{self.steps} steps {time.monotonic() - self.t0:.0f}s | reward/1000 {self.reward / n * 1000:+.1f} "
                f"(+:{self.food} -:{self.deaths}) | missed {self.missed} | think {np.mean(self.think[-200:]) * 1000:.1f}ms"
                f" | updates {agent.n_updates}")


def play_contest(args, agent, shaping=None):
    """Shared contest world: one agent, a step every 5 s.
    Never gives up on a server problem: waits RETRY_SECONDS and rejoins (after 3 failed attempts in a row it also
    rebuilds the connection). Only a non-transient error (e.g. a rejected token) stops it.
    Logged: '+' = number of steps with r > 0 (r = +5), '-' = number of `died`, 'missed' = actions not applied in time."""
    import pzclient

    env = connect(args.api_url)
    me = env.possible_agents[0]
    stats = Stats()
    failures = 0

    def report(e, what):
        print(f"{time.strftime('%H:%M:%S')} server problem during {what} ({type(e).__name__}: {str(e)[:120]}); "
              f"retry in {RETRY_SECONDS} s", flush=True)

    def rejoin():
        nonlocal env, failures
        while True:
            try:
                if env is None:
                    env = connect(args.api_url)
                o, _ = env.reset()
                failures = 0
                return o
            except pzclient.RemoteEnvError as e:
                if not is_transient(e):
                    raise
                failures += 1
                report(e, "join")
                if failures >= 3:             # rebuild the connection from scratch
                    safe_close(env)
                    env = None
                time.sleep(RETRY_SECONDS)

    try:
        obs = rejoin()
        for _ in range(args.steps):  # the episode never ends: bound the loop
            ts = time.monotonic()
            action = agent.act({me: obs[me]})
            stats.think.append(time.monotonic() - ts)
            try:
                obs2, rew, term, trunc, infos = env.step(action)
            except pzclient.RemoteEnvError as e:
                if not is_transient(e):
                    raise
                agent.forget(me)
                failures += 1
                if isinstance(e, pzclient.AgentNotConnectedError) or failures >= 3:
                    if failures >= 3:
                        report(e, "step")
                        safe_close(env)
                        env = None
                    obs = rejoin()
                else:
                    report(e, "step")
                    time.sleep(RETRY_SECONDS)
                continue
            failures = 0
            sw = infos[me].get("shared_world", {})
            applied = sw.get("action_applied", True)
            died = bool(infos[me].get("died", False))
            r = rew[me] + (shaping(obs[me], obs2[me], died) if shaping else 0.0)
            agent.observe({me: r}, {me: bool(term[me] or trunc[me])}, {me: applied})
            obs = obs2
            stats.steps += 1
            stats.reward += rew[me]
            stats.food += rew[me] > 0
            stats.deaths += died
            stats.missed += not applied
            if stats.steps % args.log_every == 0:
                print(stats.line(agent), flush=True)
                if args.save:
                    agent.save(args.save)
    finally:
        if env is not None:
            safe_close(env)
        if args.save:
            agent.save(args.save)
        print("final:", stats.line(agent), flush=True)


def play_training(args, agent, shaping=None):
    """Own training world: all its agents share the policy; the world waits for our actions.
    On a server problem: waits RETRY_SECONDS, reconnects, resets the world and goes on (learned weights are kept)."""
    import pzclient

    env = connect(args.api_url)
    stats = Stats()

    def restart(e):
        nonlocal env
        print(f"{time.strftime('%H:%M:%S')} server problem ({type(e).__name__}: {str(e)[:120]}); "
              f"reconnecting in {RETRY_SECONDS} s", flush=True)
        if args.save:
            agent.save(args.save)
        safe_close(env)
        time.sleep(RETRY_SECONDS)
        agent.forget_all()
        while True:
            env = connect(args.api_url)
            try:
                o, _ = env.reset(seed=args.seed + stats.steps)
                return o
            except pzclient.RemoteEnvError as e2:
                if not is_transient(e2):
                    raise
                safe_close(env)
                time.sleep(RETRY_SECONDS)

    try:
        try:
            obs, _ = env.reset(seed=args.seed)
        except pzclient.RemoteEnvError as e:
            if not is_transient(e):
                raise
            obs = restart(e)
        while stats.steps < args.steps:
            ts = time.monotonic()
            actions = agent.act({a: obs[a] for a in env.agents})
            stats.think.append(time.monotonic() - ts)
            try:
                obs2, rew, term, trunc, infos = env.step(actions)
            except pzclient.RemoteEnvError as e:
                if not is_transient(e):
                    raise
                obs = restart(e)
                continue
            died = {a: bool(infos[a].get("died", False)) for a in rew}
            shaped = {a: rew[a] + (shaping(obs[a], obs2[a], died[a]) if shaping else 0.0) for a in rew}
            agent.observe(shaped, {a: bool(term[a] or trunc[a]) for a in rew})
            obs = obs2
            stats.steps += 1
            stats.reward += float(np.mean(list(rew.values())))
            stats.food += sum(r > 0 for r in rew.values())
            stats.deaths += sum(died.values())
            if stats.steps % args.log_every == 0:
                print(stats.line(agent), flush=True)
                if args.save:
                    agent.save(args.save)
    finally:
        safe_close(env)
        if args.save:
            agent.save(args.save)
        print("final:", stats.line(agent), flush=True)


# ============================================================================== 6. command line


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["contest", "training"], default="contest")
    p.add_argument("--api-url", default=None, help="default: $PETTINGZOO_API_URL, else the contest server")
    p.add_argument("--steps", type=int, default=None,
                   help=f"steps to play (default {CONTEST_STEPS:,} in contest, {TRAINING_STEPS:,} in training)")
    p.add_argument("--no-learn", action="store_true", help="frozen policy (the embedded DDQN policy is always frozen)")
    p.add_argument("--lr", type=float, default=None, help="PPO learning rate (default 1e-4 in contest, 3e-4 in training)")
    p.add_argument("--rollout", type=int, default=256, help="PPO: steps per agent between two learning updates")
    p.add_argument("--epochs", type=int, default=4, help="PPO passes over each batch")
    p.add_argument("--ent-coef", type=float, default=0.01, help="PPO entropy bonus (exploration)")
    p.add_argument("--shaping", action="store_true", help="potential-based reward shaping (contest world only)")
    p.add_argument("--load", default=None, help="start from this checkpoint instead of the embedded weights")
    p.add_argument("--scratch", action="store_true", help="ignore the embedded weights (learn from scratch)")
    p.add_argument("--save", default=None, help="save the learned weights here (periodically)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anti-stall", type=int, default=None,
                   help=f"DDQN play: never repeat 'no movement' after N identical observations (default {DDQN_ANTI_STALL}, 0 = off)")
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()
    if args.steps is None:
        args.steps = CONTEST_STEPS if args.mode == "contest" else TRAINING_STEPS
    load_dotenv_if_any()
    torch.set_num_threads(1)  # small networks: one thread is fastest and leaves the machine responsive
    if args.api_url is None and args.mode == "training":
        args.api_url = os.environ.get("PZ_TRAIN_URL", TRAINING_URL)

    probe = connect(args.api_url)
    a0 = probe.possible_agents[0]
    obs_space, act_space = probe.observation_space(a0), probe.action_space(a0)
    lr = args.lr or (1e-4 if args.mode == "contest" else 3e-4)
    agent = Agent(obs_space, act_space, learn=not args.no_learn, lr=lr, seed=args.seed,
                  rollout=args.rollout, epochs=args.epochs, minibatch=min(256, args.rollout), ent_coef=args.ent_coef,
                  pretrained=None if args.scratch or args.load else "auto")
    if args.load:
        agent.load(args.load)
        agent.set_lr(lr)
    if args.anti_stall is not None:
        agent.stall_k = args.anti_stall
    shaping = ContestShaping(gamma=agent.gamma) if args.shaping and is_contest_world(obs_space, act_space) else None
    algo = ("DDQN (greedy on a dueling Q-network 64+64 = 128 hidden units)" if agent.qnet is not None else
            f"PPO (policy {agent.arch['actor']} = {agent.policy_hidden_units} hidden units, critic {agent.arch['critic']})")
    print(f"agent {a0}: obs {obs_space} act {act_space} | pretrained {agent.pretrained} | learning {agent.learn} | "
          f"{algo} | history {agent.arch['history']} | shaping {shaping is not None}", flush=True)
    (play_contest if args.mode == "contest" else play_training)(args, agent, shaping)


# ============================================================================== 7. embedded weights


def _decode(s):
    return zlib.decompress(base64.b64decode("".join(s.split())))


# DDQN policy for the contest world (Box(11) -> MultiBinary(2)): checkpoint models/FINAL_ddqn_500k.pt
# (step 500,000 of src/train/ddqn.py in the training mode), stored as zlib + base64 of a torch checkpoint.
# Replaced by src/build_submission.py, which rewrites only the text between the triple quotes.
PRETRAINED = """
eNqkuwk4Vd/7/20eE6FUkgYZI1PCXvc6oaSopAkNkiEkHFORSsZESJQIGVJJJalw1loKDUoTTRqoNKtQkUZ/ffr+/t/n+zzX/3c9
z+9Z17XP2fvsfb/u93rvve6913Xt42gvKiYkJCUl9P9oCkKKQu7BHj6+4V7TPN1D3Q2C1vvbWg8Xcv2PFi2yhS+k7TzEEHL3XxfI
F/5n1dOTH8AXcRYdWg3wCuWLbuGLaTvLDW2tDfSMMDA02Ojlu84nlC/uERo4lMHALSzU1z9Exi3Ya22Yr7+nW6hXQEhgsFu4sQxf
QlvbWXIoMGToSPd1XnzJvyEytv6B7qGL//4ow5dyFh46xpAv/U9Kj6Awvsz8aNlQvuwieyH7mfYGCfxh9gb2wgl8uSQPj0B/fy+P
UN/AgBCZhcGeXsFenrN8PUJl+MN1nPjyoXwFJ/4IZ9l/a13r6x7CV/QZEuIj6fM3kRFfyUfGfmYoX/kffjx/pL1wPH9Uks8fhEoo
f7QTf8y/u2v0X90d+x8QY77qX8i4f0HU/kDG/4Woh/InOPEn/luH0V8dk/4DYcKf/Beh8S/ElD8Izb8IrVC+thNf5986TP5Lh+5/
QEz5ej4y84UUQvlT/7o1M4GvP/Q15JbBX9K0UL6hE9/o32JM/oox/g/OdL7JXzGm/xIz/Y8Ys7+IGaF8cye+xb/FmP6XGMv/gJjx
ub8Q9C8I/IHgvxBeKH+mE9/q3zpM/+qw/g/EDL7NX8SsfyFm/0HY/kXMCeXbOfHnOv+53sP/S8K8/4g359v/jXf4Ey/8x475f+1Y
8JexMJTv6MRf5CzxD+MfBU7/QbDgLx4iCIfyl/xDiOcv/fOx7G/08lC+sxPf5R8F7v+lwPV/x4v8ubwM+Sv+nJIhwso/BLE/Glb9
1bD6L8UtlL/Gie/+jwb3vxrW/ifDiD80SuzFQvme/yDi+V5/RHj/DV8Xyvdx4vuGOUv/GaGBwRvcNni5B/D9PALCNgRFGLh5BAZ7
GWwI8w/1dQ8Odo/4MzI9hsZLaHDYn4Gy/u9hMgGef/fy/e2F4vkbPIbCPL08QmS8Av6syPAD/nFjLT/wH5n+7qG+AUb8oAQ+34kf
nMgPceKHag/5KxvPD/sX0DM0ImgoLvyfHnib8jcm7Ujkb3LiR2jbi/7DQvzIBQsWzBscav982AuF8jevTfIJcEZDext2mI3k6lMc
xHlTRjec5erT62sdoGF7w4710LC3IWMUWlZf4ApL6s81xIGYVwPDOg3pdli9IbahGvhRPkPKtjjxt4byt6395+z840tIqCc/2me9
zz9d3O4zpDvGiR/7L91xPhF/cs8YOni+hA00JNcfakjBVtb1FeAs64Zd1AOxUwNtiIH6k6O8caB5/V6M9MJwfaZWfTke15Anj/nx
f9ImOPETQ/k71v5TO318Q0L5SfZi/5gWEhgW7OHF3+msPrSxYchV/5BpNlYLZrn9qbBDfgf5u0e4GZmarDcICuUnhxk42ktKtW+O
PZQmLiT0Z3H8P5Z3FaGp/7u8G3gPddU91C3cKzhkqC7aWmv/36r8/7kZ/cl4rvtx7J/T82f5P2ccKzTj3xn/Vc/d3P191wVs8AoI
tbU2+X+d9P/SzEz/COBt5L/9c82I/LcCRgjx/reAtRGhXoF/qr+tNfc/yftfzd83NNTf64+GeOgc8+eUSfy3GoYL4f+4q04ztLW2
/P8hQORVGlb8YY5n/B7NPfzeQOr0rtMjru5o3y5CBcufQE/DBPZ9qReO/b6Jmae6wKP35exgUwZ0vtKDhEfS2LjUBdZPdScm07Lg
y7tsQcWbbmAfS6hh9hN0Vmg0WI4MBxW5a5Cu8wh+q/3iBqduhifWbjAomw0F8bq48/BFcvpsHTVTvEmnxsZDidN66NrzAA4UF0Fo
3DEAG5Ua4wdf0cUFBtR6KqudevcSRQJl8jbuOI7rKcIOn3mo3C0IJvDzocfCASze7gTNMwzfXPWJ2uw4QY6oFYM2Owxfnj6hdb/H
g4RXIYy2XAaZVApvqW6FQzv1UEzBSpp9yQByKyJQwcOFgv1bdfHRa+/RdhNPNGUaH752lcExcVk8XbwBJoSfpt7VjuhCRCqU9UzE
pKiGHZxeAI71puAWHVs77ZMvfF8rA0GZHJ1gFcduphbAQsNYiDm/HqIiblgaLU2hOzgv/P7dYxJQfYytlxWHflEhEE91JJG9c9jl
DbvI5scdRMTPEO6+1mD+YzKYjfPt2htXjtNedwmmv3wASS53RwUtmqy7KAWV3nVl+xMdsMmJbCiWUGaZSecFV8SBtmc10ihnHrhX
NkOwYQ9E2c0mCoOjsP5DRbpReCX7oBdKu59nwnaPY9xcTVumsVEcjPfshL5Fkxhc2007NoRBSeddYvwzHTatLibzf4SRcS0m2BU/
Qx417+kHrZ1ged2SKo69xd4dGcc+qlaAls91mDY4mjKBMfawcGduG8LRMnllLk5OnAplEpBbyqjyh4W4oGZsXd+NtXSalT/af9uS
cVEPyU++Np2hKs1eoI2ktGw3cd41EvhFSciNd4PRuabgr6gL7hZmlB3WgU2PRaDTtY8TGD4mnQmX4FrkbjAP0qZr9sqxWycPo8W3
zqMIR2PImb2Fxh7og2+xa+HcPh9I3FcHzoMikNprQEclbscuOQvwouoRjE3fxj4cV6d3D4/DfipB7I3+cXh8xRtaRq6DPuP9EHux
D8WP/EznNhvBiL3RzMz2IGhKudKDm3bDOkMxuLVlNTxJywDP34sheVIZWuS4h2Zv/ITUmxaDVVgdp6VrB3e3zYVT+xzRwiwdmNY/
AE89V8K2Ckcy7osSLQ5TA6Ud6+DaHorWu3Yh9Qs/UGi9EniGWcF3sTLYnDeOuhV7kUg9UyTnkAeGDytYbmJgnWXZBUjIMaPDx9qz
2Y3LmImDNjgFJINd40Q2dWkjjJB4jnScRzHXI610k8I+Kpq3ENgzPvlmnUY2ZhXTurUdJGuMEbmzs7728iqg30QCoLExjB3wrYfx
HzehzrzZ+Et/B6iEz4DjqBlih/9Gy/S/Iql5a+i62CNo/bE2uuXiMFok2YFaB9eh6JM16NbeBlxSEc9+FJvAyI2V9HbRJVpk8xzN
eC7LPlj/RGIDutjV7Bxe8kqVrl3hhh/2GUPil/XoregC1OW2i6ycGQ/TLzxHUYPTUZ6uFHtr+ogLL3lF0owXk6ZnHuh34l3Q1len
Dw6ftAhZOAmf6T/A3TwkTpMKm8nmp8ac8td98HLtW8HW7E10XfcycDmxn5uWWUYP/86Gil+etZaLr9GfV6Rp97xA0Jq9DL5rO1tu
tbKHgVsFKOaEIZyt84b+8FuwXLcDmq9u4LY+ug2qT0zQwXVlTOb5PhagLYlz6pbCqjFLmFmdHOT6uEJPRSlo2OdC5eJ1cJZlo30h
iTDzDsNlvqJ0ed4L6F4+lty8tREuvruGVuwvRMZfCHTMfIvcPoWASFYH9Iq1w8yM+9yjfj18YOh6veISiX7Nf41ayHfwZCVI1mCW
wP6OBwy8+wCbZ9tyx2etgmRj1+qp6oeRssZE1jFZj51qNWftVdVwXSFawOQPUqOJN5H/zS2k7HACyI2IwM9mKWC8eQzOPxaGK8Rd
yTeT55zXkmKuUeAM0UpzIUCUj5a+kSOEN41tMlpMxjcp4qyx11HB0+FQELsTbJLfkWHpXrDSWB7WzvKF/DJlJnYkELaU7EFHlLKp
jLI+U7z9HAkWp8KCiAGqetkRDp2WwLGb9gpGljOk7jOBnlFPJ9pvm/Gscje2uObE0PjSZYNiGNRyD6HTawrR5rEcDJRIYUcjNdAZ
L4B3l0zxKwk5vIRNRINpVZz3ZgKX3IXIJyvKTey6Qjucl6CvVg64xc2DGxkcSwVEhDae3wNp2pnU22kV9DxXJVcCvaiXbg04KyvB
o1cMzXQNp5aK/WDjfJm8Ug9Cfu2NQFvlaF2mJcrM86CWD+NoW4YcXm62Btqs78NZ+2Zwcy+FnTWT2NEGJJgZ6EYU3kzEU6tiQGrp
Eqax+BKMv6WGezTSkEdwGWt9Il6HfDLgteQIvFWynE5VMGX4yFFYg+VJt44/it54E47Xn4XIb5EUf9mPp+8oo/15hMTqILYxqpVM
Mg+F1KoqVJj1HlWo1YLw3hm45eUoLO89CWLN5sJAwmd0xzqdluu6Y1KuD8/vLoPrN+SJzwJRTHiKOP/WKnT+8zY0t/8jXPJ8aPlr
iSGacvJizVrf4Tg36QndpV2EeraUwP1XLuymVRT+lPqZvsm3YL9+9sDYb+64yMUZ3Jw/0uufrsE7q3rQElpEzbSSUOVqaxocY0dZ
uh0bnlbDtCZZwCarYbWKfgFU/FINtROVw/snlILz83t0W9lZcq5IAe/rms6Oz/tIrXd3UN8p9lC2uJMcN9dknyWfgsWwQhj5Qo4t
084HQ9U4HDVfkl3LaIBPR7YS3Wev0Y7TUrSq9iTZ/Wsb+Lel0av9mBb5iRHiXk4zE8PJ2+FKxO7QOcsnIV4sRqqZmWWkE8tRoujB
UUqTIsLAvDeKxPUPIp35T5DusM+o+5cofpXhjrs+IiyXWImWLr1INUVmwdHKCbDKp0ewLngKkjt3ApZJa5L2Ed1kNz+PVifNwNrL
umsvHghGY/08wU1tDbwsuA711uPQjh87kF1aC91hepJ2tiyEK2aS2Hy6LBiNek4lHGbivguFePROAzzO/D03ttSEhtz4SBv89oBM
427ys/UQ07kkB/4Z6SxDpgPUFs4htpEJ7MX4UfD4wiIQSVUCoRwCatN60fLJYqyj6joYHOhH7m/3gVbBaMHcR2ZgtCsZAnww8Ss/
zr0y3waH7BYi8C0iGSEVgh6rEColv0MwWbxLECfiSY98MAH9rtXkhnAxVOnPhms5E+nx1/vp6ooCKp01k9pV9AoWDV9N12wrhzc1
y9gDdBl/S7e0dBl2lt4ST2ETR12HY6HP0ZaERSwiJon6OsSyoLLRbPfppdB88yH16iLo+van6LSsEF1fowu2o+rQMV9VwGJT2JnP
hfD1/kEyZ9sJLizTnbTnOcDdZ3VIarIa+fzOCbIvq0FRTww8i9svuGMgi+e/tADxMBtaUcKn/h4xMGppNpKWnUY3zxCGdrVWVGad
C44m5jjieBIJ/jgc1u1WEMgerYWx4hfxJpMQ2rlsGLbQmUGMVN/RXfM0UP9DaSydGUsV10ehrk1i2M5ClE1ashGHnIhDE+23QExZ
ICz4+URQuWMhxC4ZjtccruMUKp6D4tEoenNUbq0AJrPGuERBxeLbcCraH5xd9qMzPdWo+c4scJXogxDpfbD0QTE92LwTPMhafGqY
JFQ01EDk2EuwacwxWvf2Dmi8rKU29Qokd3wmndUaAnXrUwX+vc+QcLorOumUykyThjFnbxt8JaicrStcyQ6X+ghWlbdzpcFxWOvj
eKwwXxEHTJ6IDwfJYL3RDtSvyx/GBr+jsiXl9MQXXRx75wKkrtyEJ935SOVXLUQzLGuRw3UNZmA5lsVPPAEppftA5nsl/T1wCoj/
Axh/bRWMcJrE5s3lYf7aI8S25xKU2jZB/JxIOCgiRTIsU7HVxlVI96wRdVXaiJPdH9CEAEvM0xvOFrgcpCLbZmC1+TNZ2Oh6fDdp
Ox18asWmSa2mVSVqbF6nKAkPt8AL5feyr5HiGE1Npd3h8lRHx5WtnbqWVl5UZJt4u8F1riZYuF3krmjMJIruH5BY8yPY1SLMwjZn
k7mXRemKZ3Iw/oUCSx+II4c8zUBzYDw3QvcXzTvjhXcGpNKTam2Wz9RmQmizIhvuawBXlXXY6dZJ0Dj9Dnw8kU7E5hnDRSwE66/E
0IoXO2G10WGaeigY3itVkXTH/UhN4iJqXqou2JhfUxscHY++j7WxwAXxVN89kad3ngc1bW4oRO8Y6D7PopXyE6D7ixWKtI1CgisL
oaQ/HT192EAm+a3gzenxNv+xbJBrCteh2il+3OAxLVBxS0CWKYSbqrUCpnvGo4CydME0BxneB/52FFV1EuVM9RRsM5uCJt5ShmNn
+rnA1hQUdkYBTGNWCT7/1K2VuaGGJs66xL1LKETjJ4RwVRoj0CctN9x64DK+88sXvuTuhVsau2jxKSkYsygefUv/IdAqLqTBF0Lh
7RxbS8Pvsvi8lirWOOAJle2jQX3NHPCaswByk5whp64RfXJKhlvPeoiniALxH98LYRX5MHzOHFCSTEZxbzTgaJ8HbNUtRN7841zM
w/cCsY5qlNn9lRszVB9fZR4V3LGXgs6P91C69w/EsxoN5xbacukTE1CQ5VROVOs6WvW6HH3/JhBkNTEkKo3Qt9OKdedlpdG9Vd2w
e/EmnJazjS6ZUYGstTPYtvFzcd5gIf09KYbcDRSQq5/TiXtkHGQdbySNfivxuQvpQNQvI8dGT+gculcdTZiKs91+ktSSo9SjTQtf
+a0CFqqXUTbxpf4L30PojDEgJSRLV3+8QO9bLETTP1Uhu65L3BeHT9h7QwLy/sDQKNOnxMpK11JcOxssjB6RpPit9EhNCrSIcTTK
bwJpPqLIKVo5gqBrOi2MqYGEJR2w4dQbNLHGG/Li8mGduyONlDjFZLUvYodX0lgtvYoeO36Rik+8DAeDFUDTTIk96E2BwQN5qPSU
LWt4mAlblX2ZxfV5dMfdsXSE4mj8xtaWKOfUgZh9BP2cqQz69StQ2+uLgqD+BXhAbRxTqhaiy205dqa0kkr7zId+BYYWOtZSO0VK
bVvHoFdWXwTv3as5VbsV7FkUg6335JHGk8uE98YL2zu30SlqZ1B9diXOM9NALq9uotzh58hlqoKv9b+DdeIT6vo+jecGjOKxVmcJ
apaaDR52PmxMuQClZw49W/aeQisKZ0Ki1jXyZFYmPSY7hZNZtQsa25tRswkCehSTjTofuIt354LW4rOWT2bmIMXyUzSwOBNtFDuB
ymzvc0WqtihhYDRct28nvucbEGciB/cOtXPHfvVxh6LfkZZ399E2tVvooM9O1NChSN4WcUhDLBS+RXxGAwItCNybz23s9oD9r4LZ
KX8PNkf5GLsYd4xlpcuB3i8b9m7wO9lqaAv66oWoOD+Cjp5sAia2fuys4UY25eRVOF96FjcIz+A+7TtEJB0nshStVlS4oIMzObVH
8G3pMhAabUa/3ZrIoFUM35EIoexBDJqn+45+ulePdMcnw85PC5nYulzU2KYA5+0w+/a7AvX+loGbcZj2VaqR8MPNRDjimfkM2xoW
NTsSxCROo4KlnuwndsJSgyfh/rY+2icfTI75e0DGuzKc33+bs2n9Tql9N91VrUmMXsyvzZp4XOD9KBe8x0sw8SV+VNZ4Htnd1sTu
pAmB9ModsHTpDfpN7TmYRWajJz0isNTvIboSNIk0HNhHDC9q0UM7VrH9I+NQwTUd8tnvKFCxpfBew45W6cqB+HBhSI19QiLnx6Hc
4oNUb1oV13DemntqNpGeupmHdv7Qoe5i+tR91AryIlyTqjlI4TKxQLZugYDsXP8IDEQnkjueNzBSG0Gry4zhXPUvUl9wCkwHhJBb
1TR4fDsPjvhetfxuIY++6Eay41q5rL4/HT6Y5tFHJ5yheUEANM+ZAR9ap4Kx7F10OdOTHmPxdILMF9TR+pDuqdiJuqt66Pw+a/Yj
OYEOJmOIkeyEK3nnYLW/BRuW8QNt+aWL5TJs4XlVHsyZ24X8zBOR4Ik48ypVrFvuWAR6xb3wpuoEKLWPxJ9CEF62ZSr012axZH9K
W3+fpegIwleG34PxSAHitfaSpfeW0mEzS1E6s6CtziZ07JwwFra0FvRuuUEoTgCHHIR10t+g59LrkGbLPXL6rRPY2r9EbQ6aaMFv
Ye52jB7Y68yzFAp3YtGJQfi6wA/CHDypFuWRBokzSHJmtEAkeVDwdKwyPhScAJYaO4nPCwtobLOHdzKnQUbPhbl9Ymj+CSe65eND
crX6FOJXK+HfWh+J/EUX2L+mHYsP8kn6jV74an+GFkcWwAvRQPi0Zhx+UzSC24EN4EZ2nuWbb+IsPG0fGInVkV34OnjlzCWTRYO4
iP5M9Fj9AfRm9CAxrgHZRythVf+zlJ1TYt80woBMS4O0rrl4WGgXGjdUH2r1ObC/k0Yixx5AE1M3gMKDVNjZcYk4eFlB7pUbqOZX
LVwqtsF+gxdZbPd+MDmzHh8Jf0uv1YpSqehx7FpYGstOuwCcfCI1vZmNbE0E1LntHPa6ORGUGgbQ+552uC70E7ls64VUyCSn7hmx
Kks5RoW16BpxCejJV4Fn2r8FJoJE2rz1HTR+HJpb/y6na2auZ4uOSIDrJXF8IcsNemw/Cd6lTaLLpaaT0LTT1DbSDm0MF4PAhpN0
w/iD6MrmUhpQ8BymHqzmPh64zj0aPIvydmM2/mkrTTvcAj0esuxS5Dd68+xHahJ+jWacSsfFQaL48iNN3GHSJDi7SpapuC6ly5RO
QEd0IfrsNwa7vqykY3p66NOpO2CMbgsWWQk0ru0F2rPPkJ1qfkn2qmyCX9XHaf5FDbqzywKmOkxgJXfT4WSCNPbb/x7slg9we/fa
Y/XOrdDzpBcEybdRfGgEbNQ7g1Se72VbbF+iu8vekrlNNuSzZj+JiRKj+nt00dGTN9iEXgW2ae5W1lo+j6wUKsCJJqdgZ18vOKAs
ckWmi82WWIHbrxpD5eBjWHxIC4QXnkern6pidccltG3KL+TZ+kWQc2Il+MkXY8nkEiQ7uYU0O1TAiMFa5D3nrPnpbe1wOGQsnI4p
BT5fG1lYtYBUvzp9MkqA6g8dhAfFCvSWtyj+MdIHnFKU8Zzng6i2djM4dN4F5XVvBEf1TenSK0asfIM+/dylwq46LcDpKvMFb14f
ws9dhNErOpl+i/xEs9+0WC4+NgqruuXQXb1JbJdjLDsx8il3eyJm81Vr2ImvxhC3Yjg6e1YR3iZJo2+XQ8mNAF2qJDkM54xoQ+Xi
sjCrdD1peX6chiWkwhmn/ehU2RdUpjEKjuevIdLbJjMf0S60+4Y77KjwJBUicuAodRjakzvR8RuNdGRnH/1ID6GgJ8HU+10CEb9U
BMF9zrj/QB33yTMGatcng62DHLkgp4lnTnspUHj/hZZcGcUml60ipUk94FxWRa/bBePupZ9wcxhB76aVwYQaeyTtlwNyb9VgqX0K
RGc1wsQrhehk3wTIvDOTnZx0m84/p055FT/AqTfHosUvlzQ9mk16qtuhZHIuSHmLo8Epp0jxzgo6yu8BbRG3AwjrIZL3B+ja5nrO
QyQBTG02QtjUUvQ6gNLQW7LINleSCp91ZteeXidfFljQy9YfBTfn74fVLnbseace+pkegr29HsGGonFYSPojDZn6FPQMy2kNtx3M
9E4iA1zBMipKaW/INpbW7gJSX+bDGMtY0LjXRzOP7kJSPy3wsAmjIHCXKbRO0cM05wSriX9DOiyEwWfQAtStfeDjnPuc7fIwOmyP
MptuVYVG7uboRkNtbFV5D29eYQXvI44QKiuOV89zJQ2H17H58qvR/qPz2esFVvC0ZwxtvK0I/ImS4NY/C75a32exIhpQNXc7EwqI
oTvsAfnyQtn5cdbs9xwdvOOZPRu78K3g4QQV0NVg2JebAF/jypHPugbUFWYFhSMWobmz59CSd/V0PjbHRTZ30MXTCJ1QWQoa+ong
tegmurvGDZKHv0ARtjwkeuIEFIo10S34DVp13R+N2C0KCrwYOvmxMCSJb+dcZu1DfTeWQETDboH8B0XIubMSHQ7Vg0k2N7j0FXok
P2sNeDTY02zrl1CaWcAiXOtoxP5YtqXDCaQ2vBc4xAVT3rVz7B29B93jdWhUQjQxPWHHxJtXQfoGSUu5Kjva8LUEnu1dBzN69ahT
9nk8OaiACmeWgWjwEfb7xho6tlpWINK1gQU8OUeKys0hJWMJSOQawolRHkTFcicdeD/07DrpLtOsuwmFQ3X6yldBzf3kUfhnyko4
JSRDfcpuWW4bV4yKDV/QZ7XmJFziCdrUlkLNPEfgVds+U8UZj1F2Eg+GbQtk034Ggu7hJzSm8SI+9HAUu3D5Jt678SXdeCcFCZ/Z
CV/yXeHbuwOQW6FHtWQNycktfhB+k4c3zZBiUTHmrNgjnMvcJIJXSRnQzlYPeLFAHg6v0KCd1TbIVksIHraE1B4VlcCP++7TdZul
4M6MANg8bB+5c3w4bL5lIhhYWUJXym+A8ubPcNnUH1pOaOO0H2X0YcI08Dyajya0Ektpg0paZJnPRE6Fg/fG0VS1dD7G9gWQttuP
WA/16eSPA/hjVSi026XS8zs0cNr4fejQYAtaUX3C0tvgJpxcuwNKCufRLw+1cc2KGrZm+wUoWHwIRmzMou7XFsEB6c9Izz4HcYMz
wczBHFyeHUYNNTI4U1QVRic540YZdTA+OwZqKpWxqrw/N649kHPJV8RrT49EnZXvoaBLAqeCBLU94iSImrUTXWjvhG+LRVnS5NPw
5VsvWmMfi87US2OLxw3kW2afoMhmH9t89A3dLFzB+rzV2fqIhVTjWxFyP7aAJXgrQV2dGDztHEHzMYPkWevQ1twHdJbvbtzppoIt
qnyJxk1HJHXeDraKFKPe8jNQkPCLk3q9CnJLK9DXzVlU6eUCfCazFdZuPgKeduWWzfEZIDC2QguuV8Fsu81IDT1EaMEyZthiyl5/
u0j1qw7Qut0vwD74OYT6jmS+KvvAq6Ea8poMYGnUWLye+wxFM6ewpNUbmd2JAVD/PAWPOWsIxfVJZKHML5L0eFidkfJlOqNxOG23
vgpq/dFE9qs+Nv4lDQkCI/wEFNDLc9EwYKqNV07ZxbYuaoasbiJoGi8DvPB00L9vwMosn8L+cT5sXIc4Hb5dEb2kudQrLAoHSFwZ
eg7RZ8uy54NGXxt9KPMM/OZlondZWmymszHEj9wJv7aFcUcvycGvO9cFl6/w8PaxGlRBWh6qF96jNeGKJDH5Dpy1Hgd6UqPw1cAP
UP4kExxUxoOu8Aj8XLMCen1VYXZXvmCv4jFocVAATxxPrp0Bmm08gnqBJOrOqCBGCao42qyBovJcNCVbVWDnFA18xXu1DoGLgNYv
pDJTfnGRr0aS+V+LBAv3HqDDMuJJ4Y3zqCL1Dop2+oGC3ypS8fmvBK/Ny+jStvMEaZ0lxxa/Q1e6DtHHRQP0zX0PZtSQBob5d9nm
9Ei8pqkWhvMZTpdPh+kL3tDQMiHYpXUKbTupCbf9FGHRtGJ0KXUXTBh+GhLOXwD/xZdBVkwZPiUOwyJe9ykN2QLbenzImuZZ+Jzd
Ws7K4CCEfyRQCO+RY9gzbt/1fM5kYCfxD3wNwUahsO2IPG58vYbpuBcC0VLHeu/7oOW6CRyfrgZR1Qth5Gltdmzn4aF6vQVcP9nD
7tUJ0NJyje12G0PwgTnYPUqK5Y4+CVFiI7DI+ltselsV66zZyfCvOMws57HHDZPZwgVeLEDwG24GlNM5nmq4UOouunTajQ1vn0k1
Thnh5U9WwLmODHrlRADOqReF3p+q2PNhwdA8+wR8+F4AvfUppDZblOv/7I9S7mRA08d04vnlBBplLyKYmy9Ef27yZqoDPsQUZXHn
ddqoTfAXOCYkzkb43RK8OCWOuu3XcufXT6ceWRmCXUsqyMH1iawlToguKrovkPyeR26GRxCJlDq6s1YW27WI1CkuXFazqA3RZOk8
+iSXT8ZY2SDxiwFIeYk+rNHciyQct1BJvQtEufM87iysQHbeTcixqwRuHG5DmqLxSLA7iqu8ug5W995GZdbO6MHN03SrkD6ObtqO
enuqBc57HhBX3hc0ZtsLlFa0E7l4KIP5pyuWc64n0yveo2jkzelc3OdT6KqsCZJoquHM3I7jBxvNcBbzxGnJKvCrJpstWmhAQ47m
w+q6zXihGscuN47G3wwcgRdYQ+cuUcJnnUThsq0UvWiSQ32muXPtCxug56okXm47mX4wvE0Py1iArVEBtpxjxpRSRpMUBX96Xd+Z
jfgtzvYsDUY2taNg2Z44ki99isTMzoFUVXH86mQcG973Au3b5kdnXJaAtb5ylJ4+CX0y6jj/3Di8IUyOZZmUQL/kKBTYoMapa3qz
oNOqPL/ASKhMvQRNFdGw0vQm+qVlD/H7bJCGXSQ5slyNpfQ/sdwDgdzU+6J1L/RN4cfIGIGOfSW5ILYMHl5VRU0fPggy1evp3Vt1
MKpdlNhsr4K8XhemkR0BF0pk4bbKEUEJfz18CH6HjM7EC960vUZpVr2ClTWSMJiXQaLlD4C0xkXUs/ArYlZKsMPNC7S2YhCdL4LO
Hs6iYbu70PPR59HoXm/y7aALUbkdyRzaxtUZu9WhreQmpfE59KL0dJgxOpGa9+ykj3zicf3KrfiYdC90KvKJtLIF9hcvgLHXW0Dn
dhCa/CiUe3QjD1YXHEWzr+5Ft2SVIemLGo2eWw4uW/LAdlsCZJRFw2mhGuQ/7yC4VC5iUzeYwnDJVto3ooqqJKjQWWd2wblVK7CN
qge6v+EeCpM+gCa9CYbyECHmm5KKVO+7sBFt4mxO49haHbEieFOXKbByMmfbui7CbeVUmGZlCCOPirKD4q/gzawK9EU4C6vOlK/7
bZ0B9cNNcYl1J/2uuYhlaF6HbWv3wj2DGniSZUitugXobc0hmOiWgDTTWuCG51Wa6O3A0JqRUNinBbk6wrRenqJL1YGQtasYfNfH
ceNchmFPxSWWCvOFiHBLNX0cqgUlb+JBJWYKYm22qEmNAgTJQqqQAj7S2QMrJI5RK6NGanbeiXpsWgLfr+xh6X3KTDP9IPGQuYtG
NnLgXtSArI96w7owcepRm4+fLZzPjv0soYrFuuzHx9X4mm4ipNW8RGNqUqjjSwLDkt4hnRVyOFIznXXfqcUVZDdntTWSLi/U4Q5U
PLRstkkkh9ldalnSAZo/JkOS0Fm6ejOBI4ZSMOXsXXQq7zu19hZj5Ypzic7488SUbQBPi2wwmT6cSuS+JLUHRrJhDstoq+8B8Lkh
B1axlWiH3WMS43iCuuvWwZoVBUCKO6Fxuw11C5+GNkqOwXfFAFLmeeCZcvdh340ceN+4FYLyjkHlEHfudzdQ0JqE8p4Lg+XWFuSZ
fYXYnRbGgclJcGi5At7/cxNZ0qECDvsyocWxCJ3JtAal5gfoqq0x4U0nXLdWLhWuJGTurn2QrXJLsEEyCl7O9ke9A29h8pfHMO1H
IvWZcI4ojYuh7OgA7LITIZd/OIHr99v0QVA809RLoNN7xAUL4xRwdlM8u6uSyU1j3fAztZipdunhIPsYTJwSoT4xjBWkxtB1KTdB
9Xxm7cBlQ2Z6KhamukZB90hzhnxW0NObzrKpFq9gIGwVvbUuBR0LT0XyazbAyPcOLHxeHiyJMIfNT8osml4PQ0/D0+hFg3ZSuFiH
fS/phRUJrYLd+k+Q+JbvSOnhLijNE9DfNQV0DBnFfAb06LKAIGiNfEXN0qyxsagI5RaqMVnFkehQnS7LvapNBbPE4Mj48fTrdTts
dM4Zv/olhD2aY2CNzwaY9PgFbv19BuVEWkD9R19aKzodks+ZsR/H3nA+g76s72QxOtMwDjasdySTn9nj+NpMNOWQgqDM9wh4j5iM
VFQZbHwrwXX16rL7u1ahys8xKDYmGLTIbyo3MxbEi8eAOhynn68mwlPRDHgfVUNagkRRovFs+Ni/BN25+0gQWp1PdHx8mZhMGt4/
Rh5/jMmmeYNl4DNpF0jzrrKz90qY4+0UHJkYTKuiS+gluh/Vy5WxnQfNoWyDB8jqXqaLG6ig1PYQ2kX8mI61KJ3XXk/fdJmTlfqv
6AtXeabYdx9d28qD9pXn6BKjYEhcMQYpd8mxiMfn2PDvDdD03BFg7CjSvkER5+wKptpb4iAA7GD5wFHodhoLl6/UUH2lKXChzAo3
Sq6FlcbpMPuXAZNPnEiO2W3EgZvsQHZFBJy+pskOxakya+e1RMsuE/svVmJdLtvp5kBMZ0uXwvYNcuzC2lVwQc0ZLf9ykdivKIcx
y5dR8++z6P6UGWjZ5bN43v4c6PC7T9tfpxPVF4lEfbcJKtq0HX5rlYHO5VSy5/pGGnjlJxJ2HEBaqvNBaPlIYjLlFvXWLSOXSktI
c9wZqn5tO4RvWUb8XE5THe88CDm1nSlEGYKnZCodM14F12b0owLymGmbRoL1ya8Cy+sCsnvUJhiQ6aQbplvDsttTeAuS1+IDV+Pp
geZTcPJzJl4gw4HDYl3OuPYp8q3zgO9uBtC4uIGgHFvm9MiC8e5lo3hzVYz8vbH8ugnQ/r0CSUl5oev9M8D250iYtjS5ZrfwXlY2
eQ7nntaBdDemQOzAD6p5Pxo69E+h5zHEsvWYPNw2kAavMdHI8o0Ri93Tgr5+e442zcqvvfWwgHO7d1ag1mnNFu9YjCa35TEH4UrY
/kuCuFaGkqvJ2+DQnVWszZCxH4oisKg3md1sDYeolpfIdfZh6tJWSa+JH4C6mOHgs6cESh81CF4VGWDVJ/fAcd1RqiZ/CcJrSiFE
sl3wuGULhFfFgvxlhPau2k1E4QvEZmbQ3x4joFCpn4x+GkejX50R/DqgIMitVKd+Qmbo2jFVJPheRlpPnILWpgooF50L/XWAT0fM
o4evxDJT1ysk1GUbvFEh8KHHjr1TnUTpcUeY2hNBu11MyBLbLCxyKAPqBEuZYfpG9mTWEbLEWY3Wlo7Asq8R6C7bQDqO7GLXte6D
sYkKvtfYDgvEnkHCSEt8eRoFqzkVdKJUtoCuwsjVdSJI7YylArNRqMF5NhvxZRZV1x7OzA4/QCW/bkPC6ESWoZ+DQrpaUZzZTvp9
SxMac3YBq6xwQqeqdUFcJZcesQmEwpirWPfSCSw/+ATdO3yHXErxxy4ncgR9p5eQZqOjeI96EFtpk0YuC9IEFnZKeO28N3DGRx1y
L3ylbesqYNKiDVBfEYz2qhtD1eBy+Fy7Enbm3zYfr3+Fuz1fEf92Gge/5SThgOQL7sPy/fDhRzEK6tnOFR9/C0bnOVr07DHKMjxb
q2OP2IwVilCwKIKoFp2BoHueYHHkMEz69hKFS6yB5l2l4O9ci46sy63dbxlKlvZPQ+035UCjtBVFmQvRsTJl3LAvHWi5fROZEqXP
pj64xOscrksj1+Zxs2TTQP/LUTrPXR/81X1J4qd0VPp+Oby999JiS0chvT1jC29mZAqohQ7n1lgnUw05QrSqTKFvnzuSfLgJeXxY
DY32TSj5azxpNFfhzR52AF3Si0ZHwnehQZsfqDVYAZbFNXCTF0xDRSsnQ87sk9zG4Vacd7sxGSatilaO2Iustv/kVkVe4OJdjrHj
tVPY6HcPqeFjUdYjdR6OKkkxYx5iEa/OodLkJhpTXYdbYl+h1JCf1CQ/nGz5do/ArV3kawOH1utVwc+G/eS9UiNdWlZEFMVqWFjL
F9L23ILhKdNgUmE8lMweA+tpheAHO4NOyQwTNK5JYzkCPmXz+CQzVx8W8C/RfWXD8MdWRXiw1QadnF1ONq+oQBJWZSQj0Z8qzPgI
E0sYV+EiXLP43gTS8nkPZOYXIyGz88zS7RJ3tU6JpZS5kHOwA+iKLfDeIwl2nJuKfx6cjwrja+jqlg/wbNcknORtAAYpAGfsZoOP
GQ/u6h8FmSm+XPkSZbb/dRo7NBAPs9ZE00nnIvFVoeFQ/OsUt8RYl146lQjaGa5oAa8L+Y9Vp88F7fBuWy7nrhWLqJ4Cm5qwGx7r
BSHfvUpgHxCDpiZc4eRHutNVj7cQryvTcWeTNmxbF2exfOA6eqguhfcqP6UivATwNSbkt+tGPCziNgQUn4Rb06bjt/fG4hk/rXA0
jWCWfl9hvZsZa30rBpJH80hFBhHs/BUMri8S4G5+FinI2gQ7r5ynk9NlLQp3rCZxe6vobd9KOF4fTtxlJVGNxC9gxh3QfT4HWX7k
sXMrs2BpizXkPX5CLoum4eqgdtT1chxxuCMFXqKWsGR6HLddIMG8I0vpfElC3054Yak7d+h+umwRkbbLYgeUD2ClseFgmmzCjnwy
h5+9OfCpWZMtUPDGSypns/Z1qnhbRw8tGyhl7z/Pw5/FtkPVW0NYkDoKLXrYAL9aulCXwV7y0G8Je9M6jl2W2MPNuufHmfoTVndQ
Bc9awYHXMW1QfmEGp1t+osQLuvSRkBJ8engYHd+bSsNLH1Kdba+pXYgVqG+vgdVWNbXYZxnRwzGoV3IP6qocjl36pkKzcg44ZcYi
ztgOzGc+ofc11XHwYz4EODnj5va7ZNJJbejIXo4jLDNI9fNtWLLhArsVwa+5PbKMvJAqpWlF4wHihYF3sRRNfqwERlvPIvcXojRK
L4UGawJXtAKjxl9qYHd8B41sOoI88yfA4IyjyKgA4P7MEVBSES+YV7mH9q0bQApNtcgrdJB7obqUhi4XBsENWZjapEke17xCGVIH
UDv1r30Yc49Oln9WM8n5MuoeGIFuvE1Af948/2yxyVDIQEjoz/LfvXku9p9vnhvZWgsJ+T6uIxNm1XMzxu3Cm3dPxcXlqziXBb+4
50M3q3uknubkdMAG2MGGjxyNf002x0pW90lVB6ExsyZCfUcPbLicCrPsQ0nOYS9u2LeRdfvKvoKVoQO+rbGEuf4cBG5YIGmwKUZ7
Dsfiuh0J7GRMJByrXQArl2SREIcaom+QjvWdVtPUp6kQ9OoaNtxSju8bX6arKp/BMx8j+lU2jrlv1WeaOhvx3uBINHjZAiObMTjm
7DAsPOOnQFvSCvyvCdXZuZjgCfGaTEg6BOtJP6bmHaa4Y5QsbewuI1HElfXkXUf3tLxw6YetuHtHICTniNHSO6X0j39etsYqQv/6
/8L/B/+M//jH/+XJe2q/kmdTjXn8cxN5V17X8B6uGsvLmQe8ErUw3us4K15yhQPPZoYTL2W8Ee/ezDKewpI03s+mVF6BfTRvWZ4y
r0vnAG+NdjHv6TPGU+vYzZusmcLrNR09c9OBIt7c/fm8h52RPP1na3n71Fbzqv35PBNRG17uxAjeu/f5vHfPEnmrIn15R1Yd57U0
r+HFLcrihWFn3tpx83glKm68lw0pvM/8MF71PSPegRDpmV0Dbrx6idO8/B3RPLWtB3gXdZN5I2KCePdcQnlvsA1PZedi3pSNmbxH
ZVd5LjrxPJ7sJN6jIW2/8rN5yz/F8+YIEnkhYwS8CLX9PP0J7bw//k0L/bL2f+CfyR//rhy8S4vGpLA34bnsENnPxPUyIMUtiol2
jWebVPzpaMvt9L3LMOa0J4GJtGWxDK8kOr0pCrcPCtGlGf21OfuyGe/cErauJ4TmHDOjpdM5Zs8tZu5u9nRvjgET2Dqz3vejWJjl
e0ZqFrPYMa20/sYa9rSilq3uMaFCEm5U+/xb1rh1KVWIlGINB01I1Bdz2rtOghmfMmdGNuJM6PF9FmQTy+qT93AbxVJZ1fqV7GiP
GKsffoGNmLWSivY+o52+quzHokNsrPnQnOtuMD0zcQqtNztKC+VXM5FN0WTKpCb4ucGEBTqm09WaBswsP475SGwT/PHPxTYp83/g
n+kf/5R+uIFc0S4oeiZE+xbwBGOz7HFryxLiPKqPyk+KxoNnEZ7xNZmOE+0mJZNG4bh4fXxmzGbof/EbPAeG40snp1iejq1HnXN3
cRsH4tElTRd8qLwInpf8hGeX1XDWwhn44fVoxsu3pzuCGgUDvBlEanwHrZLs4YrlMqnvQUr1og9Rg7mTcKiCIbqzhVFu+g/qKh5M
og+erV2qvlfwpUgR/aoh6NKeuTjNI4l6n/fE51ybQKzrFBU9XI1dbTaDq3kkwxcuIe+Zr+H+YhdidYFjEtcmY5MJwfSxgwV+5pIK
+4/PZcJJw5h35gAiN5fAT+XjYFsfi7dnLILzMwa4IFNTan/rJPf6UiJolY7FbeWhbEfKPXB6weh4XjxtlIuG8xZCdZu/i7HSvgf0
nXcU2PWG0KfiJ5n/NTEs/UMJDm0pp3nq/ZBoV80uanRSmdWbmdksIaxy8pPA3WgprfdeBbO3PyWPRh0Fz6xM2n9tLg7ix0LBDhl2
c84UGDyXDCGjvkLB/l90Vs1RwRazfrpplRrK8aKkqM2d5UvI06ojsfRp4j6utUaEud9+JEjqcIbFyANfDFODqRb1nOI3OcYMJ2GV
GSaQe2ckroh2ZOVRFlQzJRDbSM/Ga+oTgfe4FU7lFcII12E4yTGDyvk+R074JmR9SoGu2Vo0VdkKThR2QJTyCdQzvZnqvnKlx2Wf
gojZAZZ8VpYukbZH1Q6ldPGzOrplbzXqGxwk4Va/WenWGOjO34Qf7SmvO2P9nd7uyccZ26NBMU8d5sq8FVS/mEfviSrje5+SQPxl
PvsdtgOUHDvopG8jqE7jKxRCa5C0TBXluY8H9GE65HZUod1XTOmwHVNga6gtC1k0HZbHjmY2zZ9IRtN8plryhFujfRf2HbljuSnd
gg5ItVnOX5sN6yurwO6eG9OrrQGPbwXIZdZ6OC8yGR/7vR0mdfCAuxyA2LkztHTOATRp1jPq9j2JykkaUZ93U9nn4h/wftoBiHux
kBWIAQCNo56CKeBxoQcWyQqzp7dewvSvjTSupQAWxhjSiQHKMMHLBOs8d4PTY4Cu9BogRyeuratvdmdxkcPRl/7t+JOnBWcWiXD3
VMDzS/2gaWYfWFd64ki8Fr8p1CMzlp6il9xE4HqTGlb++c7y5doCsDMRpQYipXBv716a1ecN6rFGQF5K46jABjJtbCTodYVAc5Q/
vdb9Gunvd8O+r+ZDTJMq2yasy57e20SObhzH/K/vEzS/sWTT716lUo/N8LupuaD2pYGz6t5NX5eOYi4n89Fpg+8k1u8s2dmWQo0r
tgvO7Z3E3oXKg/7rz+T93s+IuFxD6nlxZJzYEzRCagTIPPpOXPy/EdsMB1pxGVGhdAyxJjF1U9K+wMkzj1F0ZgFXOZhPo36Jskfm
cyD7hB4att2T6rywg+ymBFIpdhrMV6SDYt0RZFhbRx/dvAKHciLIycljsPJ0ETrmDQeBfjX0AKdNlZY8hUbNJoiudifCsRz0BrTD
wmWIdT5XJw4LhkGw1G5I93gJwzb31MZXT+YOOw499yQaCZYYVsLV9/5c+/zJ8NXnN+XNv4h7DdLA9WQJKUnahvzVdcGC71FnrrKL
7Jwph3rKgrDPxF0ELRxEuaetYMuGV1Qodi4aNWIBLvFUYo/XXYTYjaL0qPcDlNmRw94mGbMvyxAM65/LAr3VQObTJzK/1YwdaLxK
F//0oyYJJnTzjWbUZpSFWlZchwu3HFGb8wlk8ZKivSLCtPtzCBcwW5N4zJqFB8c6wwSRYbVLYm7SkqJP9NiJSbhmpTSc8GuH9zYi
eOvOMlpwaTW3WZjPCg3W4FreJlb4ShNH7Q7Bu4Y307n5CUxcOR5GHpFm7+r0oerLAuqeLoCp3yLpg8LnFE2og4KcP+/+FMCjA5WC
pPvltM87lan5vwOenWutc/0gWbn2Laj/jKSXjU7SrKE54pTfJUAtRoLhwzMgveQRtTg7lkavlGLZrWtZvsls3HA4iy3zzkQl4rPA
oCcPjYTVWI03Ae99kEYCzJ/Ahs97iYGInMDprgTTQYAXzEqFxHV6eMU+BHsO3iHT6tuwlZQifilMkXK9PdzV0KDKTRTULeWZhbEX
HbySws5aNkAelmL+KpNwka8CjVo9Cv/OV8FJW3eA+nIxevTLG7JOtoAE3dyNj9vdpZdGCjGZTx2g+FCR3hn9EOubaGP+9mlk9700
S99aITrn7QE68/08PN8kDa+AGXTmemm8fNoiOqGVcqpNeWzDwePUf7tp3QhOHgc9+kz2Tw9Av5Vfw/hXX0DVtJLunn2IJlYeh/o8
KSp53pGJ9QSBLc3mrj5WY5GskEvry4eDuzzZLc6PBf1aTSXVP4GcpyzOvjQbq0c8gd4PCtCkcg9dsD0Hb/IiYdV8bThseYFGWKRy
JjducW/PjMD5IlOIyeODzDroB/Br5uKzrZimh/rg8JubQFJ+Zl2smzsKdcymv2t12Rrf0/T0rkDc322OR2tmkFXTBWDyYDst/5xE
LzzYBUcrzFHk4ViST+s56dOXkfT18SjvmiRLlWtj5yfq1j0X7GVBnAKec/AozDvqjcbnvUDqj4DsHd0gONVgjo0mT8SWHuNg9Ipe
EvtsHbWqNmCVmpn0wYYybv3dQGwa30TnusvDo9XJzOuzCpY1t8UxKz+RV9bi+ParA9TzsyIzH0zAUkV8an5xBSt6NB9WP1VgF6Q2
AScWx27Ge+C3fnl0980e8gvOUcmWDTjwri2BC9+5X9BLTq3h2NjWd5zqwVxY/GMLlKsII5+224BMr+DNIoVMRzQMd1QbUazvh6Xu
/qDSzmFoM3tNvI/w0djanez2hiJ6/409Dn51nCqkVtHhclm0bGwOqzk8Bp97KsT2KcUxR7NhbNyyNRSPeUC9+UlIUV+DK+w/DpJV
9qR7fCusOdeNtu3bA7fahuHXsgvYqoMxIC+0j1Nd/RPcd+9gB52HM23dfbgXovAsu3BcqRnKim4kgW68Kkvp02JPXYbmOs93s4iH
U1ijegI2eNRBR3r8QLScwbzXG+g75SLmES6BH5siNlmmExYvW0i3jzkIO05L4AmtaSzv/Ticbfua/Jy/Ct/JdaanIraxynRxtvyp
kMB/cCle2aYHMf0jIDe7AXJeR5M7AebkpvBomhW8lSW/3Q22x8Wxa/9T0PCcxoRUo1HsHgOo+zia1Xy9CAOynXTzixi6SlFH8KxQ
h2nkPKI22XPw2YVpUNx2k0b7tKPxY3zxpJ95lg97xPGU0kv0V7onLNw6DEed76bMWAiv+imLP1UL4+7ZJbTrwU4m+ekMWXpXG+eZ
puDlRyeQbZPaYKVbO3GbO459mLwbKx8zwOX225mi8W+SF70TYvf8ptpTH1Kh8e/o90fJsN+kFBa4JVGp+2+ol1okfPe6Kxg8MpbF
Gyuw2FsJ9FynEJq5JBV3VG0Aaw9FeJNxmL7okWP7PPvooxN30Z3suyTn9GKIaNXHfgXu8OqIEf7WmcxWyilh87ztoHoikR7euhy/
v+RM16wox5pidqygNIeoVXfCrU8zIOuEJfWf4wxl+zTQ5fN21KDRCwy8zoDr/yrtPYCiWpt9bzJKkiwgQTGQFBQEgVndg4IKYhYD
IgYUFAwgYgSVHBRQgihBETAQVBAEhVn9AGJGBURMqKAgKkbEHD/3+5577t6er+6tc+5UrVo1U/N7Vj89a/27u2qq++Vm/vsvT8gc
X0avlhSAmPp57vRKNVayzplTGCXOXBduwFE/TIXX+nuzkO0FtKXHjH2JEqMDD+Yy2eWmaDLRCNMnH4HuGS/5BbJf+Vmzj2DS71je
Dy+Tdbde9aTjj6F5RQjZLLhD1x2208VnETx3tY5fYh6H2675g5GNNf3Im4ujc7Ppe4wv7i9qhMG9++HYdkkYM6IcHm91Ea399o5P
wr3wcYQ+G3/9BBd6eD/zdH8PZccmwYSdVqxwgQounjgGwOwrVd1MwBkXN5DpjUXweY0choeIY0FDEVe8rpO+RYTAlu/J3I/I09Ac
FE9b247SZ8eXnKb9OBY41Bh23LVgkvcPYWCtDEaFfYbNN08JuGAenJQkhBqTR3J7v7wE6TtWrJ+bLq6+2ws3p6uAAx/O5h+7QtEL
d7ET2a3cjZokVtochHH8BtA8Ekcmu9eylbFq9GHFDDLWGk/ls51wrWcp5/RZjbod/Jj1JA/UOiTJHu37QMYRg/FEgD6zilVBiVpp
as2v4bXC3tOrPEuMC4jkDjyag3A/mlUt3Emhu6xpiYIB1jbs4RfKt/DXPOooI2oRvHhxgxOXsBQYBJ4FncW5NO6ILyiFAVpDJEWd
3QR5AXNhpGYkSYrUWEjYPKa1IY0VirlB0ZJL/BQ/SzD/YAEzg5+hwksN7HXeSS1K03HPgB3osMwT38IuJlBkGOfYHzzXfMSnJgaw
IW8DhvQsx9XN12lefj+W0NbMa90PBc1lijjLfAGZRxVB7AAd3O2wE4JV3WllxjFYPt0eK/xj8MmL8/TZLx4KjA9B1bRcdvV0Pez0
8ECrd+Js+a1leHTBMnKdZEtiR0exK10TsfLXKXqm/RRapnYzp7CzVR/a5HHFxsPEb/Ki5yOMUNnvFO56n4iXHBfgmM/Ay5buBqF0
OwQ+jaLRS9SrTQqiQFtiMbeoJ51PW+ANAaJ5EK2mDulhuRCarEe6yurVhbE3mJvPCKbd4cMiQcRWRRri2kGvYHl7POQZz4GCPGXc
KzwCNhPruTf2qeCmFcBFWVVCqZ40c/w2A05tPsC13F8JX3Zcg9htElhudYpVt7ULAmz0mGj1HarptwVmzD7E27lsxiOFYXCz6jlU
Ktzm+4dmco2L17OLwfI488xuljTTkknWMJjyqJg7e/czt9IvCBbf2MtFnY2kiQvW4dgPbSTfN1JkGZdKeGolrj00jot+/ICFCtXw
bloei1jSYr+8zpUbJNwJ0pfu8S8HDmIr1MqZuOEoeNksSQ/U5GD2ijE08qoUul11oL4XC+BATgDfryNO5DFwCzcFVtDahihRvfEM
Xnm+Etm0ZoGX2z5wXJsBmlX9OP3aZsFyvVUQfv8FbaMJ0DatL90LHo5vLAyYWOlpwtxokN0ri6luliC/OIqeTFXnvjQPJbPlnjac
52JRvX8+f3L8cdi5LYmvsxpGgZeP85usftGEQ13cnZhGeDdkAfgc3QESck/5la8+id52D+cu6V7ji7Nlme3bdKqXvMa77siyN3iU
DxtqxsPJ+Zfg8kpjWrVICSZs3ADZj01QsDWO212STD8NAvnCmakwZUIyzMu5C452V2GD82CRgpsZTDZr4+b2NtDNtC5ui40hXh5h
jmnJv+vGkkO01zmfW/bhpqgo0InGv94CN8/Is8m9J2D2uLvcQqMUVlK1AnNHLGenpmWJYrLkWJ60N3/9cxrlyyO78E6ZjfllwA7t
LqdtQ5cz51ep4Jz58Xe+/pjUZHeAgfAKzC3pC3MueXJ9Z9yFg0PdaFUrkijnBgQ4W2CS4xty2KqGBWVH+OOCWrp0eC5ljwNQWO0H
J0+L4aCho1iGbxqNW3kc+jwaylxlhlOTdAgsMPNiah/l4NAOC37/wFxu3pSlsEz8Ahdz9wOFe/nA8XmaOHnEOVo/cxIu7jZAp47v
3AyVvZD3+Tib/8mG6by0YDUdNnQ5ZDRvOP8JnNv6+zl5OJ5b83UvJ/baDWu8D1Z7jj7F0mcn4O6yu2BxdzTsLJqBcYudoGBWB6yT
FLKBDw3JJz0JJA+Ww+03Y9nYsfZYq+QMhmlTWW9YPOwQiFH1EgGaSm7if45P5CNPSLNpZTPsH443YNvpI79uaQfZkBn7pGcKd/Ky
eDmbNm7dtuE4XHoHG+oaT15H90G3/GN+YlkGrJYK5hv3c+i60xrl0ybZz/cZhg6nhdz3hWNZjsNtlLOQZx5XjVlEPzsYOk3IHStR
Zcp6u2F/P13s61NFixcncM3n6tgtz8N4zlGVmZ9O5/reUKaG1ZJ47lgDbjtnWz1p5y6YnLqVCheZsLcXxrKJg1txqu5qNn21IuvI
EtAhTWu4Na8vPHHbhrzrWhRpDKjeUHqcTWlTqG4bcpHUfAeyFp08diO8gJUEleGQzCJm/uAyZblWY62FM0seoFltVG3J5v8qg/y5
43GX/V2cfegoG/5aEjSnDmATpfRAqtKaLXKvZf4FodWSg57g1aMXyNrtIEhaqwp3PXgGDytv0q0+FtWzE5NQV/Y9F3RjDowJnYIr
BosJzDfdB5kfXwW2Cw/SjJvGcFcjn9YaekOVvzzGJReQm0ERDcYYvuOMiHMeeQoeqBow0ezvIPq2WDTviQKqKVRgv8CxmNaVjkfu
PYWtn5fQwXpL9nS0P8VlOqGp7nuIWpsh+GH/gE7OPkPaPcdYxtIzdC+5kL+nfR4tt1jwy7siaej5A1AiHEcrBstg5qypsOddKVda
Xk3pjUsp6dE16vNjCd1+VA2G7eI41TEWThcH4z0dDzr2ahTYXe0kb4NUiqhfhF/eqLMj2gfYkEdPuQf9w0mn25xjB5VxxBwJkZ2D
LG5cE8Z2KWeQ2hZbbuzUdLo+Q55FRtRxE2TPQ9DrLP5p8XYcXhbK8lzq4IvKKe5jjyLfZNxAZwuaqPedJXuxTh7fvuyC4LabNMo7
TNDc+Zqp6Qbi0m3ibJhpKKqfV8H2Y/UQ12XKAp+1gPN+fdHeGGXc0jWEPbIbQGKZgZSwPpnXPHwIJ851xe8fs3HXnDSQkLLCqvNB
oLjwDldXNpg9GKZFFUtd4cIBLdp9rQEaPmrg/OR8vlSzir57b4SLkMEJhsrSfIdEEOacgHFvLgk6bw1Cvz4agvTMFyLbLA+WczYE
Tr3bxsz3N8Gzh2fokIQPsD5fofSZLgZP+8XPjh+NNjlt9GiiQvUGtR745RVFd9pngKrdN5A3XQpeybtxyxgVXq26ADZ9aYK+se4U
ECIABR1esFzDlo2dKsNay2Og/zMJWF05Dt6va2BWt5So94oLqgb648xHttj00ZQZmlWD7LQimLW6SXBkmAR5BqVUVhxVhTjazq1+
7sBuf5lGAp/rVFTdA81myyhkSANE/3jHyaxMg41lF9he4UlYZyxL3xP12OZ8N5AsHsvCZ5zmSMcZ5+UCfTvpiJidBjVvp7CwbR/I
1mUOZK0VMe+Bivhoqj5YvJHAQw+lWZcCI8GJJHaq+z7EXnVAXmsGJ7PbgLmHeEJzlC5c112N423FoTm0GhaNUoHLxYeoVCoKvxUa
4p7+2uxRVTDOD7pKd4ONcJ3kYVoemgsO75JY2etKeDJnFNcbFcJYnBLPTz6MAX1jBR2jeVzQU0aLZ1riAqcgkeGjcWxU9QU49mYw
VF8woYGf30HevgDedL0vbhwowLD60aCu3UHJ7tHwxSgUbs0VY8nz3yGfJoJ+iudp5p4+vHZWGGvlB2PDiIewYWgCfIyYwVRVHblH
DjEY5RyLFdVpbOBMD961SINpXQ6FwkFRzKW+B7zfrYSmNcoo9tAdz4mNxENp2eSzXoycOm9R14rZcNblFJTqD2HywcfItK6Jd4zP
Y5bj9LGp7Dpf7K4Mm+IuwonpGtRnyHjSz7lNyeJ58Ph9Co4Z/Ar6+XbBx5TpzF19G5bNaUTZMyY09dkG5stXQ2hxLr0xUKT1uaEQ
ss+UmZSNp4lNqsw7fCPdozYs2KZHF7ItoUcthL4mIi6pr4aBqddAPSkaC7Ga3q+U5sfmekH64VGQabYM+KknYJDRKfAds42eqdbQ
hflRFP0sDpZk6ODCR58oK2MZbMqp5BXVAJVtZ7Pa5pOU6xXLTV8iRl9jZfBM1Ap0O6CMtuskWMfQ3byziRAefByCP0yH4XmZeK5y
3GAc/3YJrfc6CjmvaiB6mixzTDejN82xNN5kLE1uXCJomO1CHXeUYUmgrnDniwNsxrYFcCNdlz+S5l2tcu0Zf1I6CuYs8qeI5yqc
xvIFeKH3NLfZVJ0qgt4K5rbuguT7VjR45wj8ue0y/E5w6bleKMyzs6TAnH1wVN4dbTkZOtMyiH6+6CZIE8PFjTZgNMmEFlkOQxZ/
kaZo3+LThbWQO/8nt8wkUVT/2g7WNLuyp9K+qCr/im65xvNm364QMaXqpCfLoHpcB5+aF4lr6p2Z6u4ZZH9oNkyb4of2z8zZIpO7
VX7ilzmPFcCXdG/lCs63wdiqvhQzfSz8NLoBGiud6b36BhivbsTOxJ7nhqxQweLeVbR+iB4OjzVm+7684d/M9GJpe6OIJRbQjwXX
6N6wOO6x+ld7s9oAtu9iG+0L6OS/Su/F1J54in3+i/90RoNx8/TRwdWaFdtKYPSWu+TxdThXpHoJjlnW85GaH6j/821oEuwAmOzI
SrzXY1WLBJssZWGnUPkZcjKCoDVGgJudy/k+x6xZX293e5eMM5BTUw99re7DiRc9MGX8YwGbqISeoUco/FMsr/chko1JCyeeReH+
lg/sVn4KZqrPo7n33vBpxu78wWt7+aaNb+itZxjONJ8G75z74D2/w/Dkxyh20DEX6yMV2KniG9wn03h69j6M0hrl8Lggll26NdMu
8ZQrG24RiI29pRDWPxb93NLY0gcCdi41FVS1B7DybTz3/KcC/jhUhOND37DcWmKbfs6BYdnlVLxNHPvcNquWHh1NqqWj0OZrMGuo
NSV14xgM2aZJuVUlrFXoxOb47eNAeRjK2/jgtCeurHKMDT7U3kerCway9sW/684SseqiPpOrC4MKObJ8zaWtHMcNsZiKH/X1Yfyi
RDZw1GTmcvwIJk49R6dylKjhsTI8UpOi5kYfQXZgJqz2/cp3Jshx7ufDqiYUPuHGlKTTO6fZtP+9NP9kVQoEZq6jsYf2kvrVSG7l
eUuYHPsNhu5LqT44qJ4mt73iIqK6Ra+PPoTWcWNh1j51Uny0t+rpeG3MX/iGu/1TiYItpdjIojk00raZMt19uJgHVmTV3A4V/W7z
jddt6WT6TnCf08u3BdWA56tPULykgI/SXc0lJBBv5VXMHzK+Dt8vTQXt+DqB3m0hvWsu55IGFXBfjxTTm7AKbuTiUkg2MEPbKgG/
oymfRvq50yIrTaHqUgdaLY98ZUR/eOk5ARPH+DE5xQ44dLyBXj0/TTeltcnwynzqL/wFyuKakNnH2k7zShpn/noTax8qz2bH6QHJ
fYIEvdHoqvCKpk96CLn3KuhgYhY8PVmOk1o7IctxI/yMlEJ7lZu4rdoAHFdco4QnNrhlrSXGf55BU9fOxhUPyql79Uo6MtgHUjeI
sanTOHpS74wv6vbBywm99ORMHX8lIAK+eFVA9jxb8tMuw2Nra6HqlT5W1ZyGqae7f2vEYhQ/J2KTVK+Q2WwP4u7f4mJ26LOK7+do
a2UgJhUn4OiuGCxYoU5Dp0uj+7e57GwZT1ctH3BVHfGo9FqT3+m0T6QxzQKTNe5SaMIGlmKUBx2jQ8D3dwwvW6UKPde34e2E44Ia
pxCUOr0eP49ewxV+C+JPdxwghTQVNunoAZrilgldmwvZG78vZHxaxBTDrLBzdQ55XNpSnf1dEeYXizFUuwRflRRwQedYWrpjJ5l4
GtKjjCBUqZtPI68lknWWJou5OY9Ox1ij9tlQ/KH7klvQ8wOs8sfAIDV7dkE/jfkJJZhySR9cLS2kUjkHliIaTlrfVrH1G5ZyDW/F
sfJDDYw3deC1X4qzrB0alKySjx+05kPA6sfQayJXfSdYDq3N5LmavukUdWwC9AaWMIdE9WqNx4388TJxLMgUx+XBPAgWDeQ1tS+Q
3ptq0C4dBr4PB2KrzzXo86aL/O8oYW99CHScXMBkHrZRyGsdLJ1fyI1VF4rG95eFx0JxfL3yA+uz4A2lrSlnqS7KFOZ3g462e4PF
QQ2+JDiSluUpIvcsHU5/N8C7HnvBrz4MKs4UknybFDWekGQDM3Rh9Euk8dIf+cyWuaxlQQDmOiqhfUAuZIj/Yr++GMJtBzVQLR/G
i66YstS5stA7Pk4k8oqDyhsOrKjIAe+pXIYp7m4sZDUP6zL0EB4Y0P7p17mpax6Q81wb9ilqEps97RX5rsqCcOENbsCgLxz0VcTe
dmWQrD0nOHV3OF1PjRMMGDEIr86eCSYKORAUkE3pPUMo26CF2qPt2MV2TXbkvSY5/dTDgvA93PvdkfywhyvZ0S0o2JN9nssAS/74
7Jn0QX48JG0wY8HhEizCUYG11NtzAwMmUlK5DKrNlmKzd1gy2xevQN5QC6/JZgvOtCSh8nIlpl7exUuZzAQLuwwWm99Bi0PG8c1+
R1jVFRXI6gll+lZmrLYykmoHxFCfi9LVZWMjMPRbHv9ZwQL2XnwMVUJD9IzzhqCheWT2qJfMWmZAoLIZSKaU0a0GY5ifnIwe4kWw
wzzJ/mBdCaj4KIFJhyxS+nCo9tHEi74VsLikjYwfqELlggY+lxnYn7QZDqMD+/BBfcKx5es3XsGHsR1fw2FuzwqIyx+CkhfGkfue
VdR3kg3aXbVEa8kCell1DnYf6gdHioWCgEBp9Pw6FyY42djLt2vjxi+uLEqnHhRqu2nu24uQ0qbB5fzOYS4kDMM3K4bQslo5PJR9
inSG14P2PVk26uEMcMooEMm7b2XHoq1go9gAXK+tiMPqxuGE4SNJGd7TS08D5ij2EJ5bbsMiH3meHhXzh71jyPVlDFu6RY1rPnYD
Uvp/4OSjVHCrzFUY1pNMUVqK7NTN4/QreCC6nXGhaeKm/ANqAVnQZHV3f+dGs5UQlEo4n/VzKcv9Iqj+3Itz98wlffdsaLGpgc2n
dcl/cy1+dA1jnXpXBVOu98Xll7Nxha2QzWUdFNakiofNxXDmGC28aBcOT0YY8Lu95dHZcDOovJXB/EIJnL33KjWnq+HU1HjsbIkA
zzIP8havA11JCRr2cQUt27YfSh1+4lXaxedfPir4MV+cvtZZ40CFhxQnjbj4fArd0LouuHZ0Jj/bIQ2bh3BYaBzOBzuHwoP6Rqqd
FMay9+SDhawzFhoSftu5FTvW74OpRpNJqJ4n6NEroKX8Ftpy6Br2PvsKQV3yXJrvcX7fQgs4FS1NZl6l/KBsGWYcnQbxleJ04soP
ZjN0H219lwxvdet44wd7cd7JBravS0tYJ24DB37ns48nb8JHXcE4kRJZ2YckeBvbQof6zEbNsv4w/J0z1p4bRWol8uxc4Sh2d8lb
dOwzHY6O00W5rYl4sjCNhri3svjC2Ux5pwzG9qSQ2MlIzO72wuDucMofIIfhZqMwJy0cSzkPkHFLFwn0qqF552eKvioN+KoFx8Z7
4vERyjB50EBqdNvHXylZDeqPkVYWOaH4NFdKem/Jjn5wwE6bkaIRhxKr7rpHcdXxqmB8l2B3fatg8cvRbJGiuejzxC+kPy8CpeLK
6f6rF4LIo+GCD4J00We6xDu71/E5N/ewdRALj0TLeZL/QQsXrkLNQXn83KE7KXjyHeY8K4VS5ePY541edr909UE+Jk6gsOQz9+x9
PO7JCEZfv/24MH8Qm9hUyU6YRmPpOHnm8dQOBEWaOK3zKnFqE9i8s0vwlkYnPXExx13FDZi3tQLmnYxhNq8yqHm3Fmmc/cVBojgn
2TIaHw65U/nGRxHnKB+nred1QcFuBX6+/RzWbeuBrBH1NCUuhWwmLWSy+VLMJKuabRgsgBMnJnBfHZ14Hxlz1ms+AIbe5EG+2Juq
5JWBFejTXakLsO7jZhieNBWfn2qGk36H+TODRbTTeRb93CnGrKQi+RitbMzqqKYxUZ9hq+l8kL6VylzqfkEoH45fP2/gtthGs/lV
EhiRLo8rNguwRaeCttwbDR833aQTu67wt/aFM40FQ8Dmbj/092qkN64rYWaRC3OOlEZ5i/3QbGHCSgwY6O7juVPVAi75xg2IrNRl
G5xOQLp6NL6E+3RlpIhTuK9Oeqb6rE2+gv2ACkrZ7kcTGwLhqU0ZiscNxUjJNXDTxIzWdc2DO3Y/RNcNp8GZZCOWevA6H3JGv9r/
uy0E1qpj7Z5J8NBNlk0LHgip5Wt4n+JmCGhQrF6rUCTqkY+jNwUtZOst9btWG8p2tBXBluQzpMXPgqjByeCvnwEP577gg+xfwp2n
/bmSK/68crYVhnwbwsavPMurrlLARaOtIGzkPW7b5QYuYugE9m7gMrj6UhEvPuqEFfc9yD74Iu96NIsb9uY894ybCCHjbqD96Cqa
GWQEftIOeLBWHCrv9uc3qmRwZ4OyqGuvLVc2ey3rlfRCne5Ybv7nZdyAKEm044cwR1UlstsSyWvu3sK0GgqZ/wCL6uf3VKvLqvWw
c+1ZGjVsG+gopNDMI3PwfNFzGDPrGuTOUMAVvSfohO9huNVwHSpbT1BXVge8dhJi82glsNfx5fw7O8FRLo4CNKXwgdZqXO8vh0ql
pYJ7k99B1o0UuvVaFj9cd2T6lwu4bWVi+FKuLzXW3CWz81NELxZNZANFyAqCXFlbjC8mNOjjmx8L8EvdafDusOYyhx/nowTRzOdM
AfcwdRA77bye3e1s4EekaFf6OeoyNTVv7Jr2Amp2pNiNLH/IXbziD/tuaYHfaTOa0Hcn6M8D6v9lAPY0hbAqg6kUuEcNbnZPhJNN
h/FJg41g13Ep1sg/oaA710E3RMR3HDyBw0bcJ/17p5jYI3/ad6sK5o/O4dcsjYNFt/uxnxETKX2GN1Sc/k4xTnHw7ZwBN2TfA5jv
UwR955iyyszpsOZsCm+QWg+Dxdexz9+d2E+dERjAdXL633YzDd1TcK2whk/5EMyJ2h/A0K1HIXP1blCaPJx7V7gftMcshMD325n0
0/1w2E+f5JKT4FXGZLa5chHl7nGig32e0QkVNSbRUoGli6ZhxM0KuIqKrHvpTvBouEfKlAdVfQfDqUuaaFCkjhN/x05mkUIp5zgu
unMAvvx4gqJG3AaNJydhz4n9lBlTxAYYn4G2B2uhJDyd8sba8aPUn4DUGCEtmqEk8Bh9BWLNY0k1JAE8u4yxqy0FLEuWcSuYBIYO
lsCePkFMUU6HNtdFcvHzjTBq8wVeIiyWEpLOgOZQPVx7pYMO9vOBqkZD3CWywXD/Qyxq42XBBMlIXGhPNPErz5eP84epIcbsnaIz
+u40oVdNP2nHjf6cp+lG6J12Fe78eiqaMsIYdz+RYwsDvoBM9w5KpDC28L0ZeEi2sH16tbT6VhTqndVhxzX10OYEQeWuMHD4GIht
LR10r2AF+Dd1wTpvaSaTMAPObGim/qHvaNGHbxCl8AacbU1h0eoNNEdnDVwdGs2KNH/whjttIOzyOSLjRfQkZCL0GyYjKl8dCkoV
ZtAxfiGIneSq1vwYApd1WgWxTwDMxtyn7cu/UWOgOtrFdXPnw7zg9SId2FpqzM9KcSbhc2389rOdVDufcWbPT5C+5zWwMZSm/lsN
oG7TJ+7ivGQQsk7asINode4NUHnqI5ra1Z+3fGtb1Rkqhgsk4zhjrXv8cI1lfH7ueiq8OEvgZPGR4wMiaTddgNDQ+VzXrAx+b6Af
vFSKhJdGTVzglGzBoMArBLf2cy2hCYLK/eko5fWFO+Y1mR/reg7MRu9hdMERG40fCE5taYIRw7Ww644JPF0txVwqMqgreiD7IWXI
H/nxgY66mLO9t6UotLKd01n8hBwvnyCh9jAMdx8D0QIVmr3wd31QvotfcSAIr22SY9Mb7GjY7Xmo/MsU1Eb4QldgLX91xUTcPvEe
ZRoHs6aw8aA5Yga4fPXA1KHd8HniGiyPO8VacxjJNWqyDZONmTB0DXh81cesursiSWN5XHe1FR6xPJiX9ZGbYdsPu7RvivrUTkWN
dHMmdng+myibCap8rGhK9ARc/GIAGcQIafb8DPrpMoHrIzMHp98PgEEvDNjWZ6nMfEovPdAOw65rN+h3Okx6gk/wzuQ4v+nXMvpQ
GMeJF8vDMfE9ZLlDDZQcXkPFxh2gPn8qjH89EaacnYtL1OXpxCltGnzkNnwL2sWLTebZXu9WzjowlZyfvAQzy/fcyOEruD3H9WCr
2Afe8uROnFYzHIcN3gsj9/nRxR8yrKI7EoLmHKPJCR/5guBMdPr0Bq40P6QBY6/C1pOxFL3FHKwupTApw22sQcqGtp69DfeTMln3
rDO0viMYbhrPYbsH1FBP0EbuXJQnOxlhQLKaoWDtd5CaYrLAe/tAVj3kAOVzEiS9TQ2zraLg9QATDJgVifV720UxUqF8ZlRfFK/b
DxuHuMB3rg9EuRdTyYVsaD2RD5Nmn7LbqOWC1muOc3yzAjgbXqZ0rxRYur0/12o8n7ka7OM3r/9dJ3wIJT0bP/BJksH3zt9ogmgW
Wj3JY1dDHLF8VjV2nD6DqoYjmE/DaPRrbaDbQyLBq1GdTH0jOAmPxRTrL856Rl+FnbbNaNNsg9dK+/HWc83w1PR9sFAYznesDsH+
Tg7YrjyQcC6RRJcdH+C5HkQDmkWfha+5+sajuL/+NtWIl5PKZGuRheYbsh64je74PYJHiQWQs34spM3LYJq774qGlHBMLd2Fjbpy
gFomXoQfgfbYGy0NupqHONnVDVTsZU9KVWrMaPt5mOWRCO8VZtCVUDUcK7gNd4z28db3LATLOorZoCnBcGjwHToYI4dypxWY+6Re
OHNCitl/c2QznCeB/8NbkHz/CsR5q8HC3fNYpOIl8j+eAm6HTPDDbSP21V+e7ciXYKqrM3DaAGnogiv8hLYweq1ngmkrDNicRCvc
MKBS0HePEALvZ3B35r6jB6HjqP7OQdRacYlX8FfEqtZecIs7IFq5z4isXGRxZNB81mm5ATMXD8Vd1/TpfKgn3SqUwjUVW6EjfCTJ
HxDH3ZOLabFUEbVIhNPR/HV4Xz+XHygrA/NTUui6Rxy3PegprP2egePWnyWlJjE4adyHtd60I7fYDEHTc21WFhMGjbfFefqsjR0L
b3DrGtfjltuDQKX7NiSVLwLNyZfpslMLRNdYsLpjpagxJpa/teuooLPiKBw8Ek2Dns9he3Mv2onFd8NyPQ94cSmJc1FPtw9sUoOR
lvIs49UF7mZ+MiecVMftGm5q35p6m+KKPdj9uDLYiEvg+tELdO6KHS8CMVbbFgYWRVGcVG8B19nYl6hOgUX99Z/ajb/z4Xe74bqq
K16U3s4zOTemZiDBJSloi8T9V2FzUycNeP8LBszgaMb1ibRGW5a9jdzPd75o5V+964H48mmwbc1BculuFAzesozVfMmG7kkRdGtI
vOBGpi5+rAvG2y8soe9oa1zSP5z09cIFLWv1SKMpFp+KUnFNiiS78k6BlNt7QfPnHXg0IwNOyw+iOIdezjXEnNW2h6KG232YcMCN
hGGPeaeshbR6/yAIXfC7FhhQQfqTH4LX0utkvX4KbljYFxPPt8LwplQ4a26Ipwt/QOPpkWzwUh14uIp4mdv9BWU2lrjRfxiTeWZO
sjuW0pJxWmTSdgz8T3sw72Fd4HevEsLrKiDZ1JY1/TBgoYPiIP93/O3jHg6R22uxKOwI77xsO/a9YQYqL2/SJDE3/LxgB3c2egi0
p78WdEWdFwWnPeZPPzbF3OZGUYZiH0wdeZD2B3yiK1/XUUVaM0y4YopHHcbhMU6nWmzxbvZrwR1w8hLBpbvPYXg/HVx/QgxDZArh
KB9ADq2yLPSOMaYHeuHDCDmc8raR13eKhZOr8uj24QQ+3q8KfngXwYzrV+lBaTvtyfxMV/seZj3BLdA1W5b9VI+FtE827Baby004
VsMNVkog49u1cKvbnRz21bK57UPx69Hb4NY0DO7EyLLaBnFBrtUmllWowjIKa6CpaSm0Dv4Gn65286EnL9E8vTkwRSuVYqxyoea2
OQqObIVlCzJJa/9bGq7cwl88rcdWXHoMH4e607WxkugUpcTuNwbgpRQpXFkxAHNGRUKHjiKbPqGVP9u9GO80a9FhPpN273xIypv6
0y25OcxqPbKmmgQ6E9nFGf2uucVudYji9Vzwbu1nwcmgBBLF++DAk1lUERVPo3+Js803n0Pq3V30ZH6DffhwW1xxuoW+yBQAr1IK
Zr4vuXsrEvi1K9yYZP04Wr/jKqw9HQc/HA3I23ME+ncc5n9ttKfS806Unbgczlga84N6S6HoTSNUuG3n4vRzuVmBGmh9z4WkL7fT
y5wxeCWwljO+24dfmnKRH6efQo+cbSCnrZ0KP7ijxpsI/tCFCqrZngRmkRUw2OgL7bdZgu7ScrhHdJkOLnAjI81OrvusAXvV1sau
LkvlD5ubMGHwdtZoNhIfRIxhCt97ucjsXfDIZAdFKmhS5Y+qqm8/jvKqP8ezX123OCZyZIUn1uGkwEJe6T2wyhwX7ur95ZyzfTU8
HiRkPkOkMUcsltoU6pj3rFVs4lYVLNUVw4aEa5Qw5RLnccaNVUROFam6TobX0mNx7J1FvPOgs4LLS+2xYthh7plRLqxaPwlX17cL
aldmcu80J2I/ugR3/SrI+N1AWLc4hx5c/0QJM4dTgtJePnxtKz/KRoAyw5fAiOQoeHs7EV9cNUfv8Zb4cdU+nLHm93Pn6A61M5XZ
16HmMMJsK3t6/j2sO8LDj6uSzOOMLxayWGbWZi2Ql+sA6dvi6HphgNBp0S+Kq90Db3Wd+SmDLghoeggMkJBlQXaLqf7mSmqYxcDF
6SndKx6Lyt9y6DMzRoekreDY7w6Fv26Dbfvj8WmdPra4/2LaHzPZgnUfmY5UDNXMbESKeU0a7cPRyegTfdiTgx8nOaIk54VGv/Ph
Xe0LGH75xQ+5KM8cZ/nghfVDmaayPWwdG8iGr+oVXIk4wqSCD0OGvCT23kwheZ1a2LFAH6c6qtB7fUc7px9FpDW/isr4JXz8i7n4
9JERltS5gmfuUCjT3scpZo7gglftQdXzSyFjvRpaltwEn+xPvJfrWDbK8augy60Caqa4osGqRv6rriJ+2LCTft78ACLZXdCjfhxM
7utgRIY4ph3+QnJK82CF+zgKkX/BDbyazctJ3oFncx7TtIJjorzr49DwcSSo+sawF+wANkzo4UrtPFmz6hGqPXsRXjgmkafpVlot
nMZrS60HaZ1U6rvkIKxxVBU1+C5kdtv3kMnBR3D4aA7iNjEc7HQBZl8vgLoRXaT6OpOWGinjiMJZtOnnAU5fogu0Yp7TE8FmCC53
pftieXTY2YMbVVrO+a1rERi0fqfcOVFw9mw2sUYvQec2eeEM2VAK3iSGTZMmwjTZDkHHoF/807fTAOYuoMKB96HklwJu6XOEXu5S
AjENHnQjskDiQxD2Z+0wrt6cuz5DDj8YG7KdhoPtS8qSqel3rrbrviJ+yfpJSSczOLWzIgzWOI7d9i2/NTCGtx43ENq2j+R0Nwpx
RWquQN6unrZGGgPCKIxz+AqpGddEl8qq4PLcZMr80soMXS7TJYsfxIm9BrA0xJ+73sLy8z+5WN+zYFEnhaGdxaSwxkRwfG8PVBi9
g9pzkzF6igUtKlzPTzvXh7szSJN1jbwgGiIeQy020vhpXCDd2GHAksb9hOexA9jcym6YfFWeqbmZYOmlXYw5HKHHPvokmviErvG7
uAWlUyEy9wP1DzKC5Aeq2HPHkG3pjGKvz5gzkzG9ZPI9jpT3JkFriTycdN+BeGEW8dm54HJjCOa+PYjlTTls2iQ5Zi5hy2Ryg1n+
fltsMSql3P25oDnOnOSa88DkuQObEHAFmq/dZ32UX3JOFrp0MFIN7UoXw2RREj/9tpAlDXoFJ1s18ZdyMw1T2sPyk9ajdtELGvl8
NRwQPWSXT3RQ9J0m0CuOwNfjbsAU6VVolpLOXlwpgy2x70B97SMwWXuPTtwMook28TDTvpM2+eRBWsYspmdeSE2JdrROtR8rv1JL
zkMksD4rGKxXBsJa5aUADokwdbojfqjtg26jytiy83WQWCFbPeSGCoZK6LKf+Z9on94bbq8OiKxWetFLy6c4IUeE7QeSeNfGBxTo
UcAMNZ1x/YZeWBHljAfYdlzO9qL6uvksR61P9cinMWxmsBI1LoqE9tDL/LK8FP7SThcmnLufhrUsZ9YTD3JFTjFgu8CgepnOFNrh
n0363y/TMWcTDJvrzi/jO8A6vgVqHUMg4cx3elw1GOZq1NMV2c/86/eNXElyHmS6OsKVIQ9o2QMBeCzyB+6gIayIKCGJh3upt8QV
0ncoYN6rHkipqebTbQJptr8ZuzWVr0qKBnYncyV4KLqSr7SAfbYehJP13JH7vAPzxXej4rPd3K11k1AHZBm1bIPdYjmY5CKO8UZH
odTZi0zlkyhcoob7cbeKO5EfhlkrT/CHbx6kUVIOfHVhHXf1XREVTVZEEUthIQsnsJmj97JRjfvI19AMTdJugtr8h1w1a4bjNWb4
xt0DvfNTMT36OkxdkMf7P1TGA+c+iVxKZGhirSn8ijeliuZx3AluM53RW0hrR9hjUNda6MhIhINZ64A/F0eTlhWRqNWAaeFuSpja
Rq61x7HQZhtuKlxIYX0zQXJBCHb6/M7zFo1hzXrD8cm85zCgdA3zMQyvMhZq0qN9ijinR5HyLnmws1tmQqekp2Bs33iR8Xs7ZjDr
PdjdIFpXHlP1JdUColIduBG3d0LP6nn8l+ZIVnVhv+jySa3qr2ZFUM+dhapnJ0UFAedgQZEXLMzZAfkjxsG+uo3Q7l8mmCtxB+Qk
5bHT14JVFWRzQpWjLOLAZNBvUkSd2AyYYNsORb/6Q25xELerPREN7Nfx5s/6Y+Tew3zpjnh72xNL6UxhKpW62uA7z74QNV+SylMu
0h5PHh0wgOTE06kpy4Xqn23hlZZ0wJwn5jhFbA7qn50Lvd9X4hTl45SjdYluv3hILlOEOHfdXU5vojsY1SbQvXclpO+yCc1LgzDD
XJmypx/l6YUpm3mzAqxGKrN9N10pq3saWtUkQbuzMgjTL9CLVXvI0UGGs9KtRMePXnjD+xqk16vzp87dBkwppGk3D4D1oWA03poA
5adGstlrVZlSPKONEpLMYmkrZyP4/cxuNYaVMYmokpkBuu3aOMUirLLk9Tveb88SljVgJK05XkUfb0hju3sDHNP+xK0dcwScZpqL
Hp8Yhs/9eYHZlnPcwQ9y5NdHlu0ekY2f9IZg1NuR5KY5nSZfcmSqm2PZnQYB8nGO/KGVE1jBq4Wgo+mFccLD4Oq4EhwKdMBwlRE6
hXfSznNrodlOimt658aKT5vR99td1BKhhnPGmcPge37Y3U3cz9X5MOzpexgjYwg6RstB/t5fvbQmUJylJD9DZyDtf9IXtSbpsCrl
FD5nW4NgbV4JvO+QxlPvSvi5h8/BpIPNtKN1Dfs5Op6pdxWw1UNy+a5FHlQY7sZljB+EEs+r6ESADouI0sC9Z65D05V0HJeeRSzK
FeYcl2TT7nTSk+REEGxII4dy+erFnzbDt/EKMNP5d+6yPJP7NWgKqazPAlvpVjL/re2vpnyCh949LGNJP3KbX4Kx/CvoHjwCraf6
k8/7m2AtHU+nUmOZbcVGeFMQTmdf7WcmQ2xQ8/5y7BhkwKela7Koaf3Ysmvf+PRQCbb52Rc+8/0l9Ao8yJ92jWAC/gp8Vtch2fUR
LM6A5yyXFkN7pRim1sRRjEoi9X2vDs/sztIo2RvwZcBBTmJTP7iBauDif4OM+3sTa9WmJx1P+VPfvBiG6GDcLXMmWnlC5K0VjhGX
RvJSxRmwb2cUui6Pocptv2BWHw3248RPKK1aTNovDrDEXS9hXVIYy57VAzTZFe3a9NgO/8FsQNlsHN1ZxXYr1lH90+FsstJetrLh
E4zb4MnV50aC0dP9sCRjGH4okay22yDJAg6ps/wl3VQp0mbb6iPBbJI9/ZpuBkP2Ao3Njefchjmz4NFt5FJyD8ZWjMBbcZLUbjEV
I1c5wIu85ezDGXe41TALP6mZYuxiCczwH0Yrzzwmzebd2OwZCo/fRFNzyBny8bXAyA86LPyMHcndmEV3zv0g5yd94cvAwcKSpKNM
vzOL908ZiSniFsLb0mKoF6cBp1cWMtmds+HWtbOU7HQPbYQ8RJX24K7eocIdQ6/i3iQV7FtYzR3F22gvnCi8Y2kodN+viXJHvNB0
30nct0CVae6RADlPY0F5fz30VlBnVxbco8IlXWSjqEnT5MpQIXM++k0AvuJUJOYPPQE6IarCS4e1QflaXy4rIhba/JLZnbJw1mck
4quPQ1njjf04zU9S6Bz2nFxWn6LWAGPsI6PKFOVGMq/CXdg8aTPGLNwCQy+VM61DkmQRNJQ/UzFO2MlE7NqsOM742CP80mghzLPv
xTtNkrjE8w7YFvbjzZFx9h6lVYEDNZhgmzbcPTqO6U+rBJ1B5sw+zJLZHNsLdcPn8Gl9/Tn3OyN++1UZZiruhSOXy0kl5AW8OLqJ
L1R3ROkQee705yFMyjkQm4v2YVxiPhM52DL9njzevv89flZRG/m+l6G7+ZvBc/BtCPtmyI6U60NX6GhMqm+its+G7EfNSc7g1gTQ
X6mBetMmEUZ5QIDoC3fdaT9WayyGFw8mYe1daaqs6eUTlhuhnUsJXSpWYaNqZGjeSEVRzXR5vLX8CKV71kHe5Re8R9UavCKmAI/P
hdKloaZsoMM1Kh6TLxqZ9+Ff/T92/TR3EHMQE/vr+G/0D7D+q3+AWOcDvuXyNpIpdIFClZFU3Z3A1/XZxZ3nH3BG0/pyRie1QPZL
JCw4MB769xlFdbej4anOJlK9/juG1ocLJOwOQMi2MbxKdDescX3JrdEZCao3b3Lnqw/RM8cbMPHCDjr5WJsKr2+Gc/6VELy8L3Ar
bKDzFs/FFSvgc8OxbMBaS0q48JMvf3Od01MNoO/hZ/iBTjsgpKUFHi54CR3Gx3njC2I47kMxH7uqVcBbedl7PZTHDWUOcDIih7e4
X8J9tzSlLaqxsLW7oWr2WF96nFwCn6eUcRpRO0TV0n4wIPU+7MpKhyxtDVrpclb0oatZsHGP2L/6fxQGy5z4H/RfsPnLfy/O7BAa
Gu4W1l/bJYxpSBTeWX9TWLhkj3CJzybhqQPlwgHiycLdj3OERQYRwuCf64TrdEOF1w8tEb733Cn0qU4XHpxaIBw4IEf4c+wu4aVN
u4UHMFFonB0tzHjrJbQo2in8WpwgnB8cJjR2ihPOizkufP9jmbDPsyjh4svhQvfJu4UfDTYLCz+ECx+1Owu3+6wQup/OEDasOyPU
mbRKOGVxttD66QKh9OZM4av0BGHFiniha36M8MeFnUL56VHCSVrewoGqMcK5JonC73kJQkWX5cKAqmChVc8GodjbfULfiTuFBYd2
CpcdCRBaDY4WbnmSKMyacli4hiUID+5wEr4eVvyv/h8hoU+K/wf+G/OX/5RWNBPqu/Aq+Z/4X3lfqeqvfj6Jh/jPR+Zx2aPOU2bg
TrKdpcjSrDt4Z91nVGCkzFbXJPLLg25yS4e32+Og8ey03TNRTI8lTT/qT18e76fS7kGwpO2XfYpKKJv0ehqT2ObO3H/psyWtyaD/
BuzbxhhB4sUgvnvCcPZ910awXy3LqssTmJbpaWpXMGNXfe/zWjJzmNvIu4D2HdQbO53NSEzjpHuj2O12XfJrMEHdcy7s6VEdZllZ
SDcXWLCytEReVjeSM+cH0fPPL6kUX/ELmstIDBNFkwuXssCU9zDqnTwbnKnIPjab8ZPd7tHQxZL4l/9u5FSW/Q/8Z/uX/2Ivx7Pm
H8vYpmURKL5Tg/qEjhX+Onoc5AWDMKBcRlhjlYLvNe/hDlO333G2m9I0F7EJ2ANXN3WBaGwVBnuOqb5nsoatmO6MiVtj2Jp1UezU
veV4KVgew3cVMa0zIoxJcWTZpzXp7mcJYbXjAZrf44sjyn3RbuteDPtig8f0dHCJsQdmrjVD02MLWWZNDBu7TJ/F217CC3lbqOTR
LbaobDcrTqzBEfO34uoV2hiSHME++n6uKq2LZvlbj7JhNkn4I+sBrDcayKJqAtnh1kp2vc2TvZoXjKfCh7Evkmux2DeOfR6sICzW
c0Oroe/gVseHf/nvvm2NzP/Af3Z/+S9jUta/1tioNCj5r+nLUv/H6b39/jZB+N89lEb+P84PvmuvhPc6OiHiVSsfGuMLTUt9yTQm
gVy7BkCo1Wjqei7LNhc8JdyqyeU/ewGTDBw4RdsOvr/MS/7k0pMQ2a8/aN7bAV9mzATHoMfwdlgO2KpaczlDjnHTDeLB3O8tjXKT
xpTimeyR0Ts+/MErwR1BGj8Rk+jx2od06/xHLt9Ujix9BkK4vQ8fYK+JQt9hWKbpIHrS/xvf/9E0ZqJ7gzOtrIIAgwRaf68vWzh0
I783wBJdPX246IKvnPj0/aIPKWNpe3MUpzHeiT+1fr+gf0sTpNlU8hlZOmRhkAFb0tW4vN+lQfbNZDJZ+hSqAzsE3uKjKGNqLs0Q
6PJS/rpYc8kXe6Yvhg3umQxcpuMBp6Vsg9BU1Oqzj3pdq/kgwTFaVhRDWYv2w1QlPzSUsgOtC7Lg5ZTKUq+sxWhDbZjmngOSD/Tp
Oe/Fu7tmUVSjEpsLBiTTOIlFJJTxg+0aqLbwPvc0OIFblSyJS91/x5nyy2R9ehZw0y/ZL/RURJykUa3y9TuoTH3IScEsuDBgKCx0
GgwLV/VyN7LqaWnzcjRdzwO7aIU2dULRzucldLSolb4d3kQBE23BtUqKpDQaOIWlurT1ZwX/fmkyL8zsEcW29ePdvt4fE5FzhEr7
HuHCt86HxT2udM1AAlFFiWXk5LD707O5/LnxeD3Sk/o+MgLPgtNUnKgoGD3zKiiN9IbzYjb8zgtC7rNeGEs82gn1WrLMasx29JTS
YQnDP3NPloWyqPuD0K53PkWfX8qKLyTyD26ow17N6bjs8nROKWoEJEbHQ93FOO6kiSNbPdTD3n5LPH7yVkVrlgS46iQ7/TYaJ15Q
YFIXMrljKkNoYepRvr59Hvdy0m5wfW6GJzw8mV2uBNMMuEQzK0bQKsNH3Fd059TmxPKHOpTYitBu3jklmNfef47G6D+GXYuluf6b
hsFM3yzYY34B/AcOwyabtTDIWAofLWBc/7m7uOa5u0lX2Mr1eQ6wVOIEaB5eC2rJWnhzpwHIbdcALykG2sv2UF7oWP5ghjga9d9L
S1NFIL/vWtW5iZZQfWgPbVM6y2XJ67EzUbq0fhbap63TxlDPQj4U5dlXt0QovahANZ3iLOhrCrc5N4OTCu9PaW+yOKMlL0Gifx6Y
e3dDSAlxfoP74wO/w+Rh5sc/UdsFx1Y95X2WGtH9xnaR5xRTUiMPCvn4VRRhaslMt2vy1+5nwxbVJLqT6A3NU5wpoWawKPNavWj4
gWZqZMr8s9vHqD16CmydFkrjinv4UTa1sFvYIKpM1Ob/0p7N60cqiP2H+PyftMfxD+0ZNWGc8P9Fe/56VdqVwGwLExh8MZheeu6F
v+zpd/bhV+XfV1T+v2ih1X/a85+D4y3+u/ZYyf11xaYDy2f/3+e364tZ/u8B8v9ywTrvQN8lq3y3LAn6ffVFvssmjDP/bxowymqk
jZ31GEtbSyvbMWNGjbG1tRwzytbKxs7OauQoO2tb2zGjR4+xtPvXlHlmILHS+Lcdxv+yUvwvY//TzofBEYd3SYuJ/XUo/xfL//67
mQesXPUnffrN/Yj/CGBi/f9Btkn/b9rcxz9w9ZKgRf/h7T8XEW5c+/x/OVHnH4vMkPnbIuuC/AOXLPde9Ntvy9es9l4T9Oc6fx9k
r/KPdW78bR2vzUHe/oHLvAP/xP/ejVDpH/gc2X96wmLkn+zfO/H9k220/oMd9Sf79y50/2SVx/zBWv7J/r0D2z/Z6bZ/sFZ/sn+v
vv7J7rb7gx39J/v3yuOP/W7+g7X+k/171v3HfoP/YG3+ZP+ecf6x35A/2DF/sn/Ptv7Y79Y/WNs/2b9nWX/sd9sfrN2f7N9Vst8/
WPXtf94b/+XG+ruk/RPeHfEn/F/urL+r0z/hgZFif2rgn/DfRUP/H3Dy3+D/fzmb7iojM/x/iZ/kiP8kVf84e0v9+2wb9R8/oquM
7F/n/XH/fi/+r8+kZf7NqP7r+//+7v8HNcL38g==
"""

if __name__ == "__main__":
    sys.exit(main())
