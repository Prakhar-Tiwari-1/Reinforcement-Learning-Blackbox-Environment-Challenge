"""Policies used for evaluation: random, constant, frozen PPO checkpoint, rule-based heuristics.

Every policy is an object with `act(obs) -> np.array([b0, b1])` and `reset()`.
All of them are specific to the contest world's spaces (Box(11) -> MultiBinary(2)) except
`RandomPolicy`; they exist to understand the world and to create a starting point.

Observation dims as understood from the data (docs/EXPLORATION_REPORT_2.md):
    1      danger proximity (~1 when touching something that hurts)
    2/5, 3/6, 4/7   binary colour channels, left/right eye; 4/7 = "danger colour"
    8      health (falls on contact with danger, resets to 0.5 at respawn)
    9      food proximity (> 0.75: food is close, eaten if we keep moving)
    10     speed
Actions (two motors, Braitenberg-like): [0,0] stop, [1,1] forward, [0,1]/[1,0] turn (opposite ways).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

STOP, TURN_A, TURN_B, FORWARD = (np.array(a, dtype=np.int8) for a in ([0, 0], [0, 1], [1, 0], [1, 1]))
ROOT = Path(__file__).resolve().parents[2]


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, obs):
        return self.rng.integers(0, 2, size=2).astype(np.int8)


class ConstPolicy:
    def __init__(self, action):
        self.a = np.array(action, dtype=np.int8)

    def reset(self):
        pass

    def act(self, obs):
        return self.a


class HeuristicPolicy:
    """Rule-based forager. Parameters are the knobs tuned from the statistics.

    Priority: 1) avoid danger  2) don't go too fast  3) go for visible food  4) explore.
    `away_from_left` is the action believed to turn away from something seen in the left eye
    (dim 4); the opposite turn is used for the right eye (dim 7). The data suggests [1,0]
    (TURN_B), but both mappings are tested.
    """

    def __init__(self, danger=0.75, max_speed=0.3, food=0.6, away_from_left="B", explore_turn=0.3,
                 seed=0, name=None):
        self.danger, self.max_speed, self.food, self.explore_turn = danger, max_speed, food, explore_turn
        self.away_left = TURN_B if away_from_left == "B" else TURN_A
        self.away_right = TURN_A if away_from_left == "B" else TURN_B
        self.rng = np.random.default_rng(seed)
        self.name = name or f"heur(d={danger},v={max_speed},f={food},{away_from_left},t={explore_turn})"
        self.turn_left = 0

    def reset(self):
        self.turn_left = 0

    def act(self, o):
        left_danger, right_danger = o[4] > 0.5, o[7] > 0.5
        # 1) danger: turn away from the side that sees it; if both / unknown side, turn and slow down
        if o[1] > self.danger or left_danger or right_danger:
            if left_danger and not right_danger:
                return self.away_left
            if right_danger and not left_danger:
                return self.away_right
            return self.away_left if o[10] < self.max_speed else STOP
        # 2) speed limit (fast = deadly)
        if o[10] > self.max_speed:
            return STOP
        # 3) food is close: keep moving forward into it
        if o[9] > self.food:
            return FORWARD
        # 4) explore: mostly forward, sometimes a run of turns
        if self.turn_left > 0:
            self.turn_left -= 1
            return self.away_left
        if self.rng.random() < self.explore_turn * 0.2:
            self.turn_left = int(self.rng.integers(2, 8))
        return FORWARD


class HeuristicV2(HeuristicPolicy):
    """Rule-based forager v2, using the round-2 finding that health (dim 8) rises while touching
    something seen in colour channel 3/6 ("healer") and falls near colour 4/7 ("danger").

    Priority: 1) turn away from danger colour  2) graze on a healer while health < full
              3) steer toward a healer when health is low  4) speed limit  5) food  6) explore.
    Turn mapping "A" (TURN_A turns away from the left eye) was the better one in round 1."""

    def __init__(self, low_health=0.45, graze_until=0.65, **kw):
        kw.setdefault("away_from_left", "A")
        super().__init__(**kw)
        self.low_health, self.graze_until = low_health, graze_until
        self.toward_left, self.toward_right = self.away_right, self.away_left
        self.hungry = False  # hysteresis: seek/graze healers from low_health until graze_until

    def reset(self):
        super().reset()
        self.hungry = False

    def act(self, o):
        left_danger, right_danger = o[4] > 0.5, o[7] > 0.5
        left_heal, right_heal = o[3] > 0.5, o[6] > 0.5
        if left_danger or right_danger:
            if left_danger and not right_danger:
                return self.away_left
            if right_danger and not left_danger:
                return self.away_right
            return self.away_left if o[10] < self.max_speed else STOP
        if o[8] < self.low_health:
            self.hungry = True
        elif o[8] > self.graze_until:
            self.hungry = False
        if self.hungry and (left_heal or right_heal) and o[1] > 0.9:
            return STOP                                   # touching a healer: stay and heal
        if self.hungry and (left_heal or right_heal):
            if left_heal and not right_heal:
                return self.toward_left
            if right_heal and not left_heal:
                return self.toward_right
            return FORWARD if o[10] < self.max_speed else STOP
        if o[10] > self.max_speed:
            return STOP
        if o[9] > self.food:
            return FORWARD
        if self.turn_left > 0:
            self.turn_left -= 1
            return self.away_left
        if self.rng.random() < self.explore_turn * 0.2:
            self.turn_left = int(self.rng.integers(2, 8))
        return FORWARD


def _head(i, o, hidden=64):
    return nn.Sequential(nn.Linear(i, hidden), nn.LayerNorm(hidden), nn.Tanh(), nn.Linear(hidden, o))


class TorchPolicy:
    """Actor from ppo_baseline.py / bc_pretrain.py checkpoints: input = normalised obs + previous action."""

    def __init__(self, path, greedy=False, seed=0, name=None):
        ck = torch.load(path, weights_only=False)
        self.actor = _head(13, 2)
        self.actor.load_state_dict({k[len("actor."):]: v for k, v in ck["net"].items() if k.startswith("actor.")})
        self.mean, self.std = ck["norm_mean"], np.sqrt(ck["norm_var"] + 1e-8)
        self.greedy = greedy
        self.gen = torch.Generator().manual_seed(seed)
        self.name = name or f"{Path(path).stem}{'-greedy' if greedy else ''}"
        self.prev = np.zeros(2, np.float32)

    def reset(self):
        self.prev = np.zeros(2, np.float32)

    def act(self, o):
        x = np.concatenate([np.clip((o - self.mean) / self.std, -5, 5), self.prev]).astype(np.float32)
        with torch.no_grad():
            p = torch.sigmoid(self.actor(torch.from_numpy(x)))
        a = (p > 0.5) if self.greedy else torch.bernoulli(p, generator=self.gen)
        self.prev = a.numpy().astype(np.float32)
        return self.prev.astype(np.int8)


class QPolicy:
    """Greedy (or epsilon-greedy) policy on a Q-network from offline_q.py: (obs, prev action) -> 4 Q values."""

    ACTIONS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)

    def __init__(self, path, eps=0.0, seed=0, name=None):
        ck = torch.load(path, weights_only=False)
        self.net = nn.Sequential(nn.Linear(13, 64), nn.LayerNorm(64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 4))
        self.net.load_state_dict(ck["net"])
        self.mean, self.std = ck["norm_mean"], np.sqrt(ck["norm_var"] + 1e-8)
        self.eps, self.rng = eps, np.random.default_rng(seed)
        self.name = name or Path(path).stem
        self.prev = np.zeros(2, np.float32)

    def reset(self):
        self.prev = np.zeros(2, np.float32)

    def q(self, o):
        x = np.concatenate([np.clip((o - self.mean) / self.std, -5, 5), self.prev]).astype(np.float32)
        with torch.no_grad():
            return self.net(torch.from_numpy(x)).numpy()

    def act(self, o):
        j = self.rng.integers(4) if self.rng.random() < self.eps else int(np.argmax(self.q(o)))
        a = self.ACTIONS[j]
        self.prev = a.astype(np.float32)
        return a


class MPCPolicy:
    """Model predictive control on the learned world model (world_model.py).

    16 plans = (a1 for 2 steps, then a2 for H-2 steps); score = sum_t gamma^t (P(food) - P(death));
    play a1 of the best plan. All plans are rolled out in one batch."""

    ACTIONS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

    def __init__(self, path, horizon=10, gamma=0.95, death_weight=1.0, seed=0, name=None):
        ck = torch.load(path, weights_only=False)
        self.net = nn.Sequential(nn.Linear(17, 64), nn.LayerNorm(64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 14))
        self.net.load_state_dict(ck["net"])
        self.mean = torch.tensor(ck["norm_mean"], dtype=torch.float32)
        self.std = torch.tensor(np.sqrt(ck["norm_var"] + 1e-8), dtype=torch.float32)
        self.dscale = torch.tensor(ck["dscale"], dtype=torch.float32)
        self.H, self.gamma, self.dw = horizon, gamma, death_weight
        a1, a2 = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        self.plans = np.concatenate([np.repeat(a1.ravel()[:, None], 2, 1),
                                     np.repeat(a2.ravel()[:, None], horizon - 2, 1)], 1)  # (16, H) action indices
        self.name = name or f"mpc(H={horizon})"
        self.prev = np.zeros(2, np.float32)

    def reset(self):
        self.prev = np.zeros(2, np.float32)

    def act(self, o):
        P = len(self.plans)
        obs = torch.tensor(np.repeat(o[None], P, 0), dtype=torch.float32)
        prev = torch.tensor(np.repeat(self.prev[None], P, 0))
        score = torch.zeros(P)
        with torch.no_grad():
            for t in range(self.H):
                a = torch.tensor(self.ACTIONS[self.plans[:, t]])
                onehot = torch.nn.functional.one_hot(torch.tensor(self.plans[:, t]), 4).float()
                x = torch.cat([((obs - self.mean) / self.std).clamp(-5, 5), prev, onehot], 1)
                out = self.net(x)
                pr = torch.softmax(out[:, 11:], 1)
                score += self.gamma ** t * (pr[:, 1] - self.dw * pr[:, 2])
                obs = (obs + out[:, :11] * self.dscale).clamp(-1, 1)
                prev = a
        best = int(torch.argmax(score))
        act = self.ACTIONS[self.plans[best, 0]]
        self.prev = act.copy()
        return act.astype(np.int8)


def make(spec, seed=0):
    """Build a policy from a short text spec, e.g. 'random', 'const:1,1', 'ppo', 'ppo-greedy',
    'heur', 'heur:danger=0.6,max_speed=0.25,away_from_left=A', 'torch:data/bc.pt'."""
    kind, _, args = spec.partition(":")
    if kind == "random":
        return RandomPolicy(seed)
    if kind == "const":
        return ConstPolicy([int(x) for x in args.split(",")])
    if kind in ("ppo", "ppo-greedy"):
        return TorchPolicy(ROOT / "data" / "ppo_baseline.pt", greedy=kind.endswith("greedy"), seed=seed)
    if kind in ("torch", "torch-greedy"):
        return TorchPolicy(ROOT / args, greedy=kind.endswith("greedy"), seed=seed)
    if kind == "q":  # 'q:data/q_fqi.pt' or 'q:data/q_fqi.pt@0.05' (epsilon)
        path, _, eps = args.partition("@")
        return QPolicy(ROOT / path, eps=float(eps or 0), seed=seed, name=spec)
    if kind == "mpc":  # 'mpc:data/world_model.pt' or 'mpc:data/world_model.pt@H=15,death_weight=2'
        path, _, opts = args.partition("@")
        kw = {k: float(v) for k, v in (x.split("=") for x in filter(None, opts.split(",")))}
        if "H" in kw:
            kw["horizon"] = int(kw.pop("H"))
        return MPCPolicy(ROOT / path, seed=seed, name=spec, **kw)
    if kind == "heur2":  # 'heur2' or 'heur2:low_health=0.6,max_speed=0.2'
        kw = {k: (v if k == "away_from_left" else float(v))
              for k, v in (x.split("=") for x in filter(None, args.split(",")))}
        return HeuristicV2(seed=seed, name=spec, **kw)
    if kind == "heur":
        kw = {}
        for item in filter(None, args.split(",")):
            k, v = item.split("=")
            kw[k] = v if k == "away_from_left" else float(v)
        return HeuristicPolicy(seed=seed, name=spec, **kw)
    raise ValueError(spec)
