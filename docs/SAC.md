# Discrete Soft Actor-Critic (22–23 October 2026, after the contest)

Code: `src/train/sac_discrete.py` (online, training server), `src/train/sac_offline.py` (from a stored buffer),
`src/train/merge_buffers.py`, `scripts/bg_sac.sh`, `scripts/bg_sac3.sh`. Logs: `results/jobs/sac*.log`,
`results/eval_summary.jsonl`.

## Why SAC

- **Off-policy like DDQN:** it learns from a replay buffer, so it keeps the pre-filled buffer that made DDQN strong.
- **Larger critics are allowed:** only the policy must have ≤ 128 hidden units. In DDQN the Q-network is the policy,
  so it has to stay small. In SAC the actor is the policy (64 + 64) and the two critics (256 + 256 each) are used
  only for learning.
- **Built-in exploration:** an entropy bonus, with the temperature α tuned automatically so that the policy
  entropy stays near a target (0.35 × ln 4).
- **Discrete version:** the action space is 4 joint actions, so SAC-Discrete (Christodoulou 2019) sums over the
  actions exactly.

Everything else (inputs: last 4 observations + previous action; reward shaping; 5-step returns; γ = 0.99) is the
same as DDQN, so the comparison changes only the learning algorithm.

## Results (training world only)

**Training runs** (4,096-step windows, while learning, actions sampled from the policy):

| Run | Started from | Steps | Mean of last 5 windows | Best window |
|---|---|---:|---:|---:|
| `sac` | scratch | 197k | +130.8 | +138.2 |
| `sac2` | best checkpoint of `sac` | 197k | +139.8 | +152.3 |
| `sac3` | best checkpoint of `sac2` (merged buffer, 3M capacity) | 98k | +148.1 | +157.8 |

**Frozen evaluations** (6,000 steps × 4 agents):

| Policy | Reward / 1,000 | `r = +5` / 1,000 | `died` / 1,000 | `[0,0]` share |
|---|---:|---:|---:|---:|
| **`sac2` best checkpoint (184k), greedy** | **+145.8** | 31.7 | 2.50 | 12% |
| SAC offline on DDQN's buffer, greedy | +139.4 | 29.7 | 1.79 | 10% |
| SAC offline on DDQN's buffer, sampled | +125.2 | 26.8 | 1.71 | 13% |
| SAC offline + CQL, greedy | +59.0 | 13.1 | 1.29 | **62%** |
| SAC offline + CQL, sampled | +124.2 | 27.2 | 2.38 | 13% |
| DDQN 500k, greedy (same kind of test) | +134.5 | – | – | – |

## Reading

- SAC beat DDQN's training-world plateau after about 184k steps, and greedy SAC beat greedy DDQN by about +11.
  Each evaluation is only 6,000 steps (roughly ±8), so the gap is likely but not certain.
- **Greedy beat sampled play for SAC** (+139 vs +125), the opposite of PPO.
- **Offline learning works when the data comes from a good policy** (SAC on DDQN's buffer: +139). It fails when it
  doesn't (offline Q on random data: −3).
- **CQL flattened the policy:** conservative Q-learning lowers the Q-values of rare actions, and greedy play then
  chose `[0,0]` on 62% of steps.
- **Limits:** SAC was trained after the contest and never played in the contest world. Training-world scores did not
  predict contest scores in this project (section 3 of `CONTEST_LIVE.md`). More moving parts: actor, 2 critics,
  2 targets, α. The entropy target is sensitive.
