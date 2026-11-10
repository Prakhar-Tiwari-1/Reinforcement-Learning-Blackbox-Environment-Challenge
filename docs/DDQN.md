# DDQN: the algorithm behind the current submission

*17 October 2026. Code: `src/train/ddqn.py` (training), `prakhar.tiwari.py` (playing). Run: `runs/ddqn`. Final policy: `models/FINAL_ddqn_500k.pt` (embedded since 17 Oct; backup `models/BACKUP_ddqn_450k.pt`).*

The world is described here only with `r` (reward: −5, 0, +5), `died` (`infos[agent]["died"]`) and o0 … o10.

---

## 1. What it is, in plain words

**DDQN (Double Deep Q-Network)** learns, for every situation, a number Q for each of the 4 possible actions `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]`:

> Q(situation, action) = expected (discounted) sum of future rewards if I take this action now and play well afterwards.

To play, the agent computes the 4 Q values and takes the largest. To learn, it compares its Q value with "what actually happened in the next 5 steps + its own estimate of the situation reached", and corrects the difference. It learns from a large **memory of past steps** (replay buffer), so every step played on the server is reused many times.

The "double" part fixes the classic over-estimation of Q-learning: one network chooses the best next action, a second, delayed copy (the *target network*) evaluates it.

---

## 2. Inputs and outputs

| | |
|---|---|
| **Input (46 numbers)** | o0 … o10 at steps t, t−1, t−2, t−3 (44 numbers), each standardised as (value − mean) / std with **fixed** means and stds computed once from all recorded data and clipped to ±5; plus the previous action (2 numbers, 0/1). At the start of a life the missing older frames are copies of the oldest available one. |
| **Output** | 4 numbers: Q of `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]`. The agent plays the arg-max (greedy). |
| **Not used** | `r`, `died`, `env.state()`, any label. The shaping term below only changes the *training target*, never the input. |

**Why a fixed normalisation** (PPO used running statistics): the replay buffer stores already-normalised inputs; if the normalisation drifted, old and new entries would mean different things.

**Why 4 past observations:** a single observation gives o9's level, not whether it is rising or falling; the trend is what tells "approaching" from "passing" a source of `r = +5` (see §5).

---

## 3. Network (the policy: 128 hidden units)

```
46 → Linear(46,64) → LayerNorm → ReLU → Linear(64,64) → LayerNorm → ReLU     (64 + 64 = 128 hidden units)
   → V: Linear(64,1)   A: Linear(64,4)   →   Q = V + A − mean(A)               (output layer, "dueling")
```
- 7,749 parameters. The **dueling** split (value of the situation V + advantage of each action A) helps when, in most situations, the action matters little (true here: most steps have `r = 0`) — it adds no hidden units.
- **LayerNorm** keeps learning stable (it is what makes Q-learning work without exotic tricks in recent work, e.g. PQN 2024).
- The **target network** has the same shapes; it is only used during training and is not in the submission file.

---

## 4. Training: every setting and why

| Setting | Value | Why |
|---|---|---|
| Discount γ | 0.99 | horizon ≈ 100 steps; the o9 / o1 build-up before `r = +5` / `died` starts 20–40 steps ahead |
| n-step returns | 5 | the target uses the real rewards of the next 5 steps before trusting the estimate: faster propagation of rare `r ≠ 0` |
| Double-DQN target | y = Σ_{i<5} γ^i r_i + γ^5 · Q_target(s₅, argmax_a Q_online(s₅, a)) | reduces over-estimation |
| Target network refresh | every 2,000 updates (hard copy) | standard; keeps the target fixed long enough to learn against |
| Loss | Huber (smooth L1), gradient norm clipped at 10 | robust to the occasional large target |
| Optimizer | Adam, lr 1e-4 | standard for DQN-type methods |
| Batch / replay ratio | 256 per update, **2 updates per server step** (= 4 new transitions) | the server gives ~20–40 steps/s; the GPU can do far more updates, so every transition is reused ≈ 128 times on average |
| Replay buffer | 1,000,000 transitions (float16 inputs) | large enough to keep old *and* new behaviour |
| **Pre-fill** | **all recordings in `data/` and `data/moves/` (~1.2M agent-steps → 1M transitions)** | off-policy learning can reuse them; `data/moves/` contains 480k steps where agents deliberately tried random moves (one action held 1–15 steps) from good states — exactly the "what if I had done something else here" examples Q-learning needs |
| Exploration | ε-greedy, ε 0.30 → 0.02 over 300k steps; a random action is **held for 1, 2, 4 or 8 steps** ("ez-greedy", Dabney et al. 2020) | single random steps rarely change direction enough to find or re-find an o9 source; held turns do |
| Rewards used for learning | 0.2 × (r + F) | r scaled to ±1; F = shaping below |
| Shaping F | 5 · (0.99·Φ(o′) − Φ(o)), Φ = 0.5·clip(o9,0,1) − 0.5·o1·max(o4,o7) + 1.0·o8; on a `died` step F = −5·Φ(o) | potential-based (Ng et al. 1999): gives a signal 10–20 steps before `r = +5` / `died` without changing which policy is optimal |
| Deaths | not terminal | the agent continues after `died`; the "episode" never ends |
| 4 agents | all controlled by the same network, all their transitions go to the buffer | 4× more data per server step |
| Steps | 500,000 server steps × 4 agents (16 Oct 17:00 → 23:29, one automatic restart after a server timeout) | |
| Saving | `latest.pt` + `buffer.npz` every 20,480 steps (resume), `step_*` (last 3), `best_*` (top 3 by recent training reward) | crash-proof; see `scripts/rl.sh` |

---

## 5. Results

### Training (with ε exploration, 4 agents in one world)
| Steps | Reward / 1,000 | `r = +5` / 1,000 | `died` / 1,000 | ε |
|---|---:|---:|---:|---:|
| 0–20k | −8 → +10 | 1.3 → 4.2 | ≈ 2.9 | 0.30 → 0.28 |
| ~140k | ≈ +35 | ≈ 9 | ≈ 2.3 | 0.16 |
| ~210k | ≈ +55 | ≈ 14 | ≈ 2.9 | 0.13 |
| ~360k → 500k | **plateau ≈ +117** | ≈ 26.7 | ≈ 3.1 | 0.02 |

### Contest world (frozen, greedy, single agent)
| Checkpoint | Runs | Pooled |
|---|---|---:|
| step 208k | +90.8, +117.5, +110.0 (4–6k steps) and +45.0, +88.8 (4k) | ≈ +90 |
| step 389k (best training score) | +121.2 (20k) | +121 |
| step 410k (embedded until 17 Oct) | +152.5, +148.8, +146.2 (4k each), +141.0 (20k), +137.5 (10k) | +142.5 |
| step 450k (backup) | +147.0 (10k) | +147.0 |
| **step 500k (end of training, embedded)** | **+146.5 (20k), +154.0 (10k)** | **+149.0** (31.6 `r = +5`, 1.8 `died` per 1,000) |

Previous submission (PPO): ≈ +50. Random actions: −14.

### What DDQN does differently from PPO (`results/ddqn_vs_ppo_behaviour.txt`, 240k recorded states)
| Situation | DDQN | PPO |
|---|---|---|
| o9 > 0.7 and o9 fell by > 0.1 in 3 steps (just passed a source / just got `r = +5`) | **`[1,0]` 62%**, `[0,1]` 27% | `[0,0]` 49%, `[0,1]` 48%, `[1,0]` 1% |
| o10 > 0.35 | `[0,0]` 13% | `[0,0]` 42% |
| all states | `[1,0]` 24%, `[0,0]` 15% | `[1,0]` 5%, `[0,0]` 27% |
| o9 rising 0.3–0.75 | `[0,1]` 86% | `[0,1]` 93% (same) |

DDQN uses the second turn to come back to a source when o9 starts falling; PPO had nearly stopped using that action. This matches the measured gap: PPO's median gap between `r = +5` was 99 steps; DDQN averages ≈ 33.

---

## 6. DDQN vs PPO: advantages and disadvantages (for this challenge)

| | DDQN | PPO |
|---|---|---|
| Re-uses data (sample efficiency) | **yes** (replay, pre-fill) | no (each batch used for 4 passes then discarded) |
| Compares all actions in every state | **yes** (one Q per action) | only reinforces actions it sampled |
| Stability | needs target network, Double Q, LayerNorm, Huber — then fine here | **very stable** (clipped updates) |
| Action spaces | Discrete; MultiBinary(n) as 2ⁿ joint actions; **Box needs discretising** | **any** (Box / Discrete / MultiBinary) |
| Partial observability | greedy play can loop; 4-step history used | stochastic policy helps |
| Changing world (contest drift) | replay holds old experience | learns only from fresh data |
| Result here | **+144** | +50 |

**Consequence for the submission:** the file plays the DDQN weights in the contest world (Box(11) → MultiBinary(2)); for any other environment it falls back to the generic PPO learner (DDQN is not used there because of the Box-action limitation).

---

## 7. Open questions / honest limits
- No ablation yet: the share of the gain due to the pre-filled buffer, the shaping, the history or the held exploration is not measured (STATUS.md §3 item 7).
- Training has plateaued at ≈ +117 (training measure) since step 360k at lr 1e-4; a lower-lr continuation is the first thing to try (STATUS.md §3 item 1).
- The contest tests were made mostly alone in the shared world; with many participants the `r = +5` sources may be shared and the score may drop.
