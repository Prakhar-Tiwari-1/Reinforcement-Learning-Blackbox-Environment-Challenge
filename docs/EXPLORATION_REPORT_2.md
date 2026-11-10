# Exploration round 2: understanding the world and finding a starting point

> **Status update (17 Oct 2026):** the contest policy embedded in `prakhar.tiwari.py` is now a **Double DQN** (+141 per 1,000 in the contest world vs +51 for the PPO described here). See [STATUS.md](STATUS.md) and [DDQN.md](DDQN.md). The generic parts below (spaces, contest loop, PPO learner for other environments) are still valid.


*Written overnight, 15–16 October 2026. It follows [CHALLENGE_REPORT.md](CHALLENGE_REPORT.md) (round 1). The final agent is described in [FINAL_AGENT.md](FINAL_AGENT.md).*

## TL;DR

1. **Rewards are events:** +5 for "food" and −5 for "death". An agent **breaks even when it eats once per life**. Random play eats 0.16–0.21 times per life; our best agent eats **about 6 times per life**.
2. **There are two ways to die.**
   - Careless agents (random play) die mostly from **contact with something shown in one colour channel** ("danger").
   - Careful agents (trained PPO) die almost only when **their health (obs 8) runs out**.
   - Health goes **up** while the agent touches something shown in another colour channel ("healer"). It does **not** go up when the agent eats.
3. **Survival is easy to get; foraging is hard.**
   - Every method we tried learned to survive: hand-written rules, offline Q-learning, and a world model with planning (2–7× fewer deaths than random).
   - **Only PPO, learning by playing, learned to find food reliably.**
4. **Training longer is by far the biggest lever.** PPO went from about +5 to **+28**, then **+36**, then (after gentle fine-tuning) **+48 reward per 1,000 steps** in the training world. In the contest world the final agent scores **about +51**, against **−14** for random play.
5. **The training world does not drift by itself.** Over 30,000 steps the rates are flat and the entity counts are constant. Drift in the contest can only come from other players.
6. **The starting point for the final agent was the continued-PPO checkpoint** (`data/ppo_cont.pt`). It was trained a further 500,000 steps overnight; the result (`data/sub_fine.pt`) is embedded in the submission file.

---

## 0. Facts vs labels

The agent sees 11 anonymous numbers and a reward. To make the report readable we gave names to things. This table says what is **measured** and what is **our interpretation**.

| Name used here | What it really is | Status |
|---|---|---|
| **death** | `infos[agent]["died"] == True`, sent by the server. Coincides with a −5 reward (364/365). | **Fact** |
| **food** | a +5 reward on a step where the agent did not die | **Label** (the +5 is a fact; "food" is our interpretation) |
| **health** (obs 8) | falls fast before every death, resets to 0.5 after it, rises in jumps during certain contacts | Hypothesis, strong evidence |
| **contact / proximity** (obs 1) | ≈ 1 whenever something touches the agent; ≈ 0.98 before deaths | Hypothesis |
| **danger colour** (obs 4 / 7) | binary inputs that are on when health falls; death rate ×10 when on | Hypothesis, strong evidence |
| **healer colour** (obs 3 / 6) | binary inputs that are on 61–67% of the time when health rises | Hypothesis, good evidence |
| **food signal** (obs 9) | rises to ≈ 0.9 before a +5, drops after it | Hypothesis, strong evidence |
| **speed** (obs 10) | directly controlled by the action (0 after `[0,0]`) | Hypothesis, strong evidence |
| **two motors** (actions) | `[0,1]` and `[1,0]` have mirror-image effects on left/right inputs; `[1,1]` = fastest; `[0,0]` = stop | Hypothesis |

None of these labels are used by the agent. They only serve to explain its behaviour.

---

## 1. What we collected

All data is in `data/` (numpy `.npz`: observations, actions, rewards, deaths for every agent and step). All runs are on the **training server** unless marked *contest*.

| Dataset | Policy | Size (agent-steps) | Purpose |
|---|---|---:|---|
| `random.npz` | uniform random | 80,000 | baseline, statistics |
| `sticky.npz` | random actions held 1–20 steps | 40,000 | longer movements |
| `const.npz` | each agent holds one fixed action | 20,000 | effect of each action |
| `long_random.npz` | uniform random, 30,000 steps | 120,000 | does the world change over time? |
| `eval_base_s11..13.npz` | PPO, greedy PPO, random, heuristic (one per agent) | 96,000 | multi-seed baselines |
| `eval_h1`, `eval_q1`, `eval_final_*`, `eval_round3_*`, `eval_mpc_*` | various policies | ~170,000 | policy comparisons |
| `eval_contest_*.npz` | single agent in the shared contest world | ~60,000 steps | does it transfer to the contest? |

**About 600,000 agent-steps in total.** Text outputs of every analysis are in `results/`. The most important are `deep_analysis_v1.txt` (random data), `deep_analysis_ppo.txt` (PPO agent), `stationarity_long_random.txt` and `summary_*.txt`.

---

## 2. How the world works (statistics)

### 2.1 The economics: one meal per life to break even

Every death costs 5 and every meal earns 5, so **an agent is profitable when it eats more than once per life on average.**

| Agent | Meals per life | Mean life (steps) | Reward / 1,000 steps |
|---|---:|---:|---:|
| Random | 0.16–0.21 | 116–218 | −14 to −26 |
| Rule-based (v1) | 0.45 | ~180 | −15 |
| PPO baseline | 1.1–1.5 | ~420 | +4 to +6 |
| **PPO continued** | **≈ 6** | **> 650** | **+28 (training), +38 (contest)** |

In PPO's lives, **the more it eats, the longer it lives** (correlation +0.86): lives with 4–5 meals last over 1,500 steps. Eating does not heal (see 2.3), so this probably reflects good, safe behaviour overall rather than food itself keeping the agent alive.

### 2.2 How agents die

What the agent sees in the step before death:

| | Random agent (1,193 deaths) | PPO agent (47 deaths) |
|---|---:|---:|
| Health (obs 8) in the bottom 5% of all values | 41% | **98%** |
| "Danger colour" (obs 4 or 7) on | **57%** (base rate 13%) | 8.5% (base rate 5%) |
| Neither | 17% | 2% |
| Median steps since last meal or respawn | 56 | 130 |

**Reading:** random agents run into danger. PPO learned to avoid danger, so it now dies when its health runs out.

**Risk factors** (deaths per 1,000 steps, random data):

| Condition | Low | High |
|---|---:|---:|
| Contact (obs 1) | 0.2 when < 0.39 | **20.9** when > 0.76 |
| Speed (obs 10) | 1.5 when < 0.08 | **26.1** when > 0.36 |
| Health (obs 8) | 1.0 when > 0.5 | **29.9** when < 0.32 |
| Danger colour (obs 4) | 4.7 when off | **48.7** when on |

### 2.3 Health (obs 8)

- Health **resets to 0.5** when an agent respawns.
- It **falls** while the agent is near the danger colour. Average change per step: −0.0009 when stopped, −0.0031 at full speed.
- It **rises in jumps of about +0.09** while the agent touches something with the **healer colour** (obs 3/6 on 61–67% of the time when health rises, against 8% when it is flat).
- **It does not change when the agent eats** (median change on meal steps: −0.0001, and still flat 120 steps later).
- So the world has **two resources**: food (reward) and healers (survival). The good PPO agent reaches health 0.95; random agents never exceed 0.5.

### 2.4 Food (obs 9)

| obs 9 | Chance of a meal on the next step |
|---|---:|
| < 0.5 | ≈ 0 |
| 0.50–0.75 | 1.9 per 1,000 |
| 0.75–0.90 | 11.1 per 1,000 |
| > 0.90 | **19.5 per 1,000** |

- With food close (obs 9 > 0.75), **you only eat if you move**: 3 per 1,000 when stopped, 20–22 per 1,000 for any moving action.
- After a meal obs 9 drops from 0.9 to 0.5: the food is consumed.
- Obs 9 tells *how close* food is but not *in which direction*. That is why hand-written rules could not steer toward food (section 3).

### 2.5 Actions

- `[0,0]` stops (speed → 0) and is safest, but never eats.
- `[1,1]` is the fastest and the deadliest when held: 25.8 deaths per 1,000 steps.
- `[0,1]` and `[1,0]` have mirror-image effects on the left/right inputs, consistent with **turning left / right with two motors**.
- In round-1 tests, turning with `[0,1]` when danger is on the *left* avoided danger better than the opposite mapping (6.2 against 9.2 deaths per 1,000).

### 2.6 Does the world change over time? No (in training)

Long random run of 30,000 steps × 4 agents, in windows of 3,000 steps:
- food per 1,000: 0.25–1.50, with no trend (first half 0.83, second half 0.68);
- deaths per 1,000: 4.4–6.5, with no trend (5.07 against 5.02);
- the entity counts in the global state (used for checking only) stay at exactly 16 / 5 / 16 / 4 by category the whole time, so food is replaced as it is eaten.

**Consequence:** in training mode, more training on the same world is useful; it will not go stale. In the contest, change can only come from the other players.

---

## 3. Looking for a starting point: what we tried

Goal: find the best initial behaviour to improve with the algorithms from [ARCHITECTURES.md](ARCHITECTURES.md).

### 3.1 Results in the training world

Reward / food / deaths are per 1,000 agent-steps. Runs are 6,000–8,000 steps, several seeds, 4 agents side by side in the same world.

| Approach | Idea | Runs | Reward | Food | Deaths | Meals / life |
|---|---|---:|---:|---:|---:|---:|
| Random | — | 4 | −25.5 ± 4.2 | 0.97 | 6.07 | 0.16 |
| Rule-based v1 | avoid danger, go forward to food, speed limit | 4 | −15.3 ± 1.4 | 2.47 | 5.53 | 0.45 |
| Offline Q (Monte-Carlo) | Q-network trained on stored data (GPU), act greedily | 1 | −3.3 | 0.17 | 0.83 | 0.20 |
| Offline Q (fitted Q-iteration) | same, with bootstrapped targets | 1 | −4.2 | 0.00 | 0.83 | 0.00 |
| PPO baseline, greedy | most likely action instead of sampling | 3 | +0.2 ± 0.4 | 1.92 | 1.88 | 1.02 |
| PPO baseline | 41k training steps | 5 | +3.8 ± 3.2 | 2.42 | 1.67 | 1.45 |
| *(final comparison: PPO continued, PPO seed 1, world-model MPC in training)* | | | *see section 3.3* | | | |

### 3.2 Results in the contest world

Single agent, mostly alone in the shared world (we were the only connected participant), 3,000–8,000 steps per run.

| Approach | Runs | Reward / 1,000 | Food | Deaths | Meals / life |
|---|---:|---:|---:|---:|---:|
| Random | 3 | −13.9 ± 5.4 | 0.78 | 3.56 | 0.22 |
| Rule-based v2 (with healing) | 4 | −0.8 to −3.3 | 0–0.33 | 0.17–1.0 | ≈ 0 |
| World model + MPC planning | 3 | +0.6 ± 3.8 | 0.78 | 0.67 | 1.17 |
| PPO baseline | 4 | +6.0 ± 14.0 | 2.08 | 0.87 | 2.38 |
| **PPO continued (80k more steps)** | 2 | **+38.3 ± 7.1** | **9.17** | 1.50 | **6.11** |
| Submission file, frozen weights | 2 | **+38.8, +31.9** | 9.3, 7.5 | 1.5, 1.1 | 6.2, 6.7 |
| Submission file, online learning (lr 1e-4) | 2 | +13.1, +15.6 | 3.4, 4.3 | 0.9, 1.1 | 3.9, 3.8 |

### 3.3 Final comparisons (overnight)

**Training world** (3 seeds × 8,000 steps unless noted; 4 policies side by side):

| Policy | Reward / 1,000 | Food | Deaths | Meals / life |
|---|---:|---:|---:|---:|
| **Fine-tuned (200k more, lr 1e-4, entropy 0.005)**: embedded | **+47.9 ± 3.2** | 11.46 | 1.88 | 6.1 |
| Long run, final (300k more steps) | +38.5 ± 4.7 / +35.6 ± 2.9 ¹ | 9.50 / 8.58 | 1.79 / 1.46 | 5.3 / 5.9 |
| PPO continued | +31.2 ± 3.3 | 8.12 | 1.88 | 4.3 |
| PPO seed 1 (2 seeds) | +26.9 ± 6.2 | 6.94 | 1.56 | 4.4 |
| PPO baseline | +6.7 ± 2.5 | 2.29 | 0.96 | 2.4 |
| World model + MPC (2 seeds × 6,000) | +1.2 ± 1.8 | 0.83 | 0.58 | 1.4 |
| Random | −19.8 ± 2.6 | 0.67 | 4.62 | 0.14 |

¹ two separate 3-seed evaluations.

**Contest world:** the fine-tuned checkpoint and the long-run snapshot at 190k steps score **about +50** (4–7 runs each), the best of everything we tried. Full table in [FINAL_AGENT.md](FINAL_AGENT.md#6-overnight-results).

**Behaviour of the best policy** (from `results/behaviour_profile.txt`):
- It uses `[0,1]` (one motor on: turning in arcs) 51–58% of the time, stops (`[0,0]`) 26–30%, goes full speed only 11–12%, and almost never uses the other turn (4–9%).
- Its health reaches 0.98 (90th percentile), where random play stays below 0.57.

It sweeps the area in arcs in one direction, slowing down when needed. This echoes the remark in deck 2 (slide 9) about an A2C agent that "learned to only turn in one direction".

### 3.4 What we learned from the starting-point search

1. **Hand-written rules can make the agent survive but not eat.**
   - Rule set v1 found food (2.5–3.7 per 1,000 steps) but kept dying.
   - Rule set v2 used the healing discovery and nearly stopped dying (0.17 per 1,000), but stood still on healers and never ate.
   - Food is only signalled by "how close", not "where", so the right turning behaviour is hard to write by hand.
2. **Offline learning from random data learns caution, not foraging.**
   - Our dataset has about 1,500 deaths but only about 400 meals, and almost none of them come from deliberate food-seeking.
   - So the Q-networks and the world model correctly learn "stand still or move slowly = safe" but cannot discover how to reach food.
   - The world model predicts events well (AUC 0.96 food, 0.98 death) but plans only 10 steps ahead, which is too short to find food.
3. **PPO, which learns by playing, is the only method that learned to forage.** Its improvement continued well past where we first stopped it:

   | PPO run | Steps | Mean reward over 2nd half |
   |---|---|---:|
   | Seed 0 from scratch | 41k | +5.3 |
   | Seed 1 from scratch | 41k | +18.1 |
   | Seed 0 continued | +82k | **+27.7** |

   There is large **seed-to-seed variance**: seed 0 spent its first 10k steps learning only caution. Training longer from the best checkpoint is the safest bet.
4. **Sampling beats the greedy action.** The PPO policy that samples its actions does better than the same network always taking its most likely action (+3.8 against +0.2). In a partially visible world some randomness helps the agent avoid getting stuck, which is a known effect for policies that only see part of the state.
5. **Learning online during the contest hurt**, at least with the training-mode settings (see [FINAL_AGENT.md](FINAL_AGENT.md)). The world is stationary while we are alone in it, so there is nothing to adapt to, and small noisy updates only damage a good policy.

### 3.5 About the GPU

The GPU (RTX 4000 Ada, shared with another job running at 100%) was used for everything that trains on **stored data**: the offline Q-networks (164,000 transitions, trained in 2–4 s), the world model (226,000 transitions), and the neural probes.

PPO itself trains **on the CPU** on purpose. Its speed is limited by the network round trip to the server (about 15 ms per step), not by computation. One update of a 128-unit network on 1,024 samples takes milliseconds on the CPU, and moving such tiny batches to the GPU would be slower.

---

## 4. Where to go from here

See the "Next steps" section of [FINAL_AGENT.md](FINAL_AGENT.md). In short:
1. Keep training the PPO checkpoint in training mode for as long as it keeps improving, and embed the best checkpoint in `prakhar.tiwari.py` before **Wednesday 21 October 23:59**.
2. Run the contest agent **frozen by default**, with an option to learn conservatively if the contest world turns out to be very different (many players).
3. Possible improvements to try next:
   - an **entropy-annealing** schedule (less randomness late in training);
   - **reward for healing** (only if justified: careful, this adds human bias);
   - **opponent-aware evaluation** in the training world, with some of our 4 agents playing other policies.
