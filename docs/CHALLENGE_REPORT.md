# Multi-agent RL challenge: exploration report

*CSC_53439_EP. Status: exploration phase (week 2–3). Last update: 10 October 2026.*

**Contents**
1. The challenge
2. Objective
3. What we tried
4. What we observed
5. What this means for the agent
6. Next steps
7. How to reproduce
8. References

---

## 1. The challenge

| | |
|---|---|
| **World** | A multi-agent "artificial life" world (`alife/alife-v1`), hosted on the organisers' server, played through the PettingZoo parallel API with `pzclient`. |
| **What the agent sees** | 11 numbers, declared as `Box(0, 1, (11,))`. The world is partially observable: each agent sees only a small part of it. |
| **What the agent does** | 2 on/off switches (`MultiBinary(2)`), i.e. 4 possible actions: `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]`. |
| **Reward** | A number per step, which turns out to be `+5`, `−5` or `0` (section 4). |
| **Episode** | Never ends: no termination, and agents are never removed from the world. |
| **Training mode** | Our own private world with 4 agents (`agent_0` … `agent_3`). It waits for us, `seed` works, and `state()` is allowed. |
| **Contest mode** | One shared world. We control a single agent (`prakhar.tiwari@polytechnique.edu`). The world plays a step every **5 s** whether we answer or not, and other participants and organiser agents join and leave at any time. |
| **Constraints** | At most **128 hidden units** in total. Everything runs on our machine. The agent must work with Box, Discrete and MultiBinary spaces and must cope with **distribution drift**. |
| **Deliverable** | `prakhar.tiwari.py` uploaded on Moodle. The last version before **Wed 21 Oct 2026, 23:59** counts. Official contest: Mon 19 – Wed 21 Oct. |
| **Grading** | No fixed benchmark. The agent must be **clearly better than random**, and we must justify our choices and explain its performance. |

## 2. Objective

1. **Beat the random agent clearly and reliably.** Random play loses about **18 reward per 1,000 steps** (section 4.1).
2. **Generalise.** The same code must learn on an unknown world knowing only its observation and action spaces.
3. **Stay robust under drift.** The shared world changes as players come and go, so the agent should keep adapting during the contest.
4. **Stay within the 128-unit budget and the 5-second clock.**
5. **Be explainable.** We must be able to say *why* the agent behaves as it does. That is the purpose of this exploration.

## 3. What we tried

All the code is in [`src/explore/`](../src/explore/). All the data comes from the **training server**.

| Step | What | Script |
|---|---|---|
| 1 | Collected **20,000 steps × 4 agents** with uniformly random actions (80,000 agent-steps). | `collect.py --policy random` |
| 2 | Collected **10,000 steps × 4 agents** with "sticky" random actions, each held for 1–20 steps, to see longer, more coherent movement. | `collect.py --policy sticky` |
| 3 | Collected **5,000 steps** with each agent *holding one fixed action* (agent *i* always plays action *i*), to isolate what each action does. | `collect.py --policy const` |
| 4 | Statistical analysis: reward values, deaths, effect of each action, statistics of every observation dimension, what the agent sees just before rewards and deaths, lifetimes. | `analyze.py` |
| 5 | **Neural-network probes**, small MLPs with a chronological 80/20 train/test split: a *reward model* (obs, action → reward), a *dynamics model* (obs, action → next obs), and both again with 3 steps of history. | `nn_probes.py` |
| 6 | **First learning run**: a shared-parameter PPO agent (actor 64 + critic 64 = 128 units, LayerNorm, running observation normalisation, previous action as input), trained live on the training server. | `ppo_baseline.py` |
| 7 | Literature review plus the three course slide decks, giving four candidate architectures. | [ARCHITECTURES.md](ARCHITECTURES.md) |

## 4. What we observed

### 4.1 Rewards are sparse events: +5 for "food", −5 for "death"

- Only three reward values ever appear: **−5, 0, +5**. Across 80,000 random agent-steps: 364 × (−5), 79 × (+5), 79,557 × 0.
- **Every −5 coincides with `infos["died"] == True`** (364 of 365 deaths; the one exception got 0). So **−5 = dying**.
- **Every +5 happens on a non-death step**: a positive event we call "food".
- After dying, an agent reappears (respawns) and keeps playing. The "episode" never ends, only lives do.

**Baselines (reward per 1,000 agent-steps):**

| Policy | Reward / 1,000 | Food / 1,000 | Deaths / 1,000 | Data |
|---|---:|---:|---:|---|
| Uniform random | **−17.8** | 0.99 | 4.56 | 80,000 agent-steps |
| Sticky random (held 1–20 steps) | −56.6 | 2.93 | 14.3 | 40,000 agent-steps |
| Always `[0,0]` | −18.0 | 0.0 | 3.6 | 5,000 steps, 1 agent |
| Always `[0,1]` | −24.0 | 4.0 | 8.8 | 5,000 steps, 1 agent |
| Always `[1,0]` | −45.0 | 4.2 | 13.2 | 5,000 steps, 1 agent |
| Always `[1,1]` | −115.0 | 2.8 | 25.8 | 5,000 steps, 1 agent |
| **PPO baseline (steps 20k–41k, run 2)** | **+6.2** | 1.90 | 0.67 | 4 agents × 20k steps, see 4.5 |

*(The constant-action rows come from a single agent each, so they are noisy. Use them for direction, not precision.)*

**Takeaways.**
- **Every naive policy loses reward**, because deaths outnumber food by about 4.5 to 1.
- **Moving finds food but kills you.** The actions that move the agent (`[0,1]`, `[1,0]`) find food about 4× more often than random play, but die 2–3× more. `[1,1]` is by far the deadliest. Holding actions longer (sticky) also triples deaths.
- **Standing still (`[0,0]`) does not save you.** It still dies 3.6 times per 1,000 steps and never eats, which suggests something like starvation, or being caught.
- So the task is a **trade-off between foraging and survival**. A good agent must move *towards* food and *away* from danger, which requires actually using what it sees.

### 4.2 What the 11 observation numbers look like

From the data alone, without the world's source code (see the note in 4.6).

| Dim | Type | Behaviour | Hypothesis (data-driven) |
|---|---|---|---|
| 0 | binary, very rare (0.3%) | jumps to 1 right after a death | "just respawned" or some other rare event flag |
| 1 | continuous [0, 1] | very smooth (lag-1 autocorrelation 0.99); **high just before death** | an intensity of something *dangerous* nearby |
| 2, 5 | binary | rare (≈1%), correlated pair | one colour channel, left and right "eye" |
| 3, 6 | binary | ≈5%, correlated pair (r = 0.39) | a second colour channel, left / right |
| 4, 7 | binary | ≈10%, correlated pair (r = 0.60); **≈ +0.5 above average just before death** | a third colour channel, left / right: **the colour of danger** |
| 8 | continuous, never 0 | smooth; **low before death** (−0.2) | a health / energy level |
| 9 | continuous, **can be negative (min −0.74)** although declared in [0, 1] | **≈ +0.78 above average one step before food**, drops after eating; decays over a life | a "food is close / in front" signal |
| 10 | 30 discrete levels in [0, 0.58] | **directly controlled by the action**: drops under `[0,0]`, rises under the others | the agent's **speed** |

The pairing (2,5), (3,6), (4,7) fits a **two-eye, three-colour vision**, plus internal state (8), a food signal (9) and speed (10). This is a hypothesis; the agent itself does not need it.

**Important for the implementation:** the declared bounds are wrong for dim 9, so the agent must **normalise observations from running statistics**, not from the declared Box.

### 4.3 Neural-network probes: what can be learned?

Small MLPs (64 hidden units, LayerNorm), trained on the first 80% of the random data and tested on the last 20% (chronological split).

**Reward model**, scored by AUC: the chance the model ranks a real event above a non-event, where 0.5 is chance and 1.0 is perfect.

| Input | AUC (food, +5) | AUC (death, −5) |
|---|---:|---:|
| observation + action | 0.986 | 0.975 |
| observation only | 0.980 | 0.978 |
| action only | 0.60 | 0.55 |

- **Rewards are almost perfectly predictable one step ahead from what the agent sees.** The world is not random: the agent can *see* food and danger coming.
- **Food is announced by dim 9.** Shuffling it drops the AUC by 0.33.
- **Death is announced by dims 1, 8, 10, 4 and 7**: danger intensity, health, speed and the "danger colour".
- **The action alone says little about the next reward.** A single action does not decide the outcome; *sequences* of actions do (steering over several steps).

**Dynamics model** (obs, action → next obs):
- The world is **smooth**: just copying the last observation already explains most of the next one (R² 0.6–0.98 depending on the dim).
- The action strongly affects **dim 10 (speed)**: adding it cuts that dim's error by 73–82%.
- The action affects the vision dims by only a few percent per step.
- So **actions steer the agent slowly**, and their effects build up over several steps. Good methods must handle **delayed credit**: a high discount factor, eligibility traces, or multi-step planning.

**Memory test** (adding the 3 previous observations):
- It gives almost no gain for predicting rewards.
- It gives a small gain for predicting speed.
- So the **current observation is nearly sufficient**. A recurrent network is a "nice to have", not a necessity. A small "previous action" input already captures most of the useful history.

### 4.4 Lifetimes

| Policy | Mean life | Median life |
|---|---:|---:|
| Random | 218 steps | 153 steps |
| Sticky random | 69 steps | 53 steps |

Deaths cluster when the agent moves fast without looking: dims 1, 4 and 7 are high just before dying.

### 4.5 First learning run: shared-parameter PPO

Setup: 4 agents sharing one network (actor 64 + critic 64 = 128 units), rollouts of 256 steps, γ = 0.99, GAE λ = 0.95, clip 0.2, entropy bonus 0.01, reward scaled by 0.2. Metrics are measured *during training* on the policy being trained (stochastic actions), per window of 2,048 steps × 4 agents.

| Training steps | Reward / 1,000 | Food / 1,000 | Deaths / 1,000 |
|---:|---:|---:|---:|
| 2,048 | −12.2 | 2.20 | 4.64 |
| 4,096 | −10.4 | 0.73 | 2.81 |
| 6,144 | −1.8 | 0.61 | 0.98 |
| 8,192 | +1.2 | 1.10 | 0.85 |
| 10,240 | +7.9 | 2.20 | 0.61 |
| 14,336 | +4.9 | 1.83 | 0.85 |
| 18,432 | +6.1 | 2.81 | 1.59 |
| 22,528 | +4.9 | 1.71 | 0.73 |

*(Run 1 was interrupted at 22,528 steps; the table above is run 1. Run 2, below, went to completion.)*

**Run 2 (full, 40,960 steps; checkpoint saved in `data/ppo_baseline.pt`):**

| Training steps | Reward / 1,000 | Food / 1,000 | Deaths / 1,000 |
|---:|---:|---:|---:|
| 2,048 | −20.8 | 0.85 | 5.00 |
| 6,144 | +1.8 | 2.32 | 1.95 |
| 12,288 | +10.4 | 3.05 | 0.98 |
| 20,480 | +9.8 | 2.69 | 0.73 |
| 28,672 | +4.3 | 1.71 | 0.85 |
| 32,768 | +8.5 | 1.83 | 0.12 |
| 36,864 | +10.4 | 3.05 | 0.98 |
| 40,704 | +4.9 | 1.40 | 0.42 |
| **Mean, steps 20k–41k** | **+6.2** | **1.90** | **0.67** |

Run 2 reproduces run 1: positive after about 6,000 steps, then a plateau around **+6 per 1,000 steps**. Deaths are **≈ 7× lower than random**, and food is **≈ 2× higher**. The window-to-window noise (+0.6 to +10.4) reflects how rare the events are: about 15–30 per window. Learning has **plateaued but not converged**. Food stays well below what the moving policies reach (4 per 1,000), so there is room left.

**What happened:**
1. **Within about 6,000 steps the agent learned to survive.** Deaths dropped from 4.6 to under 1 per 1,000 steps, a **5–7× reduction**.
2. **Then it slowly learned to eat while staying alive.** Food per 1,000 steps went back up to about 2, twice the random rate, while deaths stayed low.
3. **Net result: from −17.8 (random) to about +6 per 1,000 steps.** That is a swing of about 24 reward per 1,000 steps, and **the first policy with a positive score**.
4. Its action probabilities settled around P(bit 1) ≈ 0.4, P(bit 2) ≈ 0.25–0.35, i.e. it moves less often and more selectively than random.

**Expectations from here.**
- The probes show the danger and the food are clearly visible to the agent. Better learners should push food higher, at least the 4 per 1,000 that the moving policies reach, while keeping deaths below 1 per 1,000.
- **A realistic target is +10 to +20 per 1,000 steps** in training mode. The contest world will be different (more agents competing for food, unknown organiser agents), so absolute numbers there will differ.

### 4.6 Notes and caveats

- **The global state.** In training mode `env.state()` returns a 200 × 36 table, of which 41 rows are used. Column 0 takes the values −1 (16 entities), 1 (5), 2 (16) and 3 (4, probably our 4 agents). We used this **only to sanity-check our understanding**. It is forbidden in the contest, and the agent never uses it.
- **The world's source code is public.** A web search shows the world is the course's ALife environment ([github.com/jmread/alife](https://github.com/jmread/alife)), by the lecturer. The challenge asks us to treat the world as a black box, so **we deliberately did not read its source code**. Everything above comes from data. Ask the organisers before using it.
- **Single runs.** The constant-action and PPO results come from single runs. The slides recommend multiple seeds (deck 1, slide 65) before drawing firm conclusions.

## 5. What this means for the agent

| Observation | Design consequence |
|---|---|
| Rewards are rare (≈ 5 events per 1,000 agent-steps) | Scale rewards, use a high discount (γ ≈ 0.99) and advantage estimation or eligibility traces to propagate them. |
| Food and danger are visible one step before they happen | A small feed-forward network is enough to "see" them; the 128-unit budget is not a real limitation here. |
| An action's effect on the world builds up over several steps | Use credit assignment over time (GAE, λ-traces) or multi-step planning, not one-step greedy decisions. |
| History adds little | Feed-forward network + previous action. A recurrent layer is optional. |
| Declared observation bounds are wrong | Running mean/std normalisation, with clipping. |
| The contest world drifts (players come and go) | Keep learning online during the contest (streaming actor-critic, or periodic updates). Use LayerNorm and plasticity-preserving tricks. |
| Spaces may be Box / Discrete / MultiBinary | Generic input encoder and generic output head (see [ARCHITECTURES.md](ARCHITECTURES.md)). |
| The contest clock is 5 s | Our network's forward pass takes well under 1 ms; the network round trip dominates. An online update per step is affordable. |

**Shortlisted architectures** (details in [ARCHITECTURES.md](ARCHITECTURES.md)):
- **A. Shared-parameter PPO** (slides). Already working: the baseline to beat and the source of pre-trained weights.
- **B. Learned world model + MPC planning** (slides, deck 3). Promising because rewards are so predictable from observations.
- **C. Stream AC(λ)** (Elsayed et al., 2024). Learns from every step with no replay buffer, so it can keep adapting during the contest.
- **D. PQN, Q-learning with LayerNorm** (Gallici et al., 2024). Natural for 4 discrete actions, and needs no replay buffer to go stale.

## 6. Next steps

1. **Finish the PPO baseline properly.** Run 3 seeds and a longer run (about 100k steps), and **evaluate the frozen policy** separately from training.
2. **Build the generic agent wrapper**: an observation encoder and action head for Box / Discrete / MultiBinary. Test it on local Gymnasium toy environments (CartPole = Discrete, Pendulum = Box) to prove the "any space" requirement.
3. **Implement Stream AC(λ)**, warm-started from the PPO weights, and compare it with frozen PPO, especially when the world changes (e.g. after a `reset()` or when other agents behave differently).
4. **Prototype the world model + MPC** from the data we already have (`data/*.npz`), and compare it in training mode.
5. **Test in contest mode** before 19 October: check that `action_applied` stays `True`, that reconnection works, and how the score changes with other players in the world.
6. Write the final `prakhar.tiwari.py`: self-contained, token from the environment, robust contest loop.

## 7. How to reproduce

From the project root (run one at a time: only one program per token on the training server):

```bash
uv run python src/explore/collect.py --policy random --steps 20000 --seed 1
uv run python src/explore/collect.py --policy sticky --steps 10000 --seed 2
uv run python src/explore/collect.py --policy const  --steps 5000  --seed 3
uv run python src/explore/analyze.py data/random.npz
uv run python src/explore/nn_probes.py data/random.npz
uv run python src/explore/ppo_baseline.py --steps 40960
```

Raw outputs are saved in `data/` (`analysis_*.txt`, `probes_random.txt`, `ppo_baseline*.log`).

## 8. References

**Course slides (`slides/`):**
1. J. Read, *Introduction to Deep RL and Multi-Agent Systems* (deck 1): MARL settings, non-stationarity, independent learners, Deep RL in practice.
2. J. Read, *Deep Policy Gradient and Actor Critic* (deck 2): PPO, DQN, DDPG, SAC.
3. J. Read, *Deep Model-based RL: From MCTS to AlphaZero and Beyond* (deck 3): planning, MPC, world models.

**Research:**
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022. [arXiv:2103.01955](https://arxiv.org/abs/2103.01955)
- Ni, Eysenbach & Salakhutdinov, *Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs*, ICML 2022. [PMLR](https://proceedings.mlr.press/v162/ni22a/ni22a.pdf)
- Nagabandi et al., *Neural Network Dynamics for Model-Based Deep RL with Model-Free Fine-Tuning*, ICRA 2018. [arXiv:1708.02596](https://arxiv.org/abs/1708.02596)
- Elsayed, Vasan & Mahmood, *Streaming Deep Reinforcement Learning Finally Works*, 2024. [arXiv:2410.14606](https://arxiv.org/abs/2410.14606), [code](https://github.com/mohmdelsayed/streaming-drl)
- Gallici et al., *Simplifying Deep Temporal Difference Learning* (PQN), 2024. [arXiv:2407.04811](https://arxiv.org/abs/2407.04811)
- Dohare et al., *Loss of plasticity in deep continual learning*, Nature 2024. [link](https://www.nature.com/articles/s41586-024-07711-7)
- Abbas et al., *Loss of Plasticity in Continual Deep Reinforcement Learning*, CoLLAs 2023. [arXiv:2303.07507](https://arxiv.org/abs/2303.07507)
