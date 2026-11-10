# Four candidate architectures, explained simply

This file explains, in plain language, the four approaches we shortlisted for the challenge: two from the course slides and two from recent research. For each one it gives the idea, an everyday analogy, why it fits *this* world, its weak points, and how it fits within the **128 hidden units** budget.

The numbers quoted ("food", "deaths", "observation 9", …) come from our own exploration. See [CHALLENGE_REPORT.md](CHALLENGE_REPORT.md).

---

## First, three words we need

- **Policy**: the agent's "brain": it reads an observation (11 numbers) and decides an action (two on/off switches).
- **Value / critic**: a second small brain that *guesses how good the situation is* ("I'm about to be eaten", "food is right in front of me"). It does not act, it judges.
- **Model / world model**: a brain that *predicts what will happen next* if I take an action. You can use it to imagine the future before acting.

All four approaches below are built from these three pieces in different combinations.

---

## A. PPO with a shared, small network (from the slides, deck 2 + deck 1)

**Idea.** Proximal Policy Optimization is a trial-and-error method. The agent plays for a while, then looks back: actions that turned out *better than expected* are made more likely, and actions that turned out *worse* are made less likely. The "proximal" part is a safety rule: each update may only change the behaviour a little, so learning never collapses suddenly.

**Analogy.** A tennis player reviewing a video of their last set: "that shot worked, do more of it", but changing their technique only slightly each session so they don't forget how to play.

**Extras we add for this world.**
- **Parameter sharing:** in training mode we control 4 agents, and they all use the same brain, so it learns 4 times faster. This is the "independent learners" idea from the multi-agent slides (deck 1, slide 46). Research shows plain independent PPO is surprisingly strong in multi-agent games ([Yu et al., 2022](https://arxiv.org/abs/2103.01955)).
- **Optional short memory (a GRU, a small recurrent layer):** the world is only partly visible, and recurrent PPO is a strong default for such worlds ([Ni et al., 2022](https://proceedings.mlr.press/v162/ni22a/ni22a.pdf)). Our tests show memory helps only a little here, so it is optional.

**Why it fits.**
- It is the most robust "default" algorithm and works with any action space: Box, Discrete or MultiBinary just need a different output layer.
- It is the slides' recommended starting point (PPO: "relative simplicity of implementation, robust to hyper-parameter variations").

**Weak points.**
- It needs a lot of experience (it is *on-policy*, so old data is thrown away).
- It does not adapt by itself once training stops. If the world changes in the contest, a frozen PPO agent keeps its old habits.

**Budget.** Actor 64 + critic 64 = 128 units. With a GRU: GRU 32 + actor 48 + critic 48.

---

## B. Learned world model + planning (MPC) (from the slides, deck 3)

**Idea.** First learn a *simulator* of the world from experience: a small network that predicts "if I see this and do that, what will I see next, and what reward will I get?". Then, at every step, **imagine** a handful of short futures (for example "go straight for 5 steps", "turn left for 5 steps", …) inside that simulator, and play the first move of the best imagined future. This is **Model Predictive Control** (deck 3, slide 17), the approach of [Nagabandi et al., 2018](https://arxiv.org/abs/1708.02596).

**Analogy.** A chess player who thinks "if I move here, they move there, then I…" a few moves ahead, picks the best line, plays *one* move, then thinks again.

**Why it fits.**
- Our exploration found that the **reward is very predictable from the current observation**: a small network predicts food and deaths with an AUC of 0.98, where 1.0 is perfect and 0.5 is a coin flip. The world also changes smoothly from step to step. Those are exactly the conditions where a learned model is useful.
- Only 4 possible actions, so imagining every short plan is cheap (4 actions × a few steps), far below the 5-second budget.
- **Adapts fast to change.** If the world drifts, we only have to update the *simulator* from fresh data. The planner then automatically changes behaviour, with no need to re-learn a policy.

**Weak points.**
- Errors of the model add up when imagining far ahead, so the horizon must stay short, roughly 3–10 steps.
- Our tests show one action changes the observation only a little (except speed), so the planner needs several steps of look-ahead to see the difference between actions.
- Planning with continuous Box actions needs sampling (e.g. the Cross-Entropy Method from deck 3) instead of trying every action.

**Budget.** Dynamics + reward model with 64 + 64 units, or one shared network of 128 units with two output heads.

---

## C. Streaming actor-critic: Stream AC(λ) (new research, 2024)

**Idea.** Most deep RL methods learn in *batches* from stored memories. **Stream AC(λ)** ([Elsayed, Vasan & Mahmood, 2024](https://arxiv.org/abs/2410.14606)) learns from **every single step, immediately, and then forgets it**, like a living animal. A few stabilising tricks make this work with neural networks:
- observations and rewards are normalised on the fly;
- LayerNorm (a layer that rescales the hidden values) keeps the network stable;
- "eligibility traces" remember *which recent decisions* deserve credit when a reward finally arrives;
- a special optimiser prevents any single update from being too large.

**Analogy.** A street musician who adjusts their playing after every song, depending on the crowd, instead of studying recordings at home once a month.

**Why it fits.**
- The challenge is literally *"learning for the unknown"* with **drift**: participants come and go, and the organisers may add agents. A streaming learner **keeps learning during the contest itself** and adapts to whoever is in the world right now. The paper shows it recovering again and again when the environment's physics switch back and forth.
- It is tiny and cheap: one update per step is milliseconds of computation, far below the 5-second clock.
- The authors' reference implementation is about 150 lines ([GitHub](https://github.com/mohmdelsayed/streaming-drl)), so it is easy to understand and justify.
- Eligibility traces help with our problem that **an action's effect shows up several steps later**.

**Weak points.**
- Learning from one step at a time is noisier. It is best to **pre-train it in training mode** (or start from the PPO weights) and let it keep adapting in the contest.
- It is newer, with fewer tutorials than PPO.

**Budget.** Actor 64 + critic 64 = 128 units.

---

## D. PQN: simplified Q-learning with LayerNorm (new research, 2024)

**Idea.** Q-learning (DQN in deck 2) learns a score for **each action** in each situation ("turn left here = +1.2, go straight = −3.0") and simply picks the best one. Classic DQN needs a big replay memory and a second "target" network to stay stable. **PQN** ([Gallici et al., 2024](https://arxiv.org/abs/2407.04811)) showed that adding **LayerNorm** (plus a little regularisation) makes Q-learning stable *without* either of them, so it can learn online from fresh data, which is exactly what you want when the world changes.

**Analogy.** A food critic who keeps a score for each dish on the menu and always orders the top-rated one, but keeps tasting occasionally in case the chef changed.

**Why it fits.**
- Our action space is small and discrete (4 combinations), which is the natural home of Q-learning, and it **directly tells us which action is best** in each situation.
- Because there is no old replay memory, stale experience from a world that no longer exists does not mislead it. The slides list misleading replay as a key MARL problem (deck 1, slide 53, "Experience replay becomes problematic").
- It learns from the 4 training agents in parallel, exactly the "vectorised" setting PQN was designed for.

**Weak points.**
- It needs to enumerate actions, so continuous Box action spaces need extra work: discretise them, or switch to approach A or C.
- Q-learning can over-estimate values. LayerNorm reduces this but we must watch it.

**Budget.** One Q-network: 128 → 4 outputs, or 64 + 64 in two layers.

---

## Cross-cutting tricks (used with any of the four)

These come from the research on **loss of plasticity**: networks that train for a long time on a changing world slowly lose their ability to learn ([Dohare et al., Nature 2024](https://www.nature.com/articles/s41586-024-07711-7); [Abbas et al., 2023](https://arxiv.org/abs/2303.07507)).
- **Running observation normalisation.** Never trust the declared bounds: observation 9 is declared in [0, 1] but reaches −0.74.
- **LayerNorm** in every network, for stable learning under drift.
- **Continual backprop / "shrink & perturb":** occasionally refresh a few unused neurons so the agent can keep learning new tricks.
- **A generic input/output adapter:**
  - inputs: Box → normalised floats, Discrete → one-hot, MultiBinary → 0/1 floats;
  - outputs: Discrete → softmax, MultiBinary → one coin flip per bit, Box → Gaussian squashed into the bounds.

  This is what makes the *same agent code* work on any of the three space types, as the challenge requires.

---

## Side-by-side

| | A. PPO (shared) | B. World model + MPC | C. Stream AC(λ) | D. PQN |
|---|---|---|---|---|
| Source | Slides (deck 2, deck 1) | Slides (deck 3) | New (2024) | New (2024) |
| Learns from | batches of recent play | a learned simulator | every single step | fresh batches, no replay |
| Keeps learning in the contest | not by default | yes, easily (update the model) | **yes, by design** | yes, possible |
| Handles drift | weak | good | **best** | good |
| Works for Box / Discrete / MultiBinary | all | all (Box needs sampling) | all | Discrete-like (Box needs discretising) |
| Difficulty to implement | low (done: see report) | medium | medium | low–medium |
| Main risk | frozen after training | model errors over long horizons | noisy learning | value over-estimation |

**Our suggestion:** use **A (PPO)** as the trained starting point and the baseline to beat, because we already have it running. Make the contest agent **C (Stream AC(λ))** initialised from it, so it keeps adapting live. Build **B** in parallel as a strong alternative, because the reward is so predictable from what the agent sees. Keep **D** as the fallback if policy-gradient training is unstable.
