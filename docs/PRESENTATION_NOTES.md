# Presentation notes (6 slides)

Deck: https://claude.ai/artifact/5t4Hrnvd59bc515E6PDWsK (the same text is in each slide's speaker notes).
About 1 minute per slide. Numbers are reward per 1,000 steps per agent = 5 × (count of r = +5 − count of died).

---

## 1. From exploration to Soft Actor-Critic

Hello, I'm Prakhar. My project is an agent for the multi-agent artificial-life world hosted on the course server. The agent only sees eleven numbers, o0 to o10, and each step it picks one of four actions: the combinations of two on/off switches. The reward is only ever +5, 0 or −5, and every −5 comes with the flag died, after which the agent respawns. So the whole score is simply five times the number of +5 rewards minus the number of deaths, per thousand steps; random play loses about 14 to 18. The rules: the policy network may have at most 128 hidden units, the contest is a shared world that advances every five seconds with other participants, and we get a private four-agent training world. I'll tell it in the order it happened: understanding the world, choosing learners, building and hardening the agent I submitted, and finally a method from the lectures, Soft Actor-Critic, that turned out even better.

## 2. Exploring the world

Before training anything, I collected data on the training server: random actions, random actions held for several steps, and each action held constant. Every naive policy loses, from −18 to −115. Then I asked: which numbers announce the two events, and how early? The chart shows the mean of o9 and o1 in the steps before them. o9 climbs from 0.13 to 0.91 before r = +5, and o1 from 0.43 to 0.96 before died. Both are readable 10 to 20 steps in advance. Small probe networks confirm it: o9 alone predicts r = +5 within ten steps with an AUC of 0.935, and all eleven inputs only add two hundredths. On the right, scrambling one input at a time shows the trained policy relies on o9, o10, o7, o1 and o8 for 94 percent of its decisions. That 10-to-20-step lead decided the design: four past observations as input, so the agent sees whether o9 is rising or falling; 5-step returns; a discount of 0.99; and a potential-based shaping term built from these same variables, which speeds up learning without changing which policy is best.

## 3. PPO first, then DDQN

Next, I looked for a starting point. Hand-written rules, offline Q-learning on stored data and a learned world model with planning all reduced deaths, but almost never got r = +5: their data came from other policies, so their estimates were wrong in the situations a new policy reaches. Only PPO, which learns by playing, found the +5 rewards. I chose PPO first because it works with any action space, which the challenge requires, and its clipped updates are very stable. DDQN was my second candidate: with only four joint actions it can score every action in every state, and its replay buffer reuses each transition about 128 times, starting from a million recorded ones. The chart shows the training curves. PPO here is one model trained in four stages. DDQN passed PPO at about 150 thousand steps and kept rising to a plateau near +117, while PPO flattened around +40 to +45 even after 600 thousand steps. So I switched to DDQN and trained it to 500 thousand steps, with the settings at the bottom. Note that these training scores include exploration; played greedily, both score higher.

## 4. The agent we submitted

In the contest world, played frozen and greedy, DDQN went from about +90 at 208 thousand steps to +149 at 500 thousand: three times PPO's best. The behaviour explains why. When o9 is high and starts to fall, which means the agent is passing the source, DDQN turns back with action [1,0] 62 percent of the time; PPO almost never did. So the gap between two +5 rewards fell from 99 steps to 33. Then, on 20 October, the contest got crowded. The same policy fell to +35, while it still scored +135 in the training world. So the rules had not changed: other agents reached the sources first, and the situations that come before died became 3.5 times more frequent. I made two changes and measured both live. First, training with co-players that play other policies, which cut deaths from 4.2 to 2.6 per thousand. Second, an anti-stall rule: greedy play could freeze on [0,0], because standing still keeps the observation identical, so it keeps choosing [0,0]. The submitted file, with both, scored +39.4 over about 125 thousand live steps. Fine-tuning on live data and learning online during the contest did not help.

## 5. Discrete Soft Actor-Critic

While going back over lecture deck 2, I looked again at Soft Actor-Critic. It is off-policy like DDQN, so it learns from a replay buffer, but it is an actor-critic with an entropy bonus that keeps the policy exploring. Two things made it attractive here. First, the 128-unit limit applies only to the policy, so SAC can use large critics, two of 256 plus 256, while the actor stays at 64 plus 64. In DDQN the Q-network is the policy, so it has to stay small. Second, it keeps replay and the pre-filled buffer that made DDQN strong. Our actions are four discrete combinations, so I used the discrete version, which sums over the four actions exactly. Everything else, inputs, shaping, 5-step returns and the buffer, is the same as DDQN. The orange curve: +41 after only 20 thousand steps, +129 at 184 thousand, already above DDQN's plateau, and about +140 after 400 thousand. Played greedily in the training world it scores +145.8, against +134.5 for DDQN 500k. Trained purely offline on DDQN's own buffer it still reaches +139, because that data comes from a good policy. The disadvantages: more networks and a sensitive entropy target, sampled play is worse than greedy, it only handles discrete actions, and the conservative CQL variant collapsed towards standing still. To be honest about the limits: these evaluations are 6,000-step runs, and SAC has not yet been played in the contest world.

## 6. Reflection and learnings

To finish, a look back at my main decisions. Exploring first was the best decision: the 10-to-20-step lead of o9 and o1 decided the input history, the 5-step returns, the discount and the shaping, and every later model used them. The switch from PPO to DDQN gave the +149 contest result, but I made it on one seed with different exploration in each method, and it moved me away from actor-critic methods altogether. SAC, an off-policy actor-critic, later beat both, so next time I would compare several seeds under equal conditions. I also tested mostly in a quiet contest world, so the crowd on 20 October surprised me; training against other policies from the start would have prepared the agent. Early on I picked checkpoints by their training score, and the contest disagreed. And I never ran ablations, so I cannot say exactly which ingredient made DDQN work. The four lessons at the bottom: read the data first; learn by playing; reuse that experience through replay, which mattered more than choosing between policy gradients and Q-learning; and offline learning only works when the data comes from a good policy. Thank you, I'm happy to take questions.

---

# PPO, DDQN and SAC compared (for questions)

| | PPO | DDQN | SAC (discrete here) |
|---|---|---|---|
| Family | policy gradient, actor-critic | value-based (Q-learning) | actor-critic, value-based critic |
| What it learns | a policy π(a\|s) and a value V(s) | Q(s, a) for every action; acts by argmax | a policy π(a\|s) and two Q(s, a) critics |
| Data | **on-policy**: fresh data from the current policy, used a few epochs, then thrown away | **off-policy**: replay buffer, each transition reused many times | **off-policy**: replay buffer |
| Exploration | sampling from π, entropy bonus | ε-greedy (here: random action held 1–8 steps) | sampling from π; α tunes the entropy automatically |
| Action spaces | any (discrete, multi-binary, continuous) | discrete only | continuous (original), discrete (SAC-Discrete) |
| Key stabiliser | clipped probability ratio | target network, Double-Q target | twin critics (min), soft target updates, entropy |
| Typical use | robotics/games with cheap simulators, LLM fine-tuning (RLHF), when stability matters more than samples | small discrete action sets, Atari-style games, when data is expensive | continuous control (robot arms, locomotion), when samples are expensive |
| Main limitations | sample-hungry; forgets old data; sensitive to advantage normalisation and batch size | only discrete actions; over-estimation; can be unstable without tricks; greedy policy can loop | many parts (actor, 2 critics, 2 targets, α); entropy target sensitive; discrete version less mature |
| In this project | +50 contest (best), +40–45 training | +149 quiet contest, plateau ≈ +117 training (with ε) | ≈ +140 training (sampled), +145.8 greedy; not tested in the contest |

Points that often come up:
- **On-policy vs off-policy** is the main split. PPO must collect new data after every update; DDQN and SAC learn from a memory of old data, which is why they could start from 1M recorded transitions and learned much faster here.
- **Why DDQN and not plain DQN:** DQN takes max over its own noisy estimates, so it over-estimates. Double DQN lets the online network choose the action and the target network score it.
- **Why SAC can be bigger here:** only the policy must have ≤ 128 hidden units. In DDQN the Q-network is the policy; in PPO and SAC the critic is a separate network used only for learning.
- **Greedy vs sampled:** value-based policies (DDQN) are deterministic; actor-critic policies (PPO, SAC) are stochastic. For PPO sampling played better (+3.8 vs +0.2); for SAC greedy played better (+139 vs +125). A deterministic policy can get stuck in a loop under partial observability, which is what the anti-stall rule fixes.
- **Entropy (SAC):** the objective is reward + α × entropy of the policy. α is adjusted so that the entropy stays near a target (here 0.35 × ln 4). Too high a target: the agent stays random; too low: it stops exploring.
- **Offline RL:** learning only from stored data works when the data comes from a good policy (offline SAC on DDQN's buffer: +139) and fails when it doesn't (offline Q on random data: −3). CQL makes Q-values of rare actions pessimistic; here it flattened the policy and greedy play chose [0,0] 62% of the time.
