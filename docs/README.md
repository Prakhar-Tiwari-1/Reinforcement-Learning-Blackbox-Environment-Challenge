# Reading guide

In the order the work happened:

1. **[CHALLENGE_REPORT.md](CHALLENGE_REPORT.md)**: round 1, the challenge, first exploration of the world, first PPO run.
2. **[ARCHITECTURES.md](ARCHITECTURES.md)**: the candidate architectures explained simply.
3. **[EXPLORATION_REPORT_2.md](EXPLORATION_REPORT_2.md)**: round 2, statistics of the world (what announces
   `r = +5` and `died`, and how early), and the starting points tried (rules, offline Q, world model, PPO stages).
4. **[V2_PLAN.md](V2_PLAN.md)**: the 128-unit-policy rule, which observations announce `r = +5` / `died`,
   potential-based shaping, observation importance, the behaviour around `o8` drops, crash-proof jobs.
5. **[FINAL_AGENT.md](FINAL_AGENT.md)**: the submission file's generic design (any space, PPO learner, robust contest
   loop, generality tests).
6. **[DDQN.md](DDQN.md)**: the Double DQN that replaced PPO (+148 in the quiet contest): network, training, why it
   beat PPO.
7. **[STATUS.md](STATUS.md)**: hand-off snapshot of 17 Oct (state before the contest: results, learnings, next steps).
8. **[EXPERIMENTS_17OCT.md](EXPERIMENTS_17OCT.md)**: attempts to improve the 500k DDQN before the contest
   (fine-tune, weight average, 8-observation history + 10-step returns).
9. **[CONTEST_LIVE.md](CONTEST_LIVE.md)**: the crowded contest (20–21 Oct): why the score fell, co-player training,
   freezing and the anti-stall rule, online learning, the final choice.
10. **[SAC.md](SAC.md)**: Discrete Soft Actor-Critic, trained after the contest.
11. **[PRESENTATION_NOTES.md](PRESENTATION_NOTES.md)**: the 6-slide summary of the whole project, plus a PPO / DDQN /
    SAC comparison.

Saved policies: [`../models/`](../models/) (see its README). Raw outputs: [`../results/`](../results/).

Reports 1–8 were written during the project and describe the state at that time. Some checkpoints, scripts and
logs they mention (e.g. intermediate checkpoints, `MORNING_SUMMARY.md`, older submission files) are not in this
repository; the numbers they report stand. Recordings (`data/`) and training runs (`runs/`) are not included (size).
