# The crowded contest (20–21 October 2026)

When the official contest opened, the score of our DDQN fell from about +148 to about +35 per 1,000 steps.
This report explains why, what we tried, and what the submitted agent does about it.

Units: reward per 1,000 steps = sum of `r` (∈ {−5, 0, +5}) per 1,000 steps of one agent = 5 × (count of `r = +5` −
count of `died`) per 1,000 steps. All contest numbers are frozen play (no learning) unless stated otherwise.
`o0 … o10` are the 11 observation values.

---

## 1. What changed

Same policy (DDQN 500k), recorded on 20 Oct (`results/live_vs_training_now.txt`, `results/live_shift.txt`):

| | Live contest, 20 Oct | Training world, 20 Oct | Contest, 16–17 Oct |
|---|---:|---:|---:|
| Reward / 1,000 | **+39** (first 5k steps) | +135 | +148 |
| `r = +5` / 1,000 | 12.0 | 30.2 | 31.6 |
| `died` / 1,000 | 4.4 | 3.3 | 2.0 |
| Steps with `o9 > 0.75` | 64.6% | 70.2% | – |
| P(`r = +5` within 10 steps \| `o9 > 0.85`) | **0.20** | 0.42 | – |
| Stretches with `o9 > 0.75` that contain `r = +5` | 62% | 92% | – |
| Steps with `o1 > 0.99` and `o4 > 0.5` | 4.9% | 1.4% | – |

**Facts.**
- The training world is unchanged (the same policy still scores +135), so the rules did not change.
- The agent is near high `o9` as often as before, but converts it into `r = +5` half as often.
- `died` doubled, and the situations that precede `died` are 3.5× more frequent.

**Most likely cause (hypothesis):** other participants now share the world; they reach the `r = +5` sources first and
are in contact more often. The policy was trained only with copies of itself and tested mostly alone.

## 2. Three options, measured live

Every candidate was played in the contest world against a reference, in alternating order, 10k-step runs, all
recorded (`src/train/live_ab.sh`, `src/contest_record.py`). Pooled results: `results/live_pool_20oct.txt`.

| Option | Idea | Best live result | Verdict |
|---|---|---|---|
| 1. Fine-tune on recorded live data (offline) | learn from the actual crowd | unconstrained: +16 to +28; anchored to the start weights: +31.7 | **worse** |
| 2. Co-player training (training world) | 2 of the 4 agents play other policies, drawn from a pool and changed every 5,000 steps | checkpoint 143k: **+39.3** over 150k steps (`died` 2.81 vs 4.20) | **better** |
| 3. Existing candidates | 8-observation history model, fine-tune 82k | +37.5 each (10k steps) | within noise |

**Why option 1 failed.** The recorded data came from greedy play only, so it never showed the outcome of other
actions. Offline Q-learning learned that high-`o9` moments convert badly in the crowd and stopped pursuing them:
`r = +5` fell to 4–7 per 1,000. Keeping the Q-values near the start weights (anchor penalty) removed the damage
but also most of the change.

**Why option 2 helped.** The learners met agents that take the same sources first (500k copies, older checkpoints,
PPO, random). Checkpoint 143k plays `[0,0]` twice as often as 500k, especially when `o1 > 0.9` (38% vs 19% of those
steps), and its `died` rate is one third lower (z > 5); its `r = +5` rate is the same.

## 3. Greedy play can freeze

Live recordings of good and bad checkpoints (`results/why_bad_checkpoints.txt`):

| Policy (live) | Reward / 1,000 | `[0,0]` share | Steps with unchanged observation | Longest `[0,0]` stretch |
|---|---:|---:|---:|---:|
| co-player 143k | +39.9 | 18% | 8.2% | 1,781 |
| DDQN 500k | +35.3 | 10% | 0.6% | 64 |
| co-player 204k | +22.8 | 43% | 32.4% | 1,874 |
| co-player 491k | +19.5 | 51% | 41.9% | 3,416 |
| phased run 266k | +8.8 | 49% | 38.4% | 1,492 |

Every frozen stretch has `o1 = 1`, usually `o4 ≈ 1` and `o6 ≈ 1`, `o10 = 0`, and action `[0,0]` 100% of the time.

**Mechanism (interpretation).**
- That situation precedes `died` in the data, so moving there has low Q-values.
- Standing still keeps the observation identical, so the greedy policy chooses `[0,0]` again.
- Its Q-value is learned from the value of the same situation, a circular estimate that training never corrects:
  1–2% random actions break the stillness after a few steps.
- Training scores cannot see this: checkpoint 491k scored +129 in training and +19.5 live.

**Anti-stall rule** (in `prakhar.tiwari.py`, `Agent.stall_k`): after 10 identical observations while the greedy action
is `[0,0]`, play the best moving action (by Q) for 5 steps. Live A/B, 3 × 10k steps each (`results/contest_ab_live8.log`):

| Policy | Without | With anti-stall | Gain |
|---|---:|---:|---:|
| co-player 143k | +44.0 | +45.0 | +1.0 |
| phased run final model | +38.0 | +43.7 | +5.7 |

With the rule, the share of steps with an unchanged observation fell from 8.2% to 1.2–1.3% for 143k.

## 4. A second co-player run (phased pool, heavier `died` penalty)

500k more steps from the first co-player run's final model; the co-player pool changed every 100k steps
(diverse → strong competitors → the learner's own past versions → other styles → everything), and in learning a
`died` counted −10 instead of −5, `r = +5` +6 instead of +5, with a doubled warning term `o1 × max(o4, o7)`
(`runs/coplay2_pools.json`, `src/train/ddqn.py --pool-schedule --died-weight --food-weight --w-danger`).

Training-world scores rose to +134–137, but no checkpoint beat co-player 143k live (+8.8 to +38.0). The heavier
`died` penalty plausibly made freezing in contact situations even more attractive (266k froze on 38% of steps).

## 5. Learning online in the contest

`src/train/ddqn_contest_online.py`: starts from co-player 143k, 3% random actions held 1–8 steps, anti-stall,
one Double-DQN update per live step at learning rate 2e-5 (half contest data, half the co-player training buffer),
and a safeguard that reverts to the best weights and halves the learning rate if the 20k-step live score falls
below +32.

Result over 156,000 live steps: **+33.9** per 1,000, one revert (window +30.5), no window above +38. It stayed
5–10 below frozen 143k, partly from the exploration it plays by design. Not used.

## 6. Final choice and final run

Submitted: **co-player 143k + anti-stall** (`prakhar.tiwari.py`). Final 4-hour live run on 21 Oct
(`results/contest_live.log`): **124,595 steps, +39.4 per 1,000** (1,383 × `r = +5`, 404 × `died`, 0 missed actions);
every 5,000-step checkpoint of the run was between +39.1 and +39.4.

## 7. Operations notes (course machine)

- **Logout:** an administrator disabled `loginctl` linger on 20 Oct, so `systemd --user` jobs stop at logout. Long
  jobs were run as detached processes instead (`setsid nohup scripts/bg_*.sh`).
  `KillUserProcesses=false` lets them survive logout.
- **Cleanup kills:** a school cleanup job (`/usr/sbin/tclean clean`, 00:33 and 06:33 on weekdays) stopped user
  processes. Jobs were written to resume from their last checkpoint.
- **One contest program per token:** every contest job holds `results/.contest.lock` (`flock`), so a second one
  cannot start by accident.
- **GPU-load bug (fixed 20 Oct):** earlier submission files embedded GPU tensors and could not load on a machine
  without a GPU. `prakhar.tiwari.py` now loads with `map_location="cpu"`, and `src/build_submission.py` stores CPU
  tensors. The old submission files are not kept in this repository.
