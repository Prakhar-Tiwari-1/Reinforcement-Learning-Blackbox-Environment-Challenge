# Project status and hand-off (17 October 2026, 00:30)

> **Update 17 Oct 04:45:** the submission now embeds **500k** (`models/FINAL_ddqn_500k.pt`, +148.0 over 220k contest steps). Fine-tuning, weight averaging and an 8-observation history + 10-step returns were tried; none beat 500k on reward. The 8-observation history cut `died` from 2.01 to 0.87 per 1,000 but also lowered `r = +5`. Details: [EXPERIMENTS_17OCT.md](EXPERIMENTS_17OCT.md).

**Read this first in a new session.** It says where everything stands, what we learned, and what to do next.
Quantities are always given as measured: the reward `r` ∈ {−5, 0, +5}, the flag `infos[agent]["died"]`, and the observation components o0…o10. "Reward per 1,000" = sum of `r` per agent over 1,000 steps.

---

## 1. Where we are

| | |
|---|---|
| **Submission** | `prakhar.tiwari.py` (68 KB, single file, token read from `PETTINGZOO_TOKEN`). **Not yet uploaded to Moodle; do it.** |
| **Policy in the submission** | DDQN checkpoint `models/FINAL_ddqn_500k.pt` (end of training, step 500,000; dueling Q-network 46 → 64 → 64 → 4, **128 hidden units**). Backup choice: `models/BACKUP_ddqn_450k.pt`. |
| **Contest-world score** | **+149.0** per 1,000 over 30,000 steps (runs +146.5 over 20k and +154.0 over 10k; 31.6 × `r = +5` and 1.8 × `died` per 1,000; 0 missed steps). 450k backup: +147.0 (10k). Previous embedded 410k: +139.8 in the same paired tests. |
| **Random actions** | −14 (contest), −18 (training) |
| **Previous best (PPO)** | ≈ +50 (contest); backup in `models/ppo_v1_fine.pt` |
| **Jobs running** | none (DDQN run finished at 23:29, 500,000 steps). Check with `scripts/rl.sh status`. |
| **Contest dates** | official contest Mon 19 Oct 00:00 → Wed 21 Oct 23:59 (Paris). Last upload before the end counts. |

### Score history (contest world, frozen policy)
| Policy | Reward / 1,000 | `r = +5` / 1,000 | `died` / 1,000 |
|---|---:|---:|---:|
| random | −14 | 0.8 | 3.6 |
| PPO v1 baseline (41k steps) | +6 | 2.1 | 0.9 |
| PPO v1 continued / long / fine-tuned | +37 / +47 / **+51** | ≈ 11 | ≈ 1.2 |
| DDQN at 208k steps | ≈ +90 (5 runs, +45 to +118) | ≈ 20 | ≈ 2 |
| DDQN at 389k steps (best training score) | +121.2 (20k steps) | 25.6 | 1.3 |
| DDQN at 410k steps (embedded until 17 Oct) | +141.0 (20k) / +137.5 (10k) | 30.2 | 2.0 |
| DDQN at 450k steps (backup) | +147.0 (10k) | 31.1 | 1.7 |
| **DDQN at 500k steps (embedded, final)** | **+146.5 (20k) / +154.0 (10k)** | **31.6** | **1.8** |

---

## 2. Learnings

### About the world (measured; `results/` has the raw numbers)
1. **Only two reward values besides 0:** `r = +5` and `r = −5`. Over 1,233,150 recorded agent-steps, every `r = −5` (4,363) has `died = True`, and no `r < 0` ever happens with `died = False`. Changes in o8 never give a reward by themselves.
2. **Before `r = +5`:** o9 rises from ≈ 0.13 (far) → 0.60 (20 steps before) → 0.80 (10) → 0.91 (1). o9 alone predicts "`r = +5` within 10 steps" with AUC 0.93. `r = +5` only happens while moving (P = 0.003 per step with action `[0,0]` vs 0.020–0.022 with the other actions when o9 > 0.75). After `r = +5`, o9 stays > 0.7 in 55% of cases (another source is signalled).
3. **Before `died = True`:** o1 rises from ≈ 0.43 → 0.64 (20 steps before) → 0.89 (5) → 0.96 (1). o4/o7 = 1 more often (5% → 25–30%). o8 falls in steps of 0.05–0.10, and `died = True` follows when o8 reaches ≈ 0.05–0.09. After the first one-step drop of o8 (> 0.01), `died = True` follows within 40 steps 49% of the time.
4. **o8 never rises by itself.** It rises in jumps of ≈ +0.09 per step on steps where o3 or o6 = 1 (61–67% of rising steps). After `died`, o8 = 0.5, o10 = 0, o2–o7 = 0.
5. **o1 = 1 for long stretches without any o8 drop**, so o1 is "something is touching / very close", not specifically harmful. The harmful cases come with o4/o7 = 1 or with o8 falling.
6. **Useful inputs:** 4–5 of the 11. Predicting `r = +5`: o9 (0.935 alone) → + o10, o8 → 0.954 with all. Predicting `died` / o8 drop: o1 (0.853 alone) → + o10, o8, o7 → 0.915 with all. All 11 are still fed (cheap and generic).
7. **The training world is stationary:** 30,000 random steps show flat rates and constant entity counts. Change in the contest can only come from other participants.

### About methods
8. **Only methods that learn online from their own play learned to get `r = +5`.** Hand-written rules, offline Q-learning on stored data, a learned world model + planning, and the "best move" table all reduced `died` but almost never got `r = +5` (−15 to +12). The reason is covariate shift: estimates learned from one policy's states are wrong in the states a different policy reaches.
9. **DDQN beat PPO by about 3×** (+141 vs +51 contest). The likely reasons, in order:
   - off-policy reuse of a 1M-transition replay buffer pre-filled with all recordings, including 480k exploration transitions with deliberate random moves near o9 > 0.75 and o1/o4/o7 situations;
   - Q-learning compares all 4 actions directly;
   - 5-step returns + 4-observation history + shaping make the trend of o9 usable;
   - exploration that holds a random action for 1–8 steps.

   **No ablation has been run yet**, so the split between these is not measured.
10. **What DDQN does differently** (`results/ddqn_vs_ppo_behaviour.txt`):
    - when o9 > 0.7 and o9 fell by more than 0.1 in 3 steps, DDQN plays `[1,0]` 62% of the time; PPO played `[0,0]` 49% and `[0,1]` 48%, `[1,0]` 1%;
    - overall DDQN uses `[1,0]` 24% (PPO 5%), and at o10 > 0.35 it plays `[0,0]` 13% (PPO 42%).

    It turns back toward a source that it is passing, instead of leaving it. Mean gap between `r = +5`: ≈ 33 steps vs 99 for PPO.
11. **Choose checkpoints in the contest world, never by training score.** Examples: DDQN 389k was best in training but +121 in the contest vs +141 for 410k; PPO stage-3's final checkpoint was best in training but worse in the contest than its 190k snapshot.
12. **Greedy play is better for DDQN** (+141–150 contest vs ≈ +118 in training with 2% random actions); for PPO, sampling was better than greedy (+3.8 vs +0.2).
13. **Online learning during the contest never helped PPO** (frozen +37 vs online +14; conservative online +36). The submission plays frozen by default.
14. **PPO details:**
    - longer training helped: +5 → +28 → +36 → +48;
    - lowering the learning rate and the entropy bonus at the end added the last +12;
    - restarting a trained policy with a high learning rate and entropy bonus made it worse (v2: +36 vs its starting point +46).
15. **Engineering:**
    - jobs must run as `systemd --user` services with linger enabled, otherwise they die with the session (this happened once);
    - save atomically and resume from `latest.pt`;
    - the organisers' server sometimes times out: the DDQN run crashed once at 18:43 and was restarted by systemd 60 s later with no loss.

---

## 3. How to run things

All commands run from `/Data/prakhar.tiwari/Reinforcement_Learning_Challenge`. Jobs survive disconnects (`Linger=yes`).
```bash
scripts/rl.sh status                         # running jobs, last progress of every run, disk use
scripts/rl.sh logs ddqn                      # progress table of a run
scripts/rl.sh contest                        # PLAY THE CONTEST (frozen submission), auto-restarts if it stops
scripts/rl.sh follow contest                 # live output; 'missed' must stay 0
scripts/rl.sh stop contest
scripts/rl.sh job NAME src/train/ddqn.py --name NAME [options]   # a DDQN training run (training server)
scripts/rl.sh train NAME [options]           # a PPO training run (src/train/train.py)
scripts/rl.sh select NAME --embed            # contest-world test of a run's checkpoints, embed the winner (+3 margin)
scripts/rl.sh prune --yes                    # free disk space
.venv/bin/python prakhar.tiwari.py --no-learn --steps 20000 --load models/X.pt   # test one checkpoint (contest)
uv run python src/build_submission.py models/X.pt                                 # embed a checkpoint by hand
```

**Rules:**
- one program per token per server (training and contest servers are separate);
- never run `select` or contest tests while `scripts/rl.sh contest` is playing;
- from Monday, contest-world runs count officially.

---

## 4. What to do next (to go beyond +141)

| # | Action | Command / how | Expected value | Cost |
|---|---|---|---|---|
| 1 | **Upload `prakhar.tiwari.py` to Moodle** | Moodle | secures +141 | 2 min |
| 2 | **Fine-tune DDQN from the best checkpoint** with a smaller learning rate and less exploration (the step that gave PPO its last +12) | `scripts/rl.sh job ddqn_ft src/train/ddqn.py --name ddqn_ft --init models/FINAL_ddqn_500k.pt --lr 3e-5 --eps-start 0.02 --eps-end 0.005 --eps-steps 200000 --total-steps 300000`, then `scripts/rl.sh select ddqn_ft --embed` | most likely gain | ~5 h |
| 3 | **Longer n-step / more updates**: `--n-step 10` (matches the 10–15 step signal lead), `--updates-per-step 4` (the GPU has spare capacity) | new run from `--init models/FINAL_ddqn_500k.pt` | possible gain | ~5 h |
| 4 | **Ablations to justify the design**: DDQN without the pre-filled buffer (`--prefill false`), and without shaping (`--shaping false`) | from scratch, 300–500k steps each | justification for the report | 4–5 h each |
| 5 | **Robustness to other participants**: train with 2 of the 4 training agents playing other policies (random, PPO), so the policy does not only know copies of itself | needs a small change in `ddqn.py` (fixed policies for agents 2–3) | protects the contest score when the world is crowded | ~5 h |
| 6 | **Reduce `died`**: 39 `died` per 20k steps cost ≈ 10 per 1,000. Analyse the steps before `died` for the DDQN policy (record a contest run with `src/explore/evaluate.py`-style logging) | analysis first | depends | 1 h |

**Arithmetic toward +200:** reward = 5 × (`r = +5` count − `died` count) per 1,000 steps. +141 = 5 × (30.2 − 2.0). Reaching +200 needs ≈ 41 `r = +5` with ≈ 1 `died`, i.e. a mean gap between `r = +5` of ≈ 24 steps (now ≈ 33).

**During the contest (Mon–Wed):**
- run `scripts/rl.sh contest` and check `scripts/rl.sh follow contest` regularly;
- keep training on the training server in parallel;
- before embedding a new checkpoint, stop the contest player, run `select`, then restart the contest player.

---

## 5. Where things are

| Path | What |
|---|---|
| `prakhar.tiwari.py` | the submission (generic PPO agent for any space + embedded DDQN for the contest world) |
| `models/` | **the saved best weights** + README with their scores (keep this folder) |
| `runs/ddqn/` | the DDQN run (log.jsonl, best/step checkpoints, latest.pt, 190 MB replay buffer, selection log, submission backups) |
| `runs/v2/`, `runs/v2_init.pt` | PPO v2 run (stopped at 89k steps, plateau +36) and its distilled starting point |
| `data/` | all recordings (`*.npz`, ~1.2M agent-steps) incl. `data/moves/` (exploration of moves) and older checkpoints |
| `results/` | text outputs of every analysis, `eval_summary.jsonl` (all evaluations), job logs |
| `src/train/` | `ddqn.py`, `train.py` (PPO), `distill.py`, `choose_checkpoint.py`, `prune.py`, `explore_moves.py`, `move_values.py`, `move_policy.py`, `eval_agents.py` |
| `src/explore/` | round 1–2 analyses (statistics, probes, lead times, observation importance, plots) |
| `scripts/rl.sh` | the job launcher |
| `docs/` | reports; reading order in `docs/README.md` |
