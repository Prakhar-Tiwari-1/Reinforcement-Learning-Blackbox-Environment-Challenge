# Improvement attempts on the 500k DDQN (17 October 2026, 01:10 – 04:45)

**Short answer:** none of the four changes raised the contest-world reward above the 500k checkpoint. Over 220,000 contest steps, 500k averages **+148.0** per 1,000; the best variants reach +148.7 and +148.5, inside the noise.
One change had a clear, measurable effect on something else: with an **8-observation history**, `died` happened **2.3× less often** (0.87 vs 2.01 per 1,000). It also got slightly fewer `r = +5`, so the total stayed the same.
The submission still embeds 500k, and the protected files are unchanged (checked by checksum): `models/FINAL_ddqn_500k.pt`, `models/prakhar.tiwari_SUBMISSION_ddqn500k.py` and `prakhar.tiwari.py`.

Units: reward per 1,000 steps = sum of `r` (∈ {−5, 0, +5}) over 1,000 steps of one agent. All contest numbers are frozen greedy play in the contest world, in runs of 10,000 steps. Candidates alternate their order in each round, so a change in the world over time affects them equally.

---

## 1. What was tried and why

| # | Change | Why | How |
|---|---|---|---|
| A | **Weight averaging** of the checkpoints at 450k, 471k, 491k and 500k | Averaging nearby checkpoints of one run often generalises a little better. It costs nothing, and the network keeps its 128 units. | `data/ddqn_avg450_500.pt` (no training) |
| B | **Fine-tune** 500k with a lower learning rate (3e-5 instead of 1e-4) and less exploration (ε 0.02 → 0.005) | This is the step that gave PPO its last +12. | Run `runs/ft500`, 90k steps. Instead of the old recordings, it reuses the 1M transitions of the 500k run itself (new option `--buffer-from`). |
| C | **8-observation history** (input 46 → 90) and **10-step returns** (instead of 5) | The network saw only t…t−3, while o9 starts rising ≈ 20 steps before `r = +5` and o1 ≈ 20 steps before `died`. 10-step returns send a reward back 10 steps per update instead of 5. | Run `runs/h8n10`, 110k steps, started from B's 30k checkpoint. |

### How C keeps what was already learned
- **Zero weights for the new inputs.** The first layer gets zero weights for the 4 extra (older) frames. The widened network therefore computes **exactly** the same Q-values at the start: checked on 300 states, max difference 0.0. The submission file plays the widened checkpoint with identical actions on 500 test steps.
- **Rebuilt replay buffer.** The 500k run's 1M stored transitions were rebuilt into 8-frame states and 10-step returns (`src/train/widen_buffer.py`). In that buffer, entries x and x+4 are the same agent one step apart.
  - The script keeps an entry only if every step of its 56-step chain links up. 973,955 of the 1,000,000 transitions passed.
  - Check: the stored 5-step next state equals the entry 20 positions later in 100% of the kept entries.

## 2. Results in the contest world

All candidates are pooled from 7 head-to-head comparisons (`results/contest_compare_*.log`, table in `results/contest_pool_17oct.txt`):

| Checkpoint | Runs × 10k | Reward / 1,000 | sd between runs | `r = +5` / 1,000 | `died` / 1,000 |
|---|---:|---:|---:|---:|---:|
| B: fine-tune, step 30k | 12 | **+148.7** | 4.5 | 31.5 | 1.76 |
| C: history 8 + 10-step, step 110k (final) | 3 | **+148.5** | 2.3 | 30.6 | **0.87** |
| **500k (submission)** | 22 | **+148.0** | 8.0 | 31.6 | 2.01 |
| A: weight average 450k–500k | 4 | +146.4 | 2.8 | 30.8 | 1.50 |
| B: fine-tune, step 90k (final) | 3 | +145.3 | 7.3 | 31.0 | 1.93 |
| C: step 41k | 3 | +144.2 | 4.8 | 30.0 | 1.17 |
| B: step 51k | 3 | +132.5 | 20.1 | 28.3 | 1.80 |
| C: step 61k | 3 | +119.8 | 35.1 | 24.8 | 0.87 |

Training-world scores at the end of each run (mean of the last 10 windows of 4,096 steps, ε = 0.005–0.01, 4 agents): B **+134.1**, C +128.0. The original run's last stretch was ≈ +117 at ε = 0.02. As before, training scores did not predict the contest ranking.

## 3. What the numbers show

**Facts:**
1. **Reward: no improvement.** The spread of single 10k runs of the same checkpoint (sd ≈ 5–8) is as large as all the differences between the top 3. With 3–12 runs per candidate, a gap of 0.5–0.7 cannot be told apart from zero.
2. **`died` with the 8-observation history: a real reduction.** The final C checkpoint had 26 `died` in 30,000 steps. At 500k's rate (442 in 220,000 steps) about 60 would be expected. That gap is about 4.4 standard deviations (Poisson), so it is not noise. All C checkpoints had fewer `died` than 500k: 1.17, 0.87, 0.87 per 1,000.
3. **The cost of C's lower `died`: fewer `r = +5`.** C got 30.6 `r = +5` per 1,000 against 31.6 for 500k. Each `died` avoided gains 5, and each `r = +5` missed loses 5, so the total stays the same.
4. **Middle checkpoints can be worse.**
   - B at 51k and C at 61k each had one very low run: +109.5 (238 × `r = +5`), and +79.5 (164 × `r = +5`, 5 × `died`).
   - Both runs combine few `r = +5` with few `died`. That fits the policy spending long stretches without approaching any `r = +5` source. **Hypothesis**, not checked step by step.
   - This is another reason to choose by contest runs.
5. **Weight averaging (A) was not better.** It scored +146.4 against +148.0.

**Hypotheses (not tested):**
- The longer window lets the network see the o1 / o8 trend over 8 steps, so it moves away from situations that end in `died` earlier. The same caution also makes it skip some `r = +5` opportunities.
- 110k steps may be too short for the new inputs, which started at zero weight, to also improve `r = +5`. A longer C run could keep the low `died` rate and recover the `r = +5` rate. That would be worth about 5 × (2.01 − 0.87) ≈ +6 per 1,000.

## 4. Decision and next steps

- **Submission: unchanged (500k).** No candidate is better on reward, and 500k has by far the most evidence (220k contest steps).
- Candidates saved in `models/`:
  - `CAND_ddqn_hist8_n10_110k.pt`: low `died`; the submission file can load it (`--load`), since it reads the history length from the checkpoint;
  - `CAND_ddqn_ft500_30k.pt`.
- If there is time before Monday 00:00:
  1. **Continue the C run** for another 200–300k steps (`scripts/rl.sh job h8n10 src/train/ddqn.py --name h8n10 --total-steps 400000`; it resumes from `latest.pt`). Test it the same way.
  2. **Test C for longer.** It has only 3 runs. Six more 10k-step runs against 500k would show whether its +148.5 holds.
  3. **Robustness when other participants join:** still untested (see `docs/STATUS.md` §4 item 5).

## 5. Files

| Path | What |
|---|---|
| `src/train/ddqn.py` | new options: `--hist N` (history length), `--buffer-from FILE` (reuse a replay buffer), and automatic widening of `--init` weights when the history is longer. The defaults are unchanged. |
| `src/train/widen_buffer.py` | rebuilds a 4-frame / 5-step buffer as 8-frame / 10-step |
| `src/train/pool_compare.py` | pools `contest_compare.sh` logs per checkpoint |
| `runs/ft500/`, `runs/h8n10/` | the two runs (logs, checkpoints, buffers) |
| `runs/h8n10_buffer.npz` | rebuilt buffer (360 MB, can be deleted) |
| `data/cand/` | every tested checkpoint |
| `results/contest_compare_{avg,ft30k,ft51k,ft90k,h8_41k,h8_late,final}.log`, `results/contest_pool_17oct.txt` | raw contest results and the pooled table |

---

## 6. Follow-up test with 20,000-step runs (17 Oct, 08:42 – 10:10)

These are the fine-tune checkpoints not tested during the night (61k, 72k, 82k), plus 3 more runs of the history-8 110k checkpoint. Every run is 20,000 steps in the contest world. Order: round 1 forward, round 2 reversed, round 3 = history-8 110k vs 500k only. Raw results: `results/contest_compare_20k_{a,b}.log`, pooled in `results/contest_pool_20k_17oct.txt`.

| Checkpoint | Runs × 20k | Reward / 1,000 | `r = +5` / 1,000 | `died` / 1,000 |
|---|---:|---:|---:|---:|
| Fine-tune 82k | 2 | **+153.7** (154.5, 152.8) | 32.2 | 1.45 |
| Fine-tune 72k | 2 | +149.7 (150.2, 149.2) | 31.2 | 1.25 |
| **500k (submission)** | 3 | +149.3 (149.2, 148.8, 149.8) | 31.6 | 1.78 |
| History-8 110k | 3 | +146.3 (146.0, 140.8, 152.2) | 30.4 | 1.17 |
| Fine-tune 61k | 2 | +133.3 (126.8, 139.8) | 28.1 | 1.50 |

**All contest tests pooled** (night + today, `results/contest_pool_all_17oct.txt`): 500k **+148.1** over 280k steps; fine-tune 82k +153.7 over 40k; fine-tune 30k +148.7 over 120k; history-8 110k +147.4 over 90k, with `died` 1.07 per 1,000 against 1.96 for 500k.

**Reading:**
- Fine-tune 82k is the best so far. Its lead over 500k is +5.6 per 1,000, about 1.1 standard deviations from the counts of `r = +5` and `died` (Poisson), so it is not yet a proven improvement.
- The history-8 checkpoint again has about half the `died` rate of 500k. With fewer `r = +5`, its reward is the same.
- Fine-tune 61k is worse. Neighbouring checkpoints of one run differ by up to 20 per 1,000 in the contest world, so each checkpoint has to be tested on its own.
- **Decision:** the submission stays 500k. Fine-tune 82k is saved as `models/CAND_ddqn_ft500_82k.pt`.
  - It would need about 6–8 more 20k-step runs against 500k to show whether its +5.6 lead is real.
  - If it is, switch with `uv run python src/build_submission.py models/CAND_ddqn_ft500_82k.pt`, and keep `models/prakhar.tiwari_SUBMISSION_ddqn500k.py` as the fallback.
