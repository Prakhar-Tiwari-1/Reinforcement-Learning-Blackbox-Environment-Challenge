# Saved policies

Scores are reward per 1,000 steps (sum of `r`, `r` ∈ {−5, 0, +5}) of one agent, frozen greedy play.
Random actions score about −14 (contest) and −18 (training world).

| File | What | Result |
|---|---|---|
| **CAND_ddqn_coplay_143k.pt** | DDQN 500k + 143,360 steps of co-player training (2 of 4 agents play other policies). **The policy embedded in `../prakhar.tiwari.py`**, played with the anti-stall rule | live contest (crowded): **+39.4** over 124,595 steps (21 Oct); +39.3 over 150k steps without anti-stall (20 Oct) |
| FINAL_ddqn_500k.pt | DDQN after 500,000 training steps with 3 copies of itself; the starting point of co-player training | contest (quiet, 16–17 Oct): **+148.0** over 220k steps; crowded (20 Oct): +35.1 over 85k steps |
| SAC_sac2_184k.pt | SAC-Discrete, run `sac2`, step 184,320: actor 46 → 64 → 64 → 4 (the policy) + two 256 + 256 critics | training world, greedy: +145.8 (6k steps × 4 agents); never played in the contest |
| ddqn_training_config.json | settings of the DDQN run (`src/train/ddqn.py`) | |

DDQN network: dueling Q-network, input = last 4 observations + previous action (46 numbers) → 64 → 64 → 4 Q-values,
128 hidden units.

- **Play** a DDQN checkpoint in the contest world: `python prakhar.tiwari.py --load models/<file>.pt`.
- **Embed** it into the submission file: `uv run python src/build_submission.py models/<file>.pt`.
- **Evaluate** the SAC actor in the training world: `uv run python src/train/sac_offline.py eval models/SAC_sac2_184k.pt --steps 6000 --mode greedy`.

## Other checkpoints (results only; files not in this repository)

| Checkpoint | Contest, quiet (16–17 Oct) | Contest, crowded (20 Oct) |
|---|---:|---:|
| DDQN 500k fine-tuned 82k more steps (lr 3e-5) | +153.7 (40k steps) | +37.5 (10k) |
| DDQN 500k fine-tuned 30k more steps | +148.7 (120k) | – |
| DDQN with 8-observation history + 10-step returns | +147.4 (90k; `died` 1.07 vs 1.96 per 1,000) | +37.5 (10k) |
| DDQN step 410k (submission until 17 Oct) | +142.5 (42k) | – |
| DDQN step 389k (best by training score) | +121 (20k) | – |
| DDQN step ≈ 208k | about +90 | – |
| PPO v1, fine-tuned (submission until 16 Oct) | about +50 | – |
