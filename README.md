# Reinforcement Learning Challenge Project

An agent for a multi-agent world hosted by the course organizers. The world runs on their server; the agent runs locally and connects over the network with the `pzclient` library, through the standard [PettingZoo](https://pettingzoo.farama.org/) parallel API.

## The challenge

The world is a **black box**: the agent knows it only through observations and rewards. The agent must:

- **Learn on unseen environments**, knowing only their observation and action spaces.
- **Handle different flat spaces**: `Box`, `Discrete` (value `o` represented as `np.array([o])`) and `MultiBinary`. The contest world itself has `Box` observations and `MultiBinary` actions.
- **Stay robust under distribution drift**: other participants join and leave at any time, and the organizers may add agents of their own (non-stationarity).

There is no benchmark to beat, but the agent should perform clearly better than a random agent, and every design choice should be justifiable.

### Constraints

- Neural networks use **at most 128 hidden units in total** (across all layers and networks).
- Training and inference run on the local machine; the server only runs the world.
- **Training mode**: the world waits for the agent.
- **Contest mode**: the world has its own clock and plays a step at most **T = 5 s** after the previous one. Agents whose action is late play a default action, so the agent must keep up with the clock.

### Schedule

| Period | What happens |
|---|---|
| Now to week 4 | Train in training mode; contest mode is open for testing only (results ignored) |
| Week 5: Mon 19 Oct 2026 00:00 to Wed 21 Oct 2026 23:59 (Paris) | Official contest |

The last agent submitted before the end of the contest is the one evaluated.

## Repository layout

```
.
├── README.md
├── .env.example     # template for local configuration (committed)
├── .env             # your token and server URLs (git-ignored, never commit)
├── pyproject.toml   # dependencies (managed with uv)
├── src/             # agent code, training and evaluation scripts
└── firstname.lastname.py   # final submission file, named after the token
```

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # create .venv and install dependencies
cp .env.example .env          # then fill in your token and the server URLs
```

`.env` holds the secret token (it identifies and names your agent on the shared world) and the training/contest server URLs listed in section 1 of the course tutorial. Keep it private.

## Approach

To be filled in as the project develops: algorithm choice, how the 128-hidden-unit budget is spent, how the agent adapts to drift, and evaluation against a random baseline.

## Submission

Save the agent in a single file named `<firstname.lastname>.py` (matching the token) and upload it on the course platform. It can be resubmitted any number of times before the deadline.
