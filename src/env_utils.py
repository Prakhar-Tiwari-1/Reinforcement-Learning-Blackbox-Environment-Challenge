"""Build the remote PettingZoo environment from the settings in .env."""

import os
from pathlib import Path

import pettingzoo
import pzclient  # noqa: F401  (importing it registers the remote environments)
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ENV_ID = "csc53439ep/csc53439ep-remote-v1"


def make_env(mode="training"):
    """Return the remote world. mode is "training" (own world) or "contest" (shared world)."""
    url = os.environ["PZ_TRAIN_URL" if mode == "training" else "PZ_CONTEST_URL"]
    return pettingzoo.make("parallel", ENV_ID, api_url=url, token=os.environ["PETTINGZOO_TOKEN"])


def my_agent_name():
    """Name of our agent in the shared world. Only reads it: does not join (no reset, no close,
    so a contest program running with the same token is not disturbed)."""
    return make_env("contest").possible_agents[0]
