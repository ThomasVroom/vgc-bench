"""
Utility module for VGC-Bench.

Contains shared constants, enums, and helper functions used throughout the
codebase. Defines observation space dimensions, loads Pokemon game data,
and provides training configuration utilities.
"""

import json
import os
import random
import re
import io
import zipfile
from pathlib import Path
from enum import Enum, auto, unique

import numpy as np
import torch
from poke_env.battle import (
    Effect,
    Field,
    MoveCategory,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_zip_file


@unique
class LearningStyle(Enum):
    """
    Training paradigm options for reinforcement learning.

    Defines different self-play and opponent sampling strategies used
    during PPO training for Pokemon VGC agents.

    Values:
        EXPLOITER: Train against a fixed opponent policy.
        PURE_SELF_PLAY: Train against current policy (both players identical).
        FICTITIOUS_PLAY: Sample historical checkpoints uniformly as opponents.
        DOUBLE_ORACLE: Sample checkpoints based on Nash equilibrium distribution.
    """

    EXPLOITER = auto()
    PURE_SELF_PLAY = auto()
    FICTITIOUS_PLAY = auto()
    DOUBLE_ORACLE = auto()

    @property
    def is_self_play(self) -> bool:
        """Check if this style involves any form of self-play training."""
        return self in {
            LearningStyle.PURE_SELF_PLAY,
            LearningStyle.FICTITIOUS_PLAY,
            LearningStyle.DOUBLE_ORACLE,
        }

    @property
    def abbrev(self) -> str:
        """Get two-letter abbreviation for logging and file naming."""
        match self:
            case LearningStyle.EXPLOITER:
                return "ex"
            case LearningStyle.PURE_SELF_PLAY:
                return "sp"
            case LearningStyle.FICTITIOUS_PLAY:
                return "fp"
            case LearningStyle.DOUBLE_ORACLE:
                return "do"


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# observation length constants
act_len = 107
glob_obs_len = len(Field) + len(Weather) + 3
side_obs_len = len(SideCondition) + 5
move_obs_len = len(MoveCategory) + len(Target) + len(PokemonType) + 12
pokemon_obs_len = (
    4 * move_obs_len
    + len(Effect)
    + len(PokemonGender)
    + 2 * len(PokemonType)
    + len(Status)
    + 39
)
chunk_obs_len = glob_obs_len + side_obs_len + pokemon_obs_len

# pokemon data
format_map = {
    "a": "gen9vgc2022rega",
    "b": "gen9vgc2023regb",
    "c": "gen9vgc2023regc",
    "d": "gen9vgc2023regd",
    "e": "gen9vgc2024rege",
    "f": "gen9vgc2024regf",
    "g": "gen9vgc2024regg",
    "h": "gen9vgc2024regh",
    "i": "gen9vgc2025regi",
    "j": "gen9vgc2025regj",
}


def is_vgc_format(fmt: str) -> bool:
    """Check if a format string is a recognized VGC format."""
    return bool(re.match(r"gen9vgc\d{4}reg[a-j]", fmt))


def get_reg_from_format(fmt: str) -> str:
    """Extract the regulation letter from a VGC format string"""
    m = re.match(r"gen9vgc\d{4}reg([a-j])", fmt)
    assert m is not None, f"not a valid VGC format: {fmt}"
    return m.group(1)


def load_policy(policy_class: type, file: str | Path, device: torch.device):
    """
    Load a policy from a checkpoint file.

    Args:
        policy_class: Class to use when creating a new policy.
        file: Path to the saved PPO checkpoint.
        device: PyTorch device for model placement.
    """
    try:
        policy = PPO.load(file, device=device).policy
    except: # parameter groups don't match -> recreate policy from data
        data, params, _ = load_from_zip_file(file, device=device)
        assert data and params, "no data found in file"

        # identify correct architecture
        if "progressive" in data["policy_kwargs"]:
            if "n_columns" in data["policy_kwargs"]:
                data["policy_kwargs"]["n_columns"] += 1
        else:
            data["policy_kwargs"]["progressive"] = False

        # initialize new policy
        policy = policy_class(
            observation_space=data["observation_space"],
            action_space=data["action_space"],
            lr_schedule=data["learning_rate"],
            **data["policy_kwargs"]
        ).to(device)
        policy.load_state_dict(params["policy"])
    return policy


def load_policy_from_zip(policy: BasePolicy, file: str | Path, device: torch.device):
    """Bypass SB3's leaky set_parameters - load state dict directly from zip"""
    with zipfile.ZipFile(file, "r") as zf:
        with zf.open("policy.pth") as f:
            state_dict = torch.load(
                io.BytesIO(f.read()), map_location=device, weights_only=True
            )
    policy.load_state_dict(state_dict)


with open("data/abilities.json") as f:
    abilities: list[str] = json.load(f)
with open("data/items.json") as f:
    items: list[str] = json.load(f)
with open("data/moves.json") as f:
    moves: list[str] = json.load(f)
