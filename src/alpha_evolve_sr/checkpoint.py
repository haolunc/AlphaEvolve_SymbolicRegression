"""Utilities for saving and loading checkpoints of the database component."""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

from .config import RunConfig
from .exceptions import CheckpointError
from .logging_config import get_logger

if TYPE_CHECKING:
    from .database import ProgramsDatabase

logger = get_logger("checkpoint")


def save_config(run_config: RunConfig, save_dir: str) -> None:
    """Save a RunConfig to a YAML file in *save_dir*."""
    os.makedirs(save_dir, exist_ok=True)
    run_config.to_yaml(os.path.join(save_dir, "run_config.yaml"))


def load_config(ckpt_dir: str) -> RunConfig:
    """Load a RunConfig from a previously saved YAML file in *ckpt_dir*."""
    yaml_path = os.path.join(ckpt_dir, "run_config.yaml")

    if os.path.exists(yaml_path):
        try:
            config = RunConfig.from_yaml(yaml_path)
            logger.info("Loaded configuration from %s", yaml_path)
            return config
        except Exception as e:
            logger.error("Failed to load config from %s: %s", yaml_path, e)
            raise CheckpointError(f"Failed to load config from {yaml_path}: {e}") from e
    else:
        raise CheckpointError(f"No run_config.yaml found at {yaml_path}")


def save_checkpoint(database: ProgramsDatabase, ckpt_path: str) -> None:
    """Pickle the database to *ckpt_path*."""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, "wb") as f:
        pickle.dump(database, f)
    logger.info("Checkpoint saved to %s", ckpt_path)


def load_checkpoint(ckpt_dir: str) -> ProgramsDatabase:
    """Load a pickled database from *ckpt_dir*/checkpoint_final.pkl."""
    path = os.path.join(ckpt_dir, "checkpoint_final.pkl")
    try:
        with open(path, "rb") as f:
            db = pickle.load(f)
        logger.info("Checkpoint loaded from %s", ckpt_dir)
        return db
    except (FileNotFoundError, pickle.UnpicklingError, EOFError, ModuleNotFoundError) as e:
        raise CheckpointError(f"Failed to load checkpoint from {path}: {e}") from e
