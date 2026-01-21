"""Utilities for saving and loading checkpoints of the database component"""
import os
import pickle
import time
from logging_utils import setup_console_logger
import json

# Get logger from utility module
logger = setup_console_logger(__name__)

def save_config(args, save_dir: str) -> None:
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f)    


def load_config(args, logger) -> None:
    # Extract directory from the checkpoint path
    config_path = os.path.join(args.resume_from_ckpt, "config.json")

    # Load config if it exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            
            # Keep original resume_from_ckpt and max_samples from command line
            resume_from_ckpt = args.resume_from_ckpt
            max_samples = args.max_samples
            
            # Update args with saved config
            for key, value in saved_config.items():
                # Skip the first two arguments which we want to keep from command line
                if key not in ['resume_from_ckpt', 'max_samples']:
                    setattr(args, key, value)
            
            # Restore the two arguments we want to keep from command line
            args.resume_from_ckpt = resume_from_ckpt
            args.max_samples = max_samples
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
    else:
        logger.warning(f"No config.json found at {config_path}, using command line arguments")

def save_checkpoint(database, ckpt_path: str) -> None:
    with open(ckpt_path, "wb") as f:
        pickle.dump(database, f)
    logger.info(f"Checkpoint saved to {ckpt_path}")

def load_checkpoint(ckpt_dir: str) -> None:
    with open(os.path.join(ckpt_dir, "checkpoint_final.pkl"), "rb") as f:
        logger.info(f"Checkpoint loaded from {ckpt_dir}")
        return pickle.load(f)
    