"""Unified logging configuration for alpha_evolve_sr."""
import logging
import os

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root-level logging for the package. Call once at startup."""
    global _configured
    if _configured:
        return
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root = logging.getLogger("alpha_evolve_sr")
    root.setLevel(level)
    root.addHandler(console)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``alpha_evolve_sr`` namespace."""
    return logging.getLogger(f"alpha_evolve_sr.{name}")


def setup_file_logger(name: str, log_dir: str = "./logger") -> logging.Logger:
    """Return a logger that writes to a file (used by monitoring worker)."""
    logger = logging.getLogger(f"alpha_evolve_sr.{name}")
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
