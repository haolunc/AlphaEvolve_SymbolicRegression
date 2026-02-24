"""Unified logging configuration for alpha_evolve_sr."""
import logging
import os

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured = False


def configure_logging(log_file: str | None = None) -> None:
    """Configure root-level logging for the package. Call once at startup.

    Args:
        log_file: If given, write INFO+ messages to this file.
            Console output is always WARNING+ to reduce terminal noise.
    """
    global _configured
    if _configured:
        return
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    root = logging.getLogger("alpha_evolve_sr")
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``alpha_evolve_sr`` namespace."""
    return logging.getLogger(f"alpha_evolve_sr.{name}")


