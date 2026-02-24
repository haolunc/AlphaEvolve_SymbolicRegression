"""Integration tests: run the real pipeline with a live LLM.

Marked ``slow`` — skipped by default.  Run explicitly with::

    pytest tests/test_integration.py -m slow -v
"""

from __future__ import annotations

import os
import sqlite3

import pytest
import yaml

from alpha_evolve_sr.cli import main

# Resolve paths relative to repo root (one level above tests/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBLEM_DIR = os.path.join(REPO_ROOT, "examples", "dft_xc")
DATA_FOLDER = os.path.join(REPO_ROOT, "examples", "data", "dft_ev_xc")


def _base_config(log_dir: str, **overrides) -> dict:
    """Return a minimal YAML-ready config dict."""
    cfg = {
        "problem_dir": PROBLEM_DIR,
        "data_folder": DATA_FOLDER,
        "log_dir": log_dir,
        "max_samples": 20,
        "num_samplers": 1,
        "num_evaluators": 1,
        "sampler": {
            "provider": "qwen",
            "samples_per_prompt": 1,
            "temperature": 1.0,
        },
        "database": {
            "functions_per_prompt": 2,
            "num_islands": 3,
            "reset_period": 100,
            "cluster_sampling_temperature_init": 0.1,
            "cluster_sampling_temperature_period": 20,
            "checkpoint_interval": 1,
            "log_frequency": 1,
        },
        "evaluator": {
            "timeout_seconds": 120,
        },
    }
    cfg.update(overrides)
    return cfg


def _write_config(tmp_path, cfg: dict) -> str:
    """Dump *cfg* to a YAML file and return its path."""
    path = str(tmp_path / "config.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


def _count_programs(db_path: str) -> int:
    """Return the number of rows in the checkpoint's programs table."""
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM programs").fetchone()
        return row[0]
    finally:
        conn.close()


def _has_tb_events(log_dir: str) -> bool:
    """Return True if *log_dir* contains at least one TensorBoard event file."""
    for fname in os.listdir(log_dir):
        if fname.startswith("events.out.tfevents"):
            return True
    return False


@pytest.fixture(autouse=True)
def _patch_main_env(monkeypatch):
    """Patch issues that arise from calling ``main()`` in-process.

    1. ``sys.argv`` — ``main()`` uses argparse which reads sys.argv.
    2. ``mp.set_start_method("spawn")`` — fails on second call in the same process.
    3. ``logging_config._configured`` guard — blocks re-initialization across tests.
    """
    import alpha_evolve_sr.logging_config as lc
    import logging

    # Neutralise mp.set_start_method (already set by first call or default)
    monkeypatch.setattr("multiprocessing.set_start_method", lambda *a, **kw: None)

    yield

    # Reset logging so the next test can re-configure
    lc._configured = False
    pkg_logger = logging.getLogger("alpha_evolve_sr")
    for h in pkg_logger.handlers[:]:
        h.close()
        pkg_logger.removeHandler(h)


@pytest.mark.slow
class TestPipelineIntegration:
    """End-to-end smoke tests hitting the real Qwen API."""

    def test_single_thread(self, tmp_path, monkeypatch):
        """20 samples, 1 sampler + 1 evaluator — basic end-to-end flow."""
        log_dir = str(tmp_path / "log")
        cfg_path = _write_config(tmp_path, _base_config(log_dir))

        monkeypatch.setattr("sys.argv", ["alpha-evolve-sr", "--config", cfg_path])
        main()

        assert os.path.isdir(log_dir)
        assert os.path.isfile(os.path.join(log_dir, "checkpoint.db"))
        assert _has_tb_events(log_dir)

    def test_multi_thread(self, tmp_path, monkeypatch):
        """20 samples, 2 samplers + 2 evaluators — no races or deadlocks."""
        log_dir = str(tmp_path / "log")
        cfg_path = _write_config(
            tmp_path,
            _base_config(log_dir, num_samplers=2, num_evaluators=2),
        )

        monkeypatch.setattr("sys.argv", ["alpha-evolve-sr", "--config", cfg_path])
        main()

        assert os.path.isdir(log_dir)
        assert os.path.isfile(os.path.join(log_dir, "checkpoint.db"))
        assert _has_tb_events(log_dir)

    def test_resume(self, tmp_path, monkeypatch):
        """Run1: 10 samples → Run2: resume to 20.  Checkpoint restore works."""
        log_dir = str(tmp_path / "log")
        ckpt_path = os.path.join(log_dir, "checkpoint.db")

        # --- Run 1: 10 samples ---
        cfg_path1 = _write_config(tmp_path, _base_config(log_dir, max_samples=10))
        monkeypatch.setattr("sys.argv", ["alpha-evolve-sr", "--config", cfg_path1])
        main()

        assert os.path.isfile(ckpt_path)
        count_after_run1 = _count_programs(ckpt_path)

        # Reset logging between runs (main() calls configure_logging)
        import alpha_evolve_sr.logging_config as lc
        import logging

        lc._configured = False
        pkg_logger = logging.getLogger("alpha_evolve_sr")
        for h in pkg_logger.handlers[:]:
            h.close()
            pkg_logger.removeHandler(h)

        # --- Run 2: resume to 20 samples ---
        cfg_path2 = _write_config(
            tmp_path,
            _base_config(log_dir, max_samples=20, resume_from_ckpt=ckpt_path),
        )
        monkeypatch.setattr("sys.argv", ["alpha-evolve-sr", "--config", cfg_path2])
        main()

        count_after_run2 = _count_programs(ckpt_path)
        assert count_after_run2 > count_after_run1, (
            f"Expected more programs after resume: run1={count_after_run1}, run2={count_after_run2}"
        )
