"""Tests for custom exception usage across the codebase."""

from __future__ import annotations

import os

import pytest

from alpha_evolve_sr.exceptions import (
    CheckpointError,
    LLMProviderError,
)


class TestCheckpointError:
    """CheckpointError is raised when checkpoint DB operations fail."""

    def test_bad_db_path(self):
        from alpha_evolve_sr.checkpoint import CheckpointDB
        with pytest.raises(CheckpointError, match="Failed to open"):
            CheckpointDB("/nonexistent/deeply/nested/test.db")

    def test_config_mismatch_on_resume(self, tmp_path):
        from alpha_evolve_sr.checkpoint import CheckpointDB
        from alpha_evolve_sr.config import ProgramsDatabaseConfig, RunConfig

        db = CheckpointDB(str(tmp_path / "test.db"))
        run_config = RunConfig(
            problem_dir="specs/test", data_folder="data/test", log_dir="test",
            database=ProgramsDatabaseConfig(num_islands=5),
        )
        with db.transaction():
            db.save_run_config(run_config, num_islands=5, complexity_bin_size=10)
        with pytest.raises(CheckpointError, match="num_islands"):
            db.validate_config(num_islands=8, complexity_bin_size=10)
        db.close()


class TestLLMProviderError:
    """LLMProviderError is raised when all retries fail."""

    def test_all_retries_exhausted(self):
        from unittest.mock import MagicMock
        from alpha_evolve_sr.config import SamplerConfig
        from alpha_evolve_sr.sampler import LLM

        config = SamplerConfig(max_retries=2, retry_delay_seconds=0)
        llm = LLM(config=config)
        llm._provider = MagicMock()
        llm._provider.generate.side_effect = RuntimeError("fail")

        with pytest.raises(LLMProviderError, match="retries failed"):
            llm._query_with_retry("test prompt")
