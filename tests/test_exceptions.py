"""Tests for custom exception usage across the codebase."""

from __future__ import annotations

import os

import pytest

from alpha_evolve_sr.exceptions import (
    CheckpointError,
    LLMProviderError,
    SpecificationError,
)


class TestSpecificationError:
    """SpecificationError is raised for invalid specifications."""

    def test_missing_evaluate_run(self):
        bad_spec = 'def foo():\n    pass\n# @equation.evolve\ndef equation(x, params):\n    return x\n'
        from alpha_evolve_sr.cli import _extract_function_names
        with pytest.raises(SpecificationError, match="@evaluate.run"):
            _extract_function_names(bad_spec)

    def test_missing_equation_evolve(self):
        bad_spec = '# @evaluate.run\ndef evaluate(data):\n    pass\n'
        from alpha_evolve_sr.cli import _extract_function_names
        with pytest.raises(SpecificationError, match="@equation.evolve"):
            _extract_function_names(bad_spec)


class TestCheckpointError:
    """CheckpointError is raised when checkpoint loading fails."""

    def test_missing_file(self, tmp_path):
        from alpha_evolve_sr.checkpoint import load_checkpoint
        with pytest.raises(CheckpointError, match="Failed to load"):
            load_checkpoint(str(tmp_path))

    def test_corrupt_file(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint_final.pkl"
        ckpt_path.write_bytes(b"not a pickle")
        from alpha_evolve_sr.checkpoint import load_checkpoint
        with pytest.raises(CheckpointError, match="Failed to load"):
            load_checkpoint(str(tmp_path))


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
