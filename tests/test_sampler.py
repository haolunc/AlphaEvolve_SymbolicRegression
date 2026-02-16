"""Tests for sampler module."""

from __future__ import annotations

from unittest.mock import patch

from alpha_evolve_sr.config import SamplerConfig
from alpha_evolve_sr.sampler import LLM


class TestLLMInit:
    """Tests for LLM initialisation."""

    def test_load_dotenv_called_on_init(self):
        """load_dotenv() should be called during LLM construction."""
        with patch("alpha_evolve_sr.sampler.load_dotenv") as mock_dotenv:
            LLM(config=SamplerConfig(samples_per_prompt=1))
            mock_dotenv.assert_called_once()


class TestLLMExecutorReuse:
    """Tests for persistent ThreadPoolExecutor."""

    def test_executor_created_in_init(self):
        """Executor is created in __init__ when samples_per_prompt > 1."""
        llm = LLM(config=SamplerConfig(samples_per_prompt=3))
        assert llm._executor is not None
        llm.clean()

    def test_executor_none_for_single_sample(self):
        """Executor is None when samples_per_prompt == 1."""
        llm = LLM(config=SamplerConfig(samples_per_prompt=1))
        assert llm._executor is None

    def test_clean_shuts_down_executor(self):
        """After clean(), executor is None."""
        llm = LLM(config=SamplerConfig(samples_per_prompt=3))
        assert llm._executor is not None
        llm.clean()
        assert llm._executor is None
