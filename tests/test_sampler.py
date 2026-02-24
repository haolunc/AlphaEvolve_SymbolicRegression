"""Tests for sampler module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from alpha_evolve_sr.config import SamplerConfig
from alpha_evolve_sr.messages import LLMResponse
from alpha_evolve_sr.sampler import LLM


class TestLLMInit:
    """Tests for LLM initialisation."""

    def test_load_dotenv_called_on_init(self):
        """load_dotenv() should be called during LLM construction."""
        with patch("alpha_evolve_sr.sampler.load_dotenv") as mock_dotenv:
            LLM(config=SamplerConfig(samples_per_prompt=1))
            mock_dotenv.assert_called_once()

    def test_clean_is_noop(self):
        """clean() should be callable without error."""
        llm = LLM(config=SamplerConfig(samples_per_prompt=1))
        llm.clean()  # should not raise


class TestQuery:
    """Tests for LLM.query() — single-call interface."""

    def test_returns_single_response(self):
        """query() returns a single LLMResponse on success."""
        mock_resp = LLMResponse(
            response_text="return x", input_tokens=10, output_tokens=20,
        )
        with patch("alpha_evolve_sr.sampler.load_dotenv"), \
             patch("alpha_evolve_sr.sampler._make_provider") as mock_mp:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = mock_resp
            mock_mp.return_value = mock_provider
            llm = LLM(config=SamplerConfig(samples_per_prompt=1))
            result = llm.query("test prompt")

        assert isinstance(result, LLMResponse)
        assert mock_provider.generate.call_count == 1

    def test_returns_none_on_failure(self):
        """query() returns None when all retries fail."""
        with patch("alpha_evolve_sr.sampler.load_dotenv"), \
             patch("alpha_evolve_sr.sampler._make_provider") as mock_mp:
            mock_provider = MagicMock()
            mock_provider.generate.side_effect = Exception("API error")
            mock_mp.return_value = mock_provider
            llm = LLM(config=SamplerConfig(
                samples_per_prompt=1, max_retries=1, retry_delay_seconds=0,
            ))
            result = llm.query("test prompt")

        assert result is None

    def test_cost_calculated(self):
        """query() populates token_cost on success."""
        mock_resp = LLMResponse(
            response_text="return x", input_tokens=1000, output_tokens=500,
        )
        cost_per_ktoken = (0.01, 0.03)
        with patch("alpha_evolve_sr.sampler.load_dotenv"), \
             patch("alpha_evolve_sr.sampler._make_provider") as mock_mp:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = mock_resp
            mock_mp.return_value = mock_provider
            llm = LLM(config=SamplerConfig(
                samples_per_prompt=1, cost_per_ktoken=cost_per_ktoken,
            ))
            result = llm.query("test prompt")

        expected_cost = (1000 * 0.01 + 500 * 0.03) / 1000
        assert result.token_cost == pytest.approx(expected_cost)

    def test_retries_before_giving_up(self):
        """query() retries max_retries times before returning None."""
        with patch("alpha_evolve_sr.sampler.load_dotenv"), \
             patch("alpha_evolve_sr.sampler._make_provider") as mock_mp:
            mock_provider = MagicMock()
            mock_provider.generate.side_effect = RuntimeError("fail")
            mock_mp.return_value = mock_provider
            llm = LLM(config=SamplerConfig(
                max_retries=3, retry_delay_seconds=0,
            ))
            result = llm.query("test prompt")

        assert result is None
        assert mock_provider.generate.call_count == 3
