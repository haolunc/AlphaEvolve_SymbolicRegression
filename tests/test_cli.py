"""Tests for cli module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from alpha_evolve_sr.code_manipulation import ParsedFunction
from alpha_evolve_sr.config import ProgramsDatabaseConfig, RunConfig, SamplerConfig
from alpha_evolve_sr.messages import EvalResult
from alpha_evolve_sr.sampler import LLMResponse
from tests.conftest import SAMPLE_SPEC


def _make_eval_result(island_id=None):
    """Helper to build a valid EvalResult."""
    func = ParsedFunction(
        name="equation",
        args="x, params",
        body="    return params[0] * x",
    )
    return EvalResult(
        function=func,
        island_id=island_id,
        result_per_test={"score": -1.0, "optimized_params": None, "complexity": 5, "complexity_detail": {}},
        sample_time=0.1,
        evaluate_time=0.2,
        sample_token_usage=(10, 20),
        sample_token_cost=0.001,
    )


class TestMainSingleSampleAccess:
    """Verify main_single accesses sample_info via dict keys (not indices)."""

    def test_sample_info_dict_keys_passed_to_analyse(self, tmp_path):
        """Ensure evaluator.analyse receives dict-key values from sample_info."""
        mock_evaluator = MagicMock()
        # Initial call: island_id=None (registers to all islands).
        # Sample call: island_id=0.
        mock_evaluator.analyse.side_effect = [
            _make_eval_result(island_id=None),
            _make_eval_result(island_id=0),
        ]

        mock_llm = MagicMock()
        mock_llm.draw_samples.return_value = [
            LLMResponse(response_text="return x", input_tokens=10, output_tokens=20),
        ]

        run_config = RunConfig(
            resume_from_ckpt=None,
            log_path=str(tmp_path / "logs"),
            save_ckpt_dir=None,
            save_ckpt_interval=300,
            max_samples=1,
            sampler=SamplerConfig(samples_per_prompt=1),
            database=ProgramsDatabaseConfig(num_islands=2, reset_period=999),
        )

        with patch("alpha_evolve_sr.cli.evaluator_mod.Evaluator", return_value=mock_evaluator), \
             patch("alpha_evolve_sr.cli.sampler_mod.LLM", return_value=mock_llm):
            from alpha_evolve_sr.cli import main_single
            main_single(run_config, SAMPLE_SPEC, {"x": [1, 2, 3]})

        # analyse should have been called twice: initial + sample
        assert mock_evaluator.analyse.call_count == 2

        # The sample call (second call) should use LLMResponse attributes
        sample_call = mock_evaluator.analyse.call_args_list[1]
        assert sample_call[0][0] == "return x"  # response_text
        assert sample_call[0][4] == (10, 20)  # token usage tuple

    def test_none_sample_info_skipped(self, tmp_path):
        """Ensure None / falsy sample_info entries are skipped."""
        mock_evaluator = MagicMock()
        mock_evaluator.analyse.side_effect = [
            _make_eval_result(island_id=None),
            _make_eval_result(island_id=0),
        ]

        mock_llm = MagicMock()
        # Return one None and one valid sample
        mock_llm.draw_samples.return_value = [
            None,
            LLMResponse(response_text="return x", input_tokens=5, output_tokens=10),
        ]

        run_config = RunConfig(
            resume_from_ckpt=None,
            log_path=str(tmp_path / "logs"),
            save_ckpt_dir=None,
            save_ckpt_interval=300,
            max_samples=1,
            sampler=SamplerConfig(samples_per_prompt=2),
            database=ProgramsDatabaseConfig(num_islands=2, reset_period=999),
        )

        with patch("alpha_evolve_sr.cli.evaluator_mod.Evaluator", return_value=mock_evaluator), \
             patch("alpha_evolve_sr.cli.sampler_mod.LLM", return_value=mock_llm):
            from alpha_evolve_sr.cli import main_single
            main_single(run_config, SAMPLE_SPEC, {"x": [1, 2, 3]})

        # Should be 2 calls: initial + 1 valid sample (None was skipped)
        assert mock_evaluator.analyse.call_count == 2


class TestSamplerConfigPassthrough:
    """Verify SamplerConfig is constructed from RunConfig."""

    def test_config_from_run_config(self):
        run_config = RunConfig(
            sampler=SamplerConfig(provider="openai", model_name="gpt-5", temperature=0.7),
        )
        assert run_config.sampler.provider == "openai"
        assert run_config.sampler.model_name == "gpt-5"
        assert run_config.sampler.temperature == 0.7
