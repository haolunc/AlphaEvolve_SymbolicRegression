"""Tests for cli module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from alpha_evolve_sr.config import ProgramsDatabaseConfig, RunConfig, SamplerConfig
from alpha_evolve_sr.messages import LLMResponse, SampleMessage
from tests.conftest import SAMPLE_EVALUATE_CODE, SAMPLE_PROMPT, SAMPLE_SEED_FUNCTION, make_eval_result


class TestMainSingleSampleAccess:
    """Verify main_single accesses sample_info via dict keys (not indices)."""

    def test_sample_info_dict_keys_passed_to_analyse(self, tmp_path):
        """Ensure evaluator.analyse receives a SampleMessage."""
        mock_evaluator = MagicMock()
        mock_evaluator.initialize.return_value = make_eval_result()
        mock_evaluator.analyse.return_value = make_eval_result()

        mock_llm = MagicMock()
        mock_llm.draw_samples.return_value = [
            LLMResponse(response_text="return x", input_tokens=10, output_tokens=20),
        ]

        run_config = RunConfig(
            log_dir=str(tmp_path / "logs"),
            save_ckpt_dir=None,
            max_samples=1,
            sampler=SamplerConfig(samples_per_prompt=1),
            database=ProgramsDatabaseConfig(num_islands=2, reset_period=999),
        )

        with patch("alpha_evolve_sr.cli.evaluator_mod.Evaluator", return_value=mock_evaluator), \
             patch("alpha_evolve_sr.cli.sampler_mod.LLM", return_value=mock_llm):
            from alpha_evolve_sr.cli import main_single
            main_single(
                run_config, SAMPLE_PROMPT, SAMPLE_EVALUATE_CODE, SAMPLE_SEED_FUNCTION,
                {"x": [1, 2, 3]},
            )

        # initialize called once for seed, analyse called once for the sample
        mock_evaluator.initialize.assert_called_once()
        assert mock_evaluator.analyse.call_count == 1

        # The analyse call should receive a SampleMessage
        sample_call = mock_evaluator.analyse.call_args
        sample_msg = sample_call[0][0]
        assert isinstance(sample_msg, SampleMessage)
        assert sample_msg.llm_response.response_text == "return x"
        assert sample_msg.llm_response.input_tokens == 10
        assert sample_msg.llm_response.output_tokens == 20

    def test_none_sample_info_skipped(self, tmp_path):
        """Ensure None / falsy sample_info entries are skipped."""
        mock_evaluator = MagicMock()
        mock_evaluator.initialize.return_value = make_eval_result()
        mock_evaluator.analyse.return_value = make_eval_result()

        mock_llm = MagicMock()
        # Return one None and one valid sample
        mock_llm.draw_samples.return_value = [
            None,
            LLMResponse(response_text="return x", input_tokens=5, output_tokens=10),
        ]

        run_config = RunConfig(
            log_dir=str(tmp_path / "logs"),
            save_ckpt_dir=None,
            max_samples=1,
            sampler=SamplerConfig(samples_per_prompt=2),
            database=ProgramsDatabaseConfig(num_islands=2, reset_period=999),
        )

        with patch("alpha_evolve_sr.cli.evaluator_mod.Evaluator", return_value=mock_evaluator), \
             patch("alpha_evolve_sr.cli.sampler_mod.LLM", return_value=mock_llm):
            from alpha_evolve_sr.cli import main_single
            main_single(
                run_config, SAMPLE_PROMPT, SAMPLE_EVALUATE_CODE, SAMPLE_SEED_FUNCTION,
                {"x": [1, 2, 3]},
            )

        # initialize once + analyse once (None sample was skipped)
        mock_evaluator.initialize.assert_called_once()
        assert mock_evaluator.analyse.call_count == 1

    def test_register_program_called_with_eval_result_and_sample_msg(self, tmp_path):
        """Ensure database.register_program receives (EvalResult, SampleMessage)."""
        eval_result = make_eval_result()

        mock_evaluator = MagicMock()
        mock_evaluator.initialize.return_value = make_eval_result()
        mock_evaluator.analyse.return_value = eval_result

        mock_llm = MagicMock()
        mock_llm.draw_samples.return_value = [
            LLMResponse(response_text="return x", input_tokens=10, output_tokens=20, token_cost=0.003),
        ]

        mock_database = MagicMock()
        mock_database.should_stop = True  # stop after first iteration

        run_config = RunConfig(
            log_dir=str(tmp_path / "logs"),
            save_ckpt_dir=None,
            max_samples=1,
            sampler=SamplerConfig(samples_per_prompt=1),
            database=ProgramsDatabaseConfig(num_islands=2, reset_period=999),
        )

        with patch("alpha_evolve_sr.cli.evaluator_mod.Evaluator", return_value=mock_evaluator), \
             patch("alpha_evolve_sr.cli.sampler_mod.LLM", return_value=mock_llm), \
             patch("alpha_evolve_sr.cli.ProgramsDatabase.restore_or_create", return_value=mock_database):
            from alpha_evolve_sr.cli import main_single
            main_single(
                run_config, SAMPLE_PROMPT, SAMPLE_EVALUATE_CODE, SAMPLE_SEED_FUNCTION,
                {"x": [1, 2, 3]},
            )

        # register_program should be called with (EvalResult, SampleMessage)
        assert mock_database.register_program.call_count == 1
        call_args = mock_database.register_program.call_args[0]
        assert isinstance(call_args[1], SampleMessage)
        assert call_args[1].llm_response.response_text == "return x"
        assert call_args[1].llm_response.token_cost == 0.003
