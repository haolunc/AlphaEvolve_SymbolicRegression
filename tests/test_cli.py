"""Tests for cli module.

These are orchestration smoke tests with heavy mocking — they verify control flow
(e.g. seed eval is skipped on resume, finalize is always called), not actual
pipeline behavior.  Real integration tests would require an LLM provider and
evaluation sandbox and are out of scope here.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from alpha_evolve_sr.config import ProgramsDatabaseConfig, RunConfig, SamplerConfig
from alpha_evolve_sr.messages import LLMResponse, SampleMessage
from tests.conftest import (
    SAMPLE_EVALUATE_CODE,
    SAMPLE_PROMPT,
    SAMPLE_SEED_FUNCTION,
    finished_future,
    make_eval_result,
)


class TestRunPipeline:
    """Tests for the unified run_pipeline function."""

    def _make_run_config(self, tmp_path, **overrides):
        defaults = dict(
            log_dir=str(tmp_path / "logs"),
            save_ckpt_dir=None,
            max_samples=1,
            num_samplers=1,
            num_evaluators=1,
            sampler=SamplerConfig(samples_per_prompt=1),
            database=ProgramsDatabaseConfig(num_islands=2, reset_period=999),
        )
        defaults.update(overrides)
        return RunConfig(**defaults)

    @pytest.fixture
    def pipeline_mocks(self, tmp_path):
        """Common mocks for run_pipeline tests."""
        mock_database = MagicMock()
        mock_eval_pool = MagicMock()
        mock_eval_pool._max_workers = 1
        mock_sampler_pool = MagicMock()
        mock_sampler_pool.shutdown = MagicMock()
        mock_llm = MagicMock()

        with patch("alpha_evolve_sr.cli.ThreadPoolExecutor") as MockTPE, \
             patch("alpha_evolve_sr.cli.sampler_mod.LLM", return_value=mock_llm), \
             patch("alpha_evolve_sr.cli.ProgramsDatabase.restore_or_create", return_value=mock_database), \
             patch("alpha_evolve_sr.cli._attach_complexity", side_effect=lambda r: r), \
             patch("alpha_evolve_sr.cli._cleanup_eval_threads"):
            MockTPE.side_effect = [mock_eval_pool, mock_sampler_pool]
            yield {
                "database": mock_database,
                "eval_pool": mock_eval_pool,
                "sampler_pool": mock_sampler_pool,
                "llm": mock_llm,
            }

    def test_registers_eval_result_with_sample_message(self, tmp_path, pipeline_mocks):
        """run_pipeline registers eval results paired with sample messages."""
        eval_result = make_eval_result()
        sample_msg = SampleMessage(
            llm_response=LLMResponse(
                response_text="return x", input_tokens=10, output_tokens=20, token_cost=0.003,
            ),
            island_id=0,
            sample_time=0.1,
        )

        mock_database = pipeline_mocks["database"]
        mock_database.should_stop = False
        call_count = 0
        def register_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                mock_database.should_stop = True
        mock_database.register_program.side_effect = register_side_effect
        mock_database.get_prompt.return_value = MagicMock(code="test prompt", island_id=0)

        pipeline_mocks["llm"].query.return_value = LLMResponse(
            response_text="return x", input_tokens=10, output_tokens=20, token_cost=0.003,
        )

        seed_future = finished_future(eval_result)
        analyse_future = finished_future((eval_result, sample_msg))
        pipeline_mocks["eval_pool"].submit.side_effect = [seed_future, analyse_future]

        run_config = self._make_run_config(tmp_path)

        from alpha_evolve_sr.cli import run_pipeline
        run_pipeline(
            run_config, SAMPLE_PROMPT, SAMPLE_EVALUATE_CODE, SAMPLE_SEED_FUNCTION,
            {"x": [1, 2, 3]},
        )

        assert mock_database.register_program.call_count >= 1

    def test_seed_eval_skipped_on_resume(self, tmp_path, pipeline_mocks):
        """When resume_from_ckpt is set, seed evaluation is skipped."""
        mock_database = pipeline_mocks["database"]
        mock_database.should_stop = True

        run_config = self._make_run_config(tmp_path, resume_from_ckpt="/fake/checkpoint.db")

        from alpha_evolve_sr.cli import run_pipeline
        run_pipeline(
            run_config, SAMPLE_PROMPT, SAMPLE_EVALUATE_CODE, SAMPLE_SEED_FUNCTION,
            {"x": [1, 2, 3]},
        )

        pipeline_mocks["eval_pool"].submit.assert_not_called()
        mock_database.finalize.assert_called_once()

    def test_finalize_called_on_normal_exit(self, tmp_path, pipeline_mocks):
        """database.finalize() is always called on exit."""
        mock_database = pipeline_mocks["database"]
        mock_database.should_stop = True

        seed_future = finished_future(make_eval_result())
        pipeline_mocks["eval_pool"].submit.return_value = seed_future

        run_config = self._make_run_config(tmp_path)

        from alpha_evolve_sr.cli import run_pipeline
        run_pipeline(
            run_config, SAMPLE_PROMPT, SAMPLE_EVALUATE_CODE, SAMPLE_SEED_FUNCTION,
            {"x": [1, 2, 3]},
        )

        mock_database.finalize.assert_called_once()
