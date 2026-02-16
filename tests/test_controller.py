"""Tests for the EvolutionController."""

from __future__ import annotations

import os

import pytest

from alpha_evolve_sr.code_manipulation import ParsedFunction, text_to_program
from alpha_evolve_sr.config import ProgramsDatabaseConfig
from alpha_evolve_sr.controller import EvolutionController
from alpha_evolve_sr.messages import EvalResult
from tests.conftest import SAMPLE_SPEC


def _make_eval_result(island_id=None, score=-1.0):
    func = ParsedFunction(name="equation", args="x, params", body="    return params[0] * x")
    return EvalResult(
        function=func,
        island_id=island_id,
        result_per_test={"score": score, "optimized_params": None, "complexity": 5, "complexity_detail": {}},
        sample_time=0.1,
        evaluate_time=0.2,
        sample_token_usage=(10, 20),
        sample_token_cost=0.001,
    )


@pytest.fixture
def controller(tmp_path):
    config = ProgramsDatabaseConfig(num_islands=2, reset_period=999)
    template = text_to_program(SAMPLE_SPEC)
    return EvolutionController(
        config, template, "equation", str(tmp_path / "logs"),
        ckpt_dir=str(tmp_path / "ckpts"), ckpt_interval=0, max_samples=5,
    )


class TestInitialize:
    def test_initialize_creates_database(self, controller):
        initial = _make_eval_result(island_id=None)
        controller.initialize(initial_result=initial)
        assert controller.database is not None
        assert controller.sample_count == 1

    def test_initialize_without_initial_result(self, controller):
        controller.initialize()
        assert controller.sample_count == 0


class TestRegisterAndStop:
    def test_register_increments_count(self, controller):
        controller.initialize(initial_result=_make_eval_result())
        assert controller.sample_count == 1

        controller.register_eval_result(_make_eval_result(island_id=0))
        assert controller.sample_count == 2

    def test_should_stop(self, controller):
        controller.initialize(initial_result=_make_eval_result())
        assert not controller.should_stop

        for _ in range(4):
            controller.register_eval_result(_make_eval_result(island_id=0))
        assert controller.should_stop


class TestCheckpoint:
    def test_maybe_checkpoint_writes_file(self, controller):
        controller.initialize(initial_result=_make_eval_result())
        # Force the last checkpoint time far enough in the past
        controller._last_checkpoint_time = 0.0
        saved = controller.maybe_checkpoint()
        assert saved
        ckpt_dir = controller._ckpt_dir
        assert any("checkpoint_" in f for f in os.listdir(ckpt_dir))

    def test_maybe_checkpoint_no_dir(self, tmp_path):
        config = ProgramsDatabaseConfig(num_islands=2, reset_period=999)
        template = text_to_program(SAMPLE_SPEC)
        ctrl = EvolutionController(
            config, template, "equation", str(tmp_path / "logs"),
            ckpt_dir=None,
        )
        ctrl.initialize(initial_result=_make_eval_result())
        assert not ctrl.maybe_checkpoint()


class TestFinalize:
    def test_finalize_writes_outputs(self, controller):
        controller.initialize(initial_result=_make_eval_result())
        controller.finalize()
        # Should write final checkpoint + best programs file
        assert os.path.exists(os.path.join(controller._ckpt_dir, "checkpoint_final.pkl"))


class TestGetPrompt:
    def test_get_prompt_returns_prompt(self, controller):
        controller.initialize(initial_result=_make_eval_result())
        prompt = controller.get_prompt()
        assert hasattr(prompt, "code")
        assert hasattr(prompt, "island_id")
        assert len(prompt.code) > 0
