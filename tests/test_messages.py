"""Tests for queue message dataclasses."""

from __future__ import annotations

import pytest

from alpha_evolve_sr.code_manipulation import ParsedFunction
from alpha_evolve_sr.messages import EvalResult, PerfMessage, SampleMessage


class TestSampleMessage:
    def test_creation_and_fields(self):
        msg = SampleMessage(
            sample="return x",
            island_id=2,
            version_generated=3,
            sample_time=0.5,
            sample_token_usage=(10, 20),
            sample_token_cost=0.001,
        )
        assert msg.sample == "return x"
        assert msg.island_id == 2
        assert msg.version_generated == 3
        assert msg.sample_time == 0.5
        assert msg.sample_token_usage == (10, 20)
        assert msg.sample_token_cost == 0.001

    def test_frozen(self):
        msg = SampleMessage("x", 0, 1, 0.1, (5, 10), 0.001)
        with pytest.raises(AttributeError):
            msg.sample = "y"


class TestEvalResult:
    def test_creation_and_fields(self):
        func = ParsedFunction(name="eq", args="x", body="    return x")
        msg = EvalResult(
            function=func,
            island_id=1,
            result_per_test={"score": 0.5},
            sample_time=0.1,
            evaluate_time=0.2,
            sample_token_usage=(10, 20),
            sample_token_cost=0.001,
        )
        assert msg.function is func
        assert msg.island_id == 1
        assert msg.evaluate_time == 0.2

    def test_frozen(self):
        func = ParsedFunction(name="eq", args="x", body="    return x")
        msg = EvalResult(func, 0, None, None, None, None, None)
        with pytest.raises(AttributeError):
            msg.island_id = 5


class TestPerfMessage:
    def test_creation_and_fields(self):
        msg = PerfMessage(
            worker_type="sampler",
            worker_id=3,
            stats={"prompts": 10},
        )
        assert msg.worker_type == "sampler"
        assert msg.worker_id == 3
        assert msg.stats["prompts"] == 10

    def test_frozen(self):
        msg = PerfMessage("db", 0, {})
        with pytest.raises(AttributeError):
            msg.worker_type = "eval"
