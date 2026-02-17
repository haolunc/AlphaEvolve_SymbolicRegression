"""Tests for queue message dataclasses."""

from __future__ import annotations

import pytest

from alpha_evolve_sr.code_manipulation import ParsedFunction
from alpha_evolve_sr.messages import EvalResult, ExecutionResult, LLMResponse, PerfMessage, SampleMessage


class TestLLMResponse:
    def test_creation_and_fields(self):
        resp = LLMResponse(
            response_text="return x",
            input_tokens=10,
            output_tokens=20,
            token_cost=0.001,
        )
        assert resp.response_text == "return x"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 20
        assert resp.token_cost == 0.001

    def test_default_cost(self):
        resp = LLMResponse(response_text="x", input_tokens=1, output_tokens=2)
        assert resp.token_cost == 0.0

    def test_frozen(self):
        resp = LLMResponse(response_text="x", input_tokens=1, output_tokens=2)
        with pytest.raises(AttributeError):
            resp.response_text = "y"


class TestSampleMessage:
    def test_creation_and_fields(self):
        llm_resp = LLMResponse(
            response_text="return x",
            input_tokens=10,
            output_tokens=20,
            token_cost=0.001,
        )
        msg = SampleMessage(
            llm_response=llm_resp,
            island_id=2,
            sample_time=0.5,
        )
        assert msg.llm_response.response_text == "return x"
        assert msg.llm_response.input_tokens == 10
        assert msg.llm_response.output_tokens == 20
        assert msg.llm_response.token_cost == 0.001
        assert msg.island_id == 2
        assert msg.sample_time == 0.5

    def test_frozen(self):
        llm_resp = LLMResponse(response_text="x", input_tokens=5, output_tokens=10)
        msg = SampleMessage(llm_resp, 0, 0.1)
        with pytest.raises(AttributeError):
            msg.llm_response = llm_resp


class TestExecutionResult:
    def test_creation_and_fields(self):
        er = ExecutionResult(
            score=0.5,
            optimized_params=[1.0, 2.0],
            complexity=10,
            complexity_detail={"BinOp": 3},
        )
        assert er.score == 0.5
        assert er.optimized_params == [1.0, 2.0]
        assert er.complexity == 10
        assert er.complexity_detail == {"BinOp": 3}

    def test_frozen(self):
        er = ExecutionResult(score=0.5, optimized_params=None, complexity=5, complexity_detail={})
        with pytest.raises(AttributeError):
            er.score = 1.0


class TestEvalResult:
    def test_creation_and_fields(self):
        func = ParsedFunction(name="eq", args="x", body="    return x")
        ex = ExecutionResult(score=0.5, optimized_params=None, complexity=5, complexity_detail={})
        msg = EvalResult(
            function=func,
            execution_result=ex,
            evaluate_time=0.2,
        )
        assert msg.function is func
        assert msg.execution_result is ex
        assert msg.evaluate_time == 0.2

    def test_frozen(self):
        func = ParsedFunction(name="eq", args="x", body="    return x")
        msg = EvalResult(func, None, None)
        with pytest.raises(AttributeError):
            msg.evaluate_time = 5.0


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
