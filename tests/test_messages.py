"""Tests for queue message dataclasses."""

from __future__ import annotations

from alpha_evolve_sr.code_manipulation import ParsedFunction
from alpha_evolve_sr.messages import EvalResult, ExecutionResult, LLMResponse, SampleMessage


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


class TestExecutionResult:
    def test_creation_and_fields(self):
        er = ExecutionResult(
            score=0.5,
            optimized_params=[1.0, 2.0],
        )
        assert er.score == 0.5
        assert er.optimized_params == [1.0, 2.0]


class TestEvalResult:
    def test_creation_and_fields(self):
        func = ParsedFunction(name="eq", args="x", body="    return x")
        ex = ExecutionResult(score=0.5, optimized_params=None)
        msg = EvalResult(
            function=func,
            execution_result=ex,
            evaluate_time=0.2,
            complexity=5,
            complexity_detail={"BinOp": 3},
        )
        assert msg.function is func
        assert msg.execution_result is ex
        assert msg.evaluate_time == 0.2
        assert msg.complexity == 5
        assert msg.complexity_detail == {"BinOp": 3}

    def test_complexity_defaults_to_none(self):
        func = ParsedFunction(name="eq", args="x", body="    return x")
        ex = ExecutionResult(score=0.5, optimized_params=None)
        msg = EvalResult(function=func, execution_result=ex, evaluate_time=0.1)
        assert msg.complexity is None
        assert msg.complexity_detail is None
