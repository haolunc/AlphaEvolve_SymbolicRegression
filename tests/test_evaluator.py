"""Tests for the evaluator module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from alpha_evolve_sr.code_manipulation import ParsedFunction
from alpha_evolve_sr.evaluator import Evaluator, Sandbox, _extract_python_text, _sample_to_program
from alpha_evolve_sr.messages import EvalResult, LLMResponse, SampleMessage
from tests.conftest import SAMPLE_EVALUATE_CODE, SAMPLE_SEED_FUNCTION


class TestExtractPython:
    """Tests for _extract_python_text helper."""

    def test_extracts_from_code_fence(self):
        text = "Here is the code:\n```python\nreturn x * 2\n```\nDone."
        assert _extract_python_text(text) == "\nreturn x * 2\n"

    def test_returns_raw_when_no_fence(self):
        text = "return x * 2"
        assert _extract_python_text(text) == "return x * 2"

    def test_extracts_first_python_block(self):
        text = "```python\nblock1\n```\nmore text\n```python\nblock2\n```"
        assert _extract_python_text(text) == "\nblock1\n"


class TestSampleToProgram:
    """Tests for _sample_to_program helper."""

    def test_returns_function_and_program_string(self):
        func, prog_str = _sample_to_program(
            "    return params[0] * x ** 2", SAMPLE_EVALUATE_CODE, SAMPLE_SEED_FUNCTION
        )
        assert func.name == "equation"
        assert "params[0] * x ** 2" in func.body
        assert isinstance(prog_str, str)
        assert "def equation" in prog_str

    def test_replaces_function_body(self):
        new_body = "    return x + 42"
        func, _ = _sample_to_program(new_body, SAMPLE_EVALUATE_CODE, SAMPLE_SEED_FUNCTION)
        assert "x + 42" in func.body


class TestSandbox:
    """Tests for Sandbox.run."""

    @pytest.fixture
    def sandbox(self):
        sb = Sandbox()
        yield sb
        sb.clean()

    def test_successful_execution(self, sandbox):
        program = "def evaluate(data):\n    return (data['x'].sum(), [1.0]), True\n"
        result, success = sandbox.run(program, {"x": __import__("numpy").array([1, 2, 3])}, 10)
        assert success
        assert result is not None

    def test_timeout_returns_none(self, sandbox):
        program = "import time\ndef evaluate(data):\n    time.sleep(100)\n    return None, True\n"
        result, success = sandbox.run(program, {}, 1)
        assert not success
        assert result is None

    def test_missing_function_returns_none(self, sandbox):
        program = "def other_func(data):\n    return 1, True\n"
        result, success = sandbox.run(program, {}, 5)
        assert not success
        assert result is None

    def test_exception_in_code_returns_none(self, sandbox):
        program = "def evaluate(data):\n    raise ValueError('boom')\n"
        result, success = sandbox.run(program, {}, 5)
        assert not success
        assert result is None


class TestEvaluator:
    """Tests for Evaluator.initialize and Evaluator.analyse."""

    @pytest.fixture
    def evaluator(self):
        eval_inst = Evaluator(
            evaluate_code=SAMPLE_EVALUATE_CODE,
            seed_function=SAMPLE_SEED_FUNCTION,
            data_dict={"x": __import__("numpy").array([1.0, 2.0, 3.0])},
        )
        yield eval_inst
        eval_inst.clean()

    def test_initialize_returns_eval_result(self, evaluator):
        """initialize() evaluates the seed function and returns EvalResult."""
        result = evaluator.initialize()
        assert isinstance(result, EvalResult)
        assert result.function is not None
        assert result.function.name == "equation"
        assert result.evaluate_time is not None
        assert result.evaluate_time >= 0

    def test_analyse_returns_eval_result_on_success(self, evaluator):
        """analyse() with a valid LLM-style sample returns EvalResult."""
        sample_msg = SampleMessage(
            llm_response=LLMResponse(
                response_text="```python\ndef equation(x, params):\n    return params[0] * x, [1.0]\n```",
                input_tokens=10,
                output_tokens=20,
                token_cost=0.001,
            ),
            island_id=0,
            sample_time=0.5,
        )
        result = evaluator.analyse(sample_msg)
        assert isinstance(result, EvalResult)
        assert result.function is not None
        assert result.function.name == "equation"
        assert result.evaluate_time is not None
        assert result.evaluate_time >= 0

    def test_analyse_returns_none_on_parse_error(self, evaluator):
        """analyse() returns None when the sample can't be parsed."""
        sample_msg = SampleMessage(
            llm_response=LLMResponse(
                response_text="this is not valid python at all {{{",
                input_tokens=10,
                output_tokens=20,
                token_cost=0.001,
            ),
            island_id=0,
            sample_time=0.5,
        )
        result = evaluator.analyse(sample_msg)
        assert result is None

    def test_analyse_eval_result_has_no_sampling_metadata(self, evaluator):
        """EvalResult from analyse() does not contain sampling metadata."""
        sample_msg = SampleMessage(
            llm_response=LLMResponse(
                response_text="```python\ndef equation(x, params):\n    return -1.0, None\n```",
                input_tokens=100,
                output_tokens=200,
                token_cost=0.05,
            ),
            island_id=5,
            sample_time=1.5,
        )
        result = evaluator.analyse(sample_msg)
        assert result is not None
        # Sampling metadata lives in SampleMessage, not EvalResult
        assert not hasattr(result, "island_id")
        assert not hasattr(result, "sample_time")
        assert not hasattr(result, "sample_token_usage")

    def test_clean_is_idempotent(self, evaluator):
        """Calling clean() multiple times should not raise."""
        evaluator.clean()
        evaluator.clean()
