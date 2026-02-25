"""Tests for the evaluator module."""

from __future__ import annotations

import pytest

from alpha_evolve_sr.code_manipulation import ParsedFunction
from alpha_evolve_sr.evaluator import (
    Evaluator,
    Sandbox,
    _extract_python_text,
    _sample_to_program,
)
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

    def test_preserves_decorators_in_program_string(self):
        seed = ParsedFunction(
            name="equation", args="x, params", body="    return x",
            decorators=("jax.jit",),
        )
        func, prog_str = _sample_to_program("    return x * 2", SAMPLE_EVALUATE_CODE, seed)
        assert func.decorators == ("jax.jit",)
        assert "@jax.jit" in prog_str
        assert prog_str.index("@jax.jit") < prog_str.index("def equation")


class TestSandbox:
    """Tests for Sandbox.run."""

    @pytest.fixture
    def sandbox(self):
        sb = Sandbox()
        yield sb
        sb.clean()

    def test_successful_execution(self, sandbox):
        program = "def evaluate(data):\n    return (data['x'].sum(), [1.0]), True\n"
        result, success, error_str, eval_output = sandbox.run(program, {"x": __import__("numpy").array([1, 2, 3])}, 10)
        assert success
        assert result is not None
        assert error_str is None

    def test_timeout_returns_none(self, sandbox):
        program = "import time\ndef evaluate(data):\n    time.sleep(100)\n    return None, True\n"
        result, success, error_str, eval_output = sandbox.run(program, {}, 1)
        assert not success
        assert result is None
        assert error_str is not None
        assert "TimeoutError" in error_str
        assert eval_output == ""

    def test_missing_function_returns_none(self, sandbox):
        program = "def other_func(data):\n    return 1, True\n"
        result, success, error_str, eval_output = sandbox.run(program, {}, 5)
        assert not success
        assert result is None
        assert error_str is not None

    def test_exception_in_code_returns_none(self, sandbox):
        program = "def evaluate(data):\n    raise ValueError('boom')\n"
        result, success, error_str, eval_output = sandbox.run(program, {}, 5)
        assert not success
        assert result is None
        assert error_str is not None
        assert "ValueError" in error_str

    def test_captures_stdout_from_eval_code(self, sandbox):
        program = "def evaluate(data):\n    print('hello from eval')\n    return (-1.0, None)\n"
        result, success, error_str, eval_output = sandbox.run(program, {}, 10)
        assert success
        assert "hello from eval" in eval_output

    def test_captures_stderr_from_eval_code(self, sandbox):
        program = (
            "import sys\n"
            "def evaluate(data):\n"
            "    print('stderr msg', file=sys.stderr)\n"
            "    return (-1.0, None)\n"
        )
        result, success, error_str, eval_output = sandbox.run(program, {}, 10)
        assert success
        assert "stderr msg" in eval_output

    def test_eval_output_empty_when_no_prints(self, sandbox):
        program = "def evaluate(data):\n    return (-1.0, None)\n"
        result, success, error_str, eval_output = sandbox.run(program, {}, 10)
        assert success
        assert eval_output == ""

    def test_eval_output_on_timeout(self, sandbox):
        program = "import time\ndef evaluate(data):\n    time.sleep(100)\n    return None\n"
        result, success, error_str, eval_output = sandbox.run(program, {}, 1)
        assert not success
        assert eval_output == ""


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

    def test_analyse_returns_error_on_parse_failure(self, evaluator):
        """analyse() returns EvalResult with error info when parsing fails."""
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
        assert isinstance(result, EvalResult)
        assert result.execution_result is None
        assert result.error_type == "parse"
        assert result.error_message is not None

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

    def test_analyse_returns_error_when_evaluate_returns_none(self):
        """analyse() returns error_type='execution' when evaluate() returns None."""
        # evaluate_code where evaluate() calls equation() and propagates None
        eval_code = (
            "import numpy as np\n"
            "def evaluate(data):\n"
            "    result = equation(data['x'], [1.0])\n"
            "    return result\n"
        )
        eval_inst = Evaluator(
            evaluate_code=eval_code,
            seed_function=SAMPLE_SEED_FUNCTION,
            data_dict={"x": __import__("numpy").array([1.0, 2.0, 3.0])},
        )
        try:
            sample_msg = SampleMessage(
                llm_response=LLMResponse(
                    response_text="```python\ndef equation(x, params):\n    return None\n```",
                    input_tokens=10,
                    output_tokens=20,
                    token_cost=0.001,
                ),
                island_id=0,
                sample_time=0.5,
            )
            result = eval_inst.analyse(sample_msg)
            assert isinstance(result, EvalResult)
            assert result.execution_result is None
            assert result.error_type == "execution"
            assert result.error_message is not None
            assert "None" in result.error_message
        finally:
            eval_inst.clean()

    def test_clean_is_idempotent(self, evaluator):
        """Calling clean() multiple times should not raise."""
        evaluator.clean()
        evaluator.clean()


