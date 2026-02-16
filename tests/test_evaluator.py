"""Tests for the evaluator module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from alpha_evolve_sr.code_manipulation import Function, text_to_program
from alpha_evolve_sr.evaluator import Evaluator, Sandbox, _extract_python, _sample_to_program
from alpha_evolve_sr.messages import EvalResult
from tests.conftest import SAMPLE_SPEC


class TestExtractPython:
    """Tests for _extract_python helper."""

    def test_extracts_from_code_fence(self):
        text = "Here is the code:\n```python\nreturn x * 2\n```\nDone."
        assert _extract_python(text) == "\nreturn x * 2\n"

    def test_returns_raw_when_no_fence(self):
        text = "return x * 2"
        assert _extract_python(text) == "return x * 2"

    def test_extracts_first_python_block(self):
        text = "```python\nblock1\n```\nmore text\n```python\nblock2\n```"
        assert _extract_python(text) == "\nblock1\n"


class TestSampleToProgram:
    """Tests for _sample_to_program helper."""

    def test_returns_function_and_program_string(self):
        template = text_to_program(SAMPLE_SPEC)
        func, prog_str = _sample_to_program(
            "    return params[0] * x ** 2", template, "equation"
        )
        assert func.name == "equation"
        assert "params[0] * x ** 2" in func.body
        assert isinstance(prog_str, str)
        assert "def equation" in prog_str

    def test_replaces_function_body(self):
        template = text_to_program(SAMPLE_SPEC)
        new_body = "    return x + 42"
        func, _ = _sample_to_program(new_body, template, "equation")
        assert "x + 42" in func.body


class TestSandbox:
    """Tests for Sandbox.run."""

    def test_successful_execution(self):
        sandbox = Sandbox()
        try:
            program = "def my_func(data):\n    return (data['x'].sum(), [1.0]), True\n"
            result, success = sandbox.run(program, "my_func", {"x": __import__("numpy").array([1, 2, 3])}, 10)
            assert success
            assert result is not None
        finally:
            sandbox.clean()

    def test_timeout_returns_none(self):
        sandbox = Sandbox()
        try:
            program = "import time\ndef slow(data):\n    time.sleep(100)\n    return None, True\n"
            result, success = sandbox.run(program, "slow", {}, 1)
            assert not success
            assert result is None
        finally:
            sandbox.clean()

    def test_missing_function_returns_none(self):
        sandbox = Sandbox()
        try:
            program = "def other_func(data):\n    return 1, True\n"
            result, success = sandbox.run(program, "missing_func", {}, 5)
            assert not success
            assert result is None
        finally:
            sandbox.clean()

    def test_exception_in_code_returns_none(self):
        sandbox = Sandbox()
        try:
            program = "def bad(data):\n    raise ValueError('boom')\n"
            result, success = sandbox.run(program, "bad", {}, 5)
            assert not success
            assert result is None
        finally:
            sandbox.clean()


class TestEvaluator:
    """Tests for Evaluator.analyse."""

    @pytest.fixture
    def evaluator(self):
        template = text_to_program(SAMPLE_SPEC)
        eval_inst = Evaluator(
            template=template,
            function_to_evolve="equation",
            function_to_run="evaluate",
            data_dict={"x": __import__("numpy").array([1.0, 2.0, 3.0])},
        )
        yield eval_inst
        eval_inst.clean()

    def test_analyse_returns_eval_result_on_success(self, evaluator):
        """analyse() with valid initial body returns EvalResult."""
        result = evaluator.analyse(
            "    return params[0] * x, [1.0]",
            island_id=None,
            version_generated=None,
        )
        # Even if evaluate() returns a bad score, we get an EvalResult
        assert isinstance(result, EvalResult)
        assert result.function is not None
        assert result.function.name == "equation"
        assert result.evaluate_time is not None
        assert result.evaluate_time >= 0

    def test_analyse_returns_none_on_parse_error(self, evaluator):
        """analyse() returns None when the sample can't be parsed."""
        result = evaluator.analyse(
            "this is not valid python at all {{{",
            island_id=0,
            version_generated=1,
        )
        assert result is None

    def test_analyse_preserves_island_id(self, evaluator):
        """analyse() passes through island_id and token usage."""
        result = evaluator.analyse(
            "    return -1.0, None",
            island_id=5,
            version_generated=None,
            sample_time=1.5,
            sample_token_usage=(100, 200),
        )
        assert result is not None
        assert result.island_id == 5
        assert result.sample_time == 1.5
        assert result.sample_token_usage == (100, 200)

    def test_clean_is_idempotent(self, evaluator):
        """Calling clean() multiple times should not raise."""
        evaluator.clean()
        evaluator.clean()
