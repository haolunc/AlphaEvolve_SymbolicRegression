"""Tests for code_manipulation module."""

import dataclasses

import pytest

from alpha_evolve_sr.code_manipulation import (
    ParsedFunction,
    get_functions_called,
    rename_function_calls,
    text_to_function,
    text_to_program,
    yield_decorated,
)


class TestYieldDecorated:
    """Tests for the comment-marker decorator parser."""

    def test_finds_evaluate_run(self):
        code = "# @evaluate.run\ndef evaluate(data):\n    pass\n"
        result = list(yield_decorated(code, "evaluate", "run"))
        assert result == ["evaluate"]

    def test_finds_equation_evolve(self):
        code = "# @equation.evolve\ndef equation(x, params):\n    return x\n"
        result = list(yield_decorated(code, "equation", "evolve"))
        assert result == ["equation"]

    def test_no_match(self):
        code = "def foo(x):\n    return x\n"
        result = list(yield_decorated(code, "evaluate", "run"))
        assert result == []

    def test_ignores_wrong_marker(self):
        code = "# @other.marker\ndef foo(x):\n    pass\n"
        result = list(yield_decorated(code, "evaluate", "run"))
        assert result == []

    def test_blank_line_between_marker_and_def(self):
        code = "# @evaluate.run\n\ndef evaluate(data):\n    pass\n"
        # The marker must be on the immediately preceding non-blank line
        result = list(yield_decorated(code, "evaluate", "run"))
        assert result == ["evaluate"]


class TestTextToFunction:
    """Tests for parsing and stringifying functions."""

    def test_roundtrip(self, sample_function):
        source = str(sample_function)
        parsed = text_to_function(source)
        assert parsed.name == sample_function.name
        assert parsed.args == sample_function.args
        assert parsed.body.strip() == sample_function.body.strip()

    def test_single_function_expected(self):
        code = "def foo():\n    pass\ndef bar():\n    pass\n"
        with pytest.raises(ValueError, match="Only one function expected"):
            text_to_function(code)


class TestTextToProgram:
    """Tests for parsing programs."""

    def test_parses_preface_and_functions(self, sample_program):
        source = str(sample_program)
        parsed = text_to_program(source)
        assert len(parsed.functions) == 1
        assert parsed.functions[0].name == "equation"

    def test_get_function_by_name(self, sample_program):
        fn = sample_program.get_function("equation")
        assert fn.name == "equation"

    def test_get_function_not_found(self, sample_program):
        with pytest.raises(ValueError, match="does not exist"):
            sample_program.get_function("nonexistent")

    def test_parses_docstring_via_ast_constant(self):
        code = 'def foo():\n    """My docstring."""\n    return 1\n'
        program = text_to_program(code)
        fn = program.functions[0]
        assert fn.docstring is not None
        assert "My docstring." in fn.docstring

    def test_parses_function_without_docstring(self):
        code = "def foo():\n    return 1\n"
        program = text_to_program(code)
        fn = program.functions[0]
        assert fn.docstring is None


class TestRenameFunctionCalls:
    """Tests for renaming function calls in code."""

    def test_renames_call(self):
        code = "def foo():\n    return foo()\n"
        result = rename_function_calls(code, "foo", "bar")
        assert "bar()" in result

    def test_does_not_rename_attribute_calls(self):
        code = "def foo():\n    return obj.foo()\n"
        result = rename_function_calls(code, "foo", "bar")
        assert "obj.foo()" in result

    def test_noop_when_name_not_present(self):
        code = "def baz():\n    return 1\n"
        result = rename_function_calls(code, "foo", "bar")
        assert result == code


class TestParsedFunctionSanitisation:
    """Tests for ParsedFunction __post_init__ sanitisation."""

    def test_body_strips_newlines(self):
        fn = ParsedFunction(name="f", args="x", body="\n    return x\n\n")
        assert not fn.body.startswith("\n")
        assert not fn.body.endswith("\n")

    def test_docstring_strips_triple_quotes(self):
        fn = ParsedFunction(name="f", args="x", body="    pass", docstring='"""hello"""')
        assert '"""' not in fn.docstring
        assert "hello" in fn.docstring

    def test_frozen(self):
        fn = ParsedFunction(name="f", args="x", body="    pass")
        with pytest.raises(dataclasses.FrozenInstanceError):
            fn.name = "g"

    def test_replace_creates_new_instance(self):
        fn = ParsedFunction(name="f", args="x", body="    pass")
        fn2 = dataclasses.replace(fn, name="g")
        assert fn.name == "f"
        assert fn2.name == "g"


class TestGetFunctionsCalled:
    """Tests for extracting called function names."""

    def test_finds_calls(self):
        code = "def foo():\n    bar()\n    baz(1)\n"
        called = get_functions_called(code)
        assert "bar" in called
        assert "baz" in called
