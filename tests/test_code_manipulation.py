"""Tests for code_manipulation module."""

import dataclasses

import pytest

from alpha_evolve_sr.code_manipulation import (
    ParsedFunction,
    text_to_function,
    text_to_program,
)


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


class TestDecoratorSupport:
    """Tests for ParsedFunction decorator parsing and round-tripping."""

    def test_parse_single_decorator(self):
        code = "@jax.jit\ndef foo(x):\n    return x\n"
        fn = text_to_function(code)
        assert fn.decorators == ("jax.jit",)

    def test_parse_multiple_decorators(self):
        code = "@decorator_a\n@decorator_b(arg=1)\ndef foo(x):\n    return x\n"
        fn = text_to_function(code)
        assert fn.decorators == ("decorator_a", "decorator_b(arg=1)")

    def test_parse_no_decorators(self):
        code = "def foo(x):\n    return x\n"
        fn = text_to_function(code)
        assert fn.decorators == ()

    def test_str_roundtrip(self):
        code = "@jax.jit\ndef foo(x):\n    return x\n"
        fn = text_to_function(code)
        reparsed = text_to_function(str(fn))
        assert reparsed.decorators == ("jax.jit",)
        assert reparsed.name == "foo"
        assert "return x" in reparsed.body

    def test_str_emits_decorator_lines(self):
        fn = ParsedFunction(
            name="f", args="x", body="    return x",
            decorators=("jax.jit", "functools.wraps(g)"),
        )
        source = str(fn)
        assert "@jax.jit\n" in source
        assert "@functools.wraps(g)\n" in source
        assert source.index("@jax.jit") < source.index("def f")

    def test_dataclasses_replace_preserves_decorators(self):
        fn = ParsedFunction(
            name="f", args="x", body="    return x", decorators=("jax.jit",),
        )
        fn2 = dataclasses.replace(fn, body="    return x * 2")
        assert fn2.decorators == ("jax.jit",)
        assert "x * 2" in fn2.body

    def test_program_with_decorated_function(self):
        code = "import jax\n\n@jax.jit\ndef foo(x):\n    return x\n"
        prog = text_to_program(code)
        assert len(prog.functions) == 1
        assert prog.functions[0].decorators == ("jax.jit",)
        assert "import jax" in prog.preface


