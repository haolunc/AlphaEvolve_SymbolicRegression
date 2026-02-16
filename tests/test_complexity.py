"""Tests for complexity scoring."""

import ast

from alpha_evolve_sr.complexity import complexity_score


class TestComplexityScore:
    """Tests for the public complexity_score function."""

    def test_simple_function(self):
        code = "def f(x):\n    return x + 1\n"
        score = complexity_score(code)
        assert isinstance(score, int)
        assert score > 0

    def test_with_breakdown(self):
        code = "def f(x):\n    return x * 2 + 1\n"
        score, breakdown = complexity_score(code, return_breakdown=True)
        assert isinstance(score, int)
        assert "BinOp" in breakdown

    def test_no_params_crash(self):
        """Function without 'params' arg should not crash (bug fix 2e)."""
        code = "def f(x, y):\n    return x + y\n"
        score = complexity_score(code)
        assert isinstance(score, int)

    def test_with_params_arg(self):
        """Function with 'params' arg should exclude it from variable counting."""
        code = "def f(x, params):\n    return x + params[0]\n"
        score = complexity_score(code)
        assert isinstance(score, int)

    def test_custom_weights(self):
        code = "def f(x):\n    return x + 1\n"
        # Custom weight: Add costs 5
        score_custom = complexity_score(code, op_weights={ast.Add: 5})
        score_default = complexity_score(code)
        assert score_custom > score_default


class TestNoGlobalMutation:
    """Verify that custom weights don't mutate global state (bug fix 2d)."""

    def test_custom_weights_dont_persist(self):
        code = "def f(x):\n    return x + 1\n"
        score1 = complexity_score(code)
        complexity_score(code, op_weights={ast.Add: 100})
        score3 = complexity_score(code)
        assert score1 == score3
