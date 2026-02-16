"""Equation complexity scoring via AST analysis."""

import ast
from collections import Counter

from .logging_config import get_logger

logger = get_logger("complexity")

# Default weight settings
_DEFAULT_OP_WEIGHTS: dict[type, int] = {
    ast.Add: 1,
    ast.Sub: 1,
    ast.Mult: 1,
    ast.Div: 1,
    ast.Pow: 1,
    ast.Mod: 1,
}

_DEFAULT_FUNC_WEIGHTS: dict[str, int] = {
    # Exponential / logarithmic
    "exp": 1,
    "log": 1,
    # Square root
    "sqrt": 1,
    # Trigonometric
    "sin": 1,
    "cos": 1,
    "tan": 1,
    # Hyperbolic
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
    # Absolute value
    "abs": 1,
}


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor that computes a weighted complexity score."""

    def __init__(
        self,
        arg_names: set[str],
        op_weights: dict[type, int] | None = None,
        func_weights: dict[str, int] | None = None,
    ):
        self.total = 0
        self.breakdown: Counter = Counter()
        self.arg_names = arg_names
        self._op_weights = op_weights or _DEFAULT_OP_WEIGHTS
        self._func_weights = func_weights or _DEFAULT_FUNC_WEIGHTS

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Count binary operations (``a + b``, ``a * b``, ``a ** b``, ...)."""
        w = self._op_weights.get(type(node.op), 1)
        self._add("BinOp", w)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Count all variable name reads."""
        if isinstance(node.ctx, ast.Load) and node.id in self.arg_names:
            self._add("Var", 1)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        """Count numeric constants."""
        if isinstance(node.value, (int, float, complex)):
            self._add("Const", 1)

    def visit_Call(self, node: ast.Call) -> None:
        """Count function / method / attribute calls."""
        fname = self._call_name(node.func)
        w = self._func_weights.get(fname, 1)
        self._add(f"Call:{fname}", w)
        self.generic_visit(node)

    def _add(self, label: str, w: int) -> None:
        self.total += w
        self.breakdown[label] += 1

    @staticmethod
    def _call_name(func_node: ast.expr) -> str:
        """Extract the leaf name from an ``ast.Call.func`` node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return ""


def complexity_score(
    fun_text: str,
    op_weights: dict[type, int] | None = None,
    func_weights: dict[str, int] | None = None,
    return_breakdown: bool = False,
) -> int | tuple[int, Counter]:
    """Compute complexity score for a Python function.

    Args:
        fun_text: Source text of a Python function.
        op_weights: Optional custom weight dict for binary operators.
        func_weights: Optional custom weight dict for function calls.
        return_breakdown: If ``True``, also return a ``Counter`` breakdown.

    Returns:
        The integer complexity score, or ``(score, breakdown)`` when
        *return_breakdown* is ``True``.
    """
    tree = ast.parse(fun_text)

    func_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    arg_names = {arg.arg for arg in func_def.args.args}
    arg_names.discard("params")

    vis = ComplexityVisitor(arg_names, op_weights, func_weights)
    vis.visit(tree)
    return (vis.total, vis.breakdown) if return_breakdown else vis.total
