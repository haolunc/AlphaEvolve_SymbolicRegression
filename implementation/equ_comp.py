from collections import Counter
from types import FunctionType
from textwrap import indent
import numpy as np
import ast

# ----------------------------
# 1. default weight settings
# ----------------------------
OP_WEIGHTS = {
    ast.Add: 1,
    ast.Sub: 1,
    ast.Mult: 1,
    ast.Div: 1,
    ast.Pow: 1,
    ast.Mod: 1,
}



FUNC_WEIGHTS = {
    # 指数 / 对数
    "exp": 1,
    "log": 1,

    # 根号
    "sqrt": 1,

    # 三角函数
    "sin": 1,
    "cos": 1,
    "tan": 1,

    # 双曲函数
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,

    # 绝对值
    "abs": 1,
}

# ----------------------------
# 2. AST visitor
# ----------------------------
class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self, arg_names):
        self.total = 0
        self.breakdown = Counter()
        self.arg_names = arg_names

    # arithmetic like a+b, a*b, a**b …
    def visit_BinOp(self, node: ast.BinOp):
        w = OP_WEIGHTS.get(type(node.op), 1)
        self._add("BinOp", w)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        """计数所有"被读取"的变量名"""
        if isinstance(node.ctx, ast.Load) and node.id in self.arg_names:
            self._add('Var', 1)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):  # Python 3.8+
        if isinstance(node.value, (int, float, complex)):
            self._add("Const", 1)

    # function / method / attribute calls
    def visit_Call(self, node: ast.Call):
        fname = self._call_name(node.func)
        w = FUNC_WEIGHTS.get(fname, 1)
        self._add(f"Call:{fname}", w)
        self.generic_visit(node)

    # ----------------------
    # utility helpers
    # ----------------------
    def _add(self, label: str, w: int):
        self.total += w
        self.breakdown[label] += 1

    @staticmethod
    def _call_name(func_node) -> str:
        """
        Extract dotted name from ast.Call.func:
          • np.exp     -> 'exp'
          • math.sin   -> 'sin'
          • exp(x)     -> 'exp'
        Returns '' if we can't figure it out.
        """
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return ''
    
# ----------------------------
# 3. public API
# ----------------------------
def complexity_score(fun_text: str,
                     op_weights: dict = None,
                     func_weights: dict = None,
                     return_breakdown: bool = False):
    """
    Compute complexity score for a Python function.

    Parameters
    ----------
    fn : Python function object
    op_weights / func_weights : optional custom weight dicts
    return_breakdown : bool, if True also return a Counter

    Returns
    -------
    score : float
    [breakdown] : Counter (only if return_breakdown)
    """
    # allow per-call override
    global OP_WEIGHTS, FUNC_WEIGHTS
    if op_weights is not None:
        OP_WEIGHTS = op_weights
    if func_weights is not None:
        FUNC_WEIGHTS = func_weights

    # tree = ast.parse(inspect.getsource(fn))
    tree = ast.parse(fun_text)
    # print(indent(ast.dump(tree, indent=4), '    '))

    func_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    arg_names = {arg.arg for arg in func_def.args.args}
    arg_names.remove("params")
    vis = ComplexityVisitor(arg_names)
    vis.visit(tree)
    return (vis.total, vis.breakdown) if return_breakdown else vis.total