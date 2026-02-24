# Copyright 2023 DeepMind Technologies Limited
# Copyright 2026 Haolun Cai
# This file has been modified by Haolun Cai for AlphaEvolve_SymbolicRegression on Jan 21, 2026.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tools for manipulating Python code.

It implements three dataclasses for representing code units:

- ``ParsedFunction`` — a frozen, immutable representation of a Python
  function's AST data (name, args, body, docstring, etc.).
- ``EvaluatedProgram`` — a mutable container that pairs a ``ParsedFunction``
  with evaluation / runtime metrics (score, complexity, timing, cost).
- ``Program`` — a sequence of ``ParsedFunction`` objects together with a
  code preface (imports, globals, etc.).
"""
import ast
import dataclasses
import io
import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ParsedFunction — immutable AST data
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ParsedFunction:
    """Immutable representation of a parsed Python function's AST data."""

    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None
    decorators: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Sanitize body: strip leading/trailing newlines.
        if isinstance(self.body, str):
            object.__setattr__(self, "body", self.body.strip("\n"))
        # Sanitize docstring: remove triple-quote wrappers.
        if isinstance(self.docstring, str) and '"""' in self.docstring:
            cleaned = self.docstring.strip().replace('"""', "")
            object.__setattr__(self, "docstring", cleaned)

    def __str__(self) -> str:
        result = ""
        for dec in self.decorators:
            result += f"@{dec}\n"

        return_type = f" -> {self.return_type}" if self.return_type else ""

        function = result + f"def {self.name}({self.args}){return_type}:\n"
        if self.docstring:
            new_line = "\n" if self.body else ""
            function += f'    """{self.docstring}"""{new_line}'
        # self.body is already indented.
        function += self.body + "\n\n"
        return function

    def save_to_file(self, filepath: str, append: bool = False) -> None:
        """Writes the function definition to ``filepath``."""
        mode = "a" if append else "w"
        function_source = str(self)

        with open(filepath, mode, encoding="utf-8") as file:
            if append:
                file.seek(0, io.SEEK_END)
                if file.tell() > 0:
                    file.write("\n")
            file.write(function_source)



# ---------------------------------------------------------------------------
# EvaluatedProgram — ParsedFunction + evaluation / runtime metrics
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EvaluatedProgram:
    """A ``ParsedFunction`` paired with evaluation results and runtime metadata."""

    parsed: ParsedFunction
    score: float | None = None
    optimized_params: list[float] | None = None
    complexity: int | None = None
    complexity_detail: dict | None = None
    global_sample_nums: int | None = None
    sample_time: float | None = None
    evaluate_time: float | None = None
    token_usage: tuple[int, int] | None = None
    token_cost: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    eval_output: str | None = None

    # -- Convenience proxies so callers don't always need ``.parsed`` ------

    @property
    def name(self) -> str:
        return self.parsed.name

    def __str__(self) -> str:
        return str(self.parsed)

    def save_to_file(self, filepath: str, append: bool = False) -> None:
        """Writes the function definition to ``filepath``."""
        self.parsed.save_to_file(filepath, append)


# ---------------------------------------------------------------------------
# Program — preface + list of ParsedFunction
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Program:
    """A parsed Python program."""

    # `preface` is everything from the beginning of the code till the first
    # function is found.
    preface: str
    functions: list[ParsedFunction]

    def __str__(self) -> str:
        program = f"{self.preface}\n" if self.preface else ""
        program += "\n".join([str(f) for f in self.functions])
        return program

    def find_function_index(self, function_name: str) -> int:
        """Returns the index of input function name."""
        function_names = [f.name for f in self.functions]
        count = function_names.count(function_name)
        if count == 0:
            raise ValueError(
                f"function {function_name} does not exist in program:\n{str(self)}"
            )
        if count > 1:
            raise ValueError(
                f"function {function_name} exists more than once in program:\n"
                f"{str(self)}"
            )
        index = function_names.index(function_name)
        return index

    def get_function(self, function_name: str) -> ParsedFunction:
        """Returns the ``ParsedFunction`` object for *function_name*."""
        index = self.find_function_index(function_name)
        return self.functions[index]


class ProgramVisitor(ast.NodeVisitor):
    """Parses code to collect all required information to produce a ``Program``."""

    def __init__(self, sourcecode: str):
        self._codelines: list[str] = sourcecode.splitlines()

        self._preface: str = ""
        self._functions: list[ParsedFunction] = []
        self._current_function: str | None = None

    def visit_FunctionDef(
        self, node: ast.FunctionDef
    ) -> None:
        """Collects all information about the function being parsed."""
        if node.col_offset == 0:  # We only care about first level functions.
            self._current_function = node.name
            if not self._functions:
                has_decorators = bool(node.decorator_list)
                if has_decorators:
                    # Find the minimum line number and retain the code
                    decorator_start_line = min(
                        decorator.lineno for decorator in node.decorator_list
                    )
                    self._preface = "\n".join(
                        self._codelines[: decorator_start_line - 1]
                    )
                else:
                    self._preface = "\n".join(self._codelines[:node.lineno - 1])

            function_end_line = node.end_lineno
            body_start_line = node.body[0].lineno - 1
            # Extract the docstring.
            docstring = None
            if isinstance(node.body[0], ast.Expr) and isinstance(
                node.body[0].value, ast.Constant
            ) and isinstance(node.body[0].value.value, str):
                docstring = f'    """{ast.literal_eval(ast.unparse(node.body[0]))}"""'
                if len(node.body) > 1:
                    body_start_line = node.body[1].lineno - 1
                else:
                    body_start_line = function_end_line

            decorators = tuple(ast.unparse(d) for d in node.decorator_list)

            self._functions.append(
                ParsedFunction(
                    name=node.name,
                    args=ast.unparse(node.args),
                    return_type=ast.unparse(node.returns) if node.returns else None,
                    docstring=docstring,
                    body="\n".join(self._codelines[body_start_line:function_end_line]),
                    decorators=decorators,
                )
            )
        self.generic_visit(node)

    def return_program(self) -> Program:
        """Returns the parsed ``Program``."""
        return Program(preface=self._preface, functions=self._functions)


def text_to_program(text: str) -> Program:
    """Returns Program object by parsing input text using Python AST."""
    try:
        tree = ast.parse(text)
        visitor = ProgramVisitor(text)
        visitor.visit(tree)
        return visitor.return_program()
    except Exception:
        logger.warning("Failed parsing %s", text)
        raise


def text_to_function(text: str) -> ParsedFunction:
    """Returns ParsedFunction object by parsing input text using Python AST."""
    program = text_to_program(text)
    if len(program.functions) != 1:
        raise ValueError(
            f"Only one function expected, got {len(program.functions)}"
            f":\n{program.functions}"
        )
    return program.functions[0]


def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
    """Renames function calls from ``source_name`` to ``target_name``."""
    if source_name not in code:
        return code
    return re.sub(
        rf"(?<!\.)\b{re.escape(source_name)}(?=\s*\()",
        target_name,
        code,
    )
