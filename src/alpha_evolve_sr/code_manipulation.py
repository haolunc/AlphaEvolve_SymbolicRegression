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
import tokenize
from collections.abc import Iterator, MutableSet, Sequence

logger = logging.getLogger(__name__)


def yield_decorated(code: str, module: str, name: str) -> Iterator[str]:
    """Yields names of functions whose preceding comment contains ``@module.name``.

    Looks for comment-style markers like ``# @evaluate.run`` on the line
    immediately before a function definition.
    """
    lines = code.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def "):
            # Check the preceding non-blank line for a comment marker
            for j in range(i - 1, -1, -1):
                prev = lines[j].strip()
                if not prev:
                    continue
                pattern = rf"#\s*@{re.escape(module)}\.{re.escape(name)}"
                if re.search(pattern, prev):
                    # Extract function name
                    match = re.match(r"def\s+(\w+)\s*\(", stripped)
                    if match:
                        yield match.group(1)
                break


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

    def __post_init__(self) -> None:
        # Sanitize body: strip leading/trailing newlines.
        if isinstance(self.body, str):
            object.__setattr__(self, "body", self.body.strip("\n"))
        # Sanitize docstring: remove triple-quote wrappers.
        if isinstance(self.docstring, str) and '"""' in self.docstring:
            cleaned = self.docstring.strip().replace('"""', "")
            object.__setattr__(self, "docstring", cleaned)

    def __str__(self) -> str:
        return_type = f" -> {self.return_type}" if self.return_type else ""

        function = f"def {self.name}({self.args}){return_type}:\n"
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
                    # Check for comment-style decorator markers above the def
                    start_line = node.lineno - 1
                    for j in range(start_line - 1, -1, -1):
                        stripped = self._codelines[j].strip()
                        if stripped.startswith("#") and "@" in stripped:
                            start_line = j
                            break
                        if stripped:
                            break
                    self._preface = "\n".join(self._codelines[:start_line])

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

            self._functions.append(
                ParsedFunction(
                    name=node.name,
                    args=ast.unparse(node.args),
                    return_type=ast.unparse(node.returns) if node.returns else None,
                    docstring=docstring,
                    body="\n".join(self._codelines[body_start_line:function_end_line]),
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
    except Exception as e:
        logger.warning("Failed parsing %s", text)
        raise e


def text_to_function(text: str) -> ParsedFunction:
    """Returns ParsedFunction object by parsing input text using Python AST."""
    program = text_to_program(text)
    if len(program.functions) != 1:
        raise ValueError(
            f"Only one function expected, got {len(program.functions)}"
            f":\n{program.functions}"
        )
    return program.functions[0]


def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
    """Transforms ``code`` into Python tokens."""
    code_bytes = code.encode()
    code_io = io.BytesIO(code_bytes)
    return tokenize.tokenize(code_io.readline)


def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
    """Transforms a list of Python tokens into code."""
    code_bytes = tokenize.untokenize(tokens)
    return code_bytes.decode()


def _yield_token_and_is_call(code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
    """Yields each token with a bool indicating whether it is a function call."""
    try:
        tokens = _tokenize(code)
        prev_token = None
        is_attribute_access = False
        for token in tokens:
            if (
                prev_token
                and prev_token.type == tokenize.NAME
                and token.type == tokenize.OP
                and token.string == "("
            ):
                yield prev_token, not is_attribute_access
                is_attribute_access = False
            else:
                if prev_token:
                    is_attribute_access = (
                        prev_token.type == tokenize.OP and prev_token.string == "."
                    )
                    yield prev_token, False
            prev_token = token
        if prev_token:
            yield prev_token, False
    except Exception as e:
        logger.warning("Failed parsing %s", code)
        raise e


def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
    """Renames function calls from ``source_name`` to ``target_name``."""
    if source_name not in code:
        return code
    modified_tokens = []
    for token, is_call in _yield_token_and_is_call(code):
        if is_call and token.string == source_name:
            modified_token = tokenize.TokenInfo(
                type=token.type,
                string=target_name,
                start=token.start,
                end=token.end,
                line=token.line,
            )
            modified_tokens.append(modified_token)
        else:
            modified_tokens.append(token)
    return _untokenize(modified_tokens)


def get_functions_called(code: str) -> MutableSet[str]:
    """Returns the set of all functions called in ``code``."""
    return set(
        token.string for token, is_call in _yield_token_and_is_call(code) if is_call
    )
