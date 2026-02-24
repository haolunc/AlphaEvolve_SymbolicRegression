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

"""Class for evaluating programs proposed by the Sampler."""

from __future__ import annotations

import contextlib
import dataclasses
import multiprocessing as mp
import os
import sys
import tempfile
import time
from typing import Any

from . import code_manipulation
from .config import EvaluatorConfig
from .logging_config import get_logger
from .messages import EvalResult, ExecutionResult, SampleMessage

logger = get_logger("evaluator")


def _extract_python_text(text: str) -> str:
    """Extract python code block from LLM response text."""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0]
    return text


def _sample_to_program(
    sample_body: str,
    evaluate_code: str,
    seed_function: code_manipulation.ParsedFunction,
) -> tuple[code_manipulation.ParsedFunction, str]:
    """Returns the equation replaced with LLM sample and the full runnable program."""
    new_func = dataclasses.replace(seed_function, body=sample_body)
    program = evaluate_code + "\n\n" + str(new_func)
    return new_func, program


@contextlib.contextmanager
def _capture_fd_output():
    """Capture all stdout/stderr at the OS fd level.

    This catches output from C extensions (e.g. CMA-ES ``disp()``, BFGS)
    that bypass Python's ``sys.stdout``/``sys.stderr``.

    Yields a callable that returns the captured text so far.
    """
    # Save original fds and Python streams
    saved_fd1 = os.dup(1)
    saved_fd2 = os.dup(2)
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr

    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        # Redirect OS-level fds to the temp file
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        # Redirect Python streams to match
        sys.stdout = os.fdopen(os.dup(tmp.fileno()), "w", closefd=True)
        sys.stderr = os.fdopen(os.dup(tmp.fileno()), "w", closefd=True)

        def _get_captured() -> str:
            sys.stdout.flush()
            sys.stderr.flush()
            os.fsync(1)
            os.fsync(2)
            tmp.seek(0)
            return tmp.read().decode("utf-8", errors="replace")

        yield _get_captured
    finally:
        # Restore OS-level fds
        sys.stdout.close()
        sys.stderr.close()
        os.dup2(saved_fd1, 1)
        os.dup2(saved_fd2, 2)
        os.close(saved_fd1)
        os.close(saved_fd2)
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        tmp.close()


def _execute_in_subprocess(
    program: str, data_dict: dict,
) -> tuple[Any, bool, str | None, str]:
    """Execute untrusted code in a subprocess and return the result.

    Returns:
        (result, success, error_str, eval_output) — *error_str* is ``None``
        on success. *eval_output* contains captured stdout/stderr text.
    """
    try:
        with _capture_fd_output() as get_captured:
            namespace: dict = {}
            exec(program, namespace)

            if "evaluate" not in namespace:
                error_str = "NameError: 'evaluate' function not defined in program"
                return None, False, error_str, get_captured()

            function = namespace["evaluate"]
            result = function(data_dict)

            return result, True, None, get_captured()
    except Exception as e:
        error_str = f"{type(e).__name__}: {e}"
        logger.error("Execution failed: %s", error_str)
        return None, False, error_str, get_captured()


class Sandbox:
    """Sandbox for executing generated code in an isolated subprocess.

    The pool is created lazily on the first ``run()`` call and reused for
    subsequent evaluations.  On timeout or crash the pool is killed and
    transparently recreated on the next call, avoiding the overhead of
    spawning a new process for every evaluation (~1-3 s each).
    """

    def __init__(self) -> None:
        self._pool: mp.pool.Pool | None = None

    def _ensure_pool(self) -> mp.pool.Pool:
        """Return the existing pool or create a new one."""
        if self._pool is None:
            self._pool = mp.get_context("spawn").Pool(processes=1)
        return self._pool

    def clean(self) -> None:
        """Terminate the current pool so the next call creates a fresh one."""
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            # If the pool is already exited
            except Exception:
                pass
            self._pool = None

    def run(
        self,
        program: str,
        data_dict: dict,
        timeout_seconds: int,
    ) -> tuple[Any, bool, str | None, str]:
        """Execute the ``evaluate`` function inside *program* with a timeout.

        Returns:
            (result, success, error_str, eval_output) — *error_str* is ``None``
            on success. *eval_output* contains captured stdout/stderr text.
        """
        try:
            pool = self._ensure_pool()
            async_result = pool.apply_async(
                _execute_in_subprocess, args=(program, data_dict)
            )
            try:
                result, success, error_str, eval_output = async_result.get(timeout=timeout_seconds)
                if not success:
                    logger.warning("Sandbox execution completed but reported failure")
                return result, success, error_str, eval_output
            except mp.TimeoutError:
                error_str = f"TimeoutError: execution exceeded {timeout_seconds}s"
                logger.warning("Process execution timed out after %d seconds", timeout_seconds)
                self.clean()
                return None, False, error_str, ""
        except KeyboardInterrupt:
            self.clean()
            raise
        except Exception as e:
            error_str = f"SandboxError: {type(e).__name__}: {e}"
            logger.error("Sandbox execution error: %s", e)
            self.clean()
            return None, False, error_str, ""


class Evaluator:
    """Evaluates functions generated by LLMs."""

    def __init__(
        self,
        evaluate_code: str,
        seed_function: code_manipulation.ParsedFunction,
        data_dict: dict,
        config: EvaluatorConfig | None = None,
    ):
        self._evaluate_code = evaluate_code
        self._seed_function = seed_function
        self._data_dict = data_dict
        self._config = config or EvaluatorConfig()
        self._sandbox = Sandbox()

    def clean(self) -> None:
        """Release sandbox resources."""
        self._sandbox.clean()

    def _evaluate_body(self, sample_body: str) -> EvalResult:
        """Build program from *sample_body*, execute in sandbox, return ``EvalResult``."""
        new_function, program = _sample_to_program(
            sample_body, self._evaluate_code, self._seed_function
        )

        time_reset = time.time()
        run_result, runs_ok, error_str, eval_output = self._sandbox.run(
            program, self._data_dict, self._config.timeout_seconds
        )
        evaluate_time = time.time() - time_reset

        execution_result: ExecutionResult | None = None
        error_type: str | None = None
        error_message: str | None = None
        if runs_ok and run_result is not None:
            execution_result = ExecutionResult(
                score=run_result[0],
                optimized_params=run_result[1],
            )
        elif not runs_ok:
            error_type = "timeout" if "TimeoutError" in (error_str or "") else "execution"
            error_message = error_str

        return EvalResult(
            function=new_function,
            execution_result=execution_result,
            evaluate_time=evaluate_time,
            error_type=error_type,
            error_message=error_message,
            eval_output=eval_output or None,
        )

    def initialize(self) -> EvalResult:
        """Evaluate the seed function. Returns ``EvalResult``."""
        return self._evaluate_body(self._seed_function.body)

    def analyse(self, sample_message: SampleMessage) -> EvalResult:
        """Parse an LLM-generated sample and evaluate it.

        Always returns an ``EvalResult`` — on parse failure, the result carries
        ``error_type='parse'`` and ``error_message``.
        """
        try:
            sample_function_body = code_manipulation.text_to_function(
                _extract_python_text(sample_message.llm_response.response_text)
            ).body
        except Exception as e:
            logger.error("Error parsing sample: %s", e)
            return EvalResult(
                function=self._seed_function,
                execution_result=None,
                evaluate_time=None,
                error_type="parse",
                error_message=str(e),
            )

        return self._evaluate_body(sample_function_body)
