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

import copy
import dataclasses
import multiprocessing as mp
import time
from typing import Any

import jax

from . import code_manipulation
from .complexity import complexity_score
from .config import EvaluatorConfig
from .logging_config import get_logger
from .messages import EvalResult

logger = get_logger("evaluator")


def _extract_python(text: str) -> str:
    """Extract python code block from LLM response text."""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0]
    return text


def _sample_to_program(
    sample_body: str,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.ParsedFunction, str]:
    """Returns the equation replaced with LLM sample and the full runnable program."""
    program = copy.deepcopy(template)
    idx = program.find_function_index(function_to_evolve)
    new_func = dataclasses.replace(program.functions[idx], body=sample_body)
    program.functions[idx] = new_func
    return new_func, str(program)


def _execute_in_subprocess(
    program: str, function_to_run: str, data_dict: dict, result_queue: mp.Queue | None = None
) -> tuple[Any, bool]:
    """Execute untrusted code in a subprocess and return the result."""
    try:
        namespace: dict = {}
        exec(program, namespace)

        if "equation" in namespace:
            namespace["equation"] = jax.jit(namespace["equation"])
        if "get_gradients" in namespace:
            namespace["get_gradients"] = jax.jit(namespace["get_gradients"])

        if function_to_run not in namespace:
            return None, False

        function = namespace[function_to_run]
        result = function(data_dict)

        if result_queue is not None:
            result_queue.put((result, True))
        return result, True
    except Exception as e:
        logger.error("Execution failed: %s", e)
        if result_queue is not None:
            result_queue.put((None, False))
        return None, False


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

    def _kill_and_recreate(self) -> None:
        """Terminate the current pool so the next call creates a fresh one."""
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
            self._pool = None

    def clean(self) -> None:
        """Terminate any active process pool."""
        self._kill_and_recreate()

    def run(
        self,
        program: str,
        function_to_run: str,
        data_dict: dict,
        timeout_seconds: int,
    ) -> tuple[Any, bool]:
        """Execute *function_to_run* inside *program* with a timeout."""
        try:
            pool = self._ensure_pool()
            async_result = pool.apply_async(
                _execute_in_subprocess, args=(program, function_to_run, data_dict, None)
            )
            try:
                result, success = async_result.get(timeout=timeout_seconds)
                if not success:
                    logger.warning("Sandbox execution completed but reported failure")
                return result, success
            except mp.TimeoutError:
                logger.warning("Process execution timed out after %d seconds", timeout_seconds)
                self._kill_and_recreate()
                return None, False
        except KeyboardInterrupt:
            self.clean()
            raise
        except Exception as e:
            logger.error("Sandbox execution error: %s", e)
            self._kill_and_recreate()
            return None, False


class Evaluator:
    """Evaluates functions generated by LLMs."""

    def __init__(
        self,
        template: code_manipulation.Program,
        function_to_evolve: str,
        function_to_run: str,
        data_dict: dict,
        config: EvaluatorConfig | None = None,
    ):
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._data_dict = data_dict
        self._config = config or EvaluatorConfig()
        self._sandbox = Sandbox()

    def clean(self) -> None:
        """Release sandbox resources."""
        self._sandbox.clean()

    def analyse(
        self,
        sample: str,
        island_id: int | None,
        version_generated: int | None,
        sample_time: float | None = None,
        sample_token_usage: tuple[int, int] | None = None,
        sample_token_cost: float | None = None,
    ) -> EvalResult | None:
        """Compile *sample* into a program and execute it.

        Returns an ``EvalResult`` on success, or ``None`` on failure.
        """
        try:
            if version_generated:
                sample_function_body = code_manipulation.text_to_function(_extract_python(sample)).body
            else:
                sample_function_body = sample

            new_function, program = _sample_to_program(
                sample_function_body, self._template, self._function_to_evolve
            )
        except Exception as e:
            logger.error("Error parsing sample: %s\n %s", e, sample)
            return None

        time_reset = time.time()
        run_result, runs_ok = self._sandbox.run(
            program, self._function_to_run, self._data_dict, self._config.timeout_seconds
        )
        evaluate_time = time.time() - time_reset

        result_per_test: dict | None = None
        if runs_ok and run_result is not None:
            try:
                complexity_val, complexity_detail = complexity_score(str(new_function), return_breakdown=True)
            except Exception as e:
                logger.error("Error calculating complexity for %s: %s", new_function.name, e)
                complexity_val = None
                complexity_detail = {}

            score = run_result[0]
            optimized_params = run_result[1]

            result_per_test = {
                "score": score,
                "optimized_params": optimized_params,
                "complexity": complexity_val,
                "complexity_detail": complexity_detail,
            }

        return EvalResult(
            function=new_function,
            island_id=island_id,
            result_per_test=result_per_test,
            sample_time=sample_time,
            evaluate_time=evaluate_time,
            sample_token_usage=sample_token_usage,
            sample_token_cost=sample_token_cost,
        )
