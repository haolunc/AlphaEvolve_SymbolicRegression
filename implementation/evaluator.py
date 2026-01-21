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
import ast
import time
import jax
from collections.abc import Sequence
import copy
from typing import Any, Type
import multiprocessing as mp
import equ_comp

import code_manipulation
from logging_utils import setup_logger

# Get logger from utility module
logger = setup_logger(__name__)

def _extract_python(text: str) -> str:
    if "```python" in text:
        return text.split("```python")[1].split("```")[0]
    else:
        return text

def _sample_to_program(
    sample_body: str,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """
    Args:
        
    Returns the equation replaced with llm sample and the full runnable program including evaluation code.
    """
    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = sample_body
    return evolved_function, str(program)


def _execute_in_subprocess(program, function_to_run, data_dict, result_queue=None):
    """Execute untrusted code in a subprocess and return the result directly or via queue."""
    try:
        # Create a namespace to execute the code
        namespace = {}
        
        # Execute the program in the namespace
        exec(program, namespace)

        if 'equation' in namespace:
            namespace['equation'] = jax.jit(namespace['equation'])
        if 'get_gradients' in namespace:
            namespace['get_gradients'] = jax.jit(namespace['get_gradients'])
                
        # Check if the function exists in the namespace
        if function_to_run not in namespace:
            return None, False
        
        # Get the function and execute it
        function = namespace[function_to_run]
        result = function(data_dict)
        
        # Either return directly or put in queue if provided
        if result_queue is not None:
            result_queue.put((result, True))
        return result, True
        
    except Exception as e:
        # Any exception means the execution failed
        logger.error(f"Execution failed: {str(e)}")
        if result_queue is not None:
            result_queue.put((None, False))
        return None, False


class Sandbox:
    """Sandbox for executing generated code."""

    def __init__(self):
        self._pool = None

    def clean(self):
        """Clean up resources by terminating any active process pool."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
            # logger.debug("Sandbox process pool terminated and joined")

    def run(
        self,
        program: str,
        function_to_run: str,
        data_dict: dict,
        timeout_seconds: int,
    ) -> tuple[Any, bool]:
        """Returns the score from running function_to_run() and a success flag."""
        # Use a context manager to ensure proper resource cleanup
        try:
            # Create a new process pool for each execution to avoid daemonic process issues
            self._pool = mp.get_context('spawn').Pool(processes=1)
            
            # Use apply_async with a timeout to execute the function
            async_result = self._pool.apply_async(_execute_in_subprocess, 
                                           args=(program, function_to_run, data_dict, None))
            
            # Wait for the result with timeout
            try:
                result, success = async_result.get(timeout=timeout_seconds)
                if not success:
                    logger.warning(f"Sandbox execution completed but reported failure")
                return result, success
            except mp.TimeoutError:
                logger.warning(f"Process execution timed out after {timeout_seconds} seconds")
                return None, False
            
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received during sandbox execution, cleaning up...")
            self.clean()
            raise
        except Exception as e:
            logger.error(f"Sandbox execution error: {str(e)}")
            return None, False
        finally:
            self.clean()



class Evaluator:
    """Class that analyses functions generated by LLMs."""

    def __init__(
        self,
        template: code_manipulation.Program,
        function_to_evolve: str,
        function_to_run: str,
        data_dict: dict,
        timeout_seconds: int = 400,
    ):
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._timeout_seconds = timeout_seconds
        self._sandbox = Sandbox()
        self._data_dict = data_dict

    def analyse(
        self,
        sample: str,
        island_id: int | None,
        version_generated: int | None,
        sample_time: float | None = None,
        sample_token_usage: tuple[int, int] | None = None,
    ) -> None:
        """Compiles the sample into a program and executes it."""
        try:
            if version_generated:
                sample_function_body = code_manipulation.text_to_function(_extract_python(sample)).body
            else:
                sample_function_body = sample

            new_function, program = _sample_to_program(
                sample_function_body, self._template, self._function_to_evolve
            )
        except Exception as e:
            logger.error(f"Error parsing sample: {str(e)}\n {sample}")
            return None

        result_per_test = {}
        time_reset = time.time()


        run_result, runs_ok = self._sandbox.run(
            program, self._function_to_run, self._data_dict, self._timeout_seconds
        )
            
        evaluate_time = time.time() - time_reset

        if (
            runs_ok
            and run_result is not None
        ):
            try:
                complexity_val, complexity_detail = equ_comp.complexity_score(str(new_function), return_breakdown=True)
            except Exception as e:
                logger.error(f"Error calculating complexity for {str(new_function)}: {e}")
                complexity_val = None
                complexity_detail = {}

            score = run_result[0]
            optimized_params = run_result[1]
                
            result_per_test = {"score": score,
                                "optimized_params": optimized_params,
                                "complexity": complexity_val,
                                "complexity_detail": complexity_detail}
        else:
            result_per_test = None
            
        result_dict = {
            "function": new_function,
            "island_id": island_id,
            "result_per_test": result_per_test,
            "sample_time": sample_time,
            "evaluate_time": evaluate_time,
            "sample_token_usage": sample_token_usage,
        }


        return result_dict

