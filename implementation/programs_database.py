# Copyright 2023 DeepMind Technologies Limited
#
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

"""A programs database that implements the evolutionary algorithm."""
from collections.abc import Mapping, Sequence
import copy
import dataclasses
from typing import Any, Dict, List
import re
import numpy as np
import scipy

from logging_utils import setup_logger
import code_manipulation
import config as config_lib
import profiler
text = "Do not include explanations, comments, or anything else. "
# Get logger from utility module
logger = setup_logger(__name__)


import math

def floor_pow10(x: float) -> float:
    return -10 ** math.floor(math.log10(-x))

def _get_prompt_mid(new_version: int) -> str:
    prompt_mid = f"""
Rules:
- You must preserve the full function signature and docstring structure.
- **Only output the full definition of new version equation `equation_v{new_version}`** in ```python    ```. **don't include any other text, like explanations.**
- **Inside the function, add inline comments explaining the physical or biological meaning of each mathematical term.**

Previous versions are given below:

```python
"""
    return prompt_mid

def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f"`logits` contains non-finite value(s): {non_finites}")
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1 :])
    return result


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

    Attributes:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the implementations
         included in the prompt. Used to direct the newly generated implementation
         into the same island.
    """

    code: str
    version_generated: int
    island_id: int


class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(
        self,
        config: config_lib.ProgramsDatabaseConfig,
        template: code_manipulation.Program,
        function_to_evolve: str,
        log_dir: str,
    ) -> None:
        self._config: config_lib.ProgramsDatabaseConfig = config
        self._score_sampling_temperature_init = config.cluster_sampling_temperature_init
        self._score_sampling_temperature_period = config.cluster_sampling_temperature_period
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._cost_per_ktoken: tuple[float, float] = config.cost_per_ktoken

        # Initialize empty islands.
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(
                    template,
                    function_to_evolve,
                    config.functions_per_prompt
                )
            )
        
        # info for reset islands
        self._best_score_per_island: list[float] = ([-float('inf')] * config.num_islands)
        self._best_program_per_island: list[code_manipulation.Function | None] = (
            [None] * config.num_islands)

        # self._waiting_reset = []
        # self._best_scores_per_island_per_c: Dict[int, List[float]] = {}
        # self._best_function_per_island_per_c: Dict[int, List[code_manipulation.Function]] = {}

        self._last_reset_step: int = 1
        self._profiler = profiler.Profiler(config.num_islands, log_dir)
        self._global_sample_nums = 0

    def get_prompt(self) -> Prompt:
        """Returns a prompt containing implementations from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        period = self._score_sampling_temperature_period
        temperature = self._score_sampling_temperature_init * (
            1 - (self._global_sample_nums % period) / period
        )
        code, version_generated = self._islands[island_id].get_prompt(temperature)
        return Prompt(code, version_generated, island_id)

    def _register_program_in_island(
        self,
        program: code_manipulation.Function,
        island_id: int,
    ) -> None:
        """Registers `program` in the specified island."""
        self._islands[island_id].register_program(program)
        score = program.score

        if score > self._best_score_per_island[island_id]:
            self._best_score_per_island[island_id] = score
            self._best_program_per_island[island_id] = program
            logger.info("Best score of island %d increased to %s", island_id, score)

        # # Initialize best scores and functions per complexity
        # if complexity not in self._best_scores_per_island_per_c:
        #     self._best_scores_per_island_per_c[complexity] = [-99999.9] * self._config.num_islands
        #     self._best_function_per_island_per_c[complexity] = [None] * self._config.num_islands

        # if score > self._best_scores_per_island_per_c[complexity][island_id]:
        #     self._best_function_per_island_per_c[complexity][island_id] = program
        #     self._best_scores_per_island_per_c[complexity][island_id] = score
        #     logger.info("Best score of island %d with complexity %d increased to %s", island_id, complexity, score)
        
        # Log debug information about island state
        island = self._islands[island_id]
        logger.debug(
            "Island %d: %d clusters, %d total programs",
            island_id,
            len(island._clusters),
            island._num_programs
        )

    def register_program(
        self,
        program: code_manipulation.Function,
        island_id: int | None,
        result_per_test: dict | None,
        sample_time: float | None = None,
        evaluate_time: float | None = None,
        sample_token_usage: tuple[int, int] | None = None,
    ) -> None:
        """Registers `program` in the database."""
        # if self._waiting_reset:
        #     c_dealing = self._waiting_reset.pop()
        #     print(f"reset with complexity {c_dealing}")
        #     print(f"waiting reset: {self._waiting_reset}")
        #     self.reset_islands(c_dealing)

        self._global_sample_nums += 1
        global_sample_nums = self._global_sample_nums
        program.global_sample_nums = global_sample_nums
        program.sample_time = sample_time
        program.evaluate_time = evaluate_time
        program.token_usage = sample_token_usage
        if sample_token_usage is not None:
            cost = (sample_token_usage[0] * self._cost_per_ktoken[0] + sample_token_usage[1] * self._cost_per_ktoken[1]) / 1000
        else:
            cost = 0
        program.token_cost = cost

        if result_per_test is not None:
            program.score = result_per_test["score"]
            program.optimized_params = result_per_test["optimized_params"]
            program.complexity = result_per_test["complexity"]
            program.complexity_detail = result_per_test["complexity_detail"]
            if island_id is None:
                # This is a program added at the beginning, so adding it to all islands.
                for island_id in range(len(self._islands)):
                    self._register_program_in_island(program, island_id)
            else:
                self._register_program_in_island(program, island_id)
 
        self._profiler.register_function(program)

        # Check whether it is time to reset an island.
        if self._global_sample_nums - self._last_reset_step > self._config.reset_period:
            self._last_reset_step = self._global_sample_nums
            self.reset_islands()

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # We sort best scores after adding minor noise to break ties.
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                    self._template,
                    self._function_to_evolve,
                    self._config.functions_per_prompt)
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id)
            logger.info("Reset island %d with founder %d", island_id, founder.global_sample_nums)

class Island:
    """A sub-population of the programs database."""

    def __init__(
        self,
        template: code_manipulation.Program,
        function_to_evolve: str,
        functions_per_prompt: int,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt

        self._clusters: dict[int, Cluster] = {}
        self._num_programs: int = 0

    def register_program(
        self,
        program: code_manipulation.Function
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        complexity_bin = program.complexity // 10
        if complexity_bin not in self._clusters:
            self._clusters[complexity_bin] = Cluster(complexity_bin, program)
            logger.debug("Created new cluster with complexity_bin %d", complexity_bin)
        else:
            self._clusters[complexity_bin].register_program(program)
            logger.debug("Added program to existing cluster with complexity_bin %d", complexity_bin)
        self._num_programs += 1

    def get_prompt(self, temperature) -> tuple[str, int]:
        """Constructs a prompt containing functions from this island."""
        complexity_bins = list(self._clusters.keys())
        # normalized_complexity = (np.array(complexity_bins) - min(complexity_bins)) / (max(complexity_bins) + 1e-6)

        # # Convert scores to probabilities using softmax with temperature schedule.
        # probabilities = _softmax(-normalized_complexity, temperature=1.0)

        # At the beginning of an experiment when we have few clusters, place fewer
        # programs into the prompt.
        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(complexity_bins), size=functions_per_prompt
        )

        chosen_bins = [complexity_bins[i] for i in idx]
        implementations = []
        scores = []
        for complexity_bin in chosen_bins:
            cluster = self._clusters[complexity_bin]
            sampled_program = cluster.sample_program(temperature)
            implementations.append(sampled_program)
            scores.append(sampled_program.score)
        
        logger.debug(f"Selected {len(chosen_bins)} clusters with complexity_bins: {chosen_bins}")

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(
        self, implementations: Sequence[code_manipulation.Function]
    ) -> str:
        """Creates a prompt containing a sequence of function `implementations`."""
        implementations = copy.deepcopy(implementations)  # We will mutate these.

        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f"{self._function_to_evolve}_v{i}"
            implementation.name = new_function_name
            # Update the docstring for all subsequent functions after `_v0`.
            if i >= 1:
                implementation.docstring = (
                    f"Improved version of `{self._function_to_evolve}_v{i - 1}`."
                )
            # If the function is recursive, replace calls to itself with its new name.
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name
            )
            versioned_functions.append(
                code_manipulation.text_to_function(implementation)
            )

        # Create the header of the function to be generated by the LLM.
        next_version = len(implementations)
        new_function_name = f"{self._function_to_evolve}_v{next_version}"
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body="",
            docstring=(
                "Improved version of "
                f"`{self._function_to_evolve}_v{next_version - 1}`."
            ),
        )
        # prompt = _get_prompt_pre(next_version - 1) + str(dataclasses.replace(self._template, functions=versioned_functions)) + "\n```\nNow define: \n```python\n" + str(header) + "\n```"

        match = re.search(r'\"\"\"(.*?)\"\"\"', self._template.preface, re.DOTALL)
        if match:
            prompt_pre = match.group(1).strip()
        else:
            logger.error("No docstring found in specification.")
            prompt_pre = ""
        prompt_mid = _get_prompt_mid(next_version - 1)
        prompt = prompt_pre + prompt_mid + "\n".join(str(fun) for fun in versioned_functions) + "\n```\nNow define: \n```python\n" + str(header) + "\n```"
        return prompt


class Cluster:
    """A cluster of programs on the same island and with the same complexity bin."""

    def __init__(self, complexity_bin: int, implementation: code_manipulation.Function):
        self._complexity_bin: int = complexity_bin
        self._programs: list[code_manipulation.Function] = [implementation]
        self._scores: list[float] = [implementation.score]

    def register_program(self, program: code_manipulation.Function) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._scores.append(program.score)
        logger.debug(f"Added program of score {program.score} to cluster with complexity_bin {self._complexity_bin}")

    def sample_program(self, temperature: float) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorther programs."""
        # Convert scores to probabilities using softmax with temperature schedule.
        probabilities = _softmax(np.array(self._scores), temperature)
        chosen_idx = np.random.choice(len(self._programs), p=probabilities)
        chosen_program = self._programs[chosen_idx]
        logger.debug(f"Sampled program with score {self._scores[chosen_idx]} from cluster with {len(self._programs)} programs")
        return chosen_program



