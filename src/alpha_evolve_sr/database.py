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

"""A programs database that implements the evolutionary algorithm."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence

import numpy as np
import scipy

from . import code_manipulation, profiler
from . import config as config_lib
from .logging_config import get_logger

logger = get_logger("database")


def _get_prompt_mid(new_version: int) -> str:
    return f"""
Rules:
- You must preserve the full function signature and docstring structure.
- **Only output the full definition of new version equation \
`equation_v{new_version}`** in ```python```. **don't include any other text.**
- **Inside the function, add inline comments explaining the physical or biological meaning of each mathematical term.**

Previous versions are given below:

```python
"""


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite ``logits``."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f"`logits` contains non-finite value(s): {non_finites}")
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

    Attributes:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is ``_v{version_generated}``.
      island_id: Identifier of the island that produced the implementations
         included in the prompt.
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
        profiler_config: config_lib.ProfilerConfig | None = None,
    ) -> None:
        self._config = config
        self._score_sampling_temperature_init = config.cluster_sampling_temperature_init
        self._score_sampling_temperature_period = config.cluster_sampling_temperature_period
        self._template = template
        self._function_to_evolve = function_to_evolve

        self._islands: list[Island] = [
            Island(
                template, function_to_evolve, config.functions_per_prompt,
                config.complexity_bin_size, config.cluster_max_size,
            )
            for _ in range(config.num_islands)
        ]

        self._best_score_per_island: list[float] = [-float("inf")] * config.num_islands
        self._best_program_per_island: list[code_manipulation.EvaluatedProgram | None] = [None] * config.num_islands

        self._last_reset_step: int = 1
        self._profiler = profiler.Profiler(config.num_islands, log_dir, config=profiler_config)
        self._global_sample_nums = 0

        # Pareto front: programs not dominated by any other in (score, complexity)
        self._pareto_front: list[code_manipulation.EvaluatedProgram] = []

    @property
    def sample_count(self) -> int:
        """The total number of samples registered so far."""
        return self._global_sample_nums

    @property
    def pareto_front(self) -> list[code_manipulation.EvaluatedProgram]:
        """Current Pareto-optimal programs (sorted by complexity)."""
        return list(self._pareto_front)

    def finalize(self) -> None:
        """Write final outputs (best program per complexity file)."""
        self._profiler.write_best_program_per_c_file()
        if self._pareto_front:
            self._profiler.write_pareto_front(self._pareto_front)

    def get_prompt(self) -> Prompt:
        """Returns a prompt containing implementations from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        period = self._score_sampling_temperature_period
        temperature = self._score_sampling_temperature_init * (
            1 - (self._global_sample_nums % period) / period
        )
        pareto = self._pareto_front if self._config.pareto_aware else None
        code, version_generated = self._islands[island_id].get_prompt(temperature, pareto)
        return Prompt(code, version_generated, island_id)

    def _register_program_in_island(
        self,
        program: code_manipulation.EvaluatedProgram,
        island_id: int,
    ) -> None:
        """Registers *program* in the specified island."""
        self._islands[island_id].register_program(program)
        score = program.score

        if score > self._best_score_per_island[island_id]:
            self._best_score_per_island[island_id] = score
            self._best_program_per_island[island_id] = program
            logger.info("Best score of island %d increased to %s", island_id, score)

        island = self._islands[island_id]
        logger.debug(
            "Island %d: %d clusters, %d total programs",
            island_id,
            island.num_clusters,
            island.num_programs,
        )

    def register_program(
        self,
        function: code_manipulation.ParsedFunction,
        island_id: int | None,
        result_per_test: dict | None,
        sample_time: float | None = None,
        evaluate_time: float | None = None,
        sample_token_usage: tuple[int, int] | None = None,
        sample_token_cost: float | None = None,
    ) -> None:
        """Registers a program in the database.

        Constructs an ``EvaluatedProgram`` from the given ``ParsedFunction``
        and evaluation results, then routes it to the appropriate island(s).
        """
        self._global_sample_nums += 1

        cost = sample_token_cost or 0.0

        evaluated = code_manipulation.EvaluatedProgram(
            parsed=function,
            score=result_per_test["score"] if result_per_test else None,
            optimized_params=result_per_test["optimized_params"] if result_per_test else None,
            complexity=result_per_test["complexity"] if result_per_test else None,
            complexity_detail=result_per_test["complexity_detail"] if result_per_test else None,
            global_sample_nums=self._global_sample_nums,
            sample_time=sample_time,
            evaluate_time=evaluate_time,
            token_usage=sample_token_usage,
            token_cost=cost,
        )

        if result_per_test is not None:
            if island_id is None:
                for iid in range(len(self._islands)):
                    self._register_program_in_island(evaluated, iid)
            else:
                self._register_program_in_island(evaluated, island_id)

        self._update_pareto_front(evaluated)
        self._profiler.register_function(evaluated, pareto_size=len(self._pareto_front))

        if self._global_sample_nums - self._last_reset_step > self._config.reset_period:
            self._last_reset_step = self._global_sample_nums
            self.reset_islands()

    def _update_pareto_front(self, program: code_manipulation.EvaluatedProgram) -> None:
        """Update the Pareto front with *program* if it is non-dominated.

        A program is *dominated* if another program exists with equal-or-lower
        complexity and equal-or-higher score (and strictly better in at least
        one dimension).  The front is kept sorted by complexity.
        """
        if program.score is None or program.complexity is None:
            return

        score = program.score
        complexity = program.complexity

        # Check if dominated by any existing front member
        for fp in self._pareto_front:
            if fp.complexity <= complexity and fp.score >= score:
                return  # dominated

        # Not dominated — add and remove any programs it dominates
        self._pareto_front = [
            fp for fp in self._pareto_front
            if not (complexity <= fp.complexity and score >= fp.score)
        ]
        self._pareto_front.append(program)
        self._pareto_front.sort(key=lambda p: p.complexity)

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island + np.random.randn(len(self._best_score_per_island)) * 1e-6
        )
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.complexity_bin_size,
                self._config.cluster_max_size,
            )
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
        complexity_bin_size: int = 10,
        cluster_max_size: int = 100,
    ) -> None:
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._functions_per_prompt = functions_per_prompt
        self._complexity_bin_size = complexity_bin_size
        self._cluster_max_size = cluster_max_size

        self._clusters: dict[int, Cluster] = {}
        self._num_programs: int = 0

    @property
    def num_clusters(self) -> int:
        """The number of complexity-bin clusters on this island."""
        return len(self._clusters)

    @property
    def num_programs(self) -> int:
        """The total number of programs registered on this island."""
        return self._num_programs

    def register_program(self, program: code_manipulation.EvaluatedProgram) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        complexity_bin = program.complexity // self._complexity_bin_size
        if complexity_bin not in self._clusters:
            self._clusters[complexity_bin] = Cluster(complexity_bin, program, self._cluster_max_size)
            logger.debug("Created new cluster with complexity_bin %d", complexity_bin)
        else:
            self._clusters[complexity_bin].register_program(program)
            logger.debug("Added program to existing cluster with complexity_bin %d", complexity_bin)
        self._num_programs += 1

    def get_prompt(
        self,
        temperature: float,
        pareto_front: list[code_manipulation.EvaluatedProgram] | None = None,
    ) -> tuple[str, int]:
        """Constructs a prompt containing functions from this island."""
        complexity_bins = list(self._clusters.keys())

        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        if pareto_front and len(pareto_front) >= 2:
            weights = self._pareto_weights(complexity_bins, pareto_front)
            idx = np.random.choice(len(complexity_bins), size=functions_per_prompt, p=weights)
        else:
            idx = np.random.choice(len(complexity_bins), size=functions_per_prompt)
        chosen_bins = [complexity_bins[i] for i in idx]

        implementations = []
        scores = []
        for complexity_bin in chosen_bins:
            cluster = self._clusters[complexity_bin]
            sampled_program = cluster.sample_program(temperature)
            implementations.append(sampled_program)
            scores.append(sampled_program.score)

        logger.debug("Selected %d clusters with complexity_bins: %s", len(chosen_bins), chosen_bins)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _pareto_weights(
        self,
        complexity_bins: list[int],
        pareto_front: list[code_manipulation.EvaluatedProgram],
    ) -> np.ndarray:
        """Compute selection weights favouring bins with Pareto improvement potential.

        For each complexity bin, the weight is ``1 + gap`` where *gap* is the
        positive difference between the Pareto-front-interpolated score at that
        complexity and the cluster's current best score.  Bins with large gaps
        (lagging behind the front) receive more exploration pressure.
        """
        pareto_complexities = np.array([p.complexity for p in pareto_front], dtype=float)
        pareto_scores = np.array([p.score for p in pareto_front], dtype=float)

        weights = np.ones(len(complexity_bins), dtype=float)
        for i, bin_key in enumerate(complexity_bins):
            bin_center = bin_key * self._complexity_bin_size + self._complexity_bin_size / 2.0
            target = float(np.interp(bin_center, pareto_complexities, pareto_scores))
            cluster_best = max(self._clusters[bin_key]._scores)
            gap = max(0.0, target - cluster_best)
            weights[i] = 1.0 + gap

        total = weights.sum()
        if total == 0:
            return np.ones(len(complexity_bins)) / len(complexity_bins)
        return weights / total

    def _generate_prompt(self, implementations: Sequence[code_manipulation.EvaluatedProgram]) -> str:
        """Creates a prompt containing a sequence of function *implementations*."""
        versioned_functions: list[code_manipulation.ParsedFunction] = []
        for i, impl in enumerate(implementations):
            new_function_name = f"{self._function_to_evolve}_v{i}"
            func = dataclasses.replace(impl.parsed, name=new_function_name)
            if i >= 1:
                doc = f"Improved version of `{self._function_to_evolve}_v{i - 1}`."
                func = dataclasses.replace(func, docstring=doc)
            renamed_code = code_manipulation.rename_function_calls(
                str(func), self._function_to_evolve, new_function_name
            )
            versioned_functions.append(code_manipulation.text_to_function(renamed_code))

        next_version = len(implementations)
        new_function_name = f"{self._function_to_evolve}_v{next_version}"
        header = dataclasses.replace(
            implementations[-1].parsed,
            name=new_function_name,
            body="",
            docstring=f"Improved version of `{self._function_to_evolve}_v{next_version - 1}`.",
        )

        match = re.search(r'"""(.*?)"""', self._template.preface, re.DOTALL)
        if match:
            prompt_pre = match.group(1).strip()
        else:
            logger.error("No docstring found in specification.")
            prompt_pre = ""
        prompt_mid = _get_prompt_mid(next_version - 1)
        prompt = (
            prompt_pre
            + prompt_mid
            + "\n".join(str(fun) for fun in versioned_functions)
            + "\n```\nNow define: \n```python\n"
            + str(header)
            + "\n```"
        )
        return prompt


class Cluster:
    """A cluster of programs on the same island and with the same complexity bin."""

    def __init__(self, complexity_bin: int, implementation: code_manipulation.EvaluatedProgram, max_size: int = 100):
        self._complexity_bin = complexity_bin
        self._max_size = max_size
        self._programs: list[code_manipulation.EvaluatedProgram] = [implementation]
        self._scores: list[float] = [implementation.score]

    def register_program(self, program: code_manipulation.EvaluatedProgram) -> None:
        """Adds *program* to the cluster, pruning if over max size."""
        self._programs.append(program)
        self._scores.append(program.score)
        logger.debug("Added program of score %s to cluster with complexity_bin %d", program.score, self._complexity_bin)
        if len(self._programs) > self._max_size:
            self._prune()

    def _prune(self) -> None:
        """Remove lowest-scoring programs to stay within *_max_size*."""
        keep_indices = np.argsort(self._scores)[-self._max_size:]
        keep_indices.sort()  # preserve insertion order among survivors
        self._programs = [self._programs[i] for i in keep_indices]
        self._scores = [self._scores[i] for i in keep_indices]
        logger.debug("Pruned cluster %d to %d programs", self._complexity_bin, len(self._programs))

    def sample_program(self, temperature: float) -> code_manipulation.EvaluatedProgram:
        """Samples a program, giving higher probability to higher-scoring programs."""
        probabilities = _softmax(np.array(self._scores), temperature)
        chosen_idx = np.random.choice(len(self._programs), p=probabilities)
        chosen_program = self._programs[chosen_idx]
        logger.debug(
            "Sampled program with score %s from cluster with %d programs",
            self._scores[chosen_idx],
            len(self._programs),
        )
        return chosen_program
