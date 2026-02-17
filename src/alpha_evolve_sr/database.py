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
import os
from collections.abc import Sequence

import numpy as np
import scipy

from . import code_manipulation, profiler
from . import config as config_lib
from .checkpoint import CheckpointDB
from .logging_config import get_logger
from .messages import EvalResult, SampleMessage

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
      island_id: Identifier of the island that produced the implementations
         included in the prompt.
    """

    code: str
    island_id: int


class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(
        self,
        config: config_lib.ProgramsDatabaseConfig,
        prompt_text: str,
        log_dir: str,
        profiler_config: config_lib.ProfilerConfig | None = None,
        *,
        ckpt_dir: str | None = None,
        checkpoint_db: CheckpointDB | None = None,
        max_samples: int = 3600,
    ) -> None:
        self._config = config
        self._score_sampling_temperature_init = config.cluster_sampling_temperature_init
        self._score_sampling_temperature_period = config.cluster_sampling_temperature_period
        self._prompt_text = prompt_text
        self._function_to_evolve = "equation"

        self._islands: list[Island] = [
            Island(
                prompt_text, config.functions_per_prompt,
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

        # Lifecycle / checkpoint management
        self._ckpt_dir = ckpt_dir
        self._checkpoint_db = checkpoint_db
        self._max_samples = max_samples

    @property
    def sample_count(self) -> int:
        """The total number of samples registered so far."""
        return self._global_sample_nums

    @property
    def pareto_front(self) -> list[code_manipulation.EvaluatedProgram]:
        """Current Pareto-optimal programs (sorted by complexity)."""
        return list(self._pareto_front)

    @property
    def should_stop(self) -> bool:
        """Whether the maximum number of samples has been reached."""
        return self._global_sample_nums >= self._max_samples

    # ------------------------------------------------------------------
    # Lifecycle (absorbed from EvolutionController)
    # ------------------------------------------------------------------

    @classmethod
    def restore_or_create(
        cls,
        config: config_lib.ProgramsDatabaseConfig,
        prompt_text: str,
        log_dir: str,
        *,
        profiler_config: config_lib.ProfilerConfig | None = None,
        ckpt_dir: str | None = None,
        max_samples: int = 3600,
        resume_path: str | None = None,
        initial_result: EvalResult | None = None,
    ) -> ProgramsDatabase:
        """Create a new database or restore from checkpoint.

        If *resume_path* is given, attempts to load from the SQLite checkpoint.
        Otherwise creates a fresh database and optionally registers *initial_result*.
        """
        import shutil

        if resume_path:
            resume_db_path = os.path.join(resume_path, "checkpoint.db")
            if os.path.exists(resume_db_path):
                try:
                    resume_db = CheckpointDB(resume_db_path)
                    if resume_db.is_populated:
                        # Determine the ongoing write DB
                        if ckpt_dir and os.path.normpath(ckpt_dir) != os.path.normpath(resume_path):
                            os.makedirs(ckpt_dir, exist_ok=True)
                            resume_db.close()
                            dest_db_path = os.path.join(ckpt_dir, "checkpoint.db")
                            shutil.copy2(resume_db_path, dest_db_path)
                            # Also copy WAL/SHM if present
                            for suffix in ("-wal", "-shm"):
                                src = resume_db_path + suffix
                                if os.path.exists(src):
                                    shutil.copy2(src, dest_db_path + suffix)
                            checkpoint_db = CheckpointDB(dest_db_path)
                        else:
                            checkpoint_db = resume_db

                        db = cls(
                            config, prompt_text, log_dir,
                            profiler_config=profiler_config,
                            ckpt_dir=ckpt_dir, checkpoint_db=checkpoint_db,
                            max_samples=max_samples,
                        )
                        db._restore_from_db(checkpoint_db)
                        logger.info(
                            "Database restored from checkpoint with %d samples",
                            db.sample_count,
                        )
                        return db
                    else:
                        resume_db.close()
                except Exception as e:
                    logger.error("Failed to load checkpoint: %s", e)
                    logger.info("Initializing new database instead")
            else:
                logger.info("No checkpoint.db found at %s, starting fresh", resume_path)

        # Fresh start
        checkpoint_db = None
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            checkpoint_db = CheckpointDB(os.path.join(ckpt_dir, "checkpoint.db"))

        db = cls(
            config, prompt_text, log_dir,
            profiler_config=profiler_config,
            ckpt_dir=ckpt_dir, checkpoint_db=checkpoint_db, max_samples=max_samples,
        )
        if initial_result is not None:
            db.register_program(initial_result)
        return db

    def _restore_from_db(self, checkpoint_db: CheckpointDB) -> None:
        """Reconstruct in-memory state from a populated CheckpointDB."""
        # Load all programs
        all_programs = checkpoint_db.load_programs()

        # Load island memberships and reconstruct islands
        memberships = checkpoint_db.load_island_memberships()
        for island_id, bins in memberships.items():
            if island_id >= len(self._islands):
                continue
            island = self._islands[island_id]
            for complexity_bin, gsn_list in bins.items():
                progs_in_bin = [all_programs[gsn] for gsn in gsn_list if gsn in all_programs]
                if not progs_in_bin:
                    continue
                if complexity_bin not in island._clusters:
                    island._clusters[complexity_bin] = Cluster(
                        complexity_bin, progs_in_bin[0], island._cluster_max_size,
                    )
                    island._num_programs += 1
                    for p in progs_in_bin[1:]:
                        island._clusters[complexity_bin].register_program(p)
                        island._num_programs += 1
                else:
                    for p in progs_in_bin:
                        island._clusters[complexity_bin].register_program(p)
                        island._num_programs += 1

        # Load pareto front
        pareto_ids = checkpoint_db.load_pareto_front_ids()
        self._pareto_front = [all_programs[gsn] for gsn in pareto_ids if gsn in all_programs]
        self._pareto_front.sort(key=lambda p: p.complexity)

        # Load metadata
        meta = checkpoint_db.load_metadata()
        self._global_sample_nums = meta.get("global_sample_nums", 0)
        self._last_reset_step = meta.get("last_reset_step", 1)

        best_scores = meta.get("best_score_per_island", [-float("inf")] * len(self._islands))
        for i, s in enumerate(best_scores):
            if i < len(self._best_score_per_island):
                self._best_score_per_island[i] = s

        best_prog_ids = meta.get("best_program_per_island", [None] * len(self._islands))
        for i, gsn in enumerate(best_prog_ids):
            if i < len(self._best_program_per_island) and gsn is not None and gsn in all_programs:
                self._best_program_per_island[i] = all_programs[gsn]

        # Restore profiler
        profiler_stats = checkpoint_db.load_profiler_stats()
        profiler_per_c = checkpoint_db.load_profiler_per_complexity()
        if profiler_stats is not None:
            self._profiler.restore_stats(profiler_stats, profiler_per_c)
            self._profiler._num_samples = self._global_sample_nums

    def maybe_checkpoint(self) -> bool:
        """No-op: persistence is now incremental via SQLite."""
        return False

    def finalize(self) -> None:
        """Close the checkpoint DB and write output files."""
        if self._checkpoint_db:
            self._checkpoint_db.close()
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
        code = self._islands[island_id].get_prompt(temperature, pareto)
        return Prompt(code, island_id)

    def _register_program_in_island(
        self,
        program: code_manipulation.EvaluatedProgram,
        island_id: int,
    ) -> list[int]:
        """Registers *program* in the specified island. Returns evicted IDs."""
        evicted = self._islands[island_id].register_program(program)
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
        return evicted

    def register_program(
        self,
        eval_result: EvalResult,
        sample_message: SampleMessage | None = None,
    ) -> None:
        """Registers a program in the database.

        Constructs an ``EvaluatedProgram`` from the given ``EvalResult``
        and optional ``SampleMessage``, then routes it to the appropriate island(s).
        """
        self._global_sample_nums += 1

        island_id = sample_message.island_id if sample_message else None
        sample_time = sample_message.sample_time if sample_message else None
        sample_token_usage = (
            (sample_message.llm_response.input_tokens, sample_message.llm_response.output_tokens)
            if sample_message else None
        )
        cost = sample_message.llm_response.token_cost if sample_message else 0.0
        llm_response_text = (
            sample_message.llm_response.response_text if sample_message else None
        )

        ex = eval_result.execution_result
        evaluated = code_manipulation.EvaluatedProgram(
            parsed=eval_result.function,
            score=ex.score if ex else None,
            optimized_params=ex.optimized_params if ex else None,
            complexity=ex.complexity if ex else None,
            complexity_detail=ex.complexity_detail if ex else None,
            global_sample_nums=self._global_sample_nums,
            sample_time=sample_time,
            evaluate_time=eval_result.evaluate_time,
            token_usage=sample_token_usage,
            token_cost=cost,
        )

        db = self._checkpoint_db

        if db:
            with db.transaction():
                self._register_and_persist(evaluated, island_id, ex, db, llm_response_text)
        else:
            self._register_and_persist(evaluated, island_id, ex, None, None)

    def _register_and_persist(
        self,
        evaluated: code_manipulation.EvaluatedProgram,
        island_id: int | None,
        ex: object | None,
        db: CheckpointDB | None,
        llm_response_text: str | None,
    ) -> None:
        """Core registration + optional DB persistence (runs inside a transaction)."""
        if db:
            db.insert_program(evaluated, island_id=island_id, llm_response_text=llm_response_text)

        if ex is not None:
            if island_id is None:
                for iid in range(len(self._islands)):
                    evicted = self._register_program_in_island(evaluated, iid)
                    if db:
                        complexity_bin = evaluated.complexity // self._config.complexity_bin_size
                        db.insert_island_program(iid, complexity_bin, evaluated.global_sample_nums)
                        db.delete_island_programs(iid, evicted)
            else:
                evicted = self._register_program_in_island(evaluated, island_id)
                if db:
                    complexity_bin = evaluated.complexity // self._config.complexity_bin_size
                    db.insert_island_program(island_id, complexity_bin, evaluated.global_sample_nums)
                    db.delete_island_programs(island_id, evicted)

        self._update_pareto_front(evaluated)
        self._profiler.register_function(evaluated, pareto_size=len(self._pareto_front))

        if db:
            db.replace_pareto_front([p.global_sample_nums for p in self._pareto_front])
            db.save_profiler_stats(self._profiler)
            per_c = self._profiler.get_best_per_complexity_snapshot()
            if evaluated.complexity is not None and evaluated.complexity in per_c:
                score, prog_str, order = per_c[evaluated.complexity]
                db.save_profiler_per_complexity(evaluated.complexity, score, prog_str, order)
            db.save_metadata("global_sample_nums", self._global_sample_nums)
            db.save_metadata("last_reset_step", self._last_reset_step)
            db.save_metadata(
                "best_score_per_island", self._best_score_per_island,
            )
            db.save_metadata(
                "best_program_per_island",
                [p.global_sample_nums if p else None for p in self._best_program_per_island],
            )

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
                self._prompt_text,
                self._config.functions_per_prompt,
                self._config.complexity_bin_size,
                self._config.cluster_max_size,
            )
            if self._checkpoint_db:
                self._checkpoint_db.reset_island(int(island_id))
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            evicted = self._register_program_in_island(founder, island_id)
            if self._checkpoint_db:
                complexity_bin = founder.complexity // self._config.complexity_bin_size
                self._checkpoint_db.insert_island_program(
                    int(island_id), complexity_bin, founder.global_sample_nums,
                )
                self._checkpoint_db.delete_island_programs(int(island_id), evicted)
            logger.info("Reset island %d with founder %d", island_id, founder.global_sample_nums)


class Island:
    """A sub-population of the programs database."""

    def __init__(
        self,
        prompt_text: str,
        functions_per_prompt: int,
        complexity_bin_size: int = 10,
        cluster_max_size: int = 100,
    ) -> None:
        self._prompt_text = prompt_text
        self._function_to_evolve = "equation"
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

    def register_program(self, program: code_manipulation.EvaluatedProgram) -> list[int]:
        """Stores a program on this island, in its appropriate cluster.

        Returns a list of ``global_sample_nums`` that were evicted (empty if none).
        """
        complexity_bin = program.complexity // self._complexity_bin_size
        if complexity_bin not in self._clusters:
            self._clusters[complexity_bin] = Cluster(complexity_bin, program, self._cluster_max_size)
            logger.debug("Created new cluster with complexity_bin %d", complexity_bin)
            evicted: list[int] = []
        else:
            evicted = self._clusters[complexity_bin].register_program(program)
            logger.debug("Added program to existing cluster with complexity_bin %d", complexity_bin)
        self._num_programs += 1
        return evicted

    def get_prompt(
        self,
        temperature: float,
        pareto_front: list[code_manipulation.EvaluatedProgram] | None = None,
    ) -> str:
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
        return self._generate_prompt(sorted_implementations)

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

        prompt_pre = self._prompt_text
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

    def register_program(self, program: code_manipulation.EvaluatedProgram) -> list[int]:
        """Adds *program* to the cluster, pruning if over max size.

        Returns a list of ``global_sample_nums`` that were evicted (empty if none).
        """
        self._programs.append(program)
        self._scores.append(program.score)
        logger.debug("Added program of score %s to cluster with complexity_bin %d", program.score, self._complexity_bin)
        if len(self._programs) > self._max_size:
            return self._prune()
        return []

    def _prune(self) -> list[int]:
        """Remove lowest-scoring programs to stay within *_max_size*.

        Returns a list of ``global_sample_nums`` that were evicted.
        """
        keep_indices = set(np.argsort(self._scores)[-self._max_size:])
        all_indices = set(range(len(self._programs)))
        evicted_indices = all_indices - keep_indices
        evicted_ids = [self._programs[i].global_sample_nums for i in evicted_indices]

        keep_sorted = sorted(keep_indices)
        self._programs = [self._programs[i] for i in keep_sorted]
        self._scores = [self._scores[i] for i in keep_sorted]
        logger.debug("Pruned cluster %d to %d programs", self._complexity_bin, len(self._programs))
        return evicted_ids

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
