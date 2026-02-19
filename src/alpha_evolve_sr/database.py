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
from .messages import EvalResult, Prompt, SampleMessage

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


class Island:
    """Lightweight in-memory index for fast sampling."""

    def __init__(
        self,
        functions_per_prompt: int,
        complexity_bin_size: int,
        cluster_max_size: int,
    ) -> None:
        self._functions_per_prompt = functions_per_prompt
        self._complexity_bin_size = complexity_bin_size
        self._cluster_max_size = cluster_max_size
        self._bins: dict[int, list[tuple[int, float]]] = {}  # complexity_bin -> [(gsn, score)]

    @property
    def num_clusters(self) -> int:
        """The number of complexity-bin clusters on this island."""
        return len(self._bins)

    @property
    def num_programs(self) -> int:
        """The total number of programs registered on this island."""
        return sum(len(entries) for entries in self._bins.values())

    def register(self, gsn: int, score: float, complexity: int) -> None:
        """Add (gsn, score) to appropriate bin, prune if needed."""
        cbin = complexity // self._complexity_bin_size
        self._bins.setdefault(cbin, []).append((gsn, score))
        if len(self._bins[cbin]) > self._cluster_max_size:
            self._prune(cbin)

    def _prune(self, cbin: int) -> None:
        """Keep top cluster_max_size entries by score."""
        entries = self._bins[cbin]
        entries.sort(key=lambda x: x[1])  # sort by score ascending
        self._bins[cbin] = entries[-self._cluster_max_size:]

    def sample_gsns(
        self,
        temperature: float,
        pareto_front: list[code_manipulation.EvaluatedProgram] | None = None,
    ) -> list[int]:
        """Sample global_sample_nums for prompt generation.

        Returns list of gsns (one per chosen complexity bin).
        """
        bins = list(self._bins.keys())
        if not bins:
            return []
        n = min(len(bins), self._functions_per_prompt)

        if pareto_front and len(pareto_front) >= 2:
            weights = self._pareto_weights(bins, pareto_front)
            idx = np.random.choice(len(bins), size=n, p=weights)
        else:
            idx = np.random.choice(len(bins), size=n)

        sampled = []
        for i in idx:
            cbin = bins[i]
            entries = self._bins[cbin]
            scores = np.array([s for _, s in entries])
            probs = _softmax(scores, temperature)
            chosen = np.random.choice(len(entries), p=probs)
            sampled.append(entries[chosen][0])  # gsn
        return sampled

    def _pareto_weights(
        self,
        bins: list[int],
        pareto_front: list[code_manipulation.EvaluatedProgram],
    ) -> np.ndarray:
        """Compute Pareto-aware bin selection weights."""
        pareto_complexities = np.array([p.complexity for p in pareto_front], dtype=float)
        pareto_scores = np.array([p.score for p in pareto_front], dtype=float)

        weights = np.ones(len(bins), dtype=float)
        for i, bin_key in enumerate(bins):
            bin_center = bin_key * self._complexity_bin_size + self._complexity_bin_size / 2.0
            target = float(np.interp(bin_center, pareto_complexities, pareto_scores))
            cluster_best = max(s for _, s in self._bins[bin_key])
            gap = max(0.0, target - cluster_best)
            weights[i] = 1.0 + gap

        total = weights.sum()
        if total == 0:
            return np.ones(len(bins)) / len(bins)
        return weights / total

    def clear(self) -> None:
        """Reset island to empty state."""
        self._bins.clear()


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
                config.functions_per_prompt,
                config.complexity_bin_size,
                config.cluster_max_size,
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
        if checkpoint_db is None:
            self._checkpoint_db = CheckpointDB(":memory:")
        else:
            self._checkpoint_db = checkpoint_db
        self._max_samples = max_samples
        self._steps_since_checkpoint = 0

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
        # Load island index directly into lightweight Island._bins
        index = checkpoint_db.load_island_index()
        for island_id, bins in index.items():
            if island_id < len(self._islands):
                self._islands[island_id]._bins = bins

        # Load pareto front (only the specific programs, not all)
        pareto_ids = checkpoint_db.load_pareto_front_ids()
        if pareto_ids:
            pareto_programs = checkpoint_db.load_programs_by_ids(pareto_ids)
            self._pareto_front = sorted(
                [pareto_programs[gsn] for gsn in pareto_ids if gsn in pareto_programs],
                key=lambda p: p.complexity,
            )

        # Load metadata
        meta = checkpoint_db.load_metadata()
        self._global_sample_nums = meta.get("global_sample_nums", 0)
        self._last_reset_step = meta.get("last_reset_step", 1)

        best_scores = meta.get("best_score_per_island", [-float("inf")] * len(self._islands))
        for i, s in enumerate(best_scores):
            if i < len(self._best_score_per_island):
                self._best_score_per_island[i] = s

        best_prog_ids = meta.get("best_program_per_island", [None] * len(self._islands))
        gsns_to_load = [gsn for gsn in best_prog_ids if gsn is not None]
        if gsns_to_load:
            best_programs = checkpoint_db.load_programs_by_ids(gsns_to_load)
            for i, gsn in enumerate(best_prog_ids):
                if i < len(self._best_program_per_island) and gsn is not None and gsn in best_programs:
                    self._best_program_per_island[i] = best_programs[gsn]

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

        # 1. Sample gsns from in-memory island (fast)
        gsns = self._islands[island_id].sample_gsns(temperature, pareto)

        # 2. Load full programs from DB (only for prompt text generation)
        programs = self._checkpoint_db.load_programs_by_ids(gsns)
        implementations = [programs[gsn] for gsn in gsns if gsn in programs]

        # 3. Sort by score and generate prompt
        implementations.sort(key=lambda p: p.score)
        code = self._generate_prompt(implementations)
        return Prompt(code, island_id)

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

    def _register_program_in_island(
        self,
        gsn: int,
        score: float,
        complexity: int,
        island_id: int,
    ) -> None:
        """Registers a program entry in the specified island's in-memory index."""
        self._islands[island_id].register(gsn, score, complexity)

        island = self._islands[island_id]
        logger.debug(
            "Island %d: %d clusters, %d total programs",
            island_id,
            island.num_clusters,
            island.num_programs,
        )

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

        with db.transaction():
            self._register_and_persist(evaluated, island_id, ex, db, llm_response_text)

    def _register_and_persist(
        self,
        evaluated: code_manipulation.EvaluatedProgram,
        island_id: int | None,
        ex: object | None,
        db: CheckpointDB,
        llm_response_text: str | None,
    ) -> None:
        """Core registration + DB persistence (runs inside a transaction)."""
        db.insert_program(evaluated, llm_response_text=llm_response_text)

        if ex is not None:
            gsn = evaluated.global_sample_nums
            score = evaluated.score
            complexity = evaluated.complexity

            if island_id is None:
                for iid in range(len(self._islands)):
                    self._register_program_in_island(gsn, score, complexity, iid)
            else:
                self._register_program_in_island(gsn, score, complexity, island_id)

            # Track best per island
            target_ids = range(len(self._islands)) if island_id is None else [island_id]
            for iid in target_ids:
                if score > self._best_score_per_island[iid]:
                    self._best_score_per_island[iid] = score
                    self._best_program_per_island[iid] = evaluated
                    logger.info("Best score of island %d increased to %s", iid, score)

        self._update_pareto_front(evaluated)
        self._profiler.register_function(evaluated, pareto_size=len(self._pareto_front))

        # Periodic checkpoint of island index
        self._steps_since_checkpoint += 1
        if self._steps_since_checkpoint >= self._config.checkpoint_interval:
            self._checkpoint_island_index()

        db.replace_pareto_front([p.global_sample_nums for p in self._pareto_front])
        db.save_profiler_stats(self._profiler)
        per_c = self._profiler.get_best_per_complexity_snapshot()
        if evaluated.complexity is not None and evaluated.complexity in per_c:
            score_val, prog_str, order = per_c[evaluated.complexity]
            db.save_profiler_per_complexity(evaluated.complexity, score_val, prog_str, order)
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

    def _checkpoint_island_index(self) -> None:
        """Persist current in-memory island index to DB."""
        self._checkpoint_db.checkpoint_island_index(
            self._islands, self._config.num_islands,
        )
        self._steps_since_checkpoint = 0

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

        # Not dominated -- add and remove any programs it dominates
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
            self._islands[island_id].clear()
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            self._islands[island_id].register(
                founder.global_sample_nums, founder.score, founder.complexity,
            )
            # Update cached best
            self._best_score_per_island[island_id] = founder.score
            self._best_program_per_island[island_id] = founder
            logger.info("Reset island %d with founder %d", island_id, founder.global_sample_nums)
        # Force checkpoint after reset
        self._checkpoint_island_index()
