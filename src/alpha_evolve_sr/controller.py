"""Unified evolution controller shared by single-process and distributed modes."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from . import checkpoint, code_manipulation
from . import database as database_mod
from .config import ProfilerConfig, ProgramsDatabaseConfig
from .logging_config import get_logger
from .messages import EvalResult

if TYPE_CHECKING:
    from .database import ProgramsDatabase

logger = get_logger("controller")


class EvolutionController:
    """Manages database lifecycle: init, register, checkpoint, finalize.

    Extracts the shared orchestration logic that was duplicated between
    ``cli.main_single`` and ``workers.database_worker``.
    """

    def __init__(
        self,
        config: ProgramsDatabaseConfig,
        template: code_manipulation.Program,
        function_to_evolve: str,
        log_dir: str,
        *,
        ckpt_dir: str | None = None,
        ckpt_interval: int = 300,
        max_samples: int = 3600,
        profiler_config: ProfilerConfig | None = None,
    ) -> None:
        self._config = config
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._log_dir = log_dir
        self._ckpt_dir = ckpt_dir
        self._ckpt_interval = ckpt_interval
        self._max_samples = max_samples
        self._profiler_config = profiler_config

        self._database: ProgramsDatabase | None = None
        self._last_checkpoint_time: float = time.time()

    def initialize(
        self,
        resume_path: str | None = None,
        initial_result: EvalResult | None = None,
    ) -> None:
        """Create or restore the database, optionally registering an initial result."""
        if resume_path:
            try:
                self._database = checkpoint.load_checkpoint(resume_path)
                logger.info(
                    "Database restored from checkpoint with %d samples",
                    self._database.sample_count,
                )
                self._last_checkpoint_time = time.time()
                return
            except Exception as e:
                logger.error("Failed to load checkpoint: %s", e)
                logger.info("Initializing new database instead")

        self._database = database_mod.ProgramsDatabase(
            self._config, self._template, self._function_to_evolve, self._log_dir,
            profiler_config=self._profiler_config,
        )
        if initial_result is not None:
            self.register_eval_result(initial_result)
        self._last_checkpoint_time = time.time()

    @property
    def database(self) -> ProgramsDatabase:
        """The underlying ProgramsDatabase."""
        assert self._database is not None, "Call initialize() first"
        return self._database

    @property
    def sample_count(self) -> int:
        return self.database.sample_count

    @property
    def should_stop(self) -> bool:
        return self.database.sample_count >= self._max_samples

    def get_prompt(self):
        """Delegate to the database's ``get_prompt``."""
        return self.database.get_prompt()

    def register_eval_result(self, result: EvalResult) -> None:
        """Register an evaluation result in the database."""
        self.database.register_program(
            result.function,
            result.island_id,
            result.result_per_test,
            sample_time=result.sample_time,
            evaluate_time=result.evaluate_time,
            sample_token_usage=result.sample_token_usage,
            sample_token_cost=result.sample_token_cost,
        )

    def maybe_checkpoint(self) -> bool:
        """Save checkpoint if the configured interval has elapsed. Returns True if saved."""
        if not self._ckpt_dir:
            return False
        current_time = time.time()
        if (current_time - self._last_checkpoint_time) > self._ckpt_interval:
            ckpt_path = os.path.join(
                self._ckpt_dir, f"checkpoint_{self.database.sample_count}.pkl",
            )
            checkpoint.save_checkpoint(self.database, ckpt_path)
            self._last_checkpoint_time = current_time
            return True
        return False

    def finalize(self) -> None:
        """Save a final checkpoint (if configured) and write output files."""
        if self._ckpt_dir:
            ckpt_path = os.path.join(self._ckpt_dir, "checkpoint_final.pkl")
            checkpoint.save_checkpoint(self.database, ckpt_path)
        self.database.finalize()
