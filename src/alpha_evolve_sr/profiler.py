"""Experiment profiling with TensorBoard."""

from __future__ import annotations

import io
import os.path
import time

import matplotlib.pyplot as plt
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from . import code_manipulation
from .config import ProfilerConfig
from .logging_config import get_logger

logger = get_logger("profiler")


# ---------------------------------------------------------------------------
# Minimal SummaryWriter using the standalone ``tensorboard`` package directly,
# avoiding a PyTorch dependency.
# ---------------------------------------------------------------------------

class SummaryWriter:
    """Minimal SummaryWriter using raw tensorboard APIs."""

    def __init__(self, log_dir: str | None = None):
        self._log_dir = log_dir or "runs"
        os.makedirs(self._log_dir, exist_ok=True)
        self._writer = EventFileWriter(self._log_dir)

    def add_scalar(
        self, tag: str, scalar_value: float, global_step: int | None = None,
    ) -> None:
        s = Summary(value=[Summary.Value(tag=tag, simple_value=scalar_value)])
        event = Event(summary=s, wall_time=time.time(), step=global_step or 0)
        self._writer.add_event(event)

    def add_scalars(
        self, main_tag: str, tag_scalar_dict: dict[str, float],
        global_step: int | None = None,
    ) -> None:
        for tag, value in tag_scalar_dict.items():
            self.add_scalar(f"{main_tag}/{tag}", value, global_step)

    def add_figure(
        self, tag: str, figure: object, global_step: int | None = None,
    ) -> None:
        buf = io.BytesIO()
        figure.savefig(buf, format="png")  # type: ignore[union-attr]
        buf.seek(0)
        image_string = buf.getvalue()
        s = Summary(value=[Summary.Value(
            tag=tag,
            image=Summary.Image(
                encoded_image_string=image_string,
                height=0,
                width=0,
                colorspace=0,
            ),
        )])
        event = Event(summary=s, wall_time=time.time(), step=global_step or 0)
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


# ---------------------------------------------------------------------------
# Component: TensorBoardWriter
# ---------------------------------------------------------------------------

class TensorBoardWriter:
    """Wraps all ``SummaryWriter`` operations."""

    def __init__(self, log_dir: str, config: ProfilerConfig):
        self._writer: SummaryWriter | None = None
        self._log_dir = log_dir
        self._config = config
        self._init_writer()

    def _init_writer(self) -> None:
        self._writer = SummaryWriter(log_dir=self._log_dir)

    def write(
        self,
        num_samples: int,
        best_score: float,
        tot_token_cost: float,
        success_count: int,
        failed_count: int,
        tot_sample_time: float,
        tot_evaluate_time: float,
        pareto_front=None,
        best_score_per_island=None,
        island_sizes=None,
    ) -> None:
        """Write all metrics to TensorBoard."""
        self._writer.add_scalar("Best Score of Function", best_score, global_step=num_samples)
        self._writer.add_scalar("Total Token Cost", tot_token_cost, global_step=num_samples)

        self._writer.add_scalars(
            "Legal/Illegal Function",
            {"legal function num": success_count, "illegal function num": failed_count},
            global_step=num_samples,
        )

        self._writer.add_scalars(
            "Total Sample/Evaluate Time",
            {"sample time": tot_sample_time, "evaluate time": tot_evaluate_time},
            global_step=num_samples,
        )

        if num_samples % self._config.log_frequency == 0:
            # Pareto front scatter+line
            if pareto_front and len(pareto_front) >= 2:
                cbins = [p.cbin for p in pareto_front]
                scores = [p.score for p in pareto_front]
                fig, ax = plt.subplots()
                ax.scatter(cbins, scores)
                ax.plot(cbins, scores, linestyle="--", alpha=0.5)
                ax.set_xlabel("Complexity Bin")
                ax.set_ylabel("Score")
                ax.set_title(f"Pareto Front (step {num_samples})")
                self._writer.add_figure("Pareto_Front", fig, global_step=num_samples)
                plt.close(fig)

            # Best score per island
            if best_score_per_island:
                self._writer.add_scalars(
                    "Best Score / Island",
                    {f"island_{i}": s for i, s in enumerate(best_score_per_island)},
                    global_step=num_samples,
                )

            # Island sizes
            if island_sizes:
                self._writer.add_scalars(
                    "Island Size",
                    {f"island_{i}": n for i, n in enumerate(island_sizes)},
                    global_step=num_samples,
                )



# ---------------------------------------------------------------------------
# Public API: Profiler
# ---------------------------------------------------------------------------

class Profiler:
    """Orchestrates experiment profiling: TensorBoard and statistics."""

    def __init__(
        self,
        log_dir: str,
        config: ProfilerConfig | None = None,
    ):
        self._config = config or ProfilerConfig()
        self._log_dir = log_dir
        self._num_samples = 0

        self._tb = TensorBoardWriter(log_dir, self._config)

        # Statistics tracking
        self._best_score: float = -float("inf")
        self._success_count: int = 0
        self._failed_count: int = 0
        self._tot_sample_time: float = 0.0
        self._tot_evaluate_time: float = 0.0
        self._tot_token_cost: float = 0.0

    def register_function(
        self,
        program: code_manipulation.EvaluatedProgram,
        *,
        pareto_front=None,
        best_score_per_island=None,
        island_sizes=None,
    ) -> None:
        """Register a newly evaluated program for logging."""
        sample_order: int = program.global_sample_nums
        if sample_order > self._num_samples:
            self._num_samples = sample_order
        self._log_program(
            program,
            pareto_front=pareto_front,
            best_score_per_island=best_score_per_island,
            island_sizes=island_sizes,
        )

    def _update_stats(self, program: code_manipulation.EvaluatedProgram) -> None:
        """Update aggregate statistics with a newly evaluated *program*."""
        score = program.score

        if score is not None and score > self._best_score:
            self._best_score = score

        if score:
            self._success_count += 1
        else:
            self._failed_count += 1
        if program.sample_time:
            self._tot_sample_time += program.sample_time
        if program.evaluate_time:
            self._tot_evaluate_time += program.evaluate_time
        if program.token_cost:
            self._tot_token_cost += program.token_cost

    def _log_program(
        self,
        program: code_manipulation.EvaluatedProgram,
        *,
        pareto_front=None,
        best_score_per_island=None,
        island_sizes=None,
    ) -> None:
        """Update stats and write TensorBoard data."""
        logger.info(
            "Evaluated Function: score=%s complexity=%s sample_order=%s",
            program.score,
            program.complexity,
            program.global_sample_nums,
        )

        self._update_stats(program)

        self._tb.write(
            num_samples=self._num_samples,
            best_score=self._best_score,
            tot_token_cost=self._tot_token_cost,
            success_count=self._success_count,
            failed_count=self._failed_count,
            tot_sample_time=self._tot_sample_time,
            tot_evaluate_time=self._tot_evaluate_time,
            pareto_front=pareto_front,
            best_score_per_island=best_score_per_island,
            island_sizes=island_sizes,
        )

    # ---- Snapshot / restore for SQLite checkpoint ----

    def get_stats_snapshot(self) -> dict:
        """Returns a dict of all aggregate counters."""
        return {
            "best_score": self._best_score,
            "success_count": self._success_count,
            "failed_count": self._failed_count,
            "tot_sample_time": self._tot_sample_time,
            "tot_evaluate_time": self._tot_evaluate_time,
            "tot_token_cost": self._tot_token_cost,
        }

    def restore_stats(self, stats: dict) -> None:
        """Sets all internal counters from loaded data."""
        self._best_score = stats.get("best_score", -float("inf"))
        self._success_count = stats.get("success_count", 0)
        self._failed_count = stats.get("failed_count", 0)
        self._tot_sample_time = stats.get("tot_sample_time", 0.0)
        self._tot_evaluate_time = stats.get("tot_evaluate_time", 0.0)
        self._tot_token_cost = stats.get("tot_token_cost", 0.0)

