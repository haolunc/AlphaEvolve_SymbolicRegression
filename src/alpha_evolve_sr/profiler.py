"""Experiment profiling with TensorBoard and JSON sample logs."""

from __future__ import annotations

import json
import os.path

import matplotlib.pyplot as plt
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from . import code_manipulation
from .config import ProfilerConfig
from .logging_config import get_logger

logger = get_logger("profiler")


# ---------------------------------------------------------------------------
# TensorBoard compatibility layer
# ---------------------------------------------------------------------------
# We use the standalone ``tensorboard`` package rather than
# ``torch.utils.tensorboard`` to avoid pulling in PyTorch.  The public API
# we need (add_scalar, add_scalars, add_text, add_figure) is provided by
# ``tensorboard.summary.writer.event_file_writer`` under the hood.  We use
# ``tensorflow.summary`` style if available, otherwise fall back to a thin
# wrapper.
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]
except ImportError:
    try:
        from tensorboardX import SummaryWriter  # type: ignore[import-untyped]
    except ImportError:
        class SummaryWriter:  # type: ignore[no-redef]
            """Minimal SummaryWriter using raw tensorboard APIs."""

            def __init__(self, log_dir: str | None = None):
                self._log_dir = log_dir or "runs"
                os.makedirs(self._log_dir, exist_ok=True)
                self._writer = EventFileWriter(self._log_dir)

            def add_scalar(
                self, tag: str, scalar_value: float, global_step: int | None = None,
            ) -> None:
                import time

                from tensorboard.compat.proto.event_pb2 import Event
                from tensorboard.compat.proto.summary_pb2 import Summary

                s = Summary(value=[Summary.Value(tag=tag, simple_value=scalar_value)])
                event = Event(summary=s, wall_time=time.time(), step=global_step or 0)
                self._writer.add_event(event)

            def add_scalars(
                self, main_tag: str, tag_scalar_dict: dict[str, float],
                global_step: int | None = None,
            ) -> None:
                for tag, value in tag_scalar_dict.items():
                    self.add_scalar(f"{main_tag}/{tag}", value, global_step)

            def add_text(
                self, tag: str, text_string: str, global_step: int | None = None,
            ) -> None:
                import time

                from tensorboard.compat.proto.event_pb2 import Event
                from tensorboard.plugins.text.summary import text_pb

                meta = text_pb(tag, text_string)
                event = Event(summary=meta, wall_time=time.time(), step=global_step or 0)
                self._writer.add_event(event)

            def add_figure(
                self, tag: str, figure: object, global_step: int | None = None,
            ) -> None:
                pass  # Not supported in fallback mode

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
        best_score_per_c: dict[int, float],
        best_progorder_per_c: dict[int, int],
        tot_token_cost: float,
        success_count: int,
        failed_count: int,
        tot_sample_time: float,
        tot_evaluate_time: float,
        cur_best_sample_order: int | None,
        cur_best_program_str: str | None,
        pareto_size: int | None = None,
    ) -> None:
        """Write all metrics to TensorBoard."""
        self._writer.add_scalar("Best Score of Function", best_score, global_step=num_samples)

        if pareto_size is not None:
            self._writer.add_scalar("Pareto Front Size", pareto_size, global_step=num_samples)

        if num_samples % self._config.log_frequency == 0:
            group_size = self._config.complexity_group_size
            grouped: dict[str, float] = {}
            for c, s in best_score_per_c.items():
                gk = c // group_size
                gname = f"C={gk * group_size}-{gk * group_size + group_size - 1}"
                if gname not in grouped or s > grouped[gname]:
                    grouped[gname] = s
            if grouped:
                self._writer.add_scalars("Best Score / Complexity Group", grouped, global_step=num_samples)

            fig, ax = plt.subplots()
            ax.scatter(list(best_score_per_c.keys()), list(best_score_per_c.values()))
            ax.set_xlabel("Complexity")
            ax.set_ylabel("Best score so far")
            ax.set_title(f"Pareto front (step {num_samples})")
            self._writer.add_figure("Score_vs_Complexity", fig, global_step=num_samples)
            plt.close(fig)

            table_lines = ["|Complexity|Score|Program|", "|---|---|---|"]
            for c, s in sorted(best_score_per_c.items()):
                progorder = best_progorder_per_c[c]
                table_lines.append(f"|{c}|{s:.4g}|{progorder}|")
            self._writer.add_text("Best Program by Complexity", "\n".join(table_lines), global_step=num_samples)

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

        if cur_best_sample_order == num_samples and cur_best_program_str is not None:
            self._writer.add_text("Best Function String", cur_best_program_str, global_step=num_samples)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Remove non-serializable writer
        state["_writer"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._init_writer()


# ---------------------------------------------------------------------------
# Component: SampleLogger
# ---------------------------------------------------------------------------

class SampleLogger:
    """Writes per-sample JSON files and equation Python files."""

    def __init__(self, json_dir: str):
        self._json_dir = json_dir
        os.makedirs(self._json_dir, exist_ok=True)

    def write(self, program: code_manipulation.EvaluatedProgram) -> None:
        """Write JSON metadata and equation source file for *program*."""
        sample_order = program.global_sample_nums or 0
        content = {
            "sample_order": sample_order,
            "score": program.score,
            "optimized_params": program.optimized_params.tolist() if program.optimized_params is not None else None,
            "complexity": program.complexity,
            "complexity_detail": program.complexity_detail,
            "sample_time": program.sample_time,
            "evaluate_time": program.evaluate_time,
            "token_usage": program.token_usage,
            "token_cost": program.token_cost,
            "function": str(program),
        }
        path = os.path.join(self._json_dir, f"samples_{sample_order}.json")
        with open(path, "w") as f:
            json.dump(content, f, indent=4)

        function_path = os.path.join(self._json_dir, f"equation_{sample_order}.py")
        program.save_to_file(function_path)


# ---------------------------------------------------------------------------
# Component: StatisticsTracker
# ---------------------------------------------------------------------------

class StatisticsTracker:
    """Tracks aggregate score, complexity, and cost statistics."""

    def __init__(self) -> None:
        self.best_score: float = -float("inf")
        self.best_program_sample_order: int | None = None
        self.best_program_str: str | None = None

        self.best_score_per_c: dict[int, float] = {}
        self.best_progstr_per_c: dict[int, str] = {}
        self.best_progorder_per_c: dict[int, int] = {}

        self.success_count: int = 0
        self.failed_count: int = 0
        self.tot_sample_time: float = 0.0
        self.tot_evaluate_time: float = 0.0
        self.tot_token_cost: float = 0.0

    def update(self, program: code_manipulation.EvaluatedProgram) -> None:
        """Update statistics with a newly evaluated *program*."""
        sample_order = program.global_sample_nums or 0
        score = program.score
        complexity = program.complexity

        if score is not None and score > self.best_score:
            self.best_score = score
            self.best_program_sample_order = sample_order
            self.best_program_str = str(program)

        if complexity is not None:
            if complexity not in self.best_score_per_c or score > self.best_score_per_c[complexity]:
                self.best_score_per_c[complexity] = score
                self.best_progstr_per_c[complexity] = str(program)
                self.best_progorder_per_c[complexity] = sample_order

        if score:
            self.success_count += 1
        else:
            self.failed_count += 1
        if program.sample_time:
            self.tot_sample_time += program.sample_time
        if program.evaluate_time:
            self.tot_evaluate_time += program.evaluate_time
        if program.token_cost:
            self.tot_token_cost += program.token_cost


# ---------------------------------------------------------------------------
# Public API: Profiler
# ---------------------------------------------------------------------------

class Profiler:
    """Orchestrates experiment profiling.

    Delegates to ``TensorBoardWriter``, ``SampleLogger``, and
    ``StatisticsTracker`` for the actual work.
    """

    def __init__(
        self,
        num_islands: int,
        log_dir: str,
        max_log_nums: int | None = None,
        config: ProfilerConfig | None = None,
    ):
        self._config = config or ProfilerConfig()
        self._log_dir = log_dir
        self._max_log_nums = max_log_nums
        self._num_samples = 0

        self._tb = TensorBoardWriter(log_dir, self._config)
        self._sample_logger = SampleLogger(os.path.join(log_dir, "samples"))
        self._stats = StatisticsTracker()

    def register_function(
        self,
        program: code_manipulation.EvaluatedProgram,
        pareto_size: int | None = None,
    ) -> None:
        """Register a newly evaluated program for logging."""
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_order: int = program.global_sample_nums
        if sample_order > self._num_samples:
            self._num_samples = sample_order
        self._log_program(program, pareto_size=pareto_size)

    def _log_program(
        self,
        program: code_manipulation.EvaluatedProgram,
        pareto_size: int | None = None,
    ) -> None:
        """Write JSON, update stats, and write TensorBoard data."""
        logger.info(
            "Evaluated Function: score=%s complexity=%s sample_order=%s",
            program.score,
            program.complexity,
            program.global_sample_nums,
        )

        self._sample_logger.write(program)
        self._stats.update(program)

        self._tb.write(
            num_samples=self._num_samples,
            best_score=self._stats.best_score,
            best_score_per_c=self._stats.best_score_per_c,
            best_progorder_per_c=self._stats.best_progorder_per_c,
            tot_token_cost=self._stats.tot_token_cost,
            success_count=self._stats.success_count,
            failed_count=self._stats.failed_count,
            tot_sample_time=self._stats.tot_sample_time,
            tot_evaluate_time=self._stats.tot_evaluate_time,
            cur_best_sample_order=self._stats.best_program_sample_order,
            cur_best_program_str=self._stats.best_program_str,
            pareto_size=pareto_size,
        )

    def write_best_program_per_c_file(self) -> None:
        """Writes the best program found for each complexity to a text file."""
        output_path = os.path.join(self._log_dir, "best_programs_per_complexity.txt")
        with open(output_path, "w") as f:
            for c in sorted(self._stats.best_score_per_c.keys()):
                score = self._stats.best_score_per_c[c]
                order = self._stats.best_progorder_per_c[c]
                prog_str = self._stats.best_progstr_per_c[c]
                f.write(f"{c},{score:.4g},{order}\n{prog_str}\n")
        logger.info("Best programs per complexity saved to %s", output_path)

    def write_pareto_front(
        self, pareto_front: list[code_manipulation.EvaluatedProgram],
    ) -> None:
        """Write Pareto-optimal programs to a Python file."""
        output_path = os.path.join(self._log_dir, "pareto_front.py")
        with open(output_path, "w") as f:
            f.write("# Pareto-optimal programs (score vs complexity)\n")
            f.write(f"# Total: {len(pareto_front)} programs\n\n")
            for prog in pareto_front:
                f.write(f"# score={prog.score:.6g}  complexity={prog.complexity}\n")
                f.write(str(prog))
                f.write("\n\n")
        logger.info("Pareto front (%d programs) saved to %s", len(pareto_front), output_path)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
