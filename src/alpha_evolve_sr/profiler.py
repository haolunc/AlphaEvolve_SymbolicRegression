"""Experiment profiling with TensorBoard."""

from __future__ import annotations

import dataclasses
import io
import os.path
import time

import matplotlib.pyplot as plt
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

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
# ProfileMetrics dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ProfileMetrics:
    """All metrics forwarded to TensorBoard in a single write call."""

    num_samples: int
    best_score: float
    tot_token_cost: float
    success_count: int
    failed_count: int
    tot_sample_time: float
    tot_evaluate_time: float
    pareto_front: list | None = None
    best_score_per_island: list[float] | None = None
    island_sizes: list[int] | None = None


# ---------------------------------------------------------------------------
# Component: TensorBoardWriter
# ---------------------------------------------------------------------------

class TensorBoardWriter:
    """Wraps all ``SummaryWriter`` operations."""

    def __init__(self, log_dir: str, log_frequency: int = 100):
        self._writer: SummaryWriter | None = None
        self._log_dir = log_dir
        self._log_frequency = log_frequency
        self._init_writer()

    def _init_writer(self) -> None:
        self._writer = SummaryWriter(log_dir=self._log_dir)

    def write(self, metrics: ProfileMetrics) -> None:
        """Write all metrics to TensorBoard."""
        self._writer.add_scalar("Best Score of Function", metrics.best_score, global_step=metrics.num_samples)

        if metrics.num_samples % self._log_frequency == 0:
            # Pareto front scatter+line
            if metrics.pareto_front and len(metrics.pareto_front) >= 2:
                cbins = [p.cbin for p in metrics.pareto_front]
                scores = [p.score for p in metrics.pareto_front]
                fig, ax = plt.subplots()
                ax.scatter(cbins, scores)
                ax.plot(cbins, scores, linestyle="--", alpha=0.5)
                ax.set_xlabel("Complexity Bin")
                ax.set_ylabel("Score")
                ax.set_title(f"Pareto Front (step {metrics.num_samples})")
                self._writer.add_figure("Pareto_Front", fig, global_step=metrics.num_samples)
                plt.close(fig)

            # Best score per island
            if metrics.best_score_per_island:
                self._writer.add_scalars(
                    "Best Score / Island",
                    {f"island_{i}": s for i, s in enumerate(metrics.best_score_per_island)},
                    global_step=metrics.num_samples,
                )

            # Island sizes
            if metrics.island_sizes:
                self._writer.add_scalars(
                    "Island Size",
                    {f"island_{i}": n for i, n in enumerate(metrics.island_sizes)},
                    global_step=metrics.num_samples,
                )


        self._writer.add_scalar("Total Token Cost", metrics.tot_token_cost, global_step=metrics.num_samples)

        self._writer.add_scalars(
            "Legal/Illegal Function",
            {"legal function num": metrics.success_count, "illegal function num": metrics.failed_count},
            global_step=metrics.num_samples,
        )

        self._writer.add_scalars(
            "Total Sample/Evaluate Time",
            {"sample time": metrics.tot_sample_time, "evaluate time": metrics.tot_evaluate_time},
            global_step=metrics.num_samples,
        )
