"""Experiment profiling with TensorBoard."""

from __future__ import annotations

import dataclasses
import io
import os
import time

import matplotlib.pyplot as plt
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
from tensorboard.plugins.custom_scalar import layout_pb2, metadata as cs_metadata
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

    def add_scalar_batch(
        self, tag_value_pairs: list[tuple[str, float]],
        global_step: int | None = None,
    ) -> None:
        """Pack multiple scalar values into a single Summary/Event."""
        values = [Summary.Value(tag=tag, simple_value=val) for tag, val in tag_value_pairs]
        event = Event(summary=Summary(value=values), wall_time=time.time(), step=global_step or 0)
        self._writer.add_event(event)

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
    # Pipeline throughput (transient, not checkpointed)
    pending_evals: int | None = None
    pending_samplers: int | None = None
    wall_time_seconds: float | None = None


# ---------------------------------------------------------------------------
# Component: TensorBoardWriter
# ---------------------------------------------------------------------------

class TensorBoardWriter:
    """Wraps all ``SummaryWriter`` operations."""

    def __init__(self, log_dir: str, num_islands: int = 10):
        self._writer: SummaryWriter | None = None
        self._log_dir = log_dir
        self._num_islands = num_islands
        self._init_writer()

    def _init_writer(self) -> None:
        self._writer = SummaryWriter(log_dir=self._log_dir)
        self._write_custom_scalars_layout()

    def _write_custom_scalars_layout(self) -> None:
        """Write a Custom Scalars layout so TensorBoard overlays related metrics."""
        # --- Island chart groups (5 per chart) ---
        score_charts: list[layout_pb2.Chart] = []
        size_charts: list[layout_pb2.Chart] = []
        for start in range(0, self._num_islands, 5):
            end = min(start + 5, self._num_islands)
            island_ids = list(range(start, end))
            score_charts.append(layout_pb2.Chart(
                title=f"Islands {start}\u2013{end - 1}",
                multiline=layout_pb2.MultilineChartContent(
                    tag=[f"Best Score / Island/island_{i}" for i in island_ids],
                ),
            ))
            size_charts.append(layout_pb2.Chart(
                title=f"Islands {start}\u2013{end - 1}",
                multiline=layout_pb2.MultilineChartContent(
                    tag=[f"Island Size/island_{i}" for i in island_ids],
                ),
            ))

        layout = layout_pb2.Layout(category=[
            layout_pb2.Category(
                title="Time",
                chart=[layout_pb2.Chart(
                    title="Sample vs Evaluate Time",
                    multiline=layout_pb2.MultilineChartContent(tag=[
                        "Total Sample/Evaluate Time/sample time",
                        "Total Sample/Evaluate Time/evaluate time",
                    ]),
                )],
            ),
            layout_pb2.Category(
                title="Function Quality",
                chart=[layout_pb2.Chart(
                    title="Legal vs Illegal",
                    multiline=layout_pb2.MultilineChartContent(tag=[
                        "Legal/Illegal Function/legal function num",
                        "Legal/Illegal Function/illegal function num",
                    ]),
                )],
            ),
            layout_pb2.Category(title="Island Scores", chart=score_charts),
            layout_pb2.Category(title="Island Sizes", chart=size_charts),
        ])

        plugin_data = SummaryMetadata.PluginData(
            plugin_name=cs_metadata.PLUGIN_NAME,
            content=layout.SerializeToString(),
        )
        summary = Summary(value=[
            Summary.Value(
                tag=cs_metadata.CONFIG_SUMMARY_TAG,
                metadata=SummaryMetadata(plugin_data=plugin_data),
            ),
        ])
        event = Event(summary=summary, wall_time=time.time())
        self._writer._writer.add_event(event)

    def write(self, metrics: ProfileMetrics) -> None:
        """Write all metrics to TensorBoard.

        Scalar values are packed into a single Summary/Event.
        Callers are responsible for gating by ``log_frequency``.
        """
        pairs: list[tuple[str, float]] = [
            ("Best Score of Function", metrics.best_score),
            ("Total Token Cost", metrics.tot_token_cost),
            ("Legal/Illegal Function/legal function num", float(metrics.success_count)),
            ("Legal/Illegal Function/illegal function num", float(metrics.failed_count)),
            ("Total Sample/Evaluate Time/sample time", metrics.tot_sample_time),
            ("Total Sample/Evaluate Time/evaluate time", metrics.tot_evaluate_time),
        ]

        # Pipeline throughput (conditional)
        if metrics.wall_time_seconds and metrics.wall_time_seconds > 0:
            pairs.append(("Pipeline/Throughput (samples per sec)",
                          metrics.num_samples / metrics.wall_time_seconds))
        if metrics.pending_evals is not None:
            pairs.append(("Pipeline/Pending Evals", float(metrics.pending_evals)))
        if metrics.pending_samplers is not None:
            pairs.append(("Pipeline/Pending Samplers", float(metrics.pending_samplers)))

        # Per-island metrics
        if metrics.best_score_per_island:
            pairs.extend(
                (f"Best Score / Island/island_{i}", s)
                for i, s in enumerate(metrics.best_score_per_island)
            )
        if metrics.island_sizes:
            pairs.extend(
                (f"Island Size/island_{i}", float(n))
                for i, n in enumerate(metrics.island_sizes)
            )

        self._writer.add_scalar_batch(pairs, global_step=metrics.num_samples)

        # Pareto figure (image — separate event)
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
