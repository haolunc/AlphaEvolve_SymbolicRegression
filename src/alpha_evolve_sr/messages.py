"""Typed dataclasses for inter-worker queue messages."""

from __future__ import annotations

import dataclasses

from . import code_manipulation


@dataclasses.dataclass(frozen=True)
class SampleMessage:
    """Produced by sampler_worker, consumed by evaluator_worker."""

    sample: str
    island_id: int
    version_generated: int
    sample_time: float
    sample_token_usage: tuple[int, int]
    sample_token_cost: float


@dataclasses.dataclass(frozen=True)
class EvalResult:
    """Result of evaluating a single candidate program.

    Returned by ``Evaluator.analyse()`` and used throughout the pipeline.
    Also serves as the message type on the result queue in distributed mode.
    """

    function: code_manipulation.ParsedFunction
    island_id: int | None
    result_per_test: dict | None
    sample_time: float | None
    evaluate_time: float | None
    sample_token_usage: tuple[int, int] | None
    sample_token_cost: float | None


@dataclasses.dataclass(frozen=True)
class PerfMessage:
    """Performance statistics from any worker, consumed by monitoring_worker."""

    worker_type: str
    worker_id: int
    stats: dict
