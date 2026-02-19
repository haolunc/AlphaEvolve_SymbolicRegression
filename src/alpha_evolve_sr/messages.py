"""Typed dataclasses for inter-worker queue messages."""

from __future__ import annotations

import dataclasses

from . import code_manipulation


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    """Standardized response from any LLM provider."""

    response_text: str
    input_tokens: int
    output_tokens: int
    token_cost: float = 0.0


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


@dataclasses.dataclass(frozen=True)
class SampleMessage:
    """Produced by sampler_worker, consumed by evaluator_worker."""

    llm_response: LLMResponse
    island_id: int
    sample_time: float


@dataclasses.dataclass(frozen=True)
class ExecutionResult:
    """Result of executing a candidate program in the sandbox."""

    score: float
    optimized_params: list[float] | None
    complexity: int | None
    complexity_detail: dict


@dataclasses.dataclass(frozen=True)
class EvalResult:
    """Result of evaluating a single candidate program.

    Returned by ``Evaluator.analyse()`` and used throughout the pipeline.
    Sampling metadata (island_id, timing, token usage) lives in ``SampleMessage``.
    """

    function: code_manipulation.ParsedFunction
    execution_result: ExecutionResult | None
    evaluate_time: float | None


@dataclasses.dataclass(frozen=True)
class PerfMessage:
    """Performance statistics from any worker, consumed by monitoring_worker."""

    worker_type: str
    worker_id: int
    stats: dict
