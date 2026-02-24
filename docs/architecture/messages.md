# Message Dataclasses

```{contents}
:local:
:depth: 2
```

---

## Overview

All data flowing between pipeline components is carried by frozen dataclasses defined in `messages.py`. These typed messages serve as **contracts** between components — each field is explicit, and the frozen constraint prevents accidental mutation after creation.

---

## LLMResponse

Standardized response from any LLM provider. Produced by `LLMProvider.generate()`, consumed by `LLM.draw_samples()` callers.

| Field | Type | Description |
|-------|------|-------------|
| `response_text` | `str` | Raw text returned by the LLM |
| `input_tokens` | `int` | Number of input tokens consumed |
| `output_tokens` | `int` | Number of output tokens generated |
| `token_cost` | `float` | Computed cost (default `0.0`, set by `LLM._query_with_retry`) |

---

## Prompt

Produced by `ProgramsDatabase.get_prompt()`, consumed by sampler threads.

| Field | Type | Description |
|-------|------|-------------|
| `code` | `str` | The full prompt string, ending with the function header to complete |
| `island_id` | `int` | Which island the sampled programs came from |

---

## SampleMessage

Produced by sampler threads, consumed by evaluator pool workers. Bundles an LLM response with sampling metadata.

| Field | Type | Description |
|-------|------|-------------|
| `llm_response` | `LLMResponse` | The LLM's response |
| `island_id` | `int` | Source island (carried from the `Prompt`) |
| `sample_time` | `float` | Wall-clock time for the LLM request (seconds) |

---

## ExecutionResult

Result of running a candidate program in the sandbox. Created inside `Evaluator._evaluate_body()`.

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Fitness score (higher is better; typically negative loss) |
| `optimized_params` | `list[float] \| None` | Numerically optimized parameter values |
| `complexity` | `int \| None` | Weighted AST node count |
| `complexity_detail` | `dict` | Breakdown of complexity by category |

---

## EvalResult

Top-level evaluation result. Returned by `Evaluator.analyse()` and `Evaluator.initialize()`, consumed by `ProgramsDatabase.register_program()`.

| Field | Type | Description |
|-------|------|-------------|
| `function` | `ParsedFunction` | The parsed equation function (with evolved body) |
| `execution_result` | `ExecutionResult \| None` | `None` if execution failed |
| `evaluate_time` | `float \| None` | Wall-clock time for sandbox execution (seconds) |

The eval pool worker function `eval_worker_analyse()` returns `(EvalResult, SampleMessage)` tuples so the main thread has both pieces for `database.register_program()`.
