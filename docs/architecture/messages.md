# Message Dataclasses

```{contents}
:local:
:depth: 2
```

---

## Overview

All data flowing between pipeline stages is carried by **frozen dataclasses** defined in `messages.py`. Frozen fields prevent accidental mutation after creation, and each dataclass acts as a typed contract between the component that creates it and the component that consumes it.

```{mermaid}
flowchart LR
    DB["ProgramsDatabase"] -- "Prompt" --> ST["Sampler Thread"]
    ST -- "SampleMessage" --> ET["Evaluator Thread"]
    ET -- "EvalResult" --> Main["Main Thread"]
    Main -- "register_program()" --> DB
```

---

## LLMResponse

Standardized wrapper around a single LLM API call. Produced by `LLMProvider.generate()`, enriched with cost by `LLM.query()`.

| Field | Type | Description |
|-------|------|-------------|
| `response_text` | `str` | Raw text returned by the LLM |
| `input_tokens` | `int` | Input tokens consumed |
| `output_tokens` | `int` | Output tokens generated |
| `token_cost` | `float` | Computed cost in dollars (default `0.0`; set by `LLM.query()`) |

**Lifecycle:** `LLMProvider.generate()` creates the initial response (cost = 0). `LLM.query()` computes the cost from `cost_per_ktoken` and returns a new `LLMResponse` with `token_cost` filled in.

---

## Prompt

A prompt produced by `ProgramsDatabase.get_prompt()`, consumed by sampler threads.

| Field | Type | Description |
|-------|------|-------------|
| `code` | `str` | Full prompt string, ending with the function header to complete |
| `island_id` | `int` | Which island produced the implementations included in the prompt |

The `code` field contains the problem specification, exemplar programs sampled from an island, and the function header the LLM must continue.

---

## SampleMessage

Produced by `_sampler_task()` in sampler threads, consumed by `_eval_thread_analyse()` in evaluator threads. Bundles an LLM response with sampling metadata.

| Field | Type | Description |
|-------|------|-------------|
| `llm_response` | `LLMResponse` | The LLM's response (with cost) |
| `island_id` | `int` | Source island (carried from the originating `Prompt`) |
| `sample_time` | `float` | Wall-clock seconds for the LLM request |

---

## ExecutionResult

Result of successfully executing a candidate program in the sandbox. Created inside `Evaluator._evaluate_body()` only when the sandboxed `evaluate()` function returns a non-`None` value.

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Fitness score (higher is better; typically negative loss) |
| `optimized_params` | `list[float] \| None` | Numerically optimized parameter values, if any |

The sandbox `evaluate()` function returns a `(score, optimized_params)` tuple. `_evaluate_body()` unpacks this into an `ExecutionResult`.

---

## EvalResult

Top-level evaluation result. Returned by `Evaluator.analyse()` and `Evaluator.initialize()`, consumed by `ProgramsDatabase.register_program()` via the main thread.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `function` | `ParsedFunction` | *(required)* | The parsed equation function (with the evolved body) |
| `execution_result` | `ExecutionResult \| None` | *(required)* | `None` if execution or parsing failed |
| `evaluate_time` | `float \| None` | *(required)* | Wall-clock seconds for sandbox execution; `None` on parse error |
| `complexity` | `int \| None` | `None` | Weighted AST node count |
| `complexity_detail` | `dict \| None` | `None` | Breakdown of complexity by node category |
| `error_type` | `str \| None` | `None` | `"parse"`, `"execution"`, or `"timeout"` |
| `error_message` | `str \| None` | `None` | Human-readable error description |
| `eval_output` | `str \| None` | `None` | Captured stdout/stderr from the sandbox |

**Important:** `complexity` and `complexity_detail` are **not** set by the evaluator. They are attached by the main thread's `_attach_complexity()` function after the `EvalResult` is returned from the evaluator thread. This keeps the evaluator focused on execution and avoids importing the complexity module in the subprocess.

### Error type semantics

| `error_type` | Meaning |
|---|---|
| `"parse"` | LLM response could not be parsed into a valid function body |
| `"execution"` | Sandbox ran but `evaluate()` raised an exception or returned `None` |
| `"timeout"` | Sandbox execution exceeded `timeout_seconds` |

---

## Message Flow

The complete lifecycle of a single sample, from prompt to database registration:

```{mermaid}
sequenceDiagram
    participant DB as ProgramsDatabase
    participant MT as Main Thread
    participant ST as Sampler Thread
    participant ET as Evaluator Thread

    MT->>DB: get_prompt()
    DB-->>MT: Prompt(code, island_id)
    MT->>ST: submit _sampler_task(prompt)
    ST->>ST: LLM.query(prompt.code)
    ST-->>MT: SampleMessage(llm_response, island_id, sample_time)
    MT->>ET: submit _eval_thread_analyse(sample_message)
    ET->>ET: Evaluator.analyse() -> EvalResult
    ET-->>MT: EvalResult (no complexity yet)
    MT->>MT: _attach_complexity(eval_result)
    MT->>DB: register_program(eval_result, sample_message)
```
