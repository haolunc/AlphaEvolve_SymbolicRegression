# Data Flow

## Pipeline Architecture

```{mermaid}
flowchart LR
    subgraph main ["Main Thread"]
        DB["ProgramsDatabase"]
        PROF["TensorBoard\nProfiler"]
    end

    subgraph threads ["ThreadPoolExecutor"]
        S1["Sampler\nThread 1"]
        S2["Sampler\nThread 2"]
        SN["Sampler\nThread N"]
    end

    subgraph pool ["mp.Pool (spawn)"]
        E1["Evaluator\nProcess 1"]
        E2["Evaluator\nProcess 2"]
        EM["Evaluator\nProcess M"]
    end

    DB -->|"Prompt"| S1
    DB -->|"Prompt"| S2
    DB -->|"Prompt"| SN
    S1 -->|"SampleMessage"| E1
    S2 -->|"SampleMessage"| E2
    SN -->|"SampleMessage"| EM
    E1 -->|"(EvalResult, SampleMessage)"| DB
    E2 -->|"(EvalResult, SampleMessage)"| DB
    EM -->|"(EvalResult, SampleMessage)"| DB
    DB -->|"ProfileMetrics"| PROF
```

The main thread acts as the orchestrator:

1. **Generates prompts** from `ProgramsDatabase.get_prompt()`
2. **Submits LLM calls** to `ThreadPoolExecutor` (sampler threads)
3. **Submits evaluations** to `mp.Pool` (evaluator processes)
4. **Registers results** back into the database
5. **Applies backpressure** — limits pending evals to `num_evaluators * 2`

---

## Initialization

Before the main loop begins, the seed function is evaluated via `eval_pool.apply(eval_worker_initialize)` (blocking). The result is passed to `ProgramsDatabase.restore_or_create()`. When resuming from a checkpoint, this step is skipped.

---

## Glossary

### Input Files

The pipeline loads four user-provided files at startup: training data (`train.csv`), a task description (`prompt.txt`), a seed equation (`equation.py`), and an evaluation harness (`evaluate.py`).

See {doc}`input-files` for format details and examples.

### Config

A single YAML file parsed by `RunConfig.from_yaml()`. It contains top-level pipeline settings and nested sections for each component (`sampler`, `database`, `evaluator`).

See {doc}`config` for the full reference.

### Evaluator

Receives an LLM-generated sample, parses it into a function body, splices it into the evaluation harness, and runs the resulting program in a sandboxed subprocess with a timeout. Returns an `EvalResult` containing the score, optimized parameters, and complexity.

See {doc}`evaluator` for internals.

### Sampler

Wraps an LLM provider (OpenAI, Qwen, or Gemini) with retry logic. Each provider creates its HTTP client once in `__init__` for connection reuse. `draw_samples()` loops sequentially `samples_per_prompt` times; threading is managed at the pipeline level via `ThreadPoolExecutor`.

See {doc}`sampler` for provider details and how to add new ones.

### Database

The `ProgramsDatabase` is the evolutionary core — it stores evaluated programs in an island-based hierarchy, builds prompts by sampling from complexity bins, tracks a Pareto front, and periodically resets weak islands.

See the {ref}`Introduction <the-program-database-island-based-evolution>` for algorithm details.

### Message Types

Typed dataclasses that flow between components:

| Message | Direction | Purpose |
|---------|-----------|---------|
| `Prompt` | Database → Sampler threads | Prompt text + source island ID |
| `SampleMessage` | Sampler threads → Evaluator pool | LLM response + timing metadata |
| `EvalResult` | Evaluator pool → Database | Parsed function + execution result |

See {doc}`messages` for the full dataclass reference.
