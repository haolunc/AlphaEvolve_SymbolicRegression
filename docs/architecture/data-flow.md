# Data Flow

## Single-Chain Mode

```{mermaid}
sequenceDiagram
    participant DB as ProgramsDatabase
    participant LLM as LLM (Sampler)
    participant Eval as Evaluator

    Note over Eval: initialize(): evaluate seed function
    Eval->>DB: EvalResult (seed score + complexity)
    Note over DB: Create database with seed program

    loop Every iteration
        DB->>DB: get_prompt() → Prompt
        DB->>LLM: Prompt.code
        LLM->>LLM: draw_samples() → LLMResponse[]
        LLM->>Eval: SampleMessage (response + island_id)
        Eval->>Eval: analyse() → parse, splice, sandbox exec
        Eval->>DB: EvalResult + SampleMessage
        DB->>DB: register_program()
    end
```

## Distributed Mode

```{mermaid}
flowchart LR
    subgraph init ["Initialization"]
        E0["Evaluator 0"]
        IRQ[("initial_result_queue")]
        E0 -->|EvalResult| IRQ
    end

    subgraph main ["Main Loop"]
        DBW["Database\nWorker"]
        PQ[("prompt_queue")]
        SW["Sampler\nWorkers ×N"]
        SQ[("sample_queue")]
        EW["Evaluator\nWorkers ×M"]
        RQ[("result_queue")]

        IRQ -->|seed result| DBW
        DBW -->|Prompt| PQ
        PQ --> SW
        SW -->|SampleMessage| SQ
        SQ --> EW
        EW -->|"(EvalResult, SampleMessage)"| RQ
        RQ --> DBW
    end

    subgraph monitor ["Monitoring"]
        PFQ[("perf_queue")]
        MON["Monitor\nWorker"]
        DBW -.->|PerfMessage| PFQ
        SW -.->|PerfMessage| PFQ
        EW -.->|PerfMessage| PFQ
        PFQ -.-> MON
    end
```

---

## Glossary

Every term on the diagrams above is briefly explained here, with links to detailed pages.

### Input Files

The pipeline loads four user-provided files at startup: training data (`train.csv`), a task description (`prompt.txt`), a seed equation (`equation.py`), and an evaluation harness (`evaluate.py`).

See {doc}`input-files` for format details and examples.

### Config

A single YAML file parsed by `RunConfig.from_yaml()`. It contains top-level pipeline settings and nested sections for each component (`sampler`, `database`, `evaluator`, `profiler`, `worker`).

See {doc}`config` for the full reference.

### Evaluator

Receives an LLM-generated sample, parses it into a function body, splices it into the evaluation harness, and runs the resulting program in a sandboxed subprocess with a timeout. Returns an `EvalResult` containing the score, optimized parameters, and complexity.

See {doc}`evaluator` for internals.

### Sampler

Wraps an LLM provider (OpenAI, Qwen, or Gemini) with retry logic and optional concurrent sampling via `ThreadPoolExecutor`. Sends a prompt string, receives `LLMResponse` objects.

See {doc}`sampler` for provider details and how to add new ones.

### Database

The `ProgramsDatabase` is the evolutionary core — it stores evaluated programs in an island-based hierarchy, builds prompts by sampling from complexity bins, tracks a Pareto front, and periodically resets weak islands.

See the {ref}`Introduction <the-program-database-island-based-evolution>` for algorithm details.

### Message Types

Typed dataclasses that flow between components:

| Message | Direction | Purpose |
|---------|-----------|---------|
| `Prompt` | Database → Sampler | Prompt text + source island ID |
| `SampleMessage` | Sampler → Evaluator | LLM response + timing metadata |
| `EvalResult` | Evaluator → Database | Parsed function + execution result |
| `PerfMessage` | Any worker → Monitor | Performance statistics |

See {doc}`messages` for the full dataclass reference.

### Queues (distributed mode)

| Queue | Carries | From → To |
|-------|---------|-----------|
| `prompt_queue` | `Prompt` | Database → Samplers |
| `sample_queue` | `SampleMessage` | Samplers → Evaluators |
| `result_queue` | `(EvalResult, SampleMessage)` | Evaluators → Database |
| `initial_result_queue` | `EvalResult` | Evaluator 0 → Database (once, at startup) |
| `perf_queue` | `PerfMessage` | All workers → Monitor |

### Initialization

Before the main loop begins, the seed function must be evaluated to give the database its first program:

- **Single-chain**: `Evaluator.initialize()` is called directly; the result is passed to `ProgramsDatabase.restore_or_create()`.
- **Distributed**: Evaluator 0 calls `initialize()` and puts the `EvalResult` on `initial_result_queue`. The database worker blocks on this queue before creating the database.
