# Architecture Document — AlphaEvolve Symbolic Regression

> **Version**: 0.3.0 | **Date**: 2026-02-17

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does) `[conceptual]`
2. [Architecture at a Glance](#2-architecture-at-a-glance) `[conceptual]`
3. [The Evolutionary Algorithm](#3-the-evolutionary-algorithm) `[conceptual]`
4. [Module Map](#4-module-map) `[reference]`
5. [The Distributed Pipeline](#5-the-distributed-pipeline) `[conceptual]`
6. [Data Flow Through the Evaluator](#6-data-flow-through-the-evaluator) `[conceptual]`
7. [Prompt Construction](#7-prompt-construction) `[conceptual]`
8. [Key Design Patterns](#8-key-design-patterns) `[reference]`
9. [Configuration Reference](#9-configuration-reference) `[reference]`
10. [Problem Specification Format](#10-problem-specification-format) `[how-to]`
11. [Common Extension Points](#11-common-extension-points) `[how-to]`
12. [Appendix: Import Dependency Matrix](#12-appendix-import-dependency-matrix) `[reference]`

---

## 1. What This System Does

**Symbolic regression** is the task of discovering a mathematical equation
that fits observed data. Unlike neural-network regression, the output is a
*human-readable formula* — compact, interpretable, and amenable to
scientific insight.

This system uses **large language models (LLMs)** as the mutation operator
inside an evolutionary algorithm. Instead of random crossover or point
mutations on expression trees, an LLM reads a prompt containing
previously discovered equations (ranked worst-to-best) and proposes a new,
improved version. The candidate is then executed in a sandbox, scored
against training data, and inserted into a population database.

**Concrete example.** Given electron-density data from DFT calculations, the
system discovers a symbolic exchange-correlation energy functional
`e_xc(rho, s, params)` that minimizes a combined energy + potential loss.
It starts from a seed (e.g. PBE exchange) and iterates until it finds a
compact formula that outperforms the seed.

The core loop is three stages, repeated thousands of times:

```
Database  ──prompt──▶  LLM Sampler  ──candidate──▶  Evaluator  ──result──▶  Database
```

---

## 2. Architecture at a Glance

```
┌────────────────────────────────────────────────────────────────────────┐
│                         CLI  (cli.py:main)                             │
│  Parses YAML config, loads problem spec + data, picks execution mode   │
└─────────────┬───────────────────────────────────┬──────────────────────┘
              │ distributed=True                  │ distributed=False
              ▼                                   ▼
┌──────────────────────────┐         ┌───────────────────────────┐
│   main_distributed()     │         │   main_single()           │
│   mp.Process workers     │         │   Sequential loop         │
│   connected by Queues    │         │   in a single process     │
└──────────────────────────┘         └───────────────────────────┘
              │                                   │
              ▼                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Shared Core Components                             │
│                                                                        │
│  ProgramsDatabase ◀──▶ Island ◀──▶ Cluster    (database.py)           │
│  LLM ◀──▶ LLMProvider (ABC)                   (sampler.py)            │
│  Evaluator ◀──▶ Sandbox                        (evaluator.py)          │
│  Profiler (TensorBoard + JSON)                 (profiler.py)           │
│  Checkpoint (pickle DB + YAML config)          (checkpoint.py)         │
└────────────────────────────────────────────────────────────────────────┘
```

### Two Execution Modes

| Mode | Entry Point | Parallelism | Use Case |
|------|-------------|-------------|----------|
| **Distributed** | `main_distributed()` | Multiple OS processes via `mp.Process` | Production runs |
| **Non-distributed** | `main_single()` | Single process, sequential | Debugging / small experiments |

Both modes share the same `ProgramsDatabase`, `LLM`, and `Evaluator` classes. The only difference is orchestration: distributed mode connects workers via `multiprocessing.Queue`; non-distributed mode calls them directly in a loop.

---

## 3. The Evolutionary Algorithm

### 3.1 Island Model

The population is partitioned into **N islands** (default 10). Each island
maintains its own set of programs and produces prompts independently. This
preserves diversity — different islands can explore different regions of
the search space without converging prematurely.

Every `reset_period` samples (default 700), the system ranks all islands
by their best score (plus a tiny noise term for tie-breaking), then
**resets the bottom 50%**. Each reset island is re-created empty and seeded
with the best program from a randomly chosen surviving island (the
"founder"). This ensures weak islands get a fresh start from proven
genetic material.

Key method: `ProgramsDatabase.reset_islands()`

### 3.2 Complexity-Binned Clusters

Within each island, programs are grouped into **clusters** by an integer
complexity bin. The complexity of a program is computed via AST analysis
(`complexity.py`) — counting binary operations, variable references,
constants, and function calls.

```
complexity_bin = program.complexity // complexity_bin_size
```

Each cluster stores up to `cluster_max_size` (default 100) programs. When
a cluster overflows, the lowest-scoring programs are pruned.

When constructing a prompt, the island randomly selects
`functions_per_prompt` (default 4) clusters, draws one program from each
via **temperature-scaled softmax sampling** over scores, sorts them
worst-to-best, and passes them to the LLM as "previous versions".

The temperature decays linearly within each period
(`cluster_sampling_temperature_period`), creating alternating phases of
exploration (high temperature, more uniform selection) and exploitation
(low temperature, best programs dominate).

Key classes: `Island`, `Cluster`

### 3.3 Pareto Front

The database maintains a **Pareto front** in the (complexity, score) space.
A program is *non-dominated* if no other program has both lower-or-equal
complexity and higher-or-equal score. The front is updated on every
`register_program()` call.

When `ProgramsDatabaseConfig.pareto_aware = True`, the Pareto front
influences cluster selection during prompt construction. Clusters whose
best score lags behind the Pareto-interpolated target at their complexity
level receive higher selection weight (via `Island._pareto_weights()`).
This steers exploration toward under-performing complexity regions.

At finalization, the Pareto front is written to `pareto_front.py` in the
log directory.

Key methods: `ProgramsDatabase._update_pareto_front()`, `Island._pareto_weights()`

---

## 4. Module Map

### Tier Diagram

Modules are organized into four tiers by dependency direction. Higher tiers
import from lower tiers, never the reverse.

```
Tier 3 — Orchestration (depends on everything below)
  ┌──────────┐  ┌──────────┐
  │  cli.py   │  │ workers.py│
  └──────────┘  └──────────┘

Tier 2 — Core Domain (depends on Tier 0–1)
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  database.py  │  │  evaluator.py │  │  sampler.py   │
  └──────────────┘  └──────────────┘  └──────────────┘

Tier 1 — Support (depends on Tier 0)
  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  ┌────────────┐
  │ profiler.py   │  │ checkpoint.py │  │ code_manipulation.py│  │complexity.py│
  └──────────────┘  └──────────────┘  └────────────────────┘  └────────────┘

Tier 0 — Foundation (no intra-package dependencies, or only logging/config)
  ┌────────────┐  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐
  │  config.py  │  │logging_config.py│  │ exceptions.py │  │  messages.py  │
  └────────────┘  └────────────────┘  └──────────────┘  └──────────────┘
```

### Module Reference

| Module | Lines | Tier | Responsibility | Key Public API |
|--------|------:|------|----------------|----------------|
| `database.py` | 528 | 2 | Evolutionary algorithm: population management, island model, prompt construction | `ProgramsDatabase`, `Island`, `Cluster`, `Prompt` |
| `workers.py` | 388 | 3 | Multiprocessing workers for distributed mode | `database_worker()`, `sampler_worker()`, `evaluator_worker()`, `monitoring_worker()` |
| `code_manipulation.py` | 340 | 1 | Python AST parsing and token-level code rewriting | `ParsedFunction`, `EvaluatedProgram`, `Program`, `text_to_function()`, `rename_function_calls()` |
| `profiler.py` | 337 | 1 | TensorBoard metrics, JSON sample logs, statistics tracking | `Profiler`, `TensorBoardWriter` |
| `cli.py` | 280 | 3 | Entry point: arg parsing, config loading, mode dispatch | `main()`, `main_distributed()`, `main_single()`, `load_problem()` |
| `evaluator.py` | 230 | 2 | Sandbox code execution and scoring | `Evaluator`, `Sandbox` |
| `config.py` | 216 | 0 | Frozen config dataclasses, YAML serialization | `RunConfig`, `ProgramsDatabaseConfig`, `SamplerConfig`, `EvaluatorConfig`, `ProfilerConfig`, `WorkerConfig` |
| `sampler.py` | 211 | 2 | LLM provider abstraction with retry logic | `LLM`, `LLMProvider` (ABC), `OpenAIProvider`, `QwenProvider`, `GeminiProvider` |
| `complexity.py` | 118 | 1 | AST-based complexity scoring with weighted op/func counting | `complexity_score()`, `ComplexityVisitor` |
| `checkpoint.py` | 58 | 1 | Pickle-based database persistence, YAML config save/load | `save_checkpoint()`, `load_checkpoint()`, `save_config()`, `load_config()` |
| `messages.py` | 45 | 0 | Typed dataclasses for inter-worker queue messages | `SampleMessage`, `EvalResult`, `PerfMessage` |
| `logging_config.py` | 45 | 0 | Unified logging setup | `configure_logging()`, `get_logger()`, `setup_file_logger()` |
| `exceptions.py` | 13 | 0 | Custom exception hierarchy | `AlphaEvolveSRError`, `LLMProviderError`, `CheckpointError` |

---

## 5. The Distributed Pipeline

### 5.1 Queue Topology

Four types of worker processes communicate through five `multiprocessing.Queue` instances:

```
                    prompt_queue                 sample_queue
  ┌─────────────┐ ─────────────▶ ┌────────────┐ ─────────────▶ ┌──────────────┐
  │  Database    │               │ Sampler(0) │               │ Evaluator(0) │
  │  Worker (1)  │               │ Sampler(1) │               │ Evaluator(1) │
  │              │ ◀───────────── │    ...     │               │     ...      │
  └──────┬───┬──┘  result_queue  │ Sampler(N) │               │ Evaluator(M) │
         │   │                   └─────┬──────┘               └──────┬───────┘
         │   │                         │         perf_queue          │
         │   │    initial_result_queue  │    ┌──────────────────┐    │
         │   │◀────────────────────────│────│  Monitor (1)     │◀───│
         │   │                         └───▶│                  │◀───┘
         │   └─────────────────────────────▶│                  │
         └─────────────────────────────────▶│                  │
                                            └──────────────────┘
```

### 5.2 Process Startup Sequence

1. `main_distributed()` creates all queues and shared counters
2. Start **Monitor** process (`monitoring_worker`)
3. Start **M Evaluator** processes (`evaluator_worker`); evaluator 0 evaluates the initial seed program
4. Sleep 2 seconds to allow initial evaluation to complete
5. Start **Database Worker** (`database_worker`); blocks on `initial_result_queue` or restores from checkpoint
6. Start **N Sampler** processes (`sampler_worker`)
7. Main process polls `termination_event` every 5 seconds
8. On termination: close queues, join processes (10 s timeout), terminate stragglers

### 5.3 Message Types

| Queue | Message Type | Producer | Consumer | Fields |
|-------|-------------|----------|----------|--------|
| `prompt_queue` | `Prompt` | Database Worker | Sampler Workers | `code`, `version_generated`, `island_id` |
| `sample_queue` | `SampleMessage` | Sampler Workers | Evaluator Workers | `sample`, `island_id`, `version_generated`, `sample_time`, `sample_token_usage`, `sample_token_cost` |
| `result_queue` | `EvalResult` | Evaluator Workers | Database Worker | `function`, `island_id`, `result_per_test`, `sample_time`, `evaluate_time`, `sample_token_usage`, `sample_token_cost` |
| `initial_result_queue` | `EvalResult` | Evaluator 0 | Database Worker | (same as above — used only once at startup) |
| `perf_queue` | `PerfMessage` | All Workers | Monitor | `worker_type`, `worker_id`, `stats` |

### 5.4 Backpressure Mechanism

Two `multiprocessing.Value` counters prevent queues from growing without
bound:

- **`prompt_pending_count`** — incremented by Database Worker on put,
  decremented by Sampler on get. The Database Worker will not enqueue a
  new prompt while `prompt_pending_count >= num_samplers`.

- **`sample_pending_count`** — incremented by Sampler on put, decremented
  by Evaluator on get. A Sampler will not enqueue a new sample while
  `sample_pending_count >= num_evaluators` (sleeps 3 s and retries).

This keeps each queue at most one item per downstream consumer, ensuring
that slow consumers are not overwhelmed.

---

## 6. Data Flow Through the Evaluator

The `Evaluator.analyse()` method transforms raw LLM text into a scored
`EvalResult` in these steps:

1. **Extract Python code** — `_extract_python(text)` strips the
   `` ```python `` fences from the LLM response. If no fences are found,
   the raw text is used as-is.

2. **Parse to function** — `code_manipulation.text_to_function()` parses
   the extracted code into a `ParsedFunction` via Python's `ast` module.

3. **Assemble runnable program** — `_sample_to_program()` splices the new
   function body into the `evaluate.py` template, producing a complete
   Python script that defines both `equation()` and `evaluate()`.

4. **Sandbox execution** — `Sandbox.run()` sends the assembled program to
   a spawned subprocess pool (`mp.get_context("spawn").Pool(1)`). The
   subprocess calls `exec()` on the program, JIT-compiles via JAX, runs
   the `evaluate()` function against the training data, and returns
   `(score, optimized_params)`.

5. **Timeout handling** — if the subprocess exceeds
   `EvaluatorConfig.timeout_seconds` (default 400), the pool is killed
   and transparently recreated on the next call.

6. **Complexity scoring** — on success, `complexity_score()` walks the
   AST of the new function to compute an integer complexity and a
   breakdown `Counter`.

7. **Package result** — an `EvalResult` dataclass is returned with the
   `ParsedFunction`, score, optimized params, complexity, and timing
   metadata.

> **Warning:** The sandbox uses `exec()` to run LLM-generated code. It
> runs in a separate *process* (not just a thread) to provide isolation,
> but there is no OS-level sandboxing (no containers, no seccomp). Do
> not run this system on untrusted LLM outputs in a production
> environment without additional safeguards.

### Sandbox Design Rationale

The `Sandbox` uses a **persistent, lazily-created** `mp.Pool(1)`:

- **Lazy creation** avoids startup cost until the first evaluation.
- **Pool reuse** avoids the ~1–3 s overhead of spawning a new process per
  evaluation.
- **Kill-and-recreate on failure** handles timeouts and crashes gracefully
  without corrupting state.

---

## 7. Prompt Construction

The prompt sent to the LLM is assembled by `Island._generate_prompt()`.
It follows a structured template designed to guide the LLM toward
producing improved equations.

### Template Structure

```
┌─────────────────────────────────────────────────────────┐
│  1. Task Description (from specs/<problem>/prompt.txt)  │
│     "Develop a compact and physically interpretable..." │
├─────────────────────────────────────────────────────────┤
│  2. Rules                                               │
│     - Preserve the full function signature              │
│     - Only output equation_v{N} in ```python```         │
│     - Add inline comments explaining each term          │
├─────────────────────────────────────────────────────────┤
│  3. Previous versions (sorted worst → best)             │
│     ```python                                           │
│     def equation_v0(rho, s, params):                    │
│         """..."""                                        │
│         ...   # (score: -0.85)                          │
│                                                         │
│     def equation_v1(rho, s, params):                    │
│         """Improved version of equation_v0."""           │
│         ...   # (score: -0.42)                          │
│                                                         │
│     def equation_v2(rho, s, params):                    │
│         """Improved version of equation_v1."""           │
│         ...   # (score: -0.15)                          │
│     ```                                                 │
├─────────────────────────────────────────────────────────┤
│  4. Completion target                                   │
│     "Now define:"                                       │
│     ```python                                           │
│     def equation_v3(rho, s, params):                    │
│         """Improved version of equation_v2."""           │
│     ```                                                 │
└─────────────────────────────────────────────────────────┘
```

### How Previous Versions Are Selected

1. **Choose clusters** — `functions_per_prompt` clusters are selected
   randomly (uniform, or Pareto-weighted if `pareto_aware=True`).
2. **Sample one program per cluster** — softmax over scores with the
   current temperature.
3. **Sort by score** (ascending, so the best program is last).
4. **Rename** — each program is renamed to `equation_v{i}` and internal
   recursive calls are updated via `code_manipulation.rename_function_calls()`.
5. **Add docstrings** — versions 1+ get `"Improved version of equation_v{i-1}."`.
6. **Build the "next version" header** — an empty `equation_v{N}` with
   the appropriate docstring is appended as the completion target.

The key insight is that presenting programs worst-to-best creates an
implicit gradient for the LLM: it sees a progression of improving
solutions and is asked to continue the trend.

---

## 8. Key Design Patterns

- **Frozen dataclasses for configuration** — all config classes except
  `RunConfig` are `@dataclass(frozen=True)`, preventing accidental
  mutation after construction. `RunConfig` is mutable because the CLI
  needs to set derived fields (`log_path`, `save_ckpt_dir`) and override
  `resume_from_ckpt` after loading.

- **ABC + factory for LLM providers** — `LLMProvider` is an abstract
  base class. Concrete implementations (`OpenAIProvider`,
  `QwenProvider`, `GeminiProvider`) are registered in the `_PROVIDERS`
  dict and instantiated by `_make_provider(name)`.

- **Lazy process pool** — `Sandbox._ensure_pool()` creates the subprocess
  pool on first use. On timeout or crash,
  `Sandbox._kill_and_recreate()` terminates and nullifies it so the
  next call creates a fresh pool.

- **Pickle-safe TensorBoard** — `TensorBoardWriter.__getstate__` drops
  the `SummaryWriter` (which holds open file handles), and
  `__setstate__` recreates it. This allows the entire
  `ProgramsDatabase` (which owns a `Profiler`) to be pickled for
  checkpointing.

- **Backpressure via shared counters** — two `multiprocessing.Value("i")`
  counters (`prompt_pending_count`, `sample_pending_count`) throttle
  producers when downstream consumers are saturated (see
  [Section 5.4](#54-backpressure-mechanism)).

- **Database absorbs controller** — lifecycle management (checkpoint
  timing, termination detection, `restore_or_create()`) lives directly
  in `ProgramsDatabase` rather than in a separate controller class.
  This eliminates a layer of indirection and reduces inter-module
  coupling.

---

## 9. Configuration Reference

All configuration is loaded from a single YAML file via `RunConfig.from_yaml()`.

> **Note:** `RunConfig` is the only mutable config dataclass. The CLI
> sets `log_path` and `save_ckpt_dir` after construction, and may
> override `resume_from_ckpt` from command-line arguments.

<details>
<summary><strong>RunConfig</strong> — top-level pipeline configuration</summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `problem_dir` | `str \| None` | `None` | Path to the problem specification directory |
| `data_folder` | `str \| None` | `None` | Path to the training data directory |
| `log_folder` | `str \| None` | `None` | Base name for the log directory |
| `log_path` | `str \| None` | `None` | Full path to the log directory (derived from `log_folder`) |
| `problem_name` | `str` | `"oscillator1"` | Human-readable problem identifier |
| `max_samples` | `int` | `3600` | Stop after this many total evaluations |
| `distributed` | `bool` | `True` | Whether to use multiprocessing |
| `num_samplers` | `int` | `8` | Number of sampler worker processes |
| `num_evaluators` | `int` | `8` | Number of evaluator worker processes |
| `save_ckpt_dir` | `str \| None` | `None` | Directory for checkpoint files (derived from `log_folder`) |
| `save_ckpt_interval` | `int` | `300` | Checkpoint interval in seconds |
| `resume_from_ckpt` | `str \| None` | `None` | Path to a checkpoint directory to resume from |
| `sampler` | `SamplerConfig` | (defaults) | Nested sampler configuration |
| `database` | `ProgramsDatabaseConfig` | (defaults) | Nested database configuration |
| `evaluator` | `EvaluatorConfig` | (defaults) | Nested evaluator configuration |
| `profiler` | `ProfilerConfig` | (defaults) | Nested profiler configuration |
| `worker` | `WorkerConfig` | (defaults) | Nested worker configuration |

</details>

<details>
<summary><strong>ProgramsDatabaseConfig</strong> — evolutionary algorithm parameters</summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `functions_per_prompt` | `int` | `4` | Number of previous programs to include in each prompt |
| `num_islands` | `int` | `10` | Number of islands for diversity |
| `reset_period` | `int` | `700` | Reset the weakest islands every N samples |
| `cluster_sampling_temperature_init` | `float` | `0.005` | Initial softmax temperature for cluster sampling |
| `cluster_sampling_temperature_period` | `int` | `200` | Period of linear temperature decay |
| `complexity_bin_size` | `int` | `10` | Width of each complexity bin |
| `cluster_max_size` | `int` | `100` | Maximum programs per cluster before pruning |
| `pareto_aware` | `bool` | `False` | Weight cluster selection by Pareto improvement potential |

</details>

<details>
<summary><strong>SamplerConfig</strong> — LLM request parameters</summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | `"qwen"` | LLM provider name (`"qwen"`, `"openai"`, or `"gemini"`) |
| `model_name` | `str \| None` | `None` | Model identifier; `None` uses the provider default |
| `temperature` | `float` | `1.0` | Sampling temperature for the LLM |
| `max_retries` | `int` | `5` | Maximum retry attempts per request |
| `retry_delay_seconds` | `float` | `5.0` | Delay between retries |
| `request_timeout_seconds` | `int` | `180` | Timeout for a single LLM request |
| `samples_per_prompt` | `int` | `1` | Number of completions to draw per prompt |
| `cost_per_ktoken` | `list[float]` | `[0.006, 0.024]` | Cost per 1K tokens `[input, output]` for tracking |

</details>

<details>
<summary><strong>EvaluatorConfig</strong> — sandbox execution parameters</summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout_seconds` | `int` | `400` | Maximum time (seconds) for a single sandbox evaluation |

</details>

<details>
<summary><strong>ProfilerConfig</strong> — logging and metrics parameters</summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `log_frequency` | `int` | `100` | Write detailed TensorBoard logs every N samples |
| `complexity_group_size` | `int` | `5` | Width of complexity groups for TensorBoard scalars |

</details>

<details>
<summary><strong>WorkerConfig</strong> — distributed worker timing</summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `perf_report_interval_seconds` | `int` | `150` | How often workers report performance stats |
| `monitor_interval_seconds` | `int` | `300` | How often the monitor prints a summary report |

</details>

---

## 10. Problem Specification Format

A problem is defined by a directory containing three files, plus a
separate data directory with training CSV(s).

### Directory Structure

```
specs/<problem_name>/
├── prompt.txt      # Task description for the LLM (free-form text)
├── equation.py     # Seed equation function (the starting point)
└── evaluate.py     # Evaluation harness (scoring + optimization)

data/<problem_name>/
└── train.csv       # Training data (columns become dict keys)
```

### `prompt.txt`

Free-form text describing the problem, background, inputs, constraints,
and philosophy. This becomes the first section of every LLM prompt. It
should be self-contained — the LLM receives no other context about the
problem domain.

### `equation.py` — Seed Equation

Defines exactly **one** top-level function named `equation`. The
signature must accept the data columns as positional arguments plus a
`params` array:

```python
def equation(rho: jnp.ndarray, s: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Computes exchange-correlation energy density (e_xc)."""
    # LDA exchange energy density: proportional to rho^(4/3)
    e_x_lda = params[0] * rho**(4/3)
    # ... enhancement factor, correlation, etc.
    return e_x, e_c
```

Rules:
- Exactly one `def` at the top level
- The function body is the seed that gets evolved
- Parameters are indexed from `params[0]`, `params[1]`, etc.
- Must be parseable by `code_manipulation.text_to_function()`

### `evaluate.py` — Evaluation Harness

Must define an `evaluate(data: dict)` function. This function:
- Receives `data_dict` (columns from `train.csv` as numpy arrays)
- Calls `equation(...)` with the data and a parameter vector
- Runs numerical optimization (e.g., BFGS + CMA-ES) to find optimal `params`
- Returns `(score, optimized_params)` on success, or `None` on failure

The `equation` function referenced inside `evaluate.py` is **replaced at
runtime** — the evaluator splices the evolved function body into the
template before execution. The harness code (imports, data preprocessing,
loss functions, optimizers) remains unchanged.

> **Important markers.** The `evaluate.py` file must import the same
> libraries used by `equation.py` (e.g., `jax.numpy as jnp`). The
> evaluator concatenates `evaluate.py` + the evolved `equation` function
> into a single script that is passed to `exec()`.

---

## 11. Common Extension Points

### 11.1 Adding an LLM Provider

1. Create a new class inheriting from `LLMProvider` in `sampler.py`:

   ```python
   class MyProvider(LLMProvider):
       def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
           # Call your API, return LLMResponse(response_text, input_tokens, output_tokens)
           ...
   ```

2. Register it in the `_PROVIDERS` dict:

   ```python
   _PROVIDERS["myprovider"] = MyProvider
   ```

3. Set `provider: "myprovider"` in the YAML config.

### 11.2 Adding a Problem

1. Create a directory `specs/<problem_name>/` with:
   - `prompt.txt` — problem description
   - `equation.py` — seed function with the signature the evaluator expects
   - `evaluate.py` — evaluation harness returning `(score, optimized_params)` or `None`

2. Create a data directory with `train.csv`.

3. Update the YAML config:

   ```yaml
   problem_dir: "specs/<problem_name>"
   data_folder: "data/<problem_name>"
   ```

### 11.3 Checkpoint and Resume

**Saving** is automatic when `save_ckpt_dir` is set (derived from
`log_folder`). The database checks `maybe_checkpoint()` after every
evaluation; if `save_ckpt_interval` seconds have elapsed, it pickles
itself to `checkpoint_<N>.pkl`. A final `checkpoint_final.pkl` is saved
on shutdown.

**Resuming** from a checkpoint:

```bash
alpha-evolve-sr --config config.yaml --resume_from_ckpt ./log/<run>/checkpoints/
```

The system loads `checkpoint_final.pkl` for the database state and
`run_config.yaml` for the configuration. The CLI re-applies any
command-line overrides after loading.

---

## 12. Appendix: Import Dependency Matrix

<details>
<summary>Click to expand the full dependency matrix</summary>

Each `▶` indicates a direct import. `TYPE` indicates a `TYPE_CHECKING`-only import.

| Module ↓ imports → | config | code_manip | database | sampler | evaluator | complexity | profiler | checkpoint | logging | exceptions | messages |
|--------------------|--------|-----------|----------|---------|-----------|-----------|----------|-----------|---------|-----------|----------|
| **cli.py** | ▶ | ▶ | ▶ | ▶ | ▶ | | | ▶ | ▶ | | |
| **workers.py** | ▶ | ▶ | ▶ | ▶ | ▶ | | | | ▶ | | ▶ |
| **database.py** | ▶ | ▶ | | | | | ▶ | ▶ | ▶ | | ▶ |
| **evaluator.py** | ▶ | ▶ | | | | ▶ | | | ▶ | | ▶ |
| **sampler.py** | ▶ | | | | | | | | ▶ | ▶ | |
| **profiler.py** | ▶ | ▶ | | | | | | | ▶ | | |
| **complexity.py** | | | | | | | | | ▶ | | |
| **checkpoint.py** | ▶ | | TYPE | | | | | | ▶ | ▶ | |
| **messages.py** | | ▶ | | | | | | | | | |
| **logging_config.py** | | | | | | | | | | | |
| **exceptions.py** | | | | | | | | | | | |

</details>
