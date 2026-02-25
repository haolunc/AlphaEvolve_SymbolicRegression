# Pipeline Architecture

```{contents}
:local:
:depth: 2
```

---

## Architecture

The pipeline uses a **thread + subprocess** hybrid with a single orchestrating main thread:

```{mermaid}
flowchart TB
    subgraph main ["Main Thread (orchestrator)"]
        direction TB
        DB["ProgramsDatabase\n(register, get_prompt)"]
        CX["_attach_complexity()"]
        TB_["TensorBoardWriter"]
        LOOP["Event Loop\n(50 ms polling)"]
    end

    subgraph spool ["ThreadPoolExecutor (num_samplers)"]
        S1["Thread 1\n_sampler_task()"]
        S2["Thread 2\n_sampler_task()"]
        SN["Thread N\n_sampler_task()"]
    end

    subgraph epool ["ThreadPoolExecutor (num_evaluators)"]
        E1["Thread 1\n_eval_thread_analyse()"]
        E2["Thread 2\n_eval_thread_analyse()"]
        EM["Thread M\n_eval_thread_analyse()"]

        E1 --- SB1["Sandbox\n(mp.Pool 1)"]
        E2 --- SB2["Sandbox\n(mp.Pool 1)"]
        EM --- SBM["Sandbox\n(mp.Pool 1)"]
    end

    LOOP -->|"1. get_prompt()"| DB
    DB -->|"Prompt"| LOOP
    LOOP -->|"2. submit(_sampler_task)"| S1
    LOOP -->|"2. submit(_sampler_task)"| S2
    LOOP -->|"2. submit(_sampler_task)"| SN

    S1 -->|"Future[SampleMessage]"| LOOP
    S2 -->|"Future[SampleMessage]"| LOOP
    SN -->|"Future[SampleMessage]"| LOOP

    LOOP -->|"3. submit(_eval_thread_analyse)"| E1
    LOOP -->|"3. submit(_eval_thread_analyse)"| E2
    LOOP -->|"3. submit(_eval_thread_analyse)"| EM

    E1 -->|"Future[(EvalResult,\nSampleMessage)]"| LOOP
    E2 -->|"Future[(EvalResult,\nSampleMessage)]"| LOOP
    EM -->|"Future[(EvalResult,\nSampleMessage)]"| LOOP

    LOOP -->|"4. _attach_complexity()"| CX
    LOOP -->|"5. register_program()"| DB
    DB -->|"ProfileMetrics"| TB_
```

**Three layers:**

| Layer | Implementation | Why |
|-------|---------------|-----|
| **Sampler pool** | `ThreadPoolExecutor(num_samplers)` | LLM API calls are pure I/O — threads release the GIL during network waits |
| **Evaluator pool** | `ThreadPoolExecutor(num_evaluators)`, each thread owns a thread-local `Evaluator` | Evaluation logic itself is I/O-bound (waiting on subprocess) |
| **Sandbox** | `mp.Pool(1)` per evaluator thread | `exec()` of untrusted code needs process isolation for timeout/crash safety |

The main thread owns the `ProgramsDatabase`, computes complexity scores, and writes TensorBoard metrics — no serialization overhead for these shared-state operations.

---

## Walk Through

### Main Thread

The main thread is the sole writer to three stateful components:

- **ProgramsDatabase** — stores evaluated programs, builds prompts, tracks Pareto front
- **`_attach_complexity()`** — computes AST complexity (weighted node count) and attaches it to `EvalResult` before registration
- **TensorBoardWriter** — logs metrics every `log_frequency` samples

### Sampler Pool

`N` threads (`num_samplers`) making LLM API calls via `_sampler_task()`:

```python
def _sampler_task(llm, prompt_code, island_id) -> SampleMessage | None:
    t0 = time.time()
    response = llm.query(prompt_code)      # single LLM call with retries
    sample_time = time.time() - t0
    return SampleMessage(response, island_id, sample_time)
```

Each thread holds no state — the `LLM` instance is shared (thread-safe HTTP client). Threads release the GIL during network I/O, so `N` concurrent API calls proceed in parallel.

### Evaluator Pool

`M` threads (`num_evaluators`), each with a **thread-local** `Evaluator` instance created by `_init_eval_thread()`:

```python
_eval_tls = threading.local()

def _init_eval_thread(evaluate_code, seed_function, data_dict, config):
    _eval_tls.evaluator = Evaluator(evaluate_code, seed_function, data_dict, config)
```

Each `Evaluator` owns a `Sandbox(mp.Pool(1))` — the only subprocess in the system. The sandbox provides crash and timeout isolation for `exec()` of untrusted LLM-generated code. The evaluator thread itself just waits on the subprocess result (I/O-bound), so threads work well here too.

---

## Initialization

Startup sequence in `run_pipeline()`:

1. **Create evaluator pool** — `ThreadPoolExecutor(num_evaluators, initializer=_init_eval_thread)` creates thread-local `Evaluator` + `Sandbox` per thread
2. **Seed evaluation** — `eval_pool.submit(_eval_thread_initialize).result()` (blocking) evaluates the seed function to establish a baseline score
3. **Restore or create database** — `ProgramsDatabase.restore_or_create()` either resumes from checkpoint or creates a fresh database, registering the seed result
4. **Create LLM and sampler pool** — `LLM(sampler_config)` + `ThreadPoolExecutor(num_samplers)`

The `mp.set_start_method("spawn")` call in `main()` ensures sandbox subprocesses start cleanly (required on macOS, recommended on Linux).

---

## Event Loop

The main thread runs a ~20 Hz polling loop:

```
while not database.should_stop and not graceful_shutdown.is_set():

    ① Collect completed eval futures
       → _attach_complexity(eval_result)
       → database.register_program(eval_result, sample_message)

    ② Collect completed sample futures
       → submit _eval_thread_analyse(sample_msg) to eval pool

    ③ Submit new prompts (if under backpressure limit)
       → prompt = database.get_prompt()
       → for _ in range(samples_per_prompt):
             submit _sampler_task(llm, prompt.code, prompt.island_id)

    ④ database.update_pipeline_stats(pending_evals, pending_samplers, wall_time)

    sleep(0.05)
```

The 50 ms polling interval is negligible compared to LLM latency (seconds) and evaluation time (seconds to minutes).

**Step details:**

1. **Collect evals** — iterate over completed eval futures. For each `(EvalResult, SampleMessage)`, compute complexity via `_attach_complexity()` (AST node counting), then register in the database. Registration triggers island updates, Pareto front maintenance, periodic checkpointing, and TensorBoard writes.

2. **Collect samples** — iterate over completed sampler futures. Each yields a `SampleMessage` (or `None` on LLM failure). Non-`None` results are submitted to the evaluator pool.

3. **Submit new prompts** — only if backpressure conditions are met (see below). Calls `database.get_prompt()` to build a prompt from sampled historical programs, then submits `samples_per_prompt` separate `_sampler_task` futures.

4. **Update stats** — feeds transient pipeline metrics (pending queue depths, wall time) to the database for TensorBoard logging.

---

## Backpressure

Two conditions must be met before new prompts are submitted:

$$\text{pending\_evals} < \text{num\_evaluators} \times 2$$

$$\text{pending\_samplers} < \text{num\_samplers}$$

The first condition prevents memory bloat from queued programs waiting for evaluation. The second ensures sampler threads aren't overloaded. When either limit is reached, the loop skips step 3 — sampling pauses until work drains.

---

## Shutdown

### Normal Exit

When `database.should_stop` becomes `True` (i.e., `max_samples` reached):

1. Stop submitting new prompts
2. Call `_drain_pipeline()` — wait for all in-flight samplers and evals to complete, register results
3. `database.finalize()` — final checkpoint + close DBs
4. `_cleanup_eval_threads()` — terminate sandbox subprocesses
5. Shutdown both thread pools

### Signal Handler (Ctrl+C)

| Trigger | Behavior |
|---------|----------|
| **1st** Ctrl+C | Set `graceful_shutdown` → exit main loop → `_drain_pipeline()` (wait up to 120 s per sampler, 60 s per eval) → finalize |
| **2nd** Ctrl+C | Set `force_shutdown` → abort drain immediately → finalize with whatever results were collected |

The `_drain_pipeline()` function processes in-flight work in two phases:

1. **Phase 1**: wait for pending sampler futures → submit results to eval pool
2. **Phase 2**: wait for pending eval futures → `_attach_complexity()` → `register_program()`

A second Ctrl+C during either phase sets a `force_event` that breaks out immediately.

---

## Component Glossary

| Component | Description | Reference |
|-----------|-------------|-----------|
| **Input Files** | 4 user-provided files: `train.csv`, `prompt.txt`, `equation.py`, `evaluate.py` | {doc}`input-files` |
| **Config** | Single YAML file parsed by `RunConfig.from_yaml()` | {doc}`config` |
| **Sampler** | `LLM.query()` with retry logic and cost tracking | {doc}`sampler` |
| **Evaluator** | Parse → splice → sandbox exec → result classification | {doc}`evaluator` |
| **Database** | Island-based evolution: stores programs, builds prompts, tracks Pareto front | {doc}`database` |
| **Messages** | Typed frozen dataclasses flowing between components | {doc}`messages` |
| **Profiler** | TensorBoard metrics: best score, cost, throughput, per-island stats, Pareto front plot | {doc}`profiler` |
| **Checkpoint** | Two SQLite databases (WAL mode): `CheckpointDB` for program state, `LogsDB` for debug logs | {doc}`checkpoint` |

---

## Config Knobs

| Field | Location | Effect |
|-------|----------|--------|
| `num_samplers` | `RunConfig` | Number of concurrent LLM threads |
| `num_evaluators` | `RunConfig` | Number of threads with thread-local Evaluator + Sandbox |
| `samples_per_prompt` | `SamplerConfig` | Separate `_sampler_task` futures submitted per prompt |
| `max_samples` | `RunConfig` | Total evaluations before stopping |
