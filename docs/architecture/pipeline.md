# Pipeline Architecture

```{contents}
:local:
:depth: 2
```

---

## Overview

The pipeline uses a **thread + process** hybrid architecture:

- **Sampler threads** (`ThreadPoolExecutor`) — LLM API calls are pure I/O; threads avoid the memory overhead of separate processes
- **Evaluator processes** (`mp.Pool` with `spawn` context) — sandbox code execution needs process isolation for timeout/crash safety
- **Main thread** — orchestrates the pipeline, owns the database, and writes profiling metrics

```{mermaid}
flowchart TB
    subgraph main ["Main Thread (orchestrator + DB + profiler)"]
        DB["ProgramsDatabase"]
        LOOP["Event Loop\n(poll futures + async results)"]
    end

    subgraph threads ["ThreadPoolExecutor (num_samplers)"]
        T1["Thread 1: LLM.draw_samples()"]
        T2["Thread 2: LLM.draw_samples()"]
        TN["Thread N: LLM.draw_samples()"]
    end

    subgraph pool ["mp.Pool (num_evaluators, spawn)"]
        P1["Worker 1: eval_worker_analyse()"]
        P2["Worker 2: eval_worker_analyse()"]
        PM["Worker M: eval_worker_analyse()"]
    end

    DB -->|"get_prompt()"| LOOP
    LOOP -->|"submit()"| T1
    LOOP -->|"submit()"| T2
    LOOP -->|"submit()"| TN

    T1 -->|"Future[SampleMessage]"| LOOP
    T2 -->|"Future[SampleMessage]"| LOOP
    TN -->|"Future[SampleMessage]"| LOOP

    LOOP -->|"apply_async()"| P1
    LOOP -->|"apply_async()"| P2
    LOOP -->|"apply_async()"| PM

    P1 -->|"AsyncResult[(EvalResult, SampleMessage)]"| LOOP
    P2 -->|"AsyncResult[(EvalResult, SampleMessage)]"| LOOP
    PM -->|"AsyncResult[(EvalResult, SampleMessage)]"| LOOP

    LOOP -->|"register_program()"| DB
```

---

## Why Threads for Samplers, Processes for Evaluators

| Component | Bottleneck | Isolation needed? | Choice |
|-----------|-----------|-------------------|--------|
| Sampler | Network I/O (LLM API, seconds per call) | No — pure HTTP client | **Thread** |
| Evaluator | CPU (sandbox `exec()` + JAX JIT) | Yes — `exec()` of untrusted code, timeouts via `mp.TimeoutError` | **Process** |

Threads share the GIL but release it during I/O, making them ideal for concurrent API calls. Processes provide crash isolation — if a generated program segfaults or hangs, only that worker is affected.

---

## Event Loop

The main thread runs a polling loop (~20 Hz, `time.sleep(0.05)`):

```
while not database.should_stop:
    1. Collect completed evals → register in DB
    2. Collect completed samples → submit to eval pool
    3. Submit new prompts to sampler threads (if under backpressure limit)
    4. Update pipeline stats for profiler
    sleep(0.05)
```

The 50 ms polling interval is negligible compared to LLM latency (seconds) and evaluation time (seconds to minutes).

---

## Backpressure

The pipeline limits the number of pending evaluations to prevent memory bloat:

$$\text{max\_pending\_evals} = \text{num\_evaluators} \times 2$$

When the pending eval count reaches this limit, no new prompts are submitted to sampler threads. This creates natural backpressure — if evaluators are slow, sampling pauses until evaluators catch up.

---

## Config Knobs

| Config Field | Location | Effect |
|-------------|----------|--------|
| `num_samplers` | `RunConfig` | Number of concurrent LLM threads |
| `num_evaluators` | `RunConfig` | Number of persistent evaluator processes |
| `samples_per_prompt` | `SamplerConfig` | Sequential LLM calls per prompt (within one thread) |
| `max_samples` | `RunConfig` | Total evaluations before stopping |

---

## Shutdown

**Normal exit** (`database.should_stop` is `True`):
1. `sampler_pool.shutdown(wait=False, cancel_futures=True)`
2. `database.finalize()` — final checkpoint
3. `eval_pool.terminate(); eval_pool.join()`

**`KeyboardInterrupt`**:
1. Cancel all pending sampler futures
2. Drain remaining evals with a 30-second timeout (save partial work)
3. `database.finalize()`
4. `eval_pool.terminate(); eval_pool.join()`

---

## Comparison with Previous Architecture

| Aspect | Old (multiprocessing) | New (thread + process hybrid) |
|--------|----------------------|-------------------------------|
| Samplers | N separate processes | N threads in `ThreadPoolExecutor` |
| Evaluators | M separate processes | M workers in `mp.Pool` |
| Database | Separate process | Main thread (no serialization overhead) |
| Monitor | Separate process | Folded into TensorBoard profiler |
| Communication | 5 `mp.Queue`s + 2 shared counters + 1 `Event` | Futures + `AsyncResult`s (no queues) |
| Memory | High (N+M+2 full Python processes) | Lower (N threads + M processes) |
| Complexity | `workers.py` (373 lines) | `run_pipeline()` (~80 lines) |
