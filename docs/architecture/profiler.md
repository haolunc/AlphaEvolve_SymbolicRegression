# Profiler -- TensorBoard Metrics

```{contents}
:local:
:depth: 2
```

---

## Overview

The profiler provides real-time visibility into the evolutionary search via TensorBoard. It logs scalar metrics (scores, costs, timing, throughput), per-island breakdowns, and Pareto front visualizations -- all without requiring PyTorch.

---

## Architecture

```{mermaid}
flowchart LR
    DB["ProgramsDatabase"] -->|"ProfileMetrics"| TBW["TensorBoardWriter"]
    TBW --> SW["SummaryWriter"]
    SW --> EFW["EventFileWriter"]
    EFW --> Files["TensorBoard<br/>event files"]
```

| Component | Role |
|-----------|------|
| `TensorBoardWriter` | Public API -- receives `ProfileMetrics`, converts to scalars + images |
| `SummaryWriter` | Minimal writer using raw `tensorboard` protobuf APIs (no PyTorch) |
| `EventFileWriter` | Low-level writer from the `tensorboard` package -- writes binary event files |

`SummaryWriter` provides two methods:
- **`add_scalar_batch(tag_value_pairs, global_step)`** -- packs multiple scalar values into a single `Summary`/`Event` protobuf, minimizing I/O.
- **`add_figure(tag, figure, global_step)`** -- renders a matplotlib figure to PNG and writes it as a TensorBoard image summary.

---

## ProfileMetrics Dataclass

`ProfileMetrics` is a frozen dataclass that snapshots all metrics for a single TensorBoard write:

```python
@dataclasses.dataclass(frozen=True)
class ProfileMetrics:
    # Core counters (checkpointed)
    num_samples: int
    best_score: float
    tot_token_cost: float
    success_count: int
    failed_count: int
    tot_sample_time: float
    tot_evaluate_time: float

    # Optional breakdowns
    pareto_front: list | None = None
    best_score_per_island: list[float] | None = None
    island_sizes: list[int] | None = None

    # Transient pipeline stats (not checkpointed)
    pending_evals: int | None = None
    pending_samplers: int | None = None
    wall_time_seconds: float | None = None
```

The top seven fields come from checkpointed counters in `ProgramsDatabase`. The optional fields are computed on-the-fly. Transient fields (`pending_evals`, `pending_samplers`, `wall_time_seconds`) are set via `update_pipeline_stats()` and are never persisted.

---

## Logged Metrics

All scalar metrics are packed into a single event per write. Per-island metrics expand to one tag per island.

| Tag | Source | Description |
|-----|--------|-------------|
| `Best Score of Function` | `max(best_score_per_island)` | Global best fitness score |
| `Total Token Cost` | cumulative | Running cost of LLM API calls |
| `Num/legal function num` | `success_count` | Programs that executed successfully |
| `Num/Illegal function num` | `failed_count` | Programs that failed (parse/exec/timeout) |
| `Time/Sample time` | cumulative | Total wall-clock LLM sampling time |
| `Time/Evaluate time` | cumulative | Total wall-clock sandbox eval time |
| `Pipeline/Throughput (samples per sec)` | `num_samples / wall_time` | End-to-end throughput (conditional on `wall_time_seconds > 0`) |
| `Pipeline/Pending Evals` | transient | Current eval queue depth (conditional) |
| `Pipeline/Pending Samplers` | transient | Current sampler queue depth (conditional) |
| `Best Score / Island/island_{i}` | per island | Best score on each island |
| `Island Size/island_{i}` | per island | Number of programs on each island |
| `Pareto_Front` | image | Scatter + trend line of the Pareto front |

Conditional metrics are only written when their source value is available (non-`None`, non-zero).

---

## Pareto Front Visualization

When the Pareto front has $\ge 2$ entries, `TensorBoardWriter.write()` generates a matplotlib figure:

- **X-axis**: complexity bin (`cbin`)
- **Y-axis**: score
- **Scatter points** for each Pareto entry
- **Dashed trend line** connecting points

The figure is rendered to PNG in-memory and written as a TensorBoard image via `add_figure()`. The matplotlib figure is closed immediately after writing to avoid memory leaks.

---

## When Metrics Are Written

`TensorBoardWriter.write()` is called from `ProgramsDatabase._register_and_persist()` every `log_frequency` samples (default 25, configurable via `ProgramsDatabaseConfig.log_frequency`).

```python
# Inside _register_and_persist():
if self._global_sample_nums % self._config.log_frequency == 0:
    self._tb_writer.write(self._build_profile_metrics())
```

The `global_step` for all TensorBoard events is `num_samples` (the global sample counter), so the x-axis in TensorBoard represents samples processed.

---

## How to View

Start TensorBoard pointing at the run's log directory:

```bash
tensorboard --logdir <log_dir> --port 6006
```

Then open `http://localhost:6006` in a browser. Metrics are organized into tag groups:

- **Scalars**: `Best Score of Function`, `Total Token Cost`, `Num/*`, `Time/*`, `Pipeline/*`
- **Scalars (per-island)**: `Best Score / Island/*`, `Island Size/*`
- **Images**: `Pareto_Front`
