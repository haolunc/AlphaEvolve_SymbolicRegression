# Configuration Reference

```{contents}
:local:
:depth: 2
```

---

## How Config is Loaded

All configuration lives in a single YAML file, parsed by `RunConfig.from_yaml()`:

```python
run_config = RunConfig.from_yaml("my_config.yaml")
```

The YAML file has top-level scalar fields plus nested sections (`sampler`, `database`, `evaluator`, `profiler`, `worker`) that map to frozen dataclasses. Unknown keys in nested sections are warned and ignored; unknown top-level keys raise `ValueError`.

> **Note:** `RunConfig` is the only mutable config dataclass. The CLI
> sets `log_path` and `save_ckpt_dir` after construction, and may
> override `resume_from_ckpt` from command-line arguments.

---

## Full Example

```{literalinclude} ../../examples/my_config.yaml
:language: yaml
```

---

## Dataclass Reference

### RunConfig

Top-level pipeline configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `problem_dir` | `str \| None` | `None` | Path to the problem directory (prompt.txt, equation.py, evaluate.py) |
| `data_folder` | `str \| None` | `None` | Path to the training data directory (train.csv) |
| `log_folder` | `str \| None` | `None` | Base name for the log directory |
| `log_path` | `str \| None` | `None` | Full path to log directory (derived from `log_folder` by CLI) |
| `problem_name` | `str` | `"oscillator1"` | Human-readable problem identifier |
| `max_samples` | `int` | `3600` | Stop after this many total evaluations |
| `distributed` | `bool` | `True` | Use multiprocessing (distributed mode) |
| `num_samplers` | `int` | `8` | Number of sampler worker processes |
| `num_evaluators` | `int` | `8` | Number of evaluator worker processes |
| `save_ckpt_dir` | `str \| None` | `None` | Directory for checkpoints (derived from `log_folder` by CLI) |
| `resume_from_ckpt` | `str \| None` | `None` | Path to a checkpoint directory to resume from |

Nested configs (each described below):

| Field | Type |
|-------|------|
| `sampler` | `SamplerConfig` |
| `database` | `ProgramsDatabaseConfig` |
| `evaluator` | `EvaluatorConfig` |
| `profiler` | `ProfilerConfig` |
| `worker` | `WorkerConfig` |

---

### ProgramsDatabaseConfig

Evolutionary algorithm parameters.

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
| `checkpoint_interval` | `int` | `10` | Save checkpoint every N registered programs |

---

### SamplerConfig

LLM request parameters.

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

---

### EvaluatorConfig

Sandbox execution parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout_seconds` | `int` | `400` | Maximum time (seconds) for a single sandbox evaluation |

---

### ProfilerConfig

Logging and metrics parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `log_frequency` | `int` | `100` | Write detailed TensorBoard logs every N samples |

---

### WorkerConfig

Distributed worker timing.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `perf_report_interval_seconds` | `int` | `150` | How often workers report performance stats |
| `monitor_interval_seconds` | `int` | `300` | How often the monitor prints a summary report |
