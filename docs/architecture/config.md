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
run_config.validate()
```

The YAML file has top-level scalar fields plus nested sections (`sampler`, `database`, `evaluator`) that map to frozen dataclasses. Unknown keys in nested sections are warned and ignored; unknown top-level keys raise `ValueError`.

```{note}
`RunConfig` is the only **mutable** config dataclass (`@dataclass` without `frozen=True`).
Its `__post_init__` normalizes `resume_from_ckpt` (directory to `checkpoint.db`) and defaults `save_ckpt_dir` from `log_dir`.
```

---

## Full Example

```{literalinclude} ../../examples/my_configs/my_config_0224_5.yaml
:language: yaml
```

---

## Dataclass Reference

### RunConfig

Top-level pipeline configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `problem_dir` | `str \| None` | `None` | Path to the problem directory (`prompt.txt`, `equation.py`, `evaluate.py`) |
| `data_folder` | `str \| None` | `None` | Path to the training data directory (`train.csv`) |
| `log_dir` | `str \| None` | `None` | Path to the log / TensorBoard directory |
| `max_samples` | `int` | `3600` | Stop after this many total evaluations |
| `num_samplers` | `int` | `8` | Number of concurrent LLM sampler threads |
| `num_evaluators` | `int` | `8` | Number of threads, each with a thread-local `Evaluator` + `Sandbox(mp.Pool(1))` |
| `save_ckpt_dir` | `str \| None` | `None` | Directory for checkpoints (defaults to `log_dir` if omitted) |
| `resume_from_ckpt` | `str \| None` | `None` | Path to a checkpoint file or directory to resume from |

Nested configs (each described below):

| Field | Type |
|-------|------|
| `sampler` | `SamplerConfig` |
| `database` | `ProgramsDatabaseConfig` |
| `evaluator` | `EvaluatorConfig` |

**Methods:**

| Method | Description |
|--------|-------------|
| `from_yaml(path)` | Load config from a YAML file |
| `from_yaml_text(yaml_text)` | Parse config from a YAML string |
| `to_yaml(path)` | Serialize config to a YAML file |
| `validate()` | Raise `ValueError` if required paths are missing or invalid |

---

### ProgramsDatabaseConfig

Evolutionary algorithm parameters. Frozen dataclass.

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
| `checkpoint_interval` | `int` | `10` | Save derived checkpoint tables every N registered programs |
| `log_frequency` | `int` | `25` | Write detailed TensorBoard logs every N samples |

---

### SamplerConfig

LLM request parameters. Frozen dataclass.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | `"qwen"` | LLM provider name (`"qwen"`, `"openai"`, or `"gemini"`) |
| `model_name` | `str \| None` | `None` | Model identifier; `None` uses the provider default |
| `temperature` | `float` | `1.0` | Sampling temperature for the LLM |
| `max_retries` | `int` | `5` | Maximum retry attempts per request |
| `retry_delay_seconds` | `float` | `5.0` | Delay between retries |
| `request_timeout_seconds` | `int` | `180` | Timeout for a single LLM request |
| `samples_per_prompt` | `int` | `1` | Number of separate LLM futures the pipeline submits per prompt (each is an independent request) |
| `cost_per_ktoken` | `list[float]` | `[0.006, 0.024]` | Cost per 1K tokens `[input, output]` for tracking |

---

### EvaluatorConfig

Sandbox execution parameters. Frozen dataclass.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout_seconds` | `int` | `400` | Maximum time (seconds) for a single sandbox evaluation |
