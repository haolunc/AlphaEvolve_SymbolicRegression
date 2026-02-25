# AlphaEvolve_SymbolicRegression

---

> Inspired by [FunSearch](https://www.nature.com/articles/s41586-023-06924-6). *Nature* (2023); [LLM-SR](https://openreview.net/forum?id=m2nmp8P5in). *ICLR Oral* (2024); [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/). (2025) - This project integrates the core insights and advancements from these three works.

![Overview](images/overview_en.png)

## Architecture Overview

The architecture consists of three main modules: the **Program Database**, the **LLM Sampler**, and the **Evaluator**. The workflow operates as an iterative loop:

1. **Prompting**: Construct a prompt by selecting historical programs from the database and combining them with specific task instructions. (`database.py`)

2. **Sampling**: Feed this prompt to the LLM Sampler, which generates new candidate equations. (`sampler.py`)

3. **Evaluation**: The Evaluator takes these candidates and uses the provided data to optimize their parameters, then calculates the accuracy of the fit. (`evaluator.py`)

4. **Update**: Valid equations are stored back into the Program Database. These serve as *in-context examples* for future prompts, continuously improving the model's performance in subsequent iterations. (`database.py`)

## Installation

```bash
# Clone the repository
git clone https://github.com/<user>/AlphaEvolve_SymbolicRegression.git
cd AlphaEvolve_SymbolicRegression

# Install in development mode (with test/lint tools)
pip install -e ".[dev]"

# Or install with Gemini support
pip install -e ".[gemini]"

# Or install with example dependencies (CMA-ES)
pip install -e ".[examples]"
```

## Environment Variables

Set the API key for your chosen LLM provider:

| Provider | Required Env Vars |
|----------|------------------|
| `qwen` (default) | `QWEN_API_KEY`, `QWEN_BASE_URL` |
| `openai` | `OPENAI_API_KEY` |
| `gemini` | `GOOGLE_API_KEY` |

You can also place them in a `.env` file in the project root.

## Quick Start

1. Create a config YAML (see `examples/my_configs/` for full examples):

```yaml
problem_dir: examples/dft_xc/
data_folder: examples/data/dft_ev_xc
log_dir: ./log/my_run/

max_samples: 50
num_samplers: 5
num_evaluators: 3

sampler:
  provider: qwen
  samples_per_prompt: 1

database:
  num_islands: 4
  functions_per_prompt: 4
```

2. Run:

```bash
alpha-evolve-sr --config my_config.yaml
```

3. Monitor with TensorBoard:

```bash
tensorboard --logdir ./log/my_run/ --port 6006
```

The pipeline logs best score, token cost, throughput, per-island stats, and a Pareto front plot.

## Resuming

Checkpoints are saved automatically to `{log_dir}/checkpoint.db`. To resume a previous run, set `resume_from_ckpt` in your YAML config:

```yaml
log_dir: ./log/my_experiment/
resume_from_ckpt: ./log/my_experiment/   # path to checkpoint.db or its parent dir
# save_ckpt_dir:                         # defaults to log_dir
```

`resume_from_ckpt` is a **YAML config field**, not a CLI flag. Structural parameters (`num_islands`, `complexity_bin_size`) must match the original run; other settings (e.g. `max_samples`, sampler config) can be changed freely.

## Graceful Shutdown

The pipeline supports a two-phase shutdown to avoid wasting in-flight LLM calls and evaluations:

| Trigger | Behavior |
|---------|----------|
| `max_samples` reached | Stop sending new prompts → drain all in-flight samplers & evals → register results → exit |
| **1st** Ctrl+C | Same as above — drain in-flight work before exiting |
| **2nd** Ctrl+C | Force immediate exit (skip drain) |

During the drain phase, the pipeline waits for each pending LLM response (up to 120 s) and each pending evaluation (up to 60 s). A second Ctrl+C at any point aborts the drain immediately.

## TensorBoard Metrics

| Metric | Description |
|--------|-------------|
| `Best Score of Function` | Global best fitness score |
| `Total Token Cost` | Running LLM API cost |
| `Num/legal function num` | Successfully evaluated programs |
| `Num/Illegal function num` | Failed programs (parse/exec/timeout) |
| `Pipeline/Throughput` | Samples per second |
| `Best Score / Island/*` | Per-island best scores |
| `Pareto_Front` | Scatter plot of Pareto-optimal (complexity, score) |

## File Structure

```
src/alpha_evolve_sr/
├── __init__.py          # Package exports
├── cli.py               # CLI entry point and unified pipeline
├── config.py            # Configuration dataclasses
├── database.py          # Program database with island-based evolutionary algorithm
├── sampler.py           # LLM provider interface and sampling
├── evaluator.py         # Thread-local sandbox evaluators (mp.Pool(1) per thread)
├── profiler.py          # TensorBoard logging for pipeline metrics
├── checkpoint.py        # SQLite checkpoint saving/loading (CheckpointDB + LogsDB)
├── code_manipulation.py # Python code parsing; core data classes for evolution
├── complexity.py        # Equation complexity calculation
├── logging_config.py    # Unified logging configuration
├── messages.py          # Pipeline message dataclasses
└── exceptions.py        # Custom exception hierarchy
```

## Documentation

Full architecture and reference documentation is built with Sphinx:

```bash
pip install -e ".[docs]"
cd docs && make html
open _build/html/index.html
```
