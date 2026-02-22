# Getting Started

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

## Quick Start

```bash
# Run with config file (see example_config.yaml for all options)
alpha-evolve-sr --config example_config.yaml

# Resume from checkpoint
alpha-evolve-sr --config example_config.yaml --resume_from_ckpt <log_path>/checkpoints

# Monitor with TensorBoard
tensorboard --logdir <log_path> --port 6006
```
