#!/usr/bin/env python
"""Minimal example that runs the AlphaEvolve SR pipeline on the DFT data.

Usage::

    pip install -e ".[examples]"
    python examples/run_dft_example.py

Set the ``QWEN_API_KEY`` and ``QWEN_BASE_URL`` environment variables (or
``OPENAI_API_KEY`` for the OpenAI provider) before running.
"""

from __future__ import annotations

import os
import sys

# Ensure the package is importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alpha_evolve_sr.config import RunConfig  # noqa: E402

if __name__ == "__main__":
    # Check for at least one API key
    has_key = any(os.getenv(k) for k in ("QWEN_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"))
    if not has_key:
        print(
            "No LLM API key found.\n"
            "Set one of QWEN_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY in your "
            "environment or in a .env file at the project root.",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = os.path.join(os.path.dirname(__file__), "example_config.yaml")
    run_config = RunConfig.from_yaml(config_path)

    # Override to non-distributed for the example
    run_config.distributed = False

    # Derive log paths
    if run_config.log_path is None and run_config.log_folder:
        run_config.log_path = "./log/" + run_config.log_folder
    if run_config.save_ckpt_dir is None and run_config.log_folder:
        run_config.save_ckpt_dir = "./log/" + run_config.log_folder + "/checkpoints"

    from alpha_evolve_sr.cli import load_problem, main_single  # noqa: E402
    from alpha_evolve_sr.logging_config import configure_logging  # noqa: E402

    configure_logging()
    prompt_text, evaluate_code, seed_function, data_dict = load_problem(
        run_config.problem_dir, run_config.data_folder,
    )
    main_single(
        run_config, prompt_text, evaluate_code, seed_function, data_dict,
    )
