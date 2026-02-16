# Copyright 2023 DeepMind Technologies Limited
# Copyright 2026 Haolun Cai
# This file has been modified by Haolun Cai for AlphaEvolve_SymbolicRegression on Jan 21, 2026.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Configuration dataclasses for alpha_evolve_sr."""

from __future__ import annotations

import dataclasses
from dataclasses import field

import yaml


@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
    """Configuration of a ProgramsDatabase.

    Attributes:
      functions_per_prompt: Number of previous programs to include in prompts.
      num_islands: Number of islands to maintain as a diversity mechanism.
      reset_period: How often (in samples) the weakest islands should be reset.
      cluster_sampling_temperature_init: Initial temperature for softmax sampling
          of clusters within an island.
      cluster_sampling_temperature_period: Period of linear decay of the cluster
          sampling temperature.
      complexity_bin_size: Size of complexity bins for clustering programs.
      cluster_max_size: Maximum number of programs kept per cluster. When
          exceeded, the lowest-scoring programs are pruned.
    """

    functions_per_prompt: int = 4
    num_islands: int = 10
    reset_period: int = 700
    cluster_sampling_temperature_init: float = 0.005
    cluster_sampling_temperature_period: int = 200
    complexity_bin_size: int = 10
    cluster_max_size: int = 100
    pareto_aware: bool = False


@dataclasses.dataclass(frozen=True)
class SamplerConfig:
    """Configuration for LLM sampling.

    Attributes:
      provider: LLM provider name ("qwen", "openai", or "gemini").
      model_name: Model identifier. None uses the provider default.
      temperature: Sampling temperature for the LLM.
      max_retries: Maximum number of retry attempts per request.
      retry_delay_seconds: Delay between retries.
      request_timeout_seconds: Timeout for a single LLM request.
      samples_per_prompt: Number of samples to draw per prompt.
      cost_per_ktoken: Cost per 1K tokens (input, output) for tracking.
    """

    provider: str = "qwen"
    model_name: str | None = None
    temperature: float = 1.0
    max_retries: int = 5
    retry_delay_seconds: float = 5.0
    request_timeout_seconds: int = 180
    samples_per_prompt: int = 1
    cost_per_ktoken: tuple[float, float] = (0.006, 0.024)


@dataclasses.dataclass(frozen=True)
class EvaluatorConfig:
    """Configuration for the program evaluator.

    Attributes:
      timeout_seconds: Maximum time allowed for sandbox execution.
    """

    timeout_seconds: int = 400


@dataclasses.dataclass(frozen=True)
class ProfilerConfig:
    """Configuration for experiment profiling.

    Attributes:
      log_frequency: How often (in samples) to write detailed TensorBoard logs.
      complexity_group_size: Size of complexity groups for TensorBoard scalars.
    """

    log_frequency: int = 100
    complexity_group_size: int = 5


@dataclasses.dataclass(frozen=True)
class WorkerConfig:
    """Configuration for distributed worker processes.

    Attributes:
      perf_report_interval_seconds: How often workers report performance stats.
      monitor_interval_seconds: How often the monitor prints a report.
    """

    perf_report_interval_seconds: int = 150
    monitor_interval_seconds: int = 300


@dataclasses.dataclass
class RunConfig:
    """Top-level configuration for a pipeline run.

    Loaded from a YAML config file via :meth:`from_yaml`.  Nested
    ``sampler`` and ``database`` sections map to :class:`SamplerConfig`
    and :class:`ProgramsDatabaseConfig` respectively.
    """

    # Paths
    spec_path: str | None = None
    data_folder: str | None = None
    log_folder: str | None = None
    log_path: str | None = None
    problem_name: str = "oscillator1"

    # Pipeline
    max_samples: int = 3600
    distributed: bool = True
    num_samplers: int = 8
    num_evaluators: int = 8

    # Checkpointing
    save_ckpt_dir: str | None = None
    save_ckpt_interval: int = 300
    resume_from_ckpt: str | None = None

    # Nested configs
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    database: ProgramsDatabaseConfig = field(default_factory=ProgramsDatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> RunConfig:
        """Load a RunConfig from a YAML file."""
        import sys

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        sampler_data = data.pop("sampler", {})
        # Migrate samples_per_prompt from top-level into sampler section
        if "samples_per_prompt" in data and "samples_per_prompt" not in sampler_data:
            sampler_data["samples_per_prompt"] = data.pop("samples_per_prompt")
        database_data = data.pop("database", {})
        evaluator_data = data.pop("evaluator", {})
        profiler_data = data.pop("profiler", {})
        worker_data = data.pop("worker", {})

        # Convert tuple fields that YAML deserialises as lists
        if "cost_per_ktoken" in sampler_data:
            sampler_data["cost_per_ktoken"] = tuple(sampler_data["cost_per_ktoken"])

        # Reject unknown top-level keys (catches typos like "smpler")
        known_fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data.keys()) - known_fields
        if unknown:
            sys.exit(
                f"Error: unknown config keys: {sorted(unknown)}. "
                f"Valid top-level keys: {sorted(known_fields)}"
            )

        return cls(
            sampler=SamplerConfig(**sampler_data),
            database=ProgramsDatabaseConfig(**database_data),
            evaluator=EvaluatorConfig(**evaluator_data),
            profiler=ProfilerConfig(**profiler_data),
            worker=WorkerConfig(**worker_data),
            **data,
        )

    def to_yaml(self, path: str) -> None:
        """Serialise this config to a YAML file."""
        import os

        def _tuples_to_lists(obj):
            if isinstance(obj, dict):
                return {k: _tuples_to_lists(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_tuples_to_lists(v) for v in obj]
            return obj

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = _tuples_to_lists(dataclasses.asdict(self))
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> None:
        """Raise :class:`SystemExit` on invalid configuration values."""
        import os
        import sys

        if not self.spec_path:
            sys.exit("Error: spec_path is required")
        if not self.data_folder:
            sys.exit("Error: data_folder is required")
        if not self.log_folder:
            sys.exit("Error: log_folder is required")
        if not os.path.isfile(self.spec_path):
            sys.exit(f"Error: spec_path does not exist: {self.spec_path}")
        if not os.path.isdir(self.data_folder):
            sys.exit(f"Error: data_folder does not exist: {self.data_folder}")
        if not isinstance(self.max_samples, int) or self.max_samples < 0:
            sys.exit(f"Error: max_samples must be a positive integer, got {self.max_samples}")
        if not isinstance(self.num_samplers, int) or self.num_samplers < 0:
            sys.exit(f"Error: num_samplers must be a positive integer, got {self.num_samplers}")
        if not isinstance(self.num_evaluators, int) or self.num_evaluators < 0:
            sys.exit(f"Error: num_evaluators must be a positive integer, got {self.num_evaluators}")
