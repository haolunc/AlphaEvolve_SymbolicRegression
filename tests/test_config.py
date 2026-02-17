"""Tests for configuration dataclasses."""

import dataclasses
import json
import os

from alpha_evolve_sr.config import (
    EvaluatorConfig,
    ProfilerConfig,
    ProgramsDatabaseConfig,
    RunConfig,
    SamplerConfig,
    WorkerConfig,
)


class TestProgramsDatabaseConfig:
    """Tests for ProgramsDatabaseConfig."""

    def test_serialization_roundtrip(self):
        config = ProgramsDatabaseConfig(functions_per_prompt=6, num_islands=5)
        as_dict = dataclasses.asdict(config)
        json_str = json.dumps(as_dict)
        restored_dict = json.loads(json_str)
        restored = ProgramsDatabaseConfig(**restored_dict)
        assert restored == config


class TestSamplerConfig:
    """Tests for SamplerConfig."""

    def test_serialization_roundtrip(self):
        config = SamplerConfig(provider="openai", cost_per_ktoken=[0.01, 0.05])
        as_dict = dataclasses.asdict(config)
        json_str = json.dumps(as_dict)
        restored_dict = json.loads(json_str)
        restored = SamplerConfig(**restored_dict)
        assert restored == config


class TestRunConfig:
    """Tests for RunConfig."""

    def test_defaults(self):
        config = RunConfig()
        assert config.max_samples == 3600
        assert config.distributed is True
        assert config.num_samplers == 8
        assert isinstance(config.sampler, SamplerConfig)
        assert isinstance(config.database, ProgramsDatabaseConfig)
        assert isinstance(config.evaluator, EvaluatorConfig)
        assert isinstance(config.profiler, ProfilerConfig)
        assert isinstance(config.worker, WorkerConfig)

    def test_from_yaml(self, tmp_path):
        yaml_content = """\
problem_dir: test_dir
data_folder: data/
max_samples: 100
distributed: false
sampler:
  provider: openai
  temperature: 0.5
database:
  num_islands: 5
  reset_period: 200
"""
        yaml_path = str(tmp_path / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = RunConfig.from_yaml(yaml_path)
        assert config.problem_dir == "test_dir"
        assert config.max_samples == 100
        assert config.distributed is False
        assert config.sampler.provider == "openai"
        assert config.sampler.temperature == 0.5
        assert config.database.num_islands == 5
        assert config.database.reset_period == 200

    def test_from_yaml_empty_file(self, tmp_path):
        """An empty YAML file should produce defaults."""
        yaml_path = str(tmp_path / "empty.yaml")
        with open(yaml_path, "w") as f:
            f.write("")

        config = RunConfig.from_yaml(yaml_path)
        assert config.max_samples == 3600
        assert config.sampler.provider == "qwen"

    def test_to_yaml_roundtrip(self, tmp_path):
        original = RunConfig(
            problem_dir="specs/test",
            max_samples=50,
            sampler=SamplerConfig(provider="gemini", temperature=0.8),
            database=ProgramsDatabaseConfig(num_islands=3),
        )
        yaml_path = str(tmp_path / "out.yaml")
        original.to_yaml(yaml_path)

        assert os.path.exists(yaml_path)

        loaded = RunConfig.from_yaml(yaml_path)
        assert loaded.problem_dir == "specs/test"
        assert loaded.max_samples == 50
        assert loaded.sampler.provider == "gemini"
        assert loaded.sampler.temperature == 0.8
        assert loaded.database.num_islands == 3

    def test_from_yaml_with_cost_per_ktoken(self, tmp_path):
        """Tuple fields like cost_per_ktoken should survive YAML round-trip."""
        yaml_content = """\
sampler:
  cost_per_ktoken: [0.01, 0.05]
"""
        yaml_path = str(tmp_path / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = RunConfig.from_yaml(yaml_path)
        assert config.sampler.cost_per_ktoken == [0.01, 0.05]

    def test_from_yaml_with_all_nested_configs(self, tmp_path):
        """All five nested config sections should be parsed from YAML."""
        yaml_content = """\
problem_dir: test_dir
sampler:
  provider: openai
database:
  num_islands: 5
evaluator:
  timeout_seconds: 200
profiler:
  log_frequency: 50
  complexity_group_size: 10
worker:
  perf_report_interval_seconds: 60
  monitor_interval_seconds: 120
"""
        yaml_path = str(tmp_path / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = RunConfig.from_yaml(yaml_path)
        assert config.sampler.provider == "openai"
        assert config.database.num_islands == 5
        assert config.evaluator.timeout_seconds == 200
        assert config.profiler.log_frequency == 50
        assert config.profiler.complexity_group_size == 10
        assert config.worker.perf_report_interval_seconds == 60
        assert config.worker.monitor_interval_seconds == 120

    def test_from_yaml_unknown_key_rejected(self, tmp_path):
        """Unknown top-level keys should raise ValueError."""
        yaml_content = """\
problem_dir: test_dir
smpler:
  provider: openai
"""
        yaml_path = str(tmp_path / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        try:
            RunConfig.from_yaml(yaml_path)
            assert False, "Should have raised ValueError for unknown key"
        except ValueError as e:
            assert "smpler" in str(e)

    def test_to_yaml_roundtrip_all_configs(self, tmp_path):
        """All nested configs should survive a to_yaml / from_yaml roundtrip."""
        original = RunConfig(
            problem_dir="specs/test",
            max_samples=50,
            sampler=SamplerConfig(provider="gemini", temperature=0.8),
            database=ProgramsDatabaseConfig(num_islands=3),
            evaluator=EvaluatorConfig(timeout_seconds=200),
            profiler=ProfilerConfig(log_frequency=50),
            worker=WorkerConfig(perf_report_interval_seconds=60),
        )
        yaml_path = str(tmp_path / "out.yaml")
        original.to_yaml(yaml_path)
        loaded = RunConfig.from_yaml(yaml_path)
        assert loaded.evaluator.timeout_seconds == 200
        assert loaded.profiler.log_frequency == 50
        assert loaded.worker.perf_report_interval_seconds == 60

    def test_validate_passes_for_valid(self, tmp_path):
        """validate() should not raise for valid config."""
        problem = tmp_path / "problem"
        problem.mkdir()
        (problem / "prompt.txt").write_text("test")
        (problem / "evaluate.py").write_text("def evaluate(data): pass")
        (problem / "equation.py").write_text("def equation(x, params): return x")
        data = tmp_path / "data"
        data.mkdir()

        config = RunConfig(problem_dir=str(problem), data_folder=str(data), log_folder="test_logs")
        config.validate()  # should not raise

    def test_validate_missing_problem_dir(self):
        """validate() should error when problem_dir is missing."""
        config = RunConfig(data_folder="/tmp", log_folder="logs")
        try:
            config.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "problem_dir" in str(e)

    def test_validate_missing_data_folder(self, tmp_path):
        """validate() should error when data_folder is missing."""
        problem = tmp_path / "problem"
        problem.mkdir()
        (problem / "prompt.txt").write_text("test")
        (problem / "evaluate.py").write_text("def evaluate(data): pass")
        (problem / "equation.py").write_text("def equation(x, params): return x")
        config = RunConfig(problem_dir=str(problem), log_folder="logs")
        try:
            config.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "data_folder" in str(e)

    def test_validate_missing_log_folder(self, tmp_path):
        """validate() should error when log_folder is missing."""
        problem = tmp_path / "problem"
        problem.mkdir()
        (problem / "prompt.txt").write_text("test")
        (problem / "evaluate.py").write_text("def evaluate(data): pass")
        (problem / "equation.py").write_text("def equation(x, params): return x")
        data = tmp_path / "data"
        data.mkdir()
        config = RunConfig(problem_dir=str(problem), data_folder=str(data))
        try:
            config.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "log_folder" in str(e)

    def test_validate_bad_problem_dir(self, tmp_path):
        config = RunConfig(problem_dir="/nonexistent/dir", data_folder="/tmp", log_folder="logs")
        try:
            config.validate()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_validate_missing_required_file(self, tmp_path):
        """validate() should error when a required file is missing from problem_dir."""
        problem = tmp_path / "problem"
        problem.mkdir()
        (problem / "prompt.txt").write_text("test")
        # Missing evaluate.py and equation.py
        data = tmp_path / "data"
        data.mkdir()
        config = RunConfig(problem_dir=str(problem), data_folder=str(data), log_folder="logs")
        try:
            config.validate()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_validate_bad_data_folder(self, tmp_path):
        problem = tmp_path / "problem"
        problem.mkdir()
        (problem / "prompt.txt").write_text("test")
        (problem / "evaluate.py").write_text("def evaluate(data): pass")
        (problem / "equation.py").write_text("def equation(x, params): return x")
        config = RunConfig(problem_dir=str(problem), data_folder="/nonexistent/folder", log_folder="logs")
        try:
            config.validate()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
