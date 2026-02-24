"""Tests for configuration dataclasses."""

import dataclasses
import json
import os
import warnings

import pytest

from alpha_evolve_sr.config import (
    EvaluatorConfig,
    ProgramsDatabaseConfig,
    RunConfig,
    SamplerConfig,
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
        assert config.num_samplers == 8
        assert isinstance(config.sampler, SamplerConfig)
        assert isinstance(config.database, ProgramsDatabaseConfig)
        assert isinstance(config.evaluator, EvaluatorConfig)
        assert config.database.log_frequency == 25

    def test_from_yaml(self, tmp_path):
        yaml_content = """\
problem_dir: test_dir
data_folder: data/
max_samples: 100
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
        """All nested config sections should be parsed from YAML."""
        yaml_content = """\
problem_dir: test_dir
sampler:
  provider: openai
database:
  num_islands: 5
  log_frequency: 50
evaluator:
  timeout_seconds: 200
"""
        yaml_path = str(tmp_path / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = RunConfig.from_yaml(yaml_path)
        assert config.sampler.provider == "openai"
        assert config.database.num_islands == 5
        assert config.database.log_frequency == 50
        assert config.evaluator.timeout_seconds == 200

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

        with pytest.raises(ValueError, match="smpler"):
            RunConfig.from_yaml(yaml_path)

    def test_to_yaml_roundtrip_all_configs(self, tmp_path):
        """All nested configs should survive a to_yaml / from_yaml roundtrip."""
        original = RunConfig(
            problem_dir="specs/test",
            max_samples=50,
            sampler=SamplerConfig(provider="gemini", temperature=0.8),
            database=ProgramsDatabaseConfig(num_islands=3, log_frequency=50),
            evaluator=EvaluatorConfig(timeout_seconds=200),
        )
        yaml_path = str(tmp_path / "out.yaml")
        original.to_yaml(yaml_path)
        loaded = RunConfig.from_yaml(yaml_path)
        assert loaded.evaluator.timeout_seconds == 200
        assert loaded.database.log_frequency == 50

    def test_validate_passes_for_valid(self, tmp_path):
        """validate() should not raise for valid config."""
        problem = tmp_path / "problem"
        problem.mkdir()
        (problem / "prompt.txt").write_text("test")
        (problem / "evaluate.py").write_text("def evaluate(data): pass")
        (problem / "equation.py").write_text("def equation(x, params): return x")
        data = tmp_path / "data"
        data.mkdir()

        config = RunConfig(problem_dir=str(problem), data_folder=str(data), log_dir="test_logs")
        config.validate()  # should not raise

    @pytest.mark.parametrize("overrides,match", [
        (dict(data_folder="/tmp", log_dir="logs"), "problem_dir"),
        (dict(problem_dir="/nonexistent/dir", data_folder="/tmp", log_dir="logs"), "problem_dir"),
    ])
    def test_validate_rejects_bad_problem_dir(self, overrides, match):
        """validate() rejects missing or nonexistent problem_dir."""
        with pytest.raises(ValueError, match=match):
            RunConfig(**overrides).validate()

    def test_validate_missing_data_folder(self, tmp_path):
        """validate() should error when data_folder is missing."""
        problem = tmp_path / "problem"
        problem.mkdir()
        (problem / "prompt.txt").write_text("test")
        (problem / "evaluate.py").write_text("def evaluate(data): pass")
        (problem / "equation.py").write_text("def equation(x, params): return x")
        with pytest.raises(ValueError, match="data_folder"):
            RunConfig(problem_dir=str(problem), log_dir="logs").validate()

    def test_validate_missing_log_dir(self, tmp_path):
        """validate() should error when log_dir is missing."""
        problem = tmp_path / "problem"
        problem.mkdir()
        (problem / "prompt.txt").write_text("test")
        (problem / "evaluate.py").write_text("def evaluate(data): pass")
        (problem / "equation.py").write_text("def equation(x, params): return x")
        data = tmp_path / "data"
        data.mkdir()
        with pytest.raises(ValueError, match="log_dir"):
            RunConfig(problem_dir=str(problem), data_folder=str(data)).validate()

    def test_validate_missing_required_file(self, tmp_path):
        """validate() should error when a required file is missing from problem_dir."""
        problem = tmp_path / "problem"
        problem.mkdir()
        (problem / "prompt.txt").write_text("test")
        # Missing evaluate.py and equation.py
        data = tmp_path / "data"
        data.mkdir()
        with pytest.raises(ValueError):
            RunConfig(problem_dir=str(problem), data_folder=str(data), log_dir="logs").validate()

    def test_validate_bad_data_folder(self, tmp_path):
        problem = tmp_path / "problem"
        problem.mkdir()
        (problem / "prompt.txt").write_text("test")
        (problem / "evaluate.py").write_text("def evaluate(data): pass")
        (problem / "equation.py").write_text("def equation(x, params): return x")
        with pytest.raises(ValueError):
            RunConfig(problem_dir=str(problem), data_folder="/nonexistent/folder", log_dir="logs").validate()

    def test_from_yaml_text(self):
        """from_yaml_text should parse a YAML string into RunConfig."""
        yaml_text = """\
problem_dir: test_dir
data_folder: data/
log_dir: logs/
max_samples: 42
database:
  num_islands: 3
  log_frequency: 25
"""
        config = RunConfig.from_yaml_text(yaml_text)
        assert config.problem_dir == "test_dir"
        assert config.max_samples == 42
        assert config.database.num_islands == 3
        assert config.database.log_frequency == 25

    def test_post_init_resume_from_dir(self, tmp_path):
        """__post_init__ should resolve a directory to the checkpoint.db inside it."""
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "checkpoint.db").write_text("fake")

        config = RunConfig(
            problem_dir="p", data_folder="d",
            log_dir="logs", resume_from_ckpt=str(ckpt),
        )
        assert config.resume_from_ckpt == str(ckpt / "checkpoint.db")

    def test_post_init_resume_from_db_file(self, tmp_path):
        """__post_init__ should pass through a .db file path unchanged."""
        db_file = tmp_path / "checkpoint.db"
        db_file.write_text("fake")

        config = RunConfig(
            problem_dir="p", data_folder="d",
            log_dir="logs", resume_from_ckpt=str(db_file),
        )
        assert config.resume_from_ckpt == str(db_file)

    def test_post_init_default_save_ckpt_dir(self):
        """__post_init__ should default save_ckpt_dir to log_dir."""
        config = RunConfig(log_dir="my_logs")
        assert config.save_ckpt_dir == "my_logs"

    def test_validate_resume_nonexistent(self, tmp_path):
        """validate() should raise ValueError for a nonexistent resume path."""
        problem = tmp_path / "problem"
        problem.mkdir()
        (problem / "prompt.txt").write_text("test")
        (problem / "evaluate.py").write_text("def evaluate(data): pass")
        (problem / "equation.py").write_text("def equation(x, params): return x")
        data = tmp_path / "data"
        data.mkdir()

        config = RunConfig(
            problem_dir=str(problem), data_folder=str(data),
            log_dir="logs", resume_from_ckpt="/nonexistent/path",
        )
        with pytest.raises(ValueError, match="checkpoint not found"):
            config.validate()

    @pytest.mark.parametrize("key,yaml_block", [
        ("distributed", "distributed: true"),
        ("worker", "worker:\n  perf_report_interval_seconds: 60"),
    ])
    def test_deprecated_key_ignored(self, tmp_path, key, yaml_block):
        """Old deprecated keys should be silently ignored with a deprecation warning."""
        yaml_content = f"""\
problem_dir: test_dir
{yaml_block}
database:
  num_islands: 5
"""
        yaml_path = str(tmp_path / "old.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = RunConfig.from_yaml(yaml_path)
            assert any(key in str(warning.message) for warning in w)
        assert config.database.num_islands == 5
        assert not hasattr(config, key)
