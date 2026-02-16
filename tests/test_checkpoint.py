"""Tests for checkpoint save/load."""

import os
import tempfile

from alpha_evolve_sr.checkpoint import load_checkpoint, load_config, save_checkpoint, save_config
from alpha_evolve_sr.code_manipulation import text_to_program
from alpha_evolve_sr.config import ProgramsDatabaseConfig, RunConfig, SamplerConfig
from alpha_evolve_sr.database import ProgramsDatabase
from tests.conftest import SAMPLE_SPEC


class TestCheckpoint:
    def test_save_load_roundtrip(self):
        """Save and load a database, verifying state is preserved."""
        config = ProgramsDatabaseConfig(num_islands=2, reset_period=999)
        template = text_to_program(SAMPLE_SPEC)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            os.makedirs(log_dir)
            db = ProgramsDatabase(config, template, "equation", log_dir)

            # Register a program so there's state to save
            func = template.get_function("equation")
            db.register_program(
                func,
                island_id=None,
                result_per_test={
                    "score": -0.5,
                    "optimized_params": None,
                    "complexity": 4,
                    "complexity_detail": {},
                },
            )

            ckpt_path = os.path.join(tmpdir, "checkpoint_final.pkl")
            save_checkpoint(db, ckpt_path)

            assert os.path.exists(ckpt_path)

            loaded_db = load_checkpoint(tmpdir)
            assert loaded_db.sample_count == db.sample_count
            assert loaded_db._config == db._config


class TestConfigSaveLoad:
    def test_save_load_run_config_roundtrip(self, tmp_path):
        """save_config + load_config should round-trip a RunConfig."""
        original = RunConfig(
            spec_path="specs/test.txt",
            data_folder="data/test",
            log_folder="test_run",
            max_samples=100,
            distributed=False,
            num_samplers=4,
            num_evaluators=2,
            sampler=SamplerConfig(provider="openai", temperature=0.5),
            database=ProgramsDatabaseConfig(num_islands=5, reset_period=200),
        )

        save_dir = str(tmp_path / "ckpt")
        save_config(original, save_dir)

        assert os.path.exists(os.path.join(save_dir, "run_config.yaml"))

        loaded = load_config(save_dir)
        assert loaded.spec_path == original.spec_path
        assert loaded.max_samples == original.max_samples
        assert loaded.distributed == original.distributed
        assert loaded.sampler.provider == "openai"
        assert loaded.sampler.temperature == 0.5
        assert loaded.database.num_islands == 5
        assert loaded.database.reset_period == 200
