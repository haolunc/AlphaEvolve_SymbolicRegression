"""Tests for SQLite-backed CheckpointDB and config save/load."""

import os
import sqlite3

import pytest

from alpha_evolve_sr.checkpoint import CheckpointDB, load_config, save_config
from alpha_evolve_sr.code_manipulation import EvaluatedProgram, ParsedFunction
from alpha_evolve_sr.config import ProgramsDatabaseConfig, RunConfig, SamplerConfig
from alpha_evolve_sr.exceptions import CheckpointError


def _make_program(gsn, score=-1.0, complexity=5):
    """Helper to build an EvaluatedProgram for testing."""
    parsed = ParsedFunction(
        name="equation", args="x, params", body="    return params[0] * x",
    )
    return EvaluatedProgram(
        parsed=parsed,
        score=score,
        optimized_params=[1.0, 2.0],
        complexity=complexity,
        complexity_detail={"BinOp": 3},
        global_sample_nums=gsn,
        sample_time=0.1,
        evaluate_time=0.2,
        token_usage=(10, 20),
        token_cost=0.001,
    )


class TestCheckpointDB:
    def test_create_and_tables(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        # Verify tables exist by querying sqlite_master
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        db.close()
        assert "programs" in tables
        assert "island_programs" in tables
        assert "pareto_front" in tables
        assert "metadata" in tables
        assert "profiler_stats" in tables
        assert "profiler_per_complexity" in tables

    def test_memory_db(self):
        """CheckpointDB supports :memory: path without errors."""
        db = CheckpointDB(":memory:")
        with db.transaction():
            db.insert_program(_make_program(1, score=-0.5))
        programs = db.load_programs()
        assert 1 in programs
        db.close()

    def test_insert_and_load_program(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        prog = _make_program(1, score=-0.5, complexity=10)
        with db.transaction():
            db.insert_program(prog, llm_response_text="return x * 2")
        programs = db.load_programs()
        db.close()
        assert 1 in programs
        loaded = programs[1]
        assert loaded.score == -0.5
        assert loaded.complexity == 10
        assert loaded.parsed.name == "equation"
        assert loaded.optimized_params == [1.0, 2.0]
        assert loaded.token_usage == (10, 20)

    def test_insert_program_none_fields(self, tmp_path):
        """Program with None score/complexity/params should round-trip."""
        db = CheckpointDB(str(tmp_path / "test.db"))
        parsed = ParsedFunction(name="f", args="x", body="    return x")
        prog = EvaluatedProgram(parsed=parsed, global_sample_nums=1)
        with db.transaction():
            db.insert_program(prog)
        loaded = db.load_programs()[1]
        db.close()
        assert loaded.score is None
        assert loaded.complexity is None
        assert loaded.optimized_params is None
        assert loaded.token_usage is None

    def test_island_programs_crud(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        prog1 = _make_program(1, score=-1.0)
        prog2 = _make_program(2, score=-0.5)
        prog3 = _make_program(3, score=-0.3)
        with db.transaction():
            db.insert_program(prog1)
            db.insert_program(prog2)
            db.insert_program(prog3)
            db.insert_island_program(0, 0, 1, score=-1.0)
            db.insert_island_program(0, 0, 2, score=-0.5)
            db.insert_island_program(0, 1, 3, score=-0.3)
        memberships = db.load_island_memberships()
        assert memberships[0][0] == [1, 2]
        assert memberships[0][1] == [3]

        # Evict program 1
        with db.transaction():
            db.delete_island_programs(0, [1])
        memberships = db.load_island_memberships()
        assert memberships[0][0] == [2]
        db.close()

    def test_reset_island(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        prog = _make_program(1)
        with db.transaction():
            db.insert_program(prog)
            db.insert_island_program(0, 0, 1, score=-1.0)
        assert db.load_island_memberships()[0][0] == [1]

        with db.transaction():
            db.reset_island(0)
        assert db.load_island_memberships() == {}
        db.close()

    def test_pareto_front(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        for i in range(1, 4):
            with db.transaction():
                db.insert_program(_make_program(i))
        with db.transaction():
            db.replace_pareto_front([1, 3])
        assert db.load_pareto_front_ids() == [1, 3]

        with db.transaction():
            db.replace_pareto_front([2])
        assert db.load_pareto_front_ids() == [2]
        db.close()

    def test_metadata(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        with db.transaction():
            db.save_metadata("global_sample_nums", 42)
            db.save_metadata("best_score_per_island", [-float("inf"), 0.5])
        meta = db.load_metadata()
        db.close()
        assert meta["global_sample_nums"] == 42
        assert meta["best_score_per_island"] == [float("-inf"), 0.5]

    def test_is_populated(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        assert not db.is_populated
        with db.transaction():
            db.save_metadata("key", "value")
        assert db.is_populated
        db.close()

    def test_transaction_rollback(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        prog = _make_program(1)
        try:
            with db.transaction():
                db.insert_program(prog)
                raise ValueError("test rollback")
        except ValueError:
            pass
        assert db.load_programs() == {}
        db.close()

    def test_roundtrip_close_reopen(self, tmp_path):
        """Data persists after close and reopen."""
        db_path = str(tmp_path / "test.db")
        db = CheckpointDB(db_path)
        with db.transaction():
            db.insert_program(_make_program(1, score=-0.3))
            db.save_metadata("global_sample_nums", 1)
        db.close()

        db2 = CheckpointDB(db_path)
        assert db2.is_populated
        progs = db2.load_programs()
        assert 1 in progs
        assert progs[1].score == -0.3
        meta = db2.load_metadata()
        assert meta["global_sample_nums"] == 1
        db2.close()

    def test_checkpoint_error_on_bad_path(self):
        with pytest.raises(CheckpointError, match="Failed to open"):
            CheckpointDB("/nonexistent/deeply/nested/path/that/cannot/exist/test.db")

    def test_load_programs_by_ids(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        with db.transaction():
            db.insert_program(_make_program(1, score=-1.0))
            db.insert_program(_make_program(2, score=-0.5))
            db.insert_program(_make_program(3, score=-0.3))
        result = db.load_programs_by_ids([1, 3])
        assert set(result.keys()) == {1, 3}
        assert result[1].score == -1.0
        assert result[3].score == -0.3

        # Empty list returns empty dict
        assert db.load_programs_by_ids([]) == {}
        db.close()

    def test_load_island_index(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        with db.transaction():
            db.insert_program(_make_program(1, score=-1.0))
            db.insert_program(_make_program(2, score=-0.5))
            db.insert_program(_make_program(3, score=-0.3))
            db.insert_island_program(0, 0, 1, score=-1.0)
            db.insert_island_program(0, 0, 2, score=-0.5)
            db.insert_island_program(1, 1, 3, score=-0.3)
        index = db.load_island_index()
        assert index[0][0] == [(1, -1.0), (2, -0.5)]
        assert index[1][1] == [(3, -0.3)]
        db.close()

    def test_checkpoint_island_index(self, tmp_path):
        """checkpoint_island_index replaces all island_programs rows."""
        from alpha_evolve_sr.database import Island

        db = CheckpointDB(str(tmp_path / "test.db"))
        with db.transaction():
            db.insert_program(_make_program(1, score=-1.0))
            db.insert_program(_make_program(2, score=-0.5))
            # Start with one entry
            db.insert_island_program(0, 0, 1, score=-1.0)

        # Create islands with new state
        island0 = Island(functions_per_prompt=2, complexity_bin_size=10, cluster_max_size=100)
        island0._bins = {0: [(1, -1.0), (2, -0.5)]}
        island1 = Island(functions_per_prompt=2, complexity_bin_size=10, cluster_max_size=100)

        with db.transaction():
            db.checkpoint_island_index([island0, island1], num_islands=2)

        index = db.load_island_index()
        assert 0 in index
        assert index[0][0] == [(1, -1.0), (2, -0.5)]
        assert 1 not in index  # island1 was empty
        db.close()


class TestConfigSaveLoad:
    def test_save_load_run_config_roundtrip(self, tmp_path):
        """save_config + load_config should round-trip a RunConfig."""
        original = RunConfig(
            problem_dir="specs/test",
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
        assert loaded.problem_dir == original.problem_dir
        assert loaded.max_samples == original.max_samples
        assert loaded.distributed == original.distributed
        assert loaded.sampler.provider == "openai"
        assert loaded.sampler.temperature == 0.5
        assert loaded.database.num_islands == 5
        assert loaded.database.reset_period == 200
