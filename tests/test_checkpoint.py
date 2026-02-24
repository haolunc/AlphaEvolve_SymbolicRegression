"""Tests for SQLite-backed CheckpointDB."""

import sqlite3

import pytest

from alpha_evolve_sr.checkpoint import CheckpointDB
from alpha_evolve_sr.code_manipulation import EvaluatedProgram, ParsedFunction
from alpha_evolve_sr.config import ProgramsDatabaseConfig, RunConfig, SamplerConfig
from alpha_evolve_sr.database import ParetoEntry
from tests.conftest import make_evaluated_program


class TestCheckpointDB:
    @pytest.fixture
    def checkpoint_db(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        yield db
        db.close()

    def test_create_and_tables(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        # Verify tables exist by querying sqlite_master
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        db.close()
        assert "programs" in tables
        assert "island_bins" in tables
        assert "pareto_front" in tables
        assert "global_stats" in tables
        assert "island_stats" in tables
        assert "run_config" in tables

    def test_memory_db(self):
        """CheckpointDB supports :memory: path without errors."""
        db = CheckpointDB(":memory:")
        with db.transaction():
            db.insert_program(make_evaluated_program(1, score=-0.5))
        programs = db.load_programs()
        assert 1 in programs
        db.close()

    def test_insert_and_load_program(self, checkpoint_db):
        prog = make_evaluated_program(1, score=-0.5, complexity=10)
        with checkpoint_db.transaction():
            checkpoint_db.insert_program(prog, llm_response_text="return x * 2")
        programs = checkpoint_db.load_programs()
        assert 1 in programs
        loaded = programs[1]
        assert loaded.score == -0.5
        assert loaded.complexity == 10
        assert loaded.parsed.name == "equation"
        assert loaded.optimized_params == [1.0, 2.0]
        assert loaded.token_usage == (10, 20)

    def test_insert_program_none_fields(self, checkpoint_db):
        """Program with None score/complexity/params should round-trip."""
        parsed = ParsedFunction(name="f", args="x", body="    return x")
        prog = EvaluatedProgram(parsed=parsed, global_sample_nums=1)
        with checkpoint_db.transaction():
            checkpoint_db.insert_program(prog)
        loaded = checkpoint_db.load_programs()[1]
        assert loaded.score is None
        assert loaded.complexity is None
        assert loaded.optimized_params is None
        assert loaded.token_usage is None

    def test_island_bins_roundtrip(self, checkpoint_db):
        """checkpoint_island_bins + load_island_index round-trip."""
        from alpha_evolve_sr.database import Island

        with checkpoint_db.transaction():
            checkpoint_db.insert_program(make_evaluated_program(1, score=-1.0))
            checkpoint_db.insert_program(make_evaluated_program(2, score=-0.5))

        island0 = Island(functions_per_prompt=2, complexity_bin_size=10, cluster_max_size=100)
        island0._bins = {0: [(1, -1.0), (2, -0.5)]}
        island1 = Island(functions_per_prompt=2, complexity_bin_size=10, cluster_max_size=100)

        with checkpoint_db.transaction():
            checkpoint_db.checkpoint_island_bins([island0, island1], num_islands=2)

        index = checkpoint_db.load_island_index()
        assert 0 in index
        assert index[0][0] == [(1, -1.0), (2, -0.5)]
        assert 1 not in index  # island1 was empty

    def test_pareto_front_roundtrip(self, checkpoint_db):
        entries = [
            ParetoEntry(cbin=0, score=-1.0, gsn=1),
            ParetoEntry(cbin=2, score=-0.5, gsn=3),
        ]
        with checkpoint_db.transaction():
            checkpoint_db.save_pareto_front(entries)
        loaded = checkpoint_db.load_pareto_front()
        assert loaded == [(0, -1.0, 1), (2, -0.5, 3)]

        # Replace with different entries
        with checkpoint_db.transaction():
            checkpoint_db.save_pareto_front([ParetoEntry(cbin=1, score=-0.3, gsn=5)])
        loaded = checkpoint_db.load_pareto_front()
        assert loaded == [(1, -0.3, 5)]

    def test_global_stats_roundtrip(self, checkpoint_db):
        with checkpoint_db.transaction():
            checkpoint_db.save_global_stats(
                global_sample_num=42,
                last_reset_step=10,
                best_score=-0.5,
                success_count=30,
                failed_count=12,
                tot_sample_time=100.0,
                tot_evaluate_time=200.0,
                tot_token_cost=5.0,
            )
        stats = checkpoint_db.load_global_stats()
        assert stats["global_sample_num"] == 42
        assert stats["last_reset_step"] == 10
        assert stats["best_score"] == -0.5
        assert stats["success_count"] == 30
        assert stats["failed_count"] == 12
        assert stats["tot_sample_time"] == 100.0
        assert stats["tot_evaluate_time"] == 200.0
        assert stats["tot_token_cost"] == 5.0

    def test_global_stats_neg_inf_best_score(self, checkpoint_db):
        """best_score of -inf is stored as NULL and restored correctly."""
        with checkpoint_db.transaction():
            checkpoint_db.save_global_stats(
                global_sample_num=0, last_reset_step=1,
                best_score=float("-inf"),
                success_count=0, failed_count=0,
                tot_sample_time=0.0, tot_evaluate_time=0.0, tot_token_cost=0.0,
            )
        stats = checkpoint_db.load_global_stats()
        assert stats["best_score"] == float("-inf")

    def test_island_stats_roundtrip(self, checkpoint_db):
        with checkpoint_db.transaction():
            checkpoint_db.save_island_stats([
                (0, 5, -0.5, 10),
                (1, 3, float("-inf"), None),
            ])
        stats = checkpoint_db.load_island_stats()
        assert len(stats) == 2
        assert stats[0] == {"island_id": 0, "size": 5, "best_score": -0.5, "best_gsn": 10}
        assert stats[1] == {"island_id": 1, "size": 3, "best_score": float("-inf"), "best_gsn": None}

    def test_is_populated(self, checkpoint_db):
        assert not checkpoint_db.is_populated
        with checkpoint_db.transaction():
            checkpoint_db.save_global_stats(
                global_sample_num=1, last_reset_step=1, best_score=-0.5,
                success_count=1, failed_count=0,
                tot_sample_time=0.1, tot_evaluate_time=0.2, tot_token_cost=0.001,
            )
        assert checkpoint_db.is_populated

    def test_transaction_rollback(self, checkpoint_db):
        prog = make_evaluated_program(1)
        try:
            with checkpoint_db.transaction():
                checkpoint_db.insert_program(prog)
                raise ValueError("test rollback")
        except ValueError:
            pass
        assert checkpoint_db.load_programs() == {}

    def test_roundtrip_close_reopen(self, tmp_path):
        """Data persists after close and reopen."""
        db_path = str(tmp_path / "test.db")
        db = CheckpointDB(db_path)
        run_config = RunConfig(
            problem_dir="specs/test", data_folder="data/test",
            log_dir="test_run", max_samples=100,
        )
        with db.transaction():
            db.insert_program(make_evaluated_program(1, score=-0.3))
            db.save_run_config(run_config, num_islands=5, complexity_bin_size=10)
            db.save_global_stats(
                global_sample_num=1, last_reset_step=1, best_score=-0.3,
                success_count=1, failed_count=0,
                tot_sample_time=0.1, tot_evaluate_time=0.2, tot_token_cost=0.001,
            )
        db.close()

        db2 = CheckpointDB(db_path)
        assert db2.is_populated
        progs = db2.load_programs()
        assert 1 in progs
        assert progs[1].score == -0.3
        stats = db2.load_global_stats()
        assert stats["global_sample_num"] == 1
        db2.close()

    def test_checkpoint_error_on_bad_path(self):
        with pytest.raises(OSError, match="Failed to open"):
            CheckpointDB("/nonexistent/deeply/nested/path/that/cannot/exist/test.db")

    def test_decorator_persistence(self, checkpoint_db):
        """Decorators survive insert → load round-trip."""
        parsed = ParsedFunction(
            name="equation", args="x, params", body="    return x",
            decorators=("jax.jit",),
        )
        prog = EvaluatedProgram(parsed=parsed, global_sample_nums=1)
        with checkpoint_db.transaction():
            checkpoint_db.insert_program(prog)
        loaded = checkpoint_db.load_programs()[1]
        assert loaded.parsed.decorators == ("jax.jit",)

    def test_decorator_persistence_empty(self, checkpoint_db):
        """Programs with no decorators round-trip as empty tuple."""
        parsed = ParsedFunction(name="f", args="x", body="    return x")
        prog = EvaluatedProgram(parsed=parsed, global_sample_nums=1)
        with checkpoint_db.transaction():
            checkpoint_db.insert_program(prog)
        loaded = checkpoint_db.load_programs()[1]
        assert loaded.parsed.decorators == ()

    def test_load_programs_by_ids(self, checkpoint_db):
        with checkpoint_db.transaction():
            checkpoint_db.insert_program(make_evaluated_program(1, score=-1.0))
            checkpoint_db.insert_program(make_evaluated_program(2, score=-0.5))
            checkpoint_db.insert_program(make_evaluated_program(3, score=-0.3))
        result = checkpoint_db.load_programs_by_ids([1, 3])
        assert set(result.keys()) == {1, 3}
        assert result[1].score == -1.0
        assert result[3].score == -0.3

        # Empty list returns empty dict
        assert checkpoint_db.load_programs_by_ids([]) == {}


class TestRunConfig:
    @pytest.fixture
    def checkpoint_db(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        yield db
        db.close()

    def test_save_load_roundtrip(self, checkpoint_db):
        """save_run_config + load_run_config should round-trip structural fields."""
        run_config = RunConfig(
            problem_dir="specs/test",
            data_folder="data/test",
            log_dir="test_run",
            max_samples=100,
            num_samplers=4,
            num_evaluators=2,
            sampler=SamplerConfig(provider="openai", temperature=0.5),
            database=ProgramsDatabaseConfig(num_islands=5, reset_period=200),
        )
        with checkpoint_db.transaction():
            checkpoint_db.save_run_config(run_config, num_islands=5, complexity_bin_size=10)

        stored = checkpoint_db.load_run_config()
        assert stored is not None
        assert stored["num_islands"] == 5
        assert stored["complexity_bin_size"] == 10
        assert "problem_dir" in stored["config_yaml"]

    def test_validate_config_passes_on_match(self, checkpoint_db):
        run_config = RunConfig(
            problem_dir="specs/test", data_folder="data/test", log_dir="test",
            database=ProgramsDatabaseConfig(num_islands=5, complexity_bin_size=10),
        )
        with checkpoint_db.transaction():
            checkpoint_db.save_run_config(run_config, num_islands=5, complexity_bin_size=10)
        # Should not raise
        checkpoint_db.validate_config(num_islands=5, complexity_bin_size=10)

    @pytest.mark.parametrize("saved,query,match", [
        (dict(num_islands=5, complexity_bin_size=10), dict(num_islands=8, complexity_bin_size=10), "num_islands"),
        (dict(num_islands=10, complexity_bin_size=10),
         dict(num_islands=10, complexity_bin_size=20), "complexity_bin_size"),
    ])
    def test_validate_config_fails_on_mismatch(self, tmp_path, saved, query, match):
        db = CheckpointDB(str(tmp_path / "test.db"))
        run_config = RunConfig(
            problem_dir="specs/test", data_folder="data/test", log_dir="test",
            database=ProgramsDatabaseConfig(**saved),
        )
        with db.transaction():
            db.save_run_config(run_config, **saved)
        with pytest.raises(ValueError, match=match):
            db.validate_config(**query)
        db.close()

    def test_validate_config_no_stored_config(self, checkpoint_db):
        """validate_config is a no-op when no config is stored yet."""
        # Should not raise
        checkpoint_db.validate_config(num_islands=5, complexity_bin_size=10)
