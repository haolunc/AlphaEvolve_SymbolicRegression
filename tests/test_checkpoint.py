"""Tests for SQLite-backed CheckpointDB and LogsDB."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from alpha_evolve_sr.checkpoint import CheckpointDB, LogsDB, export_json
from alpha_evolve_sr.code_manipulation import ParsedFunction

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
        data = db.load_bodies_by_ids([1])
        assert 1 in data
        db.close()

    def test_insert_and_load_program(self, checkpoint_db):
        prog = make_evaluated_program(1, score=-0.5, complexity=10)
        with checkpoint_db.transaction():
            checkpoint_db.insert_program(prog)
        data = checkpoint_db.load_bodies_by_ids([1])
        assert 1 in data
        body, score = data[1]
        assert score == -0.5
        assert "return" in body

    def test_insert_program_none_fields(self, checkpoint_db):
        """Program with None score/complexity should round-trip."""
        parsed = ParsedFunction(name="f", args="x", body="    return x")
        prog = SimpleNamespace(parsed=parsed, global_sample_nums=1, score=None)
        with checkpoint_db.transaction():
            checkpoint_db.insert_program(prog)
        data = checkpoint_db.load_bodies_by_ids([1])
        assert 1 in data
        _body, score = data[1]
        assert score is None

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
                (0, 5, -0.5, 10, 7),
                (1, 3, float("-inf"), None, None),
            ])
        stats = checkpoint_db.load_island_stats()
        assert len(stats) == 2
        assert stats[0] == {"island_id": 0, "size": 5, "best_score": -0.5, "best_gsn": 10, "best_complexity": 7}
        assert stats[1] == {"island_id": 1, "size": 3, "best_score": float("-inf"), "best_gsn": None, "best_complexity": None}

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
        assert checkpoint_db.load_bodies_by_ids([1]) == {}

    def test_roundtrip_close_reopen(self, tmp_path):
        """Data persists after close and reopen."""
        db_path = str(tmp_path / "test.db")
        db = CheckpointDB(db_path)
        with db.transaction():
            db.insert_program(make_evaluated_program(1, score=-0.3))
            db.save_run_config(num_islands=5, complexity_bin_size=10)
            db.save_global_stats(
                global_sample_num=1, last_reset_step=1, best_score=-0.3,
                success_count=1, failed_count=0,
                tot_sample_time=0.1, tot_evaluate_time=0.2, tot_token_cost=0.001,
            )
        db.close()

        db2 = CheckpointDB(db_path)
        assert db2.is_populated
        data = db2.load_bodies_by_ids([1])
        assert 1 in data
        assert data[1][1] == -0.3
        stats = db2.load_global_stats()
        assert stats["global_sample_num"] == 1
        db2.close()

    def test_checkpoint_error_on_bad_path(self):
        with pytest.raises(OSError, match="Failed to open"):
            CheckpointDB("/nonexistent/deeply/nested/path/that/cannot/exist/test.db")


class TestRunConfig:
    @pytest.fixture
    def checkpoint_db(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        yield db
        db.close()

    def test_save_load_roundtrip(self, checkpoint_db):
        """save_run_config + load_run_config should round-trip structural fields."""
        with checkpoint_db.transaction():
            checkpoint_db.save_run_config(num_islands=5, complexity_bin_size=10)

        stored = checkpoint_db.load_run_config()
        assert stored is not None
        assert stored["num_islands"] == 5
        assert stored["complexity_bin_size"] == 10

    def test_validate_config_passes_on_match(self, checkpoint_db):
        with checkpoint_db.transaction():
            checkpoint_db.save_run_config(num_islands=5, complexity_bin_size=10)
        # Should not raise
        checkpoint_db.validate_config(num_islands=5, complexity_bin_size=10)

    @pytest.mark.parametrize("saved,query,match", [
        (dict(num_islands=5, complexity_bin_size=10), dict(num_islands=8, complexity_bin_size=10), "num_islands"),
        (dict(num_islands=10, complexity_bin_size=10),
         dict(num_islands=10, complexity_bin_size=20), "complexity_bin_size"),
    ])
    def test_validate_config_fails_on_mismatch(self, tmp_path, saved, query, match):
        db = CheckpointDB(str(tmp_path / "test.db"))
        with db.transaction():
            db.save_run_config(**saved)
        with pytest.raises(ValueError, match=match):
            db.validate_config(**query)
        db.close()

    def test_validate_config_no_stored_config(self, checkpoint_db):
        """validate_config is a no-op when no config is stored yet."""
        # Should not raise
        checkpoint_db.validate_config(num_islands=5, complexity_bin_size=10)


class TestLogsDB:
    @pytest.fixture
    def logs_db(self, tmp_path):
        db = LogsDB(str(tmp_path / "logs.db"))
        yield db
        db.close()

    def test_insert_and_verify(self, logs_db, tmp_path):
        """insert_log writes all columns to the program_logs table."""
        logs_db.insert_log(
            global_sample_num=1,
            llm_response_text="return x * 2",
            error_type="ValueError",
            error_message="bad value",
            eval_output="CMA-ES: iter 1\n",
            complexity=9,
            complexity_detail='{"BinOp": 3}',
            optimized_params='[1.0, 2.0]',
            sample_time=0.1,
            evaluate_time=0.2,
            token_usage_input=10,
            token_usage_output=20,
            token_cost=0.001,
        )
        # Read back via raw SQL
        conn = sqlite3.connect(str(tmp_path / "logs.db"))
        row = conn.execute("SELECT * FROM program_logs WHERE global_sample_num = 1").fetchone()
        conn.close()
        assert row == (
            1, "return x * 2", "ValueError", "bad value", "CMA-ES: iter 1\n",
            9, '{"BinOp": 3}', '[1.0, 2.0]', 0.1, 0.2, 10, 20, 0.001,
        )

    def test_insert_none_fields(self, logs_db, tmp_path):
        """All-None log fields should store NULLs."""
        logs_db.insert_log(1, None, None, None, None)
        conn = sqlite3.connect(str(tmp_path / "logs.db"))
        row = conn.execute("SELECT * FROM program_logs WHERE global_sample_num = 1").fetchone()
        conn.close()
        assert row == (1, None, None, None, None, None, None, None, None, None, None, None, None)

    def test_close_and_reopen(self, tmp_path):
        """Data persists after close and reopen."""
        db_path = str(tmp_path / "logs.db")
        db = LogsDB(db_path)
        db.insert_log(1, "text", None, None, "output", sample_time=0.5)
        db.close()

        db2 = LogsDB(db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT * FROM program_logs WHERE global_sample_num = 1").fetchone()
        conn.close()
        db2.close()
        assert row[1] == "text"
        assert row[4] == "output"
        assert row[8] == 0.5  # sample_time

    def test_error_on_bad_path(self):
        with pytest.raises(OSError, match="Failed to open"):
            LogsDB("/nonexistent/deeply/nested/path/that/cannot/exist/logs.db")

    def test_load_all_logs(self, logs_db):
        logs_db.insert_log(1, "resp1", None, None, "out1", sample_time=0.1)
        logs_db.insert_log(2, "resp2", "Error", "msg", "out2", sample_time=0.2)
        rows = logs_db.load_all_logs()
        assert len(rows) == 2
        assert rows[0]["global_sample_num"] == 1
        assert rows[1]["error_type"] == "Error"

    def test_load_all_logs_empty(self, logs_db):
        assert logs_db.load_all_logs() == []


class TestLoadAllPrograms:
    @pytest.fixture
    def checkpoint_db(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "test.db"))
        yield db
        db.close()

    def test_load_all_programs(self, checkpoint_db):
        with checkpoint_db.transaction():
            checkpoint_db.insert_program(make_evaluated_program(1, score=-0.5))
            checkpoint_db.insert_program(make_evaluated_program(2, score=-0.3))
        rows = checkpoint_db.load_all_programs()
        assert len(rows) == 2
        assert rows[0]["global_sample_num"] == 1
        assert rows[0]["score"] == -0.5
        assert rows[1]["global_sample_num"] == 2

    def test_load_all_programs_empty(self, checkpoint_db):
        assert checkpoint_db.load_all_programs() == []


class TestExportJson:
    @pytest.fixture
    def checkpoint_db(self, tmp_path):
        db = CheckpointDB(str(tmp_path / "ckpt.db"))
        yield db
        db.close()

    @pytest.fixture
    def logs_db(self, tmp_path):
        db = LogsDB(str(tmp_path / "logs.db"))
        yield db
        db.close()

    def test_export_creates_valid_json(self, checkpoint_db, logs_db, tmp_path):
        """Round-trip: populate both DBs, export, and verify JSON structure."""
        with checkpoint_db.transaction():
            checkpoint_db.insert_program(make_evaluated_program(1, score=-0.5))
            checkpoint_db.save_global_stats(
                global_sample_num=1, last_reset_step=1, best_score=-0.5,
                success_count=1, failed_count=0,
                tot_sample_time=0.1, tot_evaluate_time=0.2, tot_token_cost=0.001,
            )
            checkpoint_db.save_run_config(num_islands=2, complexity_bin_size=10)
        logs_db.insert_log(1, "text", None, None, "output")

        out = str(tmp_path / "checkpoint.json")
        export_json(checkpoint_db, logs_db, out)

        with open(out, encoding="utf-8") as f:
            data = json.load(f)

        assert "checkpoint" in data
        assert "logs" in data
        assert len(data["checkpoint"]["programs"]) == 1
        assert data["checkpoint"]["global_stats"]["global_sample_num"] == 1
        assert data["checkpoint"]["run_config"]["num_islands"] == 2
        assert len(data["logs"]["program_logs"]) == 1

    def test_export_without_logs_db(self, checkpoint_db, tmp_path):
        """When logs_db is None, logs section should be empty."""
        with checkpoint_db.transaction():
            checkpoint_db.insert_program(make_evaluated_program(1, score=-0.5))
            checkpoint_db.save_global_stats(
                global_sample_num=1, last_reset_step=1, best_score=-0.5,
                success_count=1, failed_count=0,
                tot_sample_time=0.1, tot_evaluate_time=0.2, tot_token_cost=0.001,
            )

        out = str(tmp_path / "checkpoint.json")
        export_json(checkpoint_db, None, out)

        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert data["logs"] == {}

    def test_neg_inf_becomes_null(self, checkpoint_db, tmp_path):
        """-inf best_score should become null in JSON output."""
        with checkpoint_db.transaction():
            checkpoint_db.save_global_stats(
                global_sample_num=0, last_reset_step=1,
                best_score=float("-inf"),
                success_count=0, failed_count=0,
                tot_sample_time=0.0, tot_evaluate_time=0.0, tot_token_cost=0.0,
            )

        out = str(tmp_path / "checkpoint.json")
        export_json(checkpoint_db, None, out)

        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert data["checkpoint"]["global_stats"]["best_score"] is None

    def test_export_indented(self, checkpoint_db, tmp_path):
        """Output should be human-readable (indented)."""
        with checkpoint_db.transaction():
            checkpoint_db.save_global_stats(
                global_sample_num=0, last_reset_step=1, best_score=-1.0,
                success_count=0, failed_count=0,
                tot_sample_time=0.0, tot_evaluate_time=0.0, tot_token_cost=0.0,
            )

        out = str(tmp_path / "checkpoint.json")
        export_json(checkpoint_db, None, out)

        with open(out, encoding="utf-8") as f:
            text = f.read()
        # Indented JSON has newlines and leading spaces
        assert "\n" in text
        assert "  " in text
