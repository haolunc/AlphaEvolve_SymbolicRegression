"""Tests for ProgramsDatabase lifecycle (checkpoint, finalize, should_stop)."""

from __future__ import annotations

import os

import pytest

from alpha_evolve_sr.config import ProgramsDatabaseConfig
from alpha_evolve_sr.database import ProgramsDatabase
from tests.conftest import SAMPLE_PROMPT, make_eval_result, make_sample_message


@pytest.fixture
def db_lifecycle(tmp_path):
    """A ProgramsDatabase with lifecycle params for testing."""
    config = ProgramsDatabaseConfig(num_islands=2, reset_period=999)
    return ProgramsDatabase.restore_or_create(
        config, SAMPLE_PROMPT, str(tmp_path / "logs"),
        ckpt_dir=str(tmp_path / "ckpts"), max_samples=5,
    )


class TestRestoreOrCreate:
    def test_creates_with_initial_result(self, tmp_path):
        config = ProgramsDatabaseConfig(num_islands=2, reset_period=999)
        initial = make_eval_result()
        db = ProgramsDatabase.restore_or_create(
            config, SAMPLE_PROMPT, str(tmp_path / "logs"),
            ckpt_dir=str(tmp_path / "ckpts"), max_samples=5,
            initial_result=initial,
        )
        assert db.sample_count == 1
        # checkpoint.db should exist
        assert os.path.exists(os.path.join(str(tmp_path / "ckpts"), "checkpoint.db"))
        db.finalize()

    def test_creates_without_initial_result(self, tmp_path):
        config = ProgramsDatabaseConfig(num_islands=2, reset_period=999)
        db = ProgramsDatabase.restore_or_create(
            config, SAMPLE_PROMPT, str(tmp_path / "logs"),
            max_samples=5,
        )
        assert db.sample_count == 0

    def test_restore_from_checkpoint(self, tmp_path):
        """Create a DB, register programs, finalize, then restore."""
        config = ProgramsDatabaseConfig(num_islands=2, reset_period=999)
        ckpt_dir = str(tmp_path / "ckpts")

        db = ProgramsDatabase.restore_or_create(
            config, SAMPLE_PROMPT, str(tmp_path / "logs1"),
            ckpt_dir=ckpt_dir, max_samples=100,
            initial_result=make_eval_result(score=-0.5),
        )
        db.register_program(make_eval_result(score=-0.3), make_sample_message(island_id=0))
        db.register_program(make_eval_result(score=-0.1), make_sample_message(island_id=1))
        count_before = db.sample_count
        db.finalize()

        # Restore from same dir
        db2 = ProgramsDatabase.restore_or_create(
            config, SAMPLE_PROMPT, str(tmp_path / "logs2"),
            ckpt_dir=ckpt_dir, max_samples=100,
            resume_path=ckpt_dir,
        )
        assert db2.sample_count == count_before
        db2.finalize()


class TestRegisterAndStop:
    def test_register_increments_count(self, db_lifecycle):
        db_lifecycle.register_program(make_eval_result())
        assert db_lifecycle.sample_count == 1

        db_lifecycle.register_program(make_eval_result(), make_sample_message(island_id=0))
        assert db_lifecycle.sample_count == 2
        db_lifecycle.finalize()

    def test_should_stop(self, db_lifecycle):
        db_lifecycle.register_program(make_eval_result())
        assert not db_lifecycle.should_stop

        for _ in range(4):
            db_lifecycle.register_program(make_eval_result(), make_sample_message(island_id=0))
        assert db_lifecycle.should_stop
        db_lifecycle.finalize()


class TestLifecycleCheckpoint:
    def test_maybe_checkpoint_noop(self, db_lifecycle):
        """maybe_checkpoint always returns False (persistence is incremental)."""
        db_lifecycle.register_program(make_eval_result())
        assert not db_lifecycle.maybe_checkpoint()
        db_lifecycle.finalize()

    def test_checkpoint_db_exists(self, db_lifecycle):
        """checkpoint.db should be created when ckpt_dir is set."""
        db_lifecycle.register_program(make_eval_result())
        assert os.path.exists(os.path.join(db_lifecycle._ckpt_dir, "checkpoint.db"))
        db_lifecycle.finalize()


class TestLifecycleFinalize:
    def test_finalize_writes_outputs(self, db_lifecycle):
        db_lifecycle.register_program(make_eval_result())
        db_lifecycle.finalize()
        # checkpoint.db should exist (no more .pkl)
        assert os.path.exists(os.path.join(db_lifecycle._ckpt_dir, "checkpoint.db"))


class TestLifecycleGetPrompt:
    def test_get_prompt_returns_prompt(self, db_lifecycle):
        db_lifecycle.register_program(make_eval_result())
        prompt = db_lifecycle.get_prompt()
        assert hasattr(prompt, "code")
        assert hasattr(prompt, "island_id")
        assert len(prompt.code) > 0
        db_lifecycle.finalize()
