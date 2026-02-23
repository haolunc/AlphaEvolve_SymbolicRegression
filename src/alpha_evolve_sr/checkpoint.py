"""Utilities for saving and loading checkpoints of the database component."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager

import numpy as np

from . import code_manipulation
from .config import RunConfig
from .exceptions import CheckpointError
from .logging_config import get_logger

logger = get_logger("checkpoint")


# ---------------------------------------------------------------------------
# SQLite-backed incremental checkpoint
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS programs (
    global_sample_num   INTEGER PRIMARY KEY,
    func_name           TEXT    NOT NULL,
    func_args           TEXT    NOT NULL,
    func_body           TEXT    NOT NULL,
    func_return_type    TEXT,
    func_docstring      TEXT,
    score               REAL,
    optimized_params    TEXT,
    complexity          INTEGER,
    complexity_detail   TEXT,
    sample_time         REAL,
    evaluate_time       REAL,
    token_usage_input   INTEGER,
    token_usage_output  INTEGER,
    token_cost          REAL,
    llm_response_text   TEXT
);

CREATE TABLE IF NOT EXISTS island_bins (
    island_id           INTEGER NOT NULL,
    complexity_bin      INTEGER NOT NULL,
    global_sample_num   INTEGER NOT NULL,
    score               REAL    NOT NULL,
    PRIMARY KEY (island_id, complexity_bin, global_sample_num)
);

CREATE TABLE IF NOT EXISTS pareto_front (
    complexity_bin      INTEGER PRIMARY KEY,
    score               REAL    NOT NULL,
    global_sample_num   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS global_stats (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    global_sample_num   INTEGER NOT NULL DEFAULT 0,
    last_reset_step     INTEGER NOT NULL DEFAULT 1,
    best_score          REAL,
    success_count       INTEGER NOT NULL DEFAULT 0,
    failed_count        INTEGER NOT NULL DEFAULT 0,
    tot_sample_time     REAL    NOT NULL DEFAULT 0.0,
    tot_evaluate_time   REAL    NOT NULL DEFAULT 0.0,
    tot_token_cost      REAL    NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS island_stats (
    island_id           INTEGER PRIMARY KEY,
    size                INTEGER NOT NULL DEFAULT 0,
    best_score          REAL,
    best_gsn            INTEGER
);

CREATE TABLE IF NOT EXISTS run_config (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    config_yaml         TEXT    NOT NULL,
    num_islands         INTEGER NOT NULL,
    complexity_bin_size INTEGER NOT NULL
);
"""


class CheckpointDB:
    """SQLite-backed incremental checkpoint store."""

    def __init__(self, db_path: str) -> None:
        try:
            if db_path != ":memory:":
                os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            self._conn = sqlite3.connect(db_path, timeout=30)
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
        except (sqlite3.Error, OSError) as e:
            raise CheckpointError(f"Failed to open checkpoint DB at {db_path}: {e}") from e
        self._db_path = db_path
        logger.info("CheckpointDB opened at %s", db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("CheckpointDB closed")

    @contextmanager
    def transaction(self):
        """BEGIN/COMMIT/ROLLBACK wrapper."""
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ---- Write operations ------------------------------------------------

    def insert_program(
        self,
        program: code_manipulation.EvaluatedProgram,
        llm_response_text: str | None = None,
    ) -> None:
        token_input = program.token_usage[0] if program.token_usage else None
        token_output = program.token_usage[1] if program.token_usage else None
        opt_params = None
        if program.optimized_params is not None:
            params = program.optimized_params
            if isinstance(params, np.ndarray):
                params = params.tolist()
            opt_params = json.dumps(params)
        complexity_detail = None
        if program.complexity_detail is not None:
            complexity_detail = json.dumps(program.complexity_detail)

        self._conn.execute(
            """INSERT OR REPLACE INTO programs (
                global_sample_num, func_name, func_args, func_body,
                func_return_type, func_docstring, score, optimized_params,
                complexity, complexity_detail, sample_time, evaluate_time,
                token_usage_input, token_usage_output, token_cost,
                llm_response_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                program.global_sample_nums,
                program.parsed.name,
                program.parsed.args,
                program.parsed.body,
                program.parsed.return_type,
                program.parsed.docstring,
                program.score,
                opt_params,
                program.complexity,
                complexity_detail,
                program.sample_time,
                program.evaluate_time,
                token_input,
                token_output,
                program.token_cost,
                llm_response_text,
            ),
        )

    def checkpoint_island_bins(
        self, islands: list, num_islands: int,
    ) -> None:
        """Replace all island_bins rows with current in-memory state."""
        self._conn.execute("DELETE FROM island_bins")
        for island_id in range(num_islands):
            for cbin, entries in islands[island_id]._bins.items():
                for gsn, score in entries:
                    self._conn.execute(
                        "INSERT INTO island_bins VALUES (?, ?, ?, ?)",
                        (island_id, cbin, gsn, score),
                    )

    def save_pareto_front(self, pareto_entries: list) -> None:
        """Replace pareto_front table with current entries.

        Args:
            pareto_entries: list of ParetoEntry(cbin, score, gsn) namedtuples.
        """
        self._conn.execute("DELETE FROM pareto_front")
        self._conn.executemany(
            "INSERT INTO pareto_front (complexity_bin, score, global_sample_num)"
            " VALUES (?, ?, ?)",
            [(p.cbin, p.score, p.gsn) for p in pareto_entries],
        )

    def save_global_stats(
        self,
        global_sample_num: int,
        last_reset_step: int,
        best_score: float | None,
        success_count: int,
        failed_count: int,
        tot_sample_time: float,
        tot_evaluate_time: float,
        tot_token_cost: float,
    ) -> None:
        best_score_val = best_score if (best_score is not None and best_score != float("-inf")) else None
        self._conn.execute(
            """INSERT OR REPLACE INTO global_stats (
                id, global_sample_num, last_reset_step, best_score,
                success_count, failed_count,
                tot_sample_time, tot_evaluate_time, tot_token_cost
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                global_sample_num, last_reset_step, best_score_val,
                success_count, failed_count,
                tot_sample_time, tot_evaluate_time, tot_token_cost,
            ),
        )

    def save_island_stats(
        self,
        island_stats: list[tuple[int, int, float | None, int | None]],
    ) -> None:
        """Replace island_stats with current per-island data.

        Args:
            island_stats: list of (island_id, size, best_score, best_gsn) tuples.
        """
        self._conn.execute("DELETE FROM island_stats")
        for island_id, size, best_score, best_gsn in island_stats:
            best_score_val = best_score if (best_score is not None and best_score != float("-inf")) else None
            self._conn.execute(
                "INSERT INTO island_stats (island_id, size, best_score, best_gsn)"
                " VALUES (?, ?, ?, ?)",
                (island_id, size, best_score_val, best_gsn),
            )

    def save_run_config(
        self,
        run_config: RunConfig,
        num_islands: int,
        complexity_bin_size: int,
    ) -> None:
        """Store run config in the DB (called once on fresh start, updated on resume)."""
        import dataclasses
        import io

        import yaml

        buf = io.StringIO()
        yaml.dump(dataclasses.asdict(run_config), buf, default_flow_style=False, sort_keys=False)
        config_yaml = buf.getvalue()

        self._conn.execute(
            """INSERT OR REPLACE INTO run_config (
                id, config_yaml, num_islands, complexity_bin_size
            ) VALUES (1, ?, ?, ?)""",
            (config_yaml, num_islands, complexity_bin_size),
        )

    # ---- Read operations -------------------------------------------------

    def _rows_to_programs(
        self, rows: list[tuple],
    ) -> dict[int, code_manipulation.EvaluatedProgram]:
        """Convert raw SQL rows into EvaluatedProgram objects."""
        programs: dict[int, code_manipulation.EvaluatedProgram] = {}
        for row in rows:
            (
                gsn, func_name, func_args, func_body,
                func_return_type, func_docstring, score, opt_params_json,
                complexity, complexity_detail_json, sample_time, evaluate_time,
                token_input, token_output, token_cost, _llm_text,
            ) = row
            parsed = code_manipulation.ParsedFunction(
                name=func_name,
                args=func_args,
                body=func_body,
                return_type=func_return_type,
                docstring=func_docstring,
            )
            opt_params = json.loads(opt_params_json) if opt_params_json else None
            complexity_detail = json.loads(complexity_detail_json) if complexity_detail_json else None
            token_usage = (token_input, token_output) if token_input is not None else None

            programs[gsn] = code_manipulation.EvaluatedProgram(
                parsed=parsed,
                score=score,
                optimized_params=opt_params,
                complexity=complexity,
                complexity_detail=complexity_detail,
                global_sample_nums=gsn,
                sample_time=sample_time,
                evaluate_time=evaluate_time,
                token_usage=token_usage,
                token_cost=token_cost,
            )
        return programs

    def load_programs(self) -> dict[int, code_manipulation.EvaluatedProgram]:
        rows = self._conn.execute("SELECT * FROM programs").fetchall()
        return self._rows_to_programs(rows)

    def load_programs_by_ids(
        self, gsn_list: list[int],
    ) -> dict[int, code_manipulation.EvaluatedProgram]:
        """Load specific programs by their global_sample_num primary keys."""
        if not gsn_list:
            return {}
        placeholders = ",".join("?" * len(gsn_list))
        rows = self._conn.execute(
            f"SELECT * FROM programs WHERE global_sample_num IN ({placeholders})",
            gsn_list,
        ).fetchall()
        return self._rows_to_programs(rows)

    def load_island_index(self) -> dict[int, dict[int, list[tuple[int, float]]]]:
        """Returns ``{island_id: {complexity_bin: [(gsn, score), ...]}}``.

        Used to restore in-memory Island._bins from a checkpoint.
        """
        rows = self._conn.execute(
            "SELECT island_id, complexity_bin, global_sample_num, score "
            "FROM island_bins ORDER BY rowid",
        ).fetchall()
        result: dict[int, dict[int, list[tuple[int, float]]]] = {}
        for island_id, complexity_bin, gsn, score in rows:
            result.setdefault(island_id, {}).setdefault(complexity_bin, []).append(
                (gsn, score),
            )
        return result

    def load_pareto_front(self) -> list[tuple[int, float, int]]:
        """Returns list of (complexity_bin, score, global_sample_num) tuples."""
        rows = self._conn.execute(
            "SELECT complexity_bin, score, global_sample_num FROM pareto_front"
            " ORDER BY complexity_bin",
        ).fetchall()
        return [(cbin, score, gsn) for cbin, score, gsn in rows]

    def load_global_stats(self) -> dict | None:
        row = self._conn.execute(
            """SELECT global_sample_num, last_reset_step, best_score,
                      success_count, failed_count,
                      tot_sample_time, tot_evaluate_time, tot_token_cost
               FROM global_stats WHERE id = 1""",
        ).fetchone()
        if row is None:
            return None
        return {
            "global_sample_num": row[0],
            "last_reset_step": row[1],
            "best_score": row[2] if row[2] is not None else float("-inf"),
            "success_count": row[3],
            "failed_count": row[4],
            "tot_sample_time": row[5],
            "tot_evaluate_time": row[6],
            "tot_token_cost": row[7],
        }

    def load_island_stats(self) -> list[dict]:
        """Returns list of dicts with island_id, size, best_score, best_gsn."""
        rows = self._conn.execute(
            "SELECT island_id, size, best_score, best_gsn FROM island_stats"
            " ORDER BY island_id",
        ).fetchall()
        return [
            {
                "island_id": r[0],
                "size": r[1],
                "best_score": r[2] if r[2] is not None else float("-inf"),
                "best_gsn": r[3],
            }
            for r in rows
        ]

    def load_run_config(self) -> dict | None:
        """Load the stored run_config row. Returns dict or None if not stored."""
        row = self._conn.execute(
            "SELECT config_yaml, num_islands, complexity_bin_size"
            " FROM run_config WHERE id = 1",
        ).fetchone()
        if row is None:
            return None
        return {
            "config_yaml": row[0],
            "num_islands": row[1],
            "complexity_bin_size": row[2],
        }

    def validate_config(self, num_islands: int, complexity_bin_size: int) -> None:
        """Validate that structural config matches stored checkpoint.

        Raises:
            CheckpointError: if num_islands or complexity_bin_size differ.
        """
        stored = self.load_run_config()
        if stored is None:
            return  # No stored config to validate against
        if stored["num_islands"] != num_islands:
            raise CheckpointError(
                f"Config mismatch on resume: num_islands={num_islands} "
                f"but checkpoint has num_islands={stored['num_islands']}. "
                f"Changing num_islands mid-run is not supported."
            )
        if stored["complexity_bin_size"] != complexity_bin_size:
            raise CheckpointError(
                f"Config mismatch on resume: complexity_bin_size={complexity_bin_size} "
                f"but checkpoint has complexity_bin_size={stored['complexity_bin_size']}. "
                f"Changing complexity_bin_size mid-run is not supported."
            )

    @property
    def is_populated(self) -> bool:
        row = self._conn.execute("SELECT COUNT(*) FROM global_stats").fetchone()
        return row[0] > 0
