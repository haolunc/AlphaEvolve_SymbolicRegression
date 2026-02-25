"""Utilities for saving and loading checkpoints of the database component."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

from .logging_config import get_logger

logger = get_logger("checkpoint")


def _score_to_db(score: float | None) -> float | None:
    """Convert ``-inf`` to ``NULL`` for SQLite storage."""
    if score is None or score == float("-inf"):
        return None
    return score


def _score_from_db(val: float | None) -> float:
    """Convert ``NULL`` back to ``-inf`` when loading from SQLite."""
    return val if val is not None else float("-inf")


# ---------------------------------------------------------------------------
# SQLite-backed incremental checkpoint
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS programs (
    global_sample_num   INTEGER PRIMARY KEY,
    func_body           TEXT    NOT NULL,
    score               REAL
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
    best_gsn            INTEGER,
    best_complexity     INTEGER
);

CREATE TABLE IF NOT EXISTS run_config (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
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
            raise OSError(f"Failed to open checkpoint DB at {db_path}: {e}") from e
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

    def insert_program(self, program) -> None:
        """Insert a raw program row.

        ``program`` must expose ``global_sample_nums``, ``parsed.body``, and ``score``.
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO programs (
                global_sample_num, func_body, score
            ) VALUES (?, ?, ?)""",
            (
                program.global_sample_nums,
                program.parsed.body,
                program.score,
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
        self._conn.execute(
            """INSERT OR REPLACE INTO global_stats (
                id, global_sample_num, last_reset_step, best_score,
                success_count, failed_count,
                tot_sample_time, tot_evaluate_time, tot_token_cost
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                global_sample_num, last_reset_step, _score_to_db(best_score),
                success_count, failed_count,
                tot_sample_time, tot_evaluate_time, tot_token_cost,
            ),
        )

    def save_island_stats(
        self,
        island_stats: list[tuple[int, int, float | None, int | None, int | None]],
    ) -> None:
        """Replace island_stats with current per-island data.

        Args:
            island_stats: list of (island_id, size, best_score, best_gsn, best_complexity) tuples.
        """
        self._conn.execute("DELETE FROM island_stats")
        for island_id, size, best_score, best_gsn, best_complexity in island_stats:
            self._conn.execute(
                "INSERT INTO island_stats (island_id, size, best_score, best_gsn, best_complexity)"
                " VALUES (?, ?, ?, ?, ?)",
                (island_id, size, _score_to_db(best_score), best_gsn, best_complexity),
            )

    def save_run_config(
        self,
        num_islands: int,
        complexity_bin_size: int,
    ) -> None:
        """Store structural config in the DB (used for resume validation)."""
        self._conn.execute(
            """INSERT OR REPLACE INTO run_config (
                id, num_islands, complexity_bin_size
            ) VALUES (1, ?, ?)""",
            (num_islands, complexity_bin_size),
        )

    # ---- Read operations -------------------------------------------------

    def load_bodies_by_ids(self, gsn_list: list[int]) -> dict[int, tuple[str, float | None]]:
        """Return {gsn: (body, score)} for the given global_sample_nums."""
        if not gsn_list:
            return {}
        placeholders = ",".join("?" * len(gsn_list))
        rows = self._conn.execute(
            "SELECT global_sample_num, func_body, score FROM programs"
            f" WHERE global_sample_num IN ({placeholders})",
            gsn_list,
        ).fetchall()
        return {gsn: (body, score) for gsn, body, score in rows}

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
            "best_score": _score_from_db(row[2]),
            "success_count": row[3],
            "failed_count": row[4],
            "tot_sample_time": row[5],
            "tot_evaluate_time": row[6],
            "tot_token_cost": row[7],
        }

    def load_island_stats(self) -> list[dict]:
        """Returns list of dicts with island_id, size, best_score, best_gsn, best_complexity."""
        rows = self._conn.execute(
            "SELECT island_id, size, best_score, best_gsn, best_complexity"
            " FROM island_stats ORDER BY island_id",
        ).fetchall()
        return [
            {
                "island_id": r[0],
                "size": r[1],
                "best_score": _score_from_db(r[2]),
                "best_gsn": r[3],
                "best_complexity": r[4],
            }
            for r in rows
        ]

    def load_run_config(self) -> dict | None:
        """Load the stored run_config row. Returns dict or None if not stored."""
        row = self._conn.execute(
            "SELECT num_islands, complexity_bin_size"
            " FROM run_config WHERE id = 1",
        ).fetchone()
        if row is None:
            return None
        return {
            "num_islands": row[0],
            "complexity_bin_size": row[1],
        }

    def validate_config(self, num_islands: int, complexity_bin_size: int) -> None:
        """Validate that structural config matches stored checkpoint.

        Raises:
            ValueError: if num_islands or complexity_bin_size differ.
        """
        stored = self.load_run_config()
        if stored is None:
            return  # No stored config to validate against
        if stored["num_islands"] != num_islands:
            raise ValueError(
                f"Config mismatch on resume: num_islands={num_islands} "
                f"but checkpoint has num_islands={stored['num_islands']}. "
                f"Changing num_islands mid-run is not supported."
            )
        if stored["complexity_bin_size"] != complexity_bin_size:
            raise ValueError(
                f"Config mismatch on resume: complexity_bin_size={complexity_bin_size} "
                f"but checkpoint has complexity_bin_size={stored['complexity_bin_size']}. "
                f"Changing complexity_bin_size mid-run is not supported."
            )

    @property
    def is_populated(self) -> bool:
        row = self._conn.execute("SELECT COUNT(*) FROM global_stats").fetchone()
        return row[0] > 0


# ---------------------------------------------------------------------------
# Separate logs database
# ---------------------------------------------------------------------------

_LOGS_SCHEMA_SQL = """\
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS program_logs (
    global_sample_num   INTEGER PRIMARY KEY,
    llm_response_text   TEXT,
    error_type          TEXT,
    error_message       TEXT,
    eval_output         TEXT,
    complexity          INTEGER,
    complexity_detail   TEXT,
    optimized_params    TEXT,
    sample_time         REAL,
    evaluate_time       REAL,
    token_usage_input   INTEGER,
    token_usage_output  INTEGER,
    token_cost          REAL
);
"""


class LogsDB:
    """SQLite-backed store for program log columns (write-only at runtime)."""

    def __init__(self, db_path: str) -> None:
        try:
            if db_path != ":memory:":
                os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            self._conn = sqlite3.connect(db_path, timeout=30)
            self._conn.executescript(_LOGS_SCHEMA_SQL)
            self._ensure_schema_compat()
        except (sqlite3.Error, OSError) as e:
            raise OSError(f"Failed to open logs DB at {db_path}: {e}") from e
        self._db_path = db_path
        logger.info("LogsDB opened at %s", db_path)

    def insert_log(
        self,
        global_sample_num: int,
        llm_response_text: str | None,
        error_type: str | None,
        error_message: str | None,
        eval_output: str | None,
        *,
        complexity: int | None = None,
        complexity_detail: str | None = None,
        optimized_params: str | None = None,
        sample_time: float | None = None,
        evaluate_time: float | None = None,
        token_usage_input: int | None = None,
        token_usage_output: int | None = None,
        token_cost: float | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO program_logs (
                global_sample_num, llm_response_text,
                error_type, error_message, eval_output,
                complexity, complexity_detail, optimized_params,
                sample_time, evaluate_time,
                token_usage_input, token_usage_output, token_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                global_sample_num, llm_response_text,
                error_type, error_message, eval_output,
                complexity, complexity_detail, optimized_params,
                sample_time, evaluate_time,
                token_usage_input, token_usage_output, token_cost,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("LogsDB closed")
