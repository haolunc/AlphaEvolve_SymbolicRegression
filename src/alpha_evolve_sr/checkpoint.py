"""Utilities for saving and loading checkpoints of the database component."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from . import code_manipulation
from .config import RunConfig
from .exceptions import CheckpointError
from .logging_config import get_logger

if TYPE_CHECKING:
    from .profiler import Profiler

logger = get_logger("checkpoint")

_SENTINEL_NEG_INF = "__NEG_INF__"


def _json_encode(value: Any) -> str:
    """JSON-encode *value*, handling -inf and numpy types."""
    def _default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    if isinstance(value, float) and value == float("-inf"):
        return json.dumps(_SENTINEL_NEG_INF)
    if isinstance(value, list):
        value = [_SENTINEL_NEG_INF if (isinstance(v, float) and v == float("-inf")) else v for v in value]
    return json.dumps(value, default=_default)


def _json_decode(text: str) -> Any:
    """JSON-decode *text*, restoring -inf sentinels."""
    value = json.loads(text)
    if value == _SENTINEL_NEG_INF:
        return float("-inf")
    if isinstance(value, list):
        return [float("-inf") if v == _SENTINEL_NEG_INF else v for v in value]
    return value


# ---------------------------------------------------------------------------
# Config persistence (unchanged)
# ---------------------------------------------------------------------------

def save_config(run_config: RunConfig, save_dir: str) -> None:
    """Save a RunConfig to a YAML file in *save_dir*."""
    os.makedirs(save_dir, exist_ok=True)
    run_config.to_yaml(os.path.join(save_dir, "run_config.yaml"))


def load_config(ckpt_dir: str) -> RunConfig:
    """Load a RunConfig from a previously saved YAML file in *ckpt_dir*."""
    yaml_path = os.path.join(ckpt_dir, "run_config.yaml")

    if os.path.exists(yaml_path):
        try:
            config = RunConfig.from_yaml(yaml_path)
            logger.info("Loaded configuration from %s", yaml_path)
            return config
        except Exception as e:
            logger.error("Failed to load config from %s: %s", yaml_path, e)
            raise CheckpointError(f"Failed to load config from {yaml_path}: {e}") from e
    else:
        raise CheckpointError(f"No run_config.yaml found at {yaml_path}")


# ---------------------------------------------------------------------------
# SQLite-backed incremental checkpoint
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS programs (
    global_sample_num   INTEGER PRIMARY KEY,
    island_id           INTEGER,
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

CREATE TABLE IF NOT EXISTS island_programs (
    island_id           INTEGER NOT NULL,
    complexity_bin      INTEGER NOT NULL,
    global_sample_num   INTEGER NOT NULL REFERENCES programs(global_sample_num),
    PRIMARY KEY (island_id, global_sample_num)
);
CREATE INDEX IF NOT EXISTS idx_ip_island_bin ON island_programs(island_id, complexity_bin);

CREATE TABLE IF NOT EXISTS pareto_front (
    global_sample_num   INTEGER PRIMARY KEY REFERENCES programs(global_sample_num)
);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profiler_stats (
    id                        INTEGER PRIMARY KEY CHECK (id = 1),
    best_score                REAL    NOT NULL DEFAULT 0,
    best_program_sample_order INTEGER,
    best_program_str          TEXT,
    success_count             INTEGER NOT NULL DEFAULT 0,
    failed_count              INTEGER NOT NULL DEFAULT 0,
    tot_sample_time           REAL    NOT NULL DEFAULT 0.0,
    tot_evaluate_time         REAL    NOT NULL DEFAULT 0.0,
    tot_token_cost            REAL    NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS profiler_per_complexity (
    complexity          INTEGER PRIMARY KEY,
    best_score          REAL    NOT NULL,
    best_program_str    TEXT    NOT NULL,
    best_sample_order   INTEGER NOT NULL
);
"""


class CheckpointDB:
    """SQLite-backed incremental checkpoint store."""

    def __init__(self, db_path: str) -> None:
        try:
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
        island_id: int | None = None,
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
                global_sample_num, island_id, func_name, func_args, func_body,
                func_return_type, func_docstring, score, optimized_params,
                complexity, complexity_detail, sample_time, evaluate_time,
                token_usage_input, token_usage_output, token_cost,
                llm_response_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                program.global_sample_nums,
                island_id,
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

    def insert_island_program(
        self, island_id: int, complexity_bin: int, global_sample_num: int,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO island_programs (island_id, complexity_bin, global_sample_num) VALUES (?, ?, ?)",
            (island_id, complexity_bin, global_sample_num),
        )

    def delete_island_programs(self, island_id: int, evicted_ids: list[int]) -> None:
        if not evicted_ids:
            return
        placeholders = ",".join("?" * len(evicted_ids))
        self._conn.execute(
            f"DELETE FROM island_programs WHERE island_id = ? AND global_sample_num IN ({placeholders})",
            [island_id, *evicted_ids],
        )

    def reset_island(self, island_id: int) -> None:
        self._conn.execute(
            "DELETE FROM island_programs WHERE island_id = ?", (island_id,),
        )

    def replace_pareto_front(self, front_ids: list[int]) -> None:
        self._conn.execute("DELETE FROM pareto_front")
        self._conn.executemany(
            "INSERT INTO pareto_front (global_sample_num) VALUES (?)",
            [(fid,) for fid in front_ids],
        )

    def save_metadata(self, key: str, value: Any) -> None:
        encoded = _json_encode(value)
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, encoded),
        )

    def save_profiler_stats(self, profiler: Profiler) -> None:
        stats = profiler.get_stats_snapshot()
        self._conn.execute(
            """INSERT OR REPLACE INTO profiler_stats (
                id, best_score, best_program_sample_order, best_program_str,
                success_count, failed_count, tot_sample_time, tot_evaluate_time,
                tot_token_cost
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                stats["best_score"] if stats["best_score"] != float("-inf") else None,
                stats["best_program_sample_order"],
                stats["best_program_str"],
                stats["success_count"],
                stats["failed_count"],
                stats["tot_sample_time"],
                stats["tot_evaluate_time"],
                stats["tot_token_cost"],
            ),
        )

    def save_profiler_per_complexity(
        self, complexity: int, score: float, prog_str: str, order: int,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO profiler_per_complexity
               (complexity, best_score, best_program_str, best_sample_order)
               VALUES (?, ?, ?, ?)""",
            (complexity, score, prog_str, order),
        )

    # ---- Read operations -------------------------------------------------

    def load_programs(self) -> dict[int, code_manipulation.EvaluatedProgram]:
        rows = self._conn.execute("SELECT * FROM programs").fetchall()
        programs: dict[int, code_manipulation.EvaluatedProgram] = {}
        for row in rows:
            (
                gsn, _island_id, func_name, func_args, func_body,
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

    def load_island_memberships(self) -> dict[int, dict[int, list[int]]]:
        """Returns ``{island_id: {complexity_bin: [global_sample_num, ...]}}``."""
        rows = self._conn.execute(
            "SELECT island_id, complexity_bin, global_sample_num FROM island_programs ORDER BY rowid",
        ).fetchall()
        result: dict[int, dict[int, list[int]]] = {}
        for island_id, complexity_bin, gsn in rows:
            result.setdefault(island_id, {}).setdefault(complexity_bin, []).append(gsn)
        return result

    def load_pareto_front_ids(self) -> list[int]:
        rows = self._conn.execute("SELECT global_sample_num FROM pareto_front").fetchall()
        return [r[0] for r in rows]

    def load_metadata(self) -> dict[str, Any]:
        rows = self._conn.execute("SELECT key, value FROM metadata").fetchall()
        return {k: _json_decode(v) for k, v in rows}

    def load_profiler_stats(self) -> dict | None:
        row = self._conn.execute(
            """SELECT best_score, best_program_sample_order, best_program_str,
                      success_count, failed_count, tot_sample_time,
                      tot_evaluate_time, tot_token_cost
               FROM profiler_stats WHERE id = 1""",
        ).fetchone()
        if row is None:
            return None
        return {
            "best_score": row[0] if row[0] is not None else float("-inf"),
            "best_program_sample_order": row[1],
            "best_program_str": row[2],
            "success_count": row[3],
            "failed_count": row[4],
            "tot_sample_time": row[5],
            "tot_evaluate_time": row[6],
            "tot_token_cost": row[7],
        }

    def load_profiler_per_complexity(self) -> dict[int, tuple[float, str, int]]:
        rows = self._conn.execute(
            "SELECT complexity, best_score, best_program_str, best_sample_order FROM profiler_per_complexity",
        ).fetchall()
        return {c: (score, prog_str, order) for c, score, prog_str, order in rows}

    @property
    def is_populated(self) -> bool:
        row = self._conn.execute("SELECT COUNT(*) FROM metadata").fetchone()
        return row[0] > 0
