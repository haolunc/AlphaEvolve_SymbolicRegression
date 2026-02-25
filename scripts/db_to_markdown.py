#!/usr/bin/env python3
"""Generate a markdown report from AlphaEvolve SR checkpoint databases.

Reads all tables (except ``programs``) from checkpoint.db and
checkpoint_logs.db and writes a human-readable report that mirrors the
statistics shown in TensorBoard.

Usage::

    python scripts/db_to_markdown.py examples/log/0224_5 # default <log_dir>/report.md
    python scripts/db_to_markdown.py examples/log/0224_5 --output custom_report.md
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _fmt_score(val) -> str:
    if val is None:
        return "-inf"
    return f"{val:.10g}"


def _fmt_float(val, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _fmt_int(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:,}"


# ---------------------------------------------------------------------------
# Section generators — each returns a list of markdown lines
# ---------------------------------------------------------------------------

def section_run_config(conn: sqlite3.Connection) -> list[str]:
    row = conn.execute("SELECT * FROM run_config WHERE id = 1").fetchone()
    if row is None:
        return ["## Run Configuration\n", "_No data._\n"]
    return [
        "## Run Configuration\n",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Num Islands | {row['num_islands']} |",
        f"| Complexity Bin Size | {row['complexity_bin_size']} |",
        "",
    ]


def section_global_stats(conn: sqlite3.Connection) -> list[str]:
    row = conn.execute("SELECT * FROM global_stats WHERE id = 1").fetchone()
    if row is None:
        return ["## Global Summary\n", "_No data._\n"]
    return [
        "## Global Summary\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Samples (`global_sample_num`) | {_fmt_int(row['global_sample_num'])} |",
        f"| Best Score | {_fmt_score(row['best_score'])} |",
        f"| Success Count | {_fmt_int(row['success_count'])} |",
        f"| Failed Count | {_fmt_int(row['failed_count'])} |",
        f"| Last Reset Step | {_fmt_int(row['last_reset_step'])} |",
        f"| Total Sample Time (s) | {_fmt_float(row['tot_sample_time'], 2)} |",
        f"| Total Evaluate Time (s) | {_fmt_float(row['tot_evaluate_time'], 2)} |",
        f"| Total Token Cost ($) | {_fmt_float(row['tot_token_cost'], 6)} |",
        "",
    ]


def section_island_stats(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT * FROM island_stats ORDER BY island_id"
    ).fetchall()
    if not rows:
        return ["## Island Stats\n", "_No data._\n"]
    lines = [
        "## Island Stats\n",
        "| Island | Size | Best Score | Best GSN | Best Complexity |",
        "|--------|------|------------|----------|-----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['island_id']} "
            f"| {_fmt_int(r['size'])} "
            f"| {_fmt_score(r['best_score'])} "
            f"| {_fmt_int(r['best_gsn'])} "
            f"| {_fmt_int(r['best_complexity'])} |"
        )
    lines.append("")
    return lines


def section_pareto_front(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT * FROM pareto_front ORDER BY complexity_bin"
    ).fetchall()
    if not rows:
        return ["## Pareto Front\n", "_No data._\n"]
    lines = [
        "## Pareto Front\n",
        "| Complexity Bin | Score | GSN |",
        "|----------------|-------|-----|",
    ]
    for r in rows:
        lines.append(
            f"| {r['complexity_bin']} "
            f"| {_fmt_score(r['score'])} "
            f"| {_fmt_int(r['global_sample_num'])} |"
        )
    lines.append("")
    return lines


def section_island_bins(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT island_id, complexity_bin, COUNT(*) AS cnt "
        "FROM island_bins GROUP BY island_id, complexity_bin "
        "ORDER BY island_id, complexity_bin"
    ).fetchall()
    if not rows:
        return ["## Island Bins\n", "_No data._\n"]
    lines = [
        "## Island Bins\n",
        "| Island | Complexity Bin | # Programs |",
        "|--------|----------------|------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['island_id']} | {r['complexity_bin']} | {r['cnt']} |"
        )
    lines.append("")
    return lines


def section_program_logs(logs_conn: sqlite3.Connection) -> list[str]:
    total = logs_conn.execute("SELECT COUNT(*) FROM program_logs").fetchone()[0]
    if total == 0:
        return ["## Program Logs Analysis\n", "_No data._\n"]

    lines = ["## Program Logs Analysis\n"]

    # --- Error rate breakdown ---
    error_rows = logs_conn.execute(
        "SELECT COALESCE(error_type, 'success') AS etype, COUNT(*) AS cnt "
        "FROM program_logs GROUP BY error_type ORDER BY cnt DESC"
    ).fetchall()
    lines += [
        "### Error Rate Breakdown\n",
        "| Error Type | Count | % |",
        "|------------|-------|---|",
    ]
    for r in error_rows:
        pct = r["cnt"] / total * 100
        lines.append(f"| {r['etype']} | {_fmt_int(r['cnt'])} | {pct:.1f}% |")
    lines.append("")

    # --- Timing stats ---
    timing = logs_conn.execute(
        "SELECT sample_time, evaluate_time FROM program_logs "
        "WHERE sample_time IS NOT NULL OR evaluate_time IS NOT NULL"
    ).fetchall()
    sample_times = [r["sample_time"] for r in timing if r["sample_time"] is not None]
    eval_times = [r["evaluate_time"] for r in timing if r["evaluate_time"] is not None]

    lines += ["### Timing Stats (seconds)\n"]
    if sample_times or eval_times:
        lines += [
            "| Metric | Min | Max | Mean | Median |",
            "|--------|-----|-----|------|--------|",
        ]
        if sample_times:
            lines.append(
                f"| Sample Time "
                f"| {_fmt_float(min(sample_times))} "
                f"| {_fmt_float(max(sample_times))} "
                f"| {_fmt_float(statistics.mean(sample_times))} "
                f"| {_fmt_float(statistics.median(sample_times))} |"
            )
        if eval_times:
            lines.append(
                f"| Evaluate Time "
                f"| {_fmt_float(min(eval_times))} "
                f"| {_fmt_float(max(eval_times))} "
                f"| {_fmt_float(statistics.mean(eval_times))} "
                f"| {_fmt_float(statistics.median(eval_times))} |"
            )
        lines.append("")
    else:
        lines.append("_No timing data._\n")

    # --- Token usage stats ---
    token_rows = logs_conn.execute(
        "SELECT token_usage_input, token_usage_output, token_cost "
        "FROM program_logs "
        "WHERE token_usage_input IS NOT NULL"
    ).fetchall()
    inputs = [r["token_usage_input"] for r in token_rows]
    outputs = [r["token_usage_output"] for r in token_rows]
    costs = [r["token_cost"] for r in token_rows if r["token_cost"] is not None]

    lines += ["### Token Usage\n"]
    if inputs:
        lines += [
            "| Metric | Min | Max | Mean | Total |",
            "|--------|-----|-----|------|-------|",
            f"| Input Tokens "
            f"| {_fmt_int(min(inputs))} "
            f"| {_fmt_int(max(inputs))} "
            f"| {_fmt_float(statistics.mean(inputs), 0)} "
            f"| {_fmt_int(sum(inputs))} |",
            f"| Output Tokens "
            f"| {_fmt_int(min(outputs))} "
            f"| {_fmt_int(max(outputs))} "
            f"| {_fmt_float(statistics.mean(outputs), 0)} "
            f"| {_fmt_int(sum(outputs))} |",
        ]
        lines.append("")
    else:
        lines.append("_No token data._\n")

    lines += ["### Cost\n"]
    if costs:
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Cost ($) | {_fmt_float(sum(costs), 6)} |",
            f"| Mean Cost per Sample ($) | {_fmt_float(statistics.mean(costs), 6)} |",
        ]
        lines.append("")
    else:
        lines.append("_No cost data._\n")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(log_dir: Path) -> str:
    ckpt_db = log_dir / "checkpoint.db"
    logs_db = log_dir / "checkpoint_logs.db"

    if not ckpt_db.exists():
        raise FileNotFoundError(f"checkpoint.db not found in {log_dir}")

    sections: list[str] = [f"# AlphaEvolve SR — Report\n", f"Log directory: `{log_dir}`\n"]

    conn = _connect(ckpt_db)
    try:
        sections += section_run_config(conn)
        sections += section_global_stats(conn)
        sections += section_island_stats(conn)
        sections += section_pareto_front(conn)
        sections += section_island_bins(conn)
    finally:
        conn.close()

    if logs_db.exists():
        logs_conn = _connect(logs_db)
        try:
            sections += section_program_logs(logs_conn)
        finally:
            logs_conn.close()
    else:
        sections += ["## Program Logs Analysis\n", f"_`{logs_db.name}` not found._\n"]

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a markdown report from AlphaEvolve SR checkpoint databases."
    )
    parser.add_argument("log_dir", type=Path, help="Path to the log directory")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output file path (default: <log_dir>/report.md)",
    )
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    output = args.output or (log_dir / "report.md")

    report = generate_report(log_dir)
    output.write_text(report)
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
