#!/usr/bin/env python
"""Inspect a sample from the checkpoint database.

Usage:
    python examples/inspect_sample.py <checkpoint.db> <global_sample_num>

Example:
    python examples/inspect_sample.py runs/my_run/checkpoint.db 42
"""

from __future__ import annotations

import json
import sqlite3
import sys
import textwrap


def fetch_sample(db_path: str, gsn: int) -> dict | None:
    """Fetch all columns for a given global_sample_num from the programs table."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM programs WHERE global_sample_num = ?", (gsn,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def render_markdown(row: dict) -> str:
    """Render a single sample as a markdown report."""
    gsn = row["global_sample_num"]
    lines: list[str] = []
    lines.append(f"# Sample #{gsn}\n")

    # ── Scores & Complexity ──
    lines.append("## Scores & Complexity\n")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| **Score** | `{row['score']}` |")
    lines.append(f"| **Complexity** | `{row['complexity']}` |")
    if row.get("optimized_params"):
        params = json.loads(row["optimized_params"])
        lines.append(f"| **Optimized Params** | `{params}` |")
    if row.get("complexity_detail"):
        detail = json.loads(row["complexity_detail"])
        lines.append(f"| **Complexity Detail** | `{json.dumps(detail, indent=2)}` |")
    lines.append("")

    # ── Timing & Cost ──
    lines.append("## Timing & Cost\n")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| **Sample Time** | `{row.get('sample_time')}` s |")
    lines.append(f"| **Evaluate Time** | `{row.get('evaluate_time')}` s |")
    lines.append(f"| **Input Tokens** | `{row.get('token_usage_input')}` |")
    lines.append(f"| **Output Tokens** | `{row.get('token_usage_output')}` |")
    lines.append(f"| **Token Cost** | `{row.get('token_cost')}` |")
    lines.append("")

    # ── Function Signature ──
    lines.append("## Function\n")
    name = row["func_name"]
    args = row["func_args"]
    ret = row.get("func_return_type") or ""
    ret_str = f" -> {ret}" if ret else ""
    docstring = row.get("func_docstring") or ""

    body = row["func_body"]
    lines.append("```python")
    lines.append(f"def {name}({args}){ret_str}:")
    if docstring:
        lines.append(f'    """{docstring}"""')
    lines.append(textwrap.indent(body, "    "))
    lines.append("```\n")

    # ── LLM Raw Response ──
    llm_text = row.get("llm_response_text")
    if llm_text:
        lines.append("## LLM Response\n")
        lines.append("```")
        lines.append(llm_text)
        lines.append("```\n")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__.strip())
        sys.exit(1)

    db_path = sys.argv[1]
    gsn = int(sys.argv[2])

    row = fetch_sample(db_path, gsn)
    if row is None:
        print(f"No sample found with global_sample_num = {gsn}")
        sys.exit(1)

    print(render_markdown(row))


if __name__ == "__main__":
    main()
