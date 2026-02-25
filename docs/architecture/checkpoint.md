# Checkpoint Persistence

```{contents}
:local:
:depth: 2
```

---

## Overview

The checkpoint layer provides crash-resilient persistence for the evolutionary search. It consists of **two SQLite databases** with distinct roles:

| Database | Class | File | Role |
|----------|-------|------|------|
| **CheckpointDB** | `CheckpointDB` | `checkpoint.db` | Structured state for resume |
| **LogsDB** | `LogsDB` | `checkpoint_logs.db` | Debug/analysis logs |

Both live in `save_ckpt_dir` (defaults to `log_dir`).

---

## Why Two Databases

A single database would force large, write-heavy columns (LLM response text, eval output) into the same transaction as the small, structured checkpoint data. This creates two problems:

1. **Transaction blocking** -- writing multi-KB text blobs slows down the critical-path checkpoint transaction.
2. **Database bloat** -- checkpoint.db should stay small for fast copy-on-resume; log data can grow unbounded.

The split keeps CheckpointDB lean (fast resume, fast copy) while LogsDB handles bulk writes with immediate per-row commits.

---

## SQLite Configuration

Both databases are opened with the same pragmas:

```sql
PRAGMA journal_mode = WAL;    -- Write-Ahead Logging for concurrent reads
PRAGMA synchronous = NORMAL;  -- Balanced durability/performance
PRAGMA foreign_keys = ON;     -- CheckpointDB only
```

WAL mode allows the pipeline to read (e.g., `load_bodies_by_ids` for prompt generation) without blocking writes.

---

## CheckpointDB Schema

```{mermaid}
erDiagram
    programs {
        INTEGER global_sample_num PK
        TEXT func_body "NOT NULL"
        REAL score
    }

    island_bins {
        INTEGER island_id PK
        INTEGER complexity_bin PK
        INTEGER global_sample_num PK
        REAL score "NOT NULL"
    }

    pareto_front {
        INTEGER complexity_bin PK
        REAL score "NOT NULL"
        INTEGER global_sample_num "NOT NULL"
    }

    global_stats {
        INTEGER id PK "CHECK(id=1)"
        INTEGER global_sample_num
        INTEGER last_reset_step
        REAL best_score
        INTEGER success_count
        INTEGER failed_count
        REAL tot_sample_time
        REAL tot_evaluate_time
        REAL tot_token_cost
    }

    island_stats {
        INTEGER island_id PK
        INTEGER size
        REAL best_score
        INTEGER best_gsn
        INTEGER best_complexity
    }

    run_config {
        INTEGER id PK "CHECK(id=1)"
        INTEGER num_islands "NOT NULL"
        INTEGER complexity_bin_size "NOT NULL"
    }
```

### Table roles

| Table | Category | Description |
|-------|----------|-------------|
| `programs` | **Raw** | Append-only store of every evaluated program (body + score). Irreplaceable. |
| `island_bins` | **Derived** | Snapshot of in-memory `Island._bins`. Batch-replaced on checkpoint. |
| `pareto_front` | **Derived** | Current Pareto-optimal entries. Batch-replaced on checkpoint. |
| `global_stats` | **Derived** | Singleton row with aggregate counters. Batch-replaced on checkpoint. |
| `island_stats` | **Derived** | Per-island summary (size, best score/gsn/complexity). Batch-replaced. |
| `run_config` | **Structural** | Singleton row storing `num_islands` and `complexity_bin_size` for resume validation. Written once on fresh start. |

---

## LogsDB Schema

A single table for debug and analysis data:

| Column | Type | Description |
|--------|------|-------------|
| `global_sample_num` | `INTEGER PK` | Foreign key to `programs` (conceptual, not enforced) |
| `llm_response_text` | `TEXT` | Raw LLM output (potentially multi-KB) |
| `error_type` | `TEXT` | `"execution"`, `"timeout"`, or `NULL` |
| `error_message` | `TEXT` | Error string from sandbox |
| `eval_output` | `TEXT` | Captured stdout/stderr from eval code |
| `complexity` | `INTEGER` | AST complexity score |
| `complexity_detail` | `TEXT` | JSON breakdown of complexity components |
| `optimized_params` | `TEXT` | JSON-encoded optimized parameters |
| `sample_time` | `REAL` | LLM sampling wall-clock time |
| `evaluate_time` | `REAL` | Sandbox evaluation wall-clock time |
| `token_usage_input` | `INTEGER` | Input token count |
| `token_usage_output` | `INTEGER` | Output token count |
| `token_cost` | `REAL` | Estimated API cost |

### Migration

`LogsDB._ensure_schema_compat()` runs on open and adds any missing columns (e.g., `complexity` was added after initial release). This makes LogsDB backward-compatible with older DB files.

---

## Checkpoint Strategy

The checkpoint design separates **raw data** (written every step) from **derived data** (batch-replaced periodically):

```{mermaid}
flowchart LR
    subgraph "Every step"
        IP["insert_program()"] --> Programs["programs table"]
        IL["insert_log()"] --> Logs["program_logs table"]
    end

    subgraph "Every checkpoint_interval steps"
        CD["_checkpoint_derived()"] --> IB["island_bins"]
        CD --> PF["pareto_front"]
        CD --> GS["global_stats"]
        CD --> IS["island_stats"]
    end
```

- **`programs`** is append-only -- every evaluated program is written immediately (inside the main transaction). This is the irreplaceable raw data.
- **Derived tables** (`island_bins`, `pareto_front`, `global_stats`, `island_stats`) are DELETE + re-INSERT every `checkpoint_interval` steps (default 10). They can be rebuilt from `programs` + in-memory state on crash, so staleness between checkpoints is acceptable.
- **`program_logs`** is written immediately with auto-commit (separate DB, separate transaction).

### Score encoding

SQLite has no `-inf` literal. The helper functions `_score_to_db()` and `_score_from_db()` convert between Python `float("-inf")` and SQL `NULL`.

---

## Resume Flow

When `ProgramsDatabase.restore_or_create()` detects a populated checkpoint:

```{mermaid}
sequenceDiagram
    participant F as restore_or_create()
    participant CDB as CheckpointDB
    participant DB as ProgramsDatabase

    F->>CDB: validate_config(num_islands, complexity_bin_size)
    Note over CDB: Raises ValueError on mismatch

    alt save_ckpt_dir != resume_path
        F->>F: shutil.copy2(resume_db, dest_db)
        F->>F: Copy WAL/SHM if present
    end

    F->>DB: __init__(checkpoint_db=...)
    F->>DB: _restore_from_db()
    DB->>CDB: load_island_index()
    Note over DB: Populate Island._bins

    DB->>CDB: load_pareto_front()
    Note over DB: Rebuild ParetoEntry list

    DB->>CDB: load_global_stats()
    Note over DB: Restore counters

    DB->>CDB: load_island_stats()
    Note over DB: Restore per-island best scores
```

**Config validation**: `validate_config()` compares the current `num_islands` and `complexity_bin_size` against the stored `run_config` row. These are structural parameters -- changing them mid-run would corrupt the island index. A mismatch raises `ValueError`.

---

## Transaction Safety

`CheckpointDB.transaction()` is a context manager wrapping `BEGIN` / `COMMIT` / `ROLLBACK`:

```python
with checkpoint_db.transaction():
    db.insert_program(program)
    # ... more writes ...
# COMMIT on success, ROLLBACK on exception
```

- All writes in `_register_and_persist()` share a single transaction -- either all succeed or none do.
- `LogsDB` commits immediately per `insert_log()` call -- log loss is acceptable, and keeping it outside the main transaction avoids blocking.
- The `_checkpoint_derived()` batch operation (DELETE + re-INSERT for derived tables) also runs inside a transaction, ensuring atomicity of the snapshot.
