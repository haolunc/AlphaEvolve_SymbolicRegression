# Program Database

```{contents}
:local:
:depth: 2
```

---

## Overview

`ProgramsDatabase` is the evolutionary engine at the center of the search loop. It owns the population of candidate programs, orchestrates island-based diversity, maintains a Pareto front, and coordinates persistence through two SQLite databases.

The database receives `EvalResult` messages from the evaluator, registers them into the appropriate island bins, and generates prompts by sampling previously seen programs.

---

## Class Hierarchy

```{mermaid}
classDiagram
    class ProgramsDatabase {
        -_islands: list[Island]
        -_pareto_front: list[ParetoEntry]
        -_checkpoint_db: CheckpointDB
        -_logs_db: LogsDB
        -_tb_writer: TensorBoardWriter
        +restore_or_create()$ ProgramsDatabase
        +register_program(eval_result, sample_message)
        +get_prompt() Prompt
        +reset_islands()
        +finalize()
    }

    class Island {
        -_bins: dict[int, list[tuple]]
        +register(gsn, score, complexity)
        +sample_gsns(temperature, pareto_front) list[int]
        +clear()
        +num_programs: int
        +num_clusters: int
    }

    class ParetoEntry {
        <<NamedTuple>>
        cbin: int
        score: float
        gsn: int
    }

    ProgramsDatabase --> "N" Island
    ProgramsDatabase --> "*" ParetoEntry
    ProgramsDatabase --> CheckpointDB
    ProgramsDatabase --> LogsDB
    ProgramsDatabase --> TensorBoardWriter
```

**`ParetoEntry`** -- lightweight `NamedTuple` representing one point on the Pareto front:

| Field | Type | Description |
|-------|------|-------------|
| `cbin` | `int` | `complexity // complexity_bin_size` |
| `score` | `float` | Fitness score |
| `gsn` | `int` | `global_sample_num` (unique program ID) |

**`Island`** -- in-memory index for fast sampling. Bins are `dict[int, list[tuple[int, float]]]` mapping complexity bin to `(gsn, score)` pairs. Each bin is capped at `cluster_max_size`; lowest-scoring entries are pruned when exceeded.

**`ProgramsDatabase`** -- orchestrator that ties together islands, Pareto front, persistence (CheckpointDB + LogsDB), and profiling (TensorBoardWriter).

---

## Lifecycle

### `restore_or_create()` (class method)

The sole factory for `ProgramsDatabase`. Two paths:

```{mermaid}
flowchart TD
    Start["restore_or_create()"] --> HasResume{resume_path<br/>provided?}
    HasResume -->|Yes| LoadDB["Open CheckpointDB"]
    LoadDB --> Populated{DB populated?}
    Populated -->|Yes| Validate["validate_config()"]
    Validate --> CopyDB{save_ckpt_dir !=<br/>resume_path?}
    CopyDB -->|Yes| Copy["Copy DB + WAL/SHM"]
    CopyDB -->|No| Reuse["Reuse existing DB"]
    Copy --> Restore["_restore_from_db()"]
    Reuse --> Restore
    Populated -->|No| Fresh
    HasResume -->|No| Fresh["Fresh start"]
    Fresh --> CreateDB["Create CheckpointDB + LogsDB"]
    CreateDB --> SaveConfig["save_run_config()"]
    SaveConfig --> InitResult{initial_result<br/>provided?}
    InitResult -->|Yes| Register["register_program()"]
    InitResult -->|No| Done["Return DB"]
    Register --> Done
    Restore --> Done
```

- **Resume**: validates that `num_islands` and `complexity_bin_size` match the checkpoint (these are structural and cannot change mid-run). Copies the DB file if the save directory differs from the resume source.
- **Fresh start**: creates new SQLite files and stores the run config for future resume validation.

### `_restore_from_db()`

Reconstructs in-memory state from the checkpoint:

1. `load_island_index()` -- populates each `Island._bins`
2. `load_pareto_front()` -- rebuilds `_pareto_front` as `ParetoEntry` list
3. `load_global_stats()` -- restores counters (`_global_sample_nums`, `_last_reset_step`, `_success_count`, etc.)
4. `load_island_stats()` -- restores `_best_score_per_island` and `_best_program_per_island`

### `finalize()`

Called at shutdown. Performs a final `_checkpoint_derived()` inside a transaction, then closes both databases.

---

## Registration Flow

`register_program(eval_result, sample_message)` is called once per evaluated candidate. Here is the step-by-step trace:

```{mermaid}
sequenceDiagram
    participant CLI as Pipeline
    participant DB as ProgramsDatabase
    participant CDB as CheckpointDB
    participant LDB as LogsDB
    participant ISL as Island
    participant TB as TensorBoardWriter

    CLI->>DB: register_program(eval_result, sample_message)
    DB->>DB: _global_sample_nums += 1
    DB->>DB: Build _RegisteredProgram
    DB->>CDB: BEGIN transaction
    DB->>CDB: insert_program(program)
    DB->>LDB: insert_log(...) [separate DB, auto-commit]
    DB->>ISL: register(gsn, score, complexity)
    DB->>DB: _update_pareto_front(program)
    DB->>DB: _update_profiling_stats(program)
    alt every log_frequency samples
        DB->>TB: write(ProfileMetrics)
    end
    alt every checkpoint_interval steps
        DB->>CDB: _checkpoint_derived()
    end
    alt every reset_period samples
        DB->>DB: reset_islands()
    end
    DB->>CDB: COMMIT
```

Key details:

1. **Increment** `_global_sample_nums` -- the monotonic program counter.
2. **Build** `_RegisteredProgram` -- internal record combining fields from `EvalResult` and `SampleMessage` (score, complexity, timing, token usage, errors).
3. **Transaction** wraps all CheckpointDB writes:
   - `insert_program()` -- append-only raw data (body + score).
   - `LogsDB.insert_log()` -- large columns (LLM text, eval output) go to a separate DB with immediate commit.
   - `Island.register()` -- in-memory index update. If execution succeeded, the program is registered on its origin island (or all islands if no island_id).
   - `_update_pareto_front()` -- check dominance, update front.
4. **Periodic tasks**:
   - `_checkpoint_derived()` every `checkpoint_interval` steps (default 10).
   - `TensorBoardWriter.write()` every `log_frequency` steps (default 25).
   - `reset_islands()` every `reset_period` steps (default 700).

---

## Prompt Generation

`get_prompt()` builds the LLM prompt from sampled programs:

1. **Pick island** -- uniform random over all islands.
2. **Compute temperature** -- cyclic linear decay (see [Two-Stage Sampling](two-stage-sampling)).
3. **Sample GSNs** -- `island.sample_gsns(temperature, pareto_front)` returns `k` global sample numbers.
4. **Load bodies** -- `checkpoint_db.load_bodies_by_ids(gsns)` fetches function bodies and scores from SQLite.
5. **Sort** -- ascending by score (worst-to-best), so the LLM sees a progression.
6. **Generate prompt** -- `_generate_prompt(bodies)`:
   - Wraps each body as a versioned function (`equation_v0`, `equation_v1`, ...).
   - Appends the task description, rules, and a skeleton for the next version.

Returns a `Prompt(code, island_id)` message.

---

## Two-Stage Sampling

Sampling happens inside `Island.sample_gsns()`. For full mathematical details, see the [introduction](two-stage-sampling).

**Stage 1 -- Bin selection**: pick $k$ bins ($k$ = `functions_per_prompt`).

- Default: uniform random.
- Pareto-aware (when $\ge 2$ Pareto entries exist): weight by gap to Pareto front.

  $$w_i = 1 + \max(0,\; s^*_{\text{nearest}} - s_{\text{best},i})$$

  `_pareto_weights()` finds the nearest Pareto cbin via binary search, computes the gap, and normalizes.

**Stage 2 -- Program selection**: within each chosen bin, softmax over scores:

$$P(j) = \frac{\exp(s_j / T)}{\sum_k \exp(s_k / T)}$$

where $T$ is the current temperature.

### Temperature Schedule

$$T = T_{\text{init}} \times \left(1 - \frac{n \bmod P}{P}\right)$$

| Symbol | Config field | Default |
|--------|-------------|---------|
| $T_{\text{init}}$ | `cluster_sampling_temperature_init` | 0.005 |
| $P$ | `cluster_sampling_temperature_period` | 200 |
| $n$ | `_global_sample_nums` | -- |

---

## Island Reset

Every `reset_period` samples (default 700), `reset_islands()`:

1. Rank islands by best score (+ small noise to break ties).
2. Reset the bottom 50% -- `island.clear()`.
3. Re-seed each reset island with the best program from a randomly chosen surviving island.
4. Force a `_checkpoint_derived()` to persist the new state.

This prevents stagnation while preserving the most promising lineages.

---

## Pareto Front Maintenance

`_update_pareto_front(program)` maintains the global Pareto front:

- **Dominance rule**: $(c_1, s_1)$ dominates $(c_2, s_2)$ iff $c_1 \le c_2$ and $s_1 \ge s_2$ (strictly better in at least one dimension).
- If the new program is dominated by any existing entry, it is discarded.
- Otherwise, it is added and any entries it dominates are removed.
- The front is kept sorted by `cbin`.

The Pareto front serves two purposes:
- **Pareto-aware sampling** -- biases bin selection toward under-performing complexity regions.
- **Visualization** -- plotted as a scatter + trend line in TensorBoard (see [profiler](profiler.md#pareto-front-visualization)).

---

## Two-DB Design

`ProgramsDatabase` uses two separate SQLite databases:

| Database | File | Purpose |
|----------|------|---------|
| `CheckpointDB` | `checkpoint.db` | Small, structured data for resume |
| `LogsDB` | `checkpoint_logs.db` | Large columns for post-hoc analysis |

See [Checkpoint Persistence](checkpoint.md) for schema details, transaction safety, and the checkpoint strategy.
