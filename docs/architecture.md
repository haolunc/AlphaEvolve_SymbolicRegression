# Architecture Document — AlphaEvolve Symbolic Regression

> **Version**: 0.2.0 | **Date**: 2026-02-13 | **Status**: As-Is Analysis + Refactoring Proposal

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Component Responsibilities](#2-component-responsibilities)
3. [Data Flow](#3-data-flow)
4. [Class Relationships](#4-class-relationships)
5. [Sequence Diagrams](#5-sequence-diagrams)
6. [Issue Inventory](#6-issue-inventory)
7. [Refactoring Plan](#7-refactoring-plan)

---

## 1. System Overview

<!-- 系统整体是一个 LLM 引导的进化搜索框架，用于符号回归。
     核心循环：Database 提供 prompt → Sampler 调用 LLM 生成候选方程 → Evaluator 沙箱执行+参数优化 → 结果写回 Database -->

```mermaid
graph TB
    subgraph Core Loop
        DB["ProgramsDatabase<br/>(Island-based EA)"]
        SAM["LLM Sampler<br/>(OpenAI / Qwen / Gemini)"]
        EVAL["Evaluator<br/>(Sandbox + Param Optimization)"]

        DB -- "1. Prompt<br/>(historical programs)" --> SAM
        SAM -- "2. Candidate equation<br/>(raw text)" --> EVAL
        EVAL -- "3. Result<br/>(score, params, complexity)" --> DB
    end

    subgraph Infrastructure
        CKPT["Checkpoint<br/>(pickle)"]
        PROF["Profiler<br/>(TensorBoard + JSON)"]
        LOG["Logging<br/>(console + file)"]
    end

    DB --> CKPT
    DB --> PROF
    SAM --> LOG
    EVAL --> LOG

    subgraph Orchestration
        CLI["CLI Entry Point"]
        CTRL["EvolutionController"]
        WK["Workers<br/>(multiprocessing)"]
    end

    CLI --> CTRL
    CLI --> WK
    CTRL --> DB
    WK --> CTRL
    WK --> SAM
    WK --> EVAL

    SPEC["Specification File<br/>(user-defined task)"] -.-> CLI
    DATA["Training Data<br/>(CSV)"] -.-> CLI
```

### Two Execution Modes

<!-- 系统支持两种运行模式，共享核心组件但编排方式不同 -->

| Mode | Entry Point | Parallelism | Use Case |
|------|-------------|-------------|----------|
| **Distributed** | `main_distributed()` | Multiple OS processes via `mp.Process` | Production runs |
| **Non-distributed** | `main_single()` | Single process, sequential | Debugging / small experiments |

---

## 2. Component Responsibilities

<!-- 每个模块的单一职责定义 -->

```mermaid
graph LR
    subgraph "Core Domain"
        CM["code_manipulation.py<br/>─────────────<br/>Python AST parsing<br/>ParsedFunction / EvaluatedProgram<br/>Program dataclass<br/>Token-level code rewriting"]
        DB2["database.py<br/>─────────────<br/>ProgramsDatabase (top-level)<br/>Island (sub-population)<br/>Cluster (complexity-binned)<br/>Prompt construction"]
        EV["evaluator.py<br/>─────────────<br/>Code extraction from LLM output<br/>Sandbox subprocess execution<br/>Result packaging"]
        CX["complexity.py<br/>─────────────<br/>AST-based complexity scoring<br/>Weighted op/func counting"]
        MSG["messages.py<br/>─────────────<br/>SampleMessage<br/>EvalResult<br/>PerfMessage"]
    end

    subgraph "Integration"
        SM["sampler.py<br/>─────────────<br/>LLMProvider interface (ABC)<br/>OpenAI/Qwen/Gemini impls<br/>Retry logic + threading"]
        CTRL2["controller.py<br/>─────────────<br/>EvolutionController<br/>Database lifecycle<br/>Checkpoint orchestration"]
        WK2["workers.py<br/>─────────────<br/>database_worker()<br/>sampler_worker()<br/>evaluator_worker()<br/>monitoring_worker()"]
        CL["cli.py<br/>─────────────<br/>ArgumentParser<br/>main_distributed()<br/>main_single()<br/>Data loading"]
    end

    subgraph "Support"
        CF["config.py<br/>─────────────<br/>5 frozen dataclasses<br/>ProgramsDatabaseConfig<br/>SamplerConfig<br/>EvaluatorConfig<br/>ProfilerConfig<br/>WorkerConfig"]
        CK["checkpoint.py<br/>─────────────<br/>save/load database (pickle)<br/>save/load CLI config (JSON)"]
        PF["profiler.py<br/>─────────────<br/>TensorBoardWriter<br/>SampleLogger (JSON + .py)<br/>StatisticsTracker"]
        LC["logging_config.py<br/>─────────────<br/>configure_logging()<br/>get_logger()<br/>setup_file_logger()"]
        EX["exceptions.py<br/>─────────────<br/>SpecificationError<br/>CheckpointError<br/>LLMProviderError<br/>SandboxTimeoutError<br/>ConfigurationError"]
    end
```

### Module Size Distribution

<!-- 模块大小分布反映了复杂度分配 -->

```
workers.py           ████████████████████████████████████████
controller.py        ██████████████████████████████
profiler.py          ████████████████████████████████████
cli.py               █████████████████████████████████
code_manipulation.py █████████████████████████████████
database.py          ████████████████████████████████
sampler.py           ████████████████████
evaluator.py         ███████████████████
complexity.py        ███████████
messages.py          ███████
config.py            ██████████
checkpoint.py        ██████
logging_config.py    ████
exceptions.py        ██
```

---

## 3. Data Flow

### 3.1 Distributed Mode — Queue-based Pipeline

<!-- 分布式模式通过 5 个 multiprocessing.Queue 连接 4 类 worker 进程 -->

```mermaid
graph LR
    subgraph "Process: Database Worker (1)"
        DBW["database_worker()"]
    end

    subgraph "Process: Sampler Workers (N)"
        SW1["sampler_worker(0)"]
        SW2["sampler_worker(1)"]
        SWN["sampler_worker(N-1)"]
    end

    subgraph "Process: Evaluator Workers (M)"
        EW1["evaluator_worker(0)"]
        EW2["evaluator_worker(1)"]
        EWM["evaluator_worker(M-1)"]
    end

    subgraph "Process: Monitor (1)"
        MON["monitoring_worker()"]
    end

    DBW -- "prompt_queue<br/>Prompt" --> SW1
    DBW -- "prompt_queue" --> SW2
    DBW -- "prompt_queue" --> SWN

    SW1 -- "sample_queue<br/>SampleMessage" --> EW1
    SW2 -- "sample_queue" --> EW2
    SWN -- "sample_queue" --> EWM

    EW1 -- "result_queue<br/>EvalResult" --> DBW
    EW2 -- "result_queue" --> DBW
    EWM -- "result_queue" --> DBW

    EW1 -. "initial_result_queue<br/>(first eval only)" .-> DBW

    SW1 -. "perf_queue<br/>PerfMessage" .-> MON
    EW1 -. "perf_queue" .-> MON
    DBW -. "perf_queue" .-> MON
```

### 3.2 Message Types Flowing Through Queues

<!-- 队列中传递的消息现在全部是 typed dataclasses -->

```mermaid
graph TD
    subgraph "prompt_queue"
        PQ["Prompt (dataclass)<br/>├── code: str<br/>├── version_generated: int<br/>└── island_id: int"]
    end

    subgraph "sample_queue (SampleMessage)"
        SQ["SampleMessage (frozen dataclass)<br/>├── sample: str<br/>├── island_id: int<br/>├── version_generated: int<br/>├── sample_time: float<br/>└── sample_token_usage: tuple[int,int]"]
    end

    subgraph "result_queue (EvalResult)"
        RQ["EvalResult (frozen dataclass)<br/>├── function: ParsedFunction<br/>├── island_id: int|None<br/>├── result_per_test: dict|None<br/>│   ├── score: float<br/>│   ├── optimized_params: list<br/>│   ├── complexity: int<br/>│   └── complexity_detail: Counter<br/>├── sample_time: float|None<br/>├── evaluate_time: float|None<br/>└── sample_token_usage: tuple|None"]
    end

    subgraph "perf_queue (PerfMessage)"
        PERF["PerfMessage (frozen dataclass)<br/>├── worker_type: str<br/>├── worker_id: int<br/>└── stats: dict"]
    end
```

### 3.3 Backpressure Mechanism

<!-- 背压机制通过两个共享计数器实现，防止队列无限增长 -->

```mermaid
sequenceDiagram
    participant DB as Database Worker
    participant S as Sampler Worker
    participant E as Evaluator Worker

    Note over DB,S: prompt_pending_count (shared Value)
    Note over S,E: sample_pending_count (shared Value)

    DB->>DB: Check prompt_pending_count < num_samplers
    DB->>S: Put prompt → prompt_queue
    DB->>DB: prompt_pending_count += 1

    S->>S: prompt_pending_count -= 1
    S->>S: Check sample_pending_count < num_evaluators
    S->>E: Put sample → sample_queue
    S->>S: sample_pending_count += 1

    E->>E: sample_pending_count -= 1
    E->>DB: Put result → result_queue
```

---

## 4. Class Relationships

### 4.1 Core Domain Classes

<!-- 核心领域模型的类图 -->

```mermaid
classDiagram
    class ParsedFunction {
        <<frozen>>
        +str name
        +str args
        +str body
        +str|None return_type
        +str|None docstring
        +list|None decorators
        +__str__() str
        +save_to_file(filepath, append)
    }

    class EvaluatedProgram {
        +ParsedFunction parsed
        +float|None score
        +list|None optimized_params
        +int|None complexity
        +dict|None complexity_detail
        +int|None global_sample_nums
        +float|None sample_time
        +float|None evaluate_time
        +tuple|None token_usage
        +float|None token_cost
        +name: str (property)
        +__str__() str
        +save_to_file(filepath, append)
    }

    class Program {
        <<frozen>>
        +str preface
        +list~ParsedFunction~ functions
        +__str__() str
        +find_function_index(name) int
        +get_function(name) ParsedFunction
    }

    class Prompt {
        +str code
        +int version_generated
        +int island_id
    }

    class ProgramsDatabase {
        -ProgramsDatabaseConfig _config
        -list~Island~ _islands
        -list~float~ _best_score_per_island
        -list~EvaluatedProgram|None~ _best_program_per_island
        -int _last_reset_step
        -Profiler _profiler
        -int _global_sample_nums
        +get_prompt() Prompt
        +register_program(program, island_id, ...)
        +reset_islands()
        +sample_count: int (property)
        +finalize()
    }

    class Island {
        -Program _template
        -str _function_to_evolve
        -int _functions_per_prompt
        -int _complexity_bin_size
        -dict~int,Cluster~ _clusters
        -int _num_programs
        +num_clusters: int (property)
        +num_programs: int (property)
        +register_program(program)
        +get_prompt(temperature) tuple
        -_generate_prompt(implementations) str
    }

    class Cluster {
        -int _complexity_bin
        -int _max_size
        -list~EvaluatedProgram~ _programs
        -list~float~ _scores
        +register_program(program)
        +sample_program(temperature) EvaluatedProgram
        -_prune()
    }

    EvaluatedProgram *-- ParsedFunction : contains (parsed)
    ProgramsDatabase *-- Island : contains N
    Island *-- Cluster : contains by complexity_bin
    Cluster o-- EvaluatedProgram : stores programs
    ProgramsDatabase ..> Prompt : creates
    Island ..> Program : uses template
    Program *-- ParsedFunction : contains
```

### 4.2 Sampler Layer

<!-- Sampler 使用策略模式支持多 LLM 提供商 -->

```mermaid
classDiagram
    class LLMProvider {
        <<abstract>>
        +generate(prompt, config) LLMResponse*
    }

    class LLMResponse {
        +str response_text
        +int input_tokens
        +int output_tokens
    }

    class OpenAIProvider {
        +generate(prompt, config) LLMResponse
    }

    class QwenProvider {
        +generate(prompt, config) LLMResponse
    }

    class GeminiProvider {
        +generate(prompt, config) LLMResponse
    }

    class LLM {
        -int _samples_per_prompt
        -SamplerConfig _config
        -LLMProvider _provider
        -ThreadPoolExecutor _executor
        +draw_samples(prompt) Collection~LLMResponse~|None
        +clean()
        -_query_with_retry(prompt) LLMResponse|None
    }

    LLMProvider <|-- OpenAIProvider
    LLMProvider <|-- QwenProvider
    LLMProvider <|-- GeminiProvider
    LLM --> LLMProvider : uses
    LLM --> SamplerConfig : configured by
    LLMProvider ..> LLMResponse : returns

    note for LLM "ThreadPoolExecutor is persistent\n(created in __init__)"
```

### 4.3 Evaluator + Sandbox

<!-- Evaluator 在沙箱子进程中执行 LLM 生成的代码 -->

```mermaid
classDiagram
    class Evaluator {
        -Program _template
        -str _function_to_evolve
        -str _function_to_run
        -dict _data_dict
        -EvaluatorConfig _config
        -Sandbox _sandbox
        +analyse(sample, island_id, ...) EvalResult|None
        +clean()
    }

    class Sandbox {
        -Pool|None _pool
        +run(program, func, data, timeout) tuple
        +clean()
    }

    Evaluator *-- Sandbox : owns
    Evaluator --> Program : uses template
    Evaluator --> EvaluatorConfig : configured by

    note for Sandbox "Uses mp.get_context('spawn').Pool\nto isolate exec() calls.\nPool is persistent (lazy creation,\nkill-and-recreate on timeout)."
```

### 4.4 Profiler Decomposition

<!-- Profiler 内部已做了良好的职责拆分 -->

```mermaid
classDiagram
    class Profiler {
        -ProfilerConfig _config
        -str _log_dir
        -int _num_samples
        -TensorBoardWriter _tb
        -SampleLogger _sample_logger
        -StatisticsTracker _stats
        +register_function(program)
        +write_best_program_per_c_file()
    }

    class TensorBoardWriter {
        -SummaryWriter _writer
        -str _log_dir
        -ProfilerConfig _config
        +write(num_samples, best_score, ...)
    }

    class SampleLogger {
        -str _json_dir
        +write(program)
    }

    class StatisticsTracker {
        +float best_score
        +dict best_score_per_c
        +int success_count
        +int failed_count
        +float tot_token_cost
        +update(program)
    }

    Profiler *-- TensorBoardWriter
    Profiler *-- SampleLogger
    Profiler *-- StatisticsTracker
```

### 4.5 Configuration Hierarchy

```mermaid
classDiagram
    class ProgramsDatabaseConfig {
        <<frozen>>
        +int functions_per_prompt = 4
        +int num_islands = 10
        +int reset_period = 700
        +float cluster_sampling_temperature_init = 0.005
        +int cluster_sampling_temperature_period = 200
        +int complexity_bin_size = 10
        +int cluster_max_size = 100
    }

    class SamplerConfig {
        <<frozen>>
        +str provider = "qwen"
        +str|None model_name = None
        +float temperature = 1.0
        +int max_retries = 5
        +float retry_delay_seconds = 5.0
        +int request_timeout_seconds = 180
        +tuple cost_per_ktoken = (0.006, 0.024)
    }

    class EvaluatorConfig {
        <<frozen>>
        +int timeout_seconds = 400
    }

    class ProfilerConfig {
        <<frozen>>
        +int log_frequency = 100
        +int complexity_group_size = 5
    }

    class WorkerConfig {
        <<frozen>>
        +int perf_report_interval_seconds = 150
        +int monitor_interval_seconds = 300
    }
```

---

## 5. Sequence Diagrams

### 5.1 Main Evolution Loop (Non-Distributed)

<!-- 非分布式模式的完整执行流程 -->

```mermaid
sequenceDiagram
    participant CLI as cli.main_single()
    participant DB as ProgramsDatabase
    participant ISL as Island
    participant LLM as LLM Sampler
    participant EVAL as Evaluator
    participant SB as Sandbox (subprocess)
    participant PROF as Profiler

    CLI->>EVAL: analyse(initial_body)
    EVAL->>SB: run(program, func, data, timeout)
    SB-->>EVAL: (result, success)
    EVAL-->>CLI: EvalResult
    CLI->>DB: register_program(initial)

    loop Until max_samples reached
        CLI->>DB: get_prompt()
        DB->>ISL: get_prompt(temperature)
        ISL->>ISL: Select clusters → sample programs
        ISL->>ISL: _generate_prompt(sorted implementations)
        ISL-->>DB: (prompt_code, version_generated)
        DB-->>CLI: Prompt(code, version, island_id)

        CLI->>LLM: draw_samples(prompt.code)
        LLM->>LLM: _query_with_retry()
        LLM-->>CLI: [LLMResponse, ...]

        loop For each sample
            CLI->>EVAL: analyse(sample, island_id, version)
            EVAL->>EVAL: _extract_python(text)
            EVAL->>EVAL: _sample_to_program(body, template)
            EVAL->>SB: run(program, func, data, timeout)
            Note over SB: exec() in spawned subprocess<br/>JAX JIT compilation<br/>BFGS + CMA-ES optimization
            SB-->>EVAL: (score, optimized_params) or None
            EVAL->>EVAL: complexity_score(function)
            EVAL-->>CLI: EvalResult or None

            CLI->>DB: register_program(func, island_id, result)
            DB->>ISL: register_program(func)
            ISL->>ISL: Route to Cluster by complexity_bin
            DB->>PROF: register_function(func)
            PROF->>PROF: Write JSON + TensorBoard

            opt Every reset_period samples
                DB->>DB: reset_islands()
                Note over DB: Rank islands by best score<br/>Reset bottom 50%<br/>Seed with founder from survivors
            end
        end
    end

    CLI->>PROF: write_best_program_per_c_file()
```

### 5.2 Distributed Mode — Process Lifecycle

<!-- 分布式模式的进程生命周期管理 -->

```mermaid
sequenceDiagram
    participant MAIN as cli.main_distributed()
    participant MON as Monitor Process
    participant EW as Evaluator Workers [0..M-1]
    participant DBW as Database Worker
    participant SW as Sampler Workers [0..N-1]

    MAIN->>MON: mp.Process(monitoring_worker).start()
    MAIN->>EW: mp.Process(evaluator_worker).start() x M
    Note over EW: evaluator_worker(0) processes<br/>initial program evaluation
    MAIN->>MAIN: sleep(2) — wait for initial eval

    EW-->>DBW: initial_result_queue.put(result)

    MAIN->>DBW: mp.Process(database_worker).start()
    Note over DBW: Blocks on initial_result_queue.get()<br/>or loads from checkpoint

    MAIN->>SW: mp.Process(sampler_worker).start() x N

    loop Until termination_event.is_set()
        DBW->>SW: prompt_queue.put(prompt)
        SW->>EW: sample_queue.put(SampleMessage)
        EW->>DBW: result_queue.put(EvalResult)

        opt global_sample_nums >= max_samples
            DBW->>DBW: termination_event.set()
        end
    end

    MAIN->>MAIN: Close all queues
    MAIN->>MAIN: Join/terminate all processes
```

### 5.3 Prompt Construction Detail

<!-- Prompt 构造是核心竞争力所在，值得详细展开 -->

```mermaid
sequenceDiagram
    participant ISL as Island
    participant CL as Cluster
    participant CM as code_manipulation

    ISL->>ISL: Choose clusters randomly
    loop For each chosen cluster
        ISL->>CL: sample_program(temperature)
        CL->>CL: softmax(scores / temperature)
        CL->>CL: np.random.choice(programs, p=probabilities)
        CL-->>ISL: EvaluatedProgram (with score)
    end

    ISL->>ISL: Sort by score (ascending)

    loop For i, implementation in enumerate(sorted)
        ISL->>ISL: Rename to equation_v{i}
        ISL->>CM: rename_function_calls(old_name → new_name)
        ISL->>ISL: Update docstring for v1+
    end

    ISL->>ISL: Extract task docstring from template.preface
    ISL->>ISL: Build header for equation_v{N} (next version)

    Note over ISL: Final prompt structure:<br/>1. Task description<br/>2. Rules (preserve signature, etc.)<br/>3. Previous versions v0..v{N-1}<br/>4. "Now define: equation_v{N}"
```

### 5.4 Island Reset Mechanism

```mermaid
sequenceDiagram
    participant DB as ProgramsDatabase
    participant ISL_W as Weak Islands (bottom 50%)
    participant ISL_S as Strong Islands (top 50%)

    Note over DB: Triggered every reset_period samples

    DB->>DB: Rank islands by best_score + noise
    DB->>DB: Split into weak/strong halves

    loop For each weak island
        DB->>ISL_W: Replace with new Island()
        DB->>ISL_S: Pick random strong island
        ISL_S-->>DB: best_program (founder)
        DB->>ISL_W: register_program(founder)
        Note over ISL_W: New island starts with<br/>one high-quality seed program
    end
```

---

## 6. Issue Inventory

<!-- 问题清单，按严重程度和模块分类 -->

### P0 — Runtime Bugs

| # | Module | Line | Issue | Impact |
|---|--------|------|-------|--------|
| 1 | `cli.py` | 214 | ~~`main_single` 用 `sample_info[0]` 访问 dict~~  | ✅ RESOLVED — uses `sample_info["response_text"]` + `None` guard |
| 2 | `code_manipulation.py` | 194 | ~~使用 `ast.Str`（Python 3.12 已移除）~~ | ✅ RESOLVED — replaced with `ast.Constant` + `isinstance(value, str)` |

### P1 — Encapsulation Violations

| # | Module | Line | Issue |
|---|--------|------|-------|
| 3 | `workers.py` | 38, 125 | ~~访问 `db._global_sample_nums`~~ | ✅ RESOLVED — uses `db.sample_count` property |
| 4 | `workers.py` | 159 | ~~访问 `db._profiler.write_best_program_per_c_file()`~~ | ✅ RESOLVED — uses `db.finalize()` |
| 5 | `cli.py` | 188, 235 | ~~访问 `database._global_sample_nums`~~ | ✅ RESOLVED — uses `database.sample_count` property |
| 6 | `cli.py` | 241 | ~~访问 `database._profiler.write_best_program_per_c_file()`~~ | ✅ RESOLVED — uses `database.finalize()` |
| 7 | `database.py` | 139 | ~~访问 `island._clusters`, `island._num_programs`~~ | ✅ RESOLVED — uses `island.num_clusters`, `island.num_programs` properties |

### P2 — Design Smells

| # | Issue | Affected Modules |
|---|-------|-----------------|
| 8 | ~~`Function` dataclass 混合了 3 种关注点 (14 fields)~~ | ✅ RESOLVED — split into `ParsedFunction` (frozen, 6 fields) + `EvaluatedProgram` (10 eval/runtime fields) |
| 9 | ~~`Function.__setattr__` 隐式修改入参~~ | ✅ RESOLVED — `value: object` type hint + `isinstance` guards |
| 10 | ~~`exceptions.py` 定义了 5 个异常但从未使用~~ | ✅ RESOLVED — `SpecificationError`, `CheckpointError`, `LLMProviderError` wired up |
| 11 | ~~Queue 消息为 raw dict，无类型合约~~ | ✅ RESOLVED — `SampleMessage`, `EvalResult`, `PerfMessage` dataclasses in `messages.py` |
| 12 | ~~Workers 用 `object` 类型标注~~ | ✅ RESOLVED — `argparse.Namespace`, `code_manipulation.Program`, `logging.Logger` |
| 13 | ~~`SamplerConfig` 未传递给 worker 进程~~ | ✅ RESOLVED — CLI args `--llm_provider/model/temperature` → `SamplerConfig` → workers |
| 14 | ~~`ThreadPoolExecutor` 每次 `draw_samples` 重建~~ | ✅ RESOLVED — persistent executor created in `__init__` |
| 15 | ~~`load_dotenv()` 在模块导入时执行~~ | ✅ RESOLVED — moved to `LLM.__init__` |

### P3 — Reliability / Resource

| # | Issue | Module | Line |
|---|-------|--------|------|
| 16 | ~~`Cluster` 无限增长，无剪枝机制~~ | ~~`database.py`~~ | ✅ RESOLVED — `Cluster` accepts `max_size` (default 100) and prunes lowest-scoring programs |
| 17 | ~~Monitoring worker 硬编码日志路径 `"./logger"`~~ | ~~`workers.py`~~ | ✅ RESOLVED — uses `args.log_path` with `"./logger"` fallback |
| 18 | ~~Profiler 计数逻辑在乱序 sample 下会丢失~~ | ~~`profiler.py`~~ | ✅ RESOLVED — high-water mark tracking; all samples logged unconditionally |
| 19 | Pickle checkpoint 对类重命名/移动脆弱 | `checkpoint.py` | — |

### P4 — Security

| # | Issue | Module |
|---|-------|--------|
| 20 | `exec()` 沙箱无文件系统/网络/内存限制 | `evaluator.py` |

### New Issues (Post-Refactoring)

| # | Sev | Issue | Status |
|---|-----|-------|--------|
| 21 | P2 | `Evaluator.analyse()` returns raw `dict\|None` — last untyped boundary | ✅ RESOLVED — returns `EvalResult` |
| 22 | P2 | Sandbox creates/destroys `mp.Pool` per evaluation (~3600 cycles/run) | ✅ RESOLVED — persistent pool (lazy creation, kill-and-recreate on timeout) |
| 23 | P2 | No Pareto-aware selection — score only, complexity unused in ranking | Being implemented |
| 24 | P2 | Duplicated orchestration in `main_single` vs `database_worker` (~70% overlap) | ✅ RESOLVED — `EvolutionController` |
| 25 | P3 | `_generate_prompt()` does `copy.deepcopy` per prompt | ✅ RESOLVED — uses `dataclasses.replace()` |
| 26 | P3 | `LLM._query_with_retry` returns raw dict, discards typed `LLMResponse` | ✅ RESOLVED — returns `LLMResponse` |
| 27 | P3 | Specification uses DFT-specific prompt templates | Open |
| 28 | P3 | Zero test coverage for `evaluator.py` | ✅ RESOLVED — tests in `test_evaluator.py` |

---

## 7. Refactoring Plan

<!-- 重构方案分三个阶段，每个阶段内保持系统可运行 -->

### Phase 1: Fix Bugs + Improve Type Safety ✅ COMPLETE

<!-- 第一阶段：修 bug、加类型、不改架构 -->

**Goal**: Fix runtime bugs and make the codebase type-safe without changing architecture.

```
Estimated effort: 1-2 days
Risk: Low — no architectural changes
Status: ✅ COMPLETE
```

#### 1.1 Fix P0 Bugs ✅

```python
# cli.py:214 — Fix dict access in main_single
# Before (broken):
eval_result = evaluators.analyse(
    sample_info[0], prompt.island_id, ...
)
# After:
eval_result = evaluators.analyse(
    sample_info["response_text"], prompt.island_id, ...,
    sample_token_usage=(sample_info["input_tokens"], sample_info["output_tokens"]),
)

# code_manipulation.py:194 — Fix ast.Str deprecation
# Before:
isinstance(node.body[0].value, ast.Str)
# After:
isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)
```

#### 1.2 Add Public Properties to Database/Island ✅

```python
# database.py — Add properties
class ProgramsDatabase:
    @property
    def sample_count(self) -> int:
        return self._global_sample_nums

    def finalize(self) -> None:
        """Write final outputs and close resources."""
        self._profiler.write_best_program_per_c_file()

class Island:
    @property
    def num_clusters(self) -> int:
        return len(self._clusters)

    @property
    def num_programs(self) -> int:
        return self._num_programs
```

#### 1.3 Define Message Types ✅

```python
# messages.py
@dataclasses.dataclass(frozen=True)
class SampleMessage:
    sample: str
    island_id: int
    version_generated: int
    sample_time: float
    sample_token_usage: tuple[int, int]

@dataclasses.dataclass(frozen=True)
class EvalResult:
    function: ParsedFunction
    island_id: int | None
    result_per_test: dict | None
    sample_time: float | None
    evaluate_time: float | None
    sample_token_usage: tuple[int, int] | None

@dataclasses.dataclass(frozen=True)
class PerfMessage:
    worker_type: str
    worker_id: int
    stats: dict
```

#### 1.4 Use Custom Exceptions ✅

```python
# evaluator.py — replace generic exceptions
from .exceptions import SandboxTimeoutError
# In Sandbox.run():
except mp.TimeoutError:
    raise SandboxTimeoutError(f"Timed out after {timeout_seconds}s")

# sampler.py
from .exceptions import LLMProviderError
# In _query_with_retry():
raise LLMProviderError("All retries failed")
```

#### 1.5 Pass SamplerConfig Through CLI → Workers ✅

```python
# cli.py — Add sampler config CLI args
parser.add_argument("--llm_provider", default="qwen")
parser.add_argument("--llm_model", default=None)
parser.add_argument("--llm_temperature", type=float, default=1.0)

# workers.py — sampler_worker receives config
def sampler_worker(..., sampler_config: SamplerConfig | None = None):
    llm = sampler_mod.LLM(config=sampler_config)
```

---

### Phase 2: Separate Concerns in Function Dataclass ✅ COMPLETE

<!-- 第二阶段：拆分 Function 的职责，已完成 -->

**Goal**: Split `Function` into `ParsedFunction` (immutable AST data) and `EvaluatedProgram` (runtime results).

```
Estimated effort: 2-3 days
Risk: Medium — touches many modules, needs careful migration
Status: ✅ COMPLETE
```

#### Result

```mermaid
graph LR
    subgraph "Before"
        F1["Function<br/>14 fields<br/>parsing + eval + runtime"]
    end

    subgraph "After (implemented)"
        F2["ParsedFunction<br/>frozen=True<br/>name, args, body,<br/>return_type, docstring,<br/>decorators"]
        F3["EvaluatedProgram<br/>parsed: ParsedFunction<br/>score, optimized_params,<br/>complexity, complexity_detail,<br/>global_sample_nums, sample_time,<br/>evaluate_time, token_usage,<br/>token_cost"]
    end

    F1 -.->|split into| F2
    F1 -.->|split into| F3
    F3 *-- F2
```

- `ParsedFunction` is a frozen dataclass with 6 fields (pure AST data)
- `EvaluatedProgram` composes a `ParsedFunction` with evaluation and runtime metrics
- `Program.functions` is now `list[ParsedFunction]`
- `Cluster` stores `EvaluatedProgram` instances
- `Function` remains as a backward-compatibility alias for `ParsedFunction`

---

### Phase 2.5: EvolutionController ✅ COMPLETE

**Goal**: Extract duplicated orchestration logic from `main_single` and `database_worker` into a shared `EvolutionController`.

```
Status: ✅ COMPLETE — controller.py implements EvolutionController
```

The `EvolutionController` manages database lifecycle (init, register, checkpoint, finalize) and is used by both `cli.py` and `workers.py`, eliminating the ~70% code overlap.

---

### Phase 3: Architectural Improvements

<!-- 第三阶段：架构级改动，需要根据项目方向决定 -->

**Goal**: Improve reliability, extensibility, and operational maturity.

```
Estimated effort: 3-5 days
Risk: Higher — architectural changes, needs design decisions
```

#### 3.1 Cluster Pruning ✅ COMPLETE

```python
class Cluster:
    def __init__(self, complexity_bin, implementation, max_size=100):
        ...

    def register_program(self, program):
        self._programs.append(program)
        self._scores.append(program.score)
        if len(self._programs) > self._max_size:
            self._prune()

    def _prune(self):
        """Remove lowest-scoring programs to stay within max_size."""
        ...
```

Cluster max size is configurable via `ProgramsDatabaseConfig.cluster_max_size` (default 100).

#### 3.2 Pareto-Aware Selection (Next Step)

Currently, selection in `Cluster.sample_program()` ranks by score only; complexity is used only for binning. The next step is to implement Pareto-aware selection that considers both score and complexity when ranking candidates.

#### 3.3 Checkpoint Format Migration

```
Option A: JSON + separate .py files (human-readable, version-resilient)
Option B: SQLite (queryable, atomic writes)
Option C: Keep pickle but add version header + migration support
```

#### 3.4 Prompt Template Extraction

```python
class PromptTemplate:
    """Configurable prompt builder, separating content from construction logic."""
    def __init__(self, task_description: str, rules: list[str]):
        ...

    def build(self, implementations: list[EvaluatedProgram], next_version: int) -> str:
        ...
```

#### 3.5 Enhanced Sandbox (if needed for production)

```
- Use subprocess with resource limits (ulimit)
- Network namespace isolation (Linux only)
- Filesystem restrictions via tempdir + chroot
- Memory limit via cgroups or resource module
```

---

### Refactoring Dependency Graph

<!-- 重构阶段之间的依赖关系 -->

```mermaid
graph TD
    P1_1["1.1 Fix P0 Bugs ✅"] --> P1_2["1.2 Add Public Properties ✅"]
    P1_2 --> P1_3["1.3 Define Message Types ✅"]
    P1_3 --> P1_4["1.4 Use Custom Exceptions ✅"]
    P1_4 --> P1_5["1.5 Pass SamplerConfig ✅"]

    P1_5 --> P2["2. Split Function Dataclass ✅"]
    P2 --> P2_5["2.5 EvolutionController ✅"]
    P2_5 --> P3_1["3.1 Cluster Pruning ✅"]
    P2_5 --> P3_2["3.2 Pareto-Aware Selection"]
    P2 --> P3_3["3.3 Checkpoint Format"]
    P2 --> P3_4["3.4 Prompt Template"]
    P1_1 --> P3_5["3.5 Enhanced Sandbox"]

    style P1_1 fill:#4caf50,color:#fff
    style P1_2 fill:#4caf50,color:#fff
    style P1_3 fill:#4caf50,color:#fff
    style P1_4 fill:#4caf50,color:#fff
    style P1_5 fill:#4caf50,color:#fff
    style P2 fill:#4caf50,color:#fff
    style P2_5 fill:#4caf50,color:#fff
    style P3_1 fill:#4caf50,color:#fff
    style P3_2 fill:#87ceeb
    style P3_3 fill:#98fb98
    style P3_4 fill:#98fb98
    style P3_5 fill:#98fb98
```

**Legend**: Green = completed, Blue = in progress, Light green = planned

---

## Appendix: Module Dependency Matrix

<!-- 模块间依赖矩阵，▶ 表示直接导入 -->

| Module ↓ imports → | config | code_manip | database | sampler | evaluator | complexity | profiler | checkpoint | logging | exceptions | messages | controller |
|--------------------|--------|-----------|----------|---------|-----------|-----------|----------|-----------|---------|-----------|----------|-----------|
| **cli.py** | ▶ | ▶ | | ▶ | ▶ | | | ▶ | ▶ | ▶ | | ▶ |
| **workers.py** | ▶ | ▶ | | ▶ | ▶ | | | | ▶ | | ▶ | ▶ |
| **controller.py** | ▶ | ▶ | ▶ | | | | | ▶ | ▶ | | ▶ | |
| **database.py** | ▶ | ▶ | | | | | ▶ | | ▶ | | | |
| **evaluator.py** | ▶ | ▶ | | | | ▶ | | | ▶ | | ▶ | |
| **sampler.py** | ▶ | | | | | | | | ▶ | ▶ | | |
| **profiler.py** | ▶ | ▶ | | | | | | | ▶ | | | |
| **complexity.py** | | | | | | | | | ▶ | | | |
| **checkpoint.py** | | | TYPE_ONLY | | | | | | ▶ | ▶ | | |
| **messages.py** | | ▶ | | | | | | | | | | |
| **exceptions.py** | | | | | | | | | | | | |
