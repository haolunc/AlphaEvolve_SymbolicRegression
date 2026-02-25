# Evaluator

```{contents}
:local:
:depth: 2
```

---

## Overview

The `Evaluator` takes an LLM-generated code sample, parses it into a function body, splices it into the user's evaluation harness, and executes the resulting program in a sandboxed subprocess. It returns an `EvalResult` that always carries a `ParsedFunction` and detailed error information when something goes wrong.

Evaluators run inside **evaluator threads** (a `ThreadPoolExecutor` in `cli.py`). Each thread owns a thread-local `Evaluator` instance, initialized by `_init_eval_thread()`. The only subprocess is inside `Sandbox` (an `mp.Pool(1)`).

```{mermaid}
flowchart LR
    SM["SampleMessage"] --> Parse["Parse\n(text_to_function)"]
    Parse -->|success| Splice["Splice\n(_sample_to_program)"]
    Parse -->|failure| PE["EvalResult\nerror_type='parse'"]
    Splice --> Sandbox["Sandbox\n(mp.Pool subprocess)"]
    Sandbox --> ER["EvalResult"]
```

---

## Analysis Flow

`analyse(sample_message)` is the main entry point during the evolution loop:

1. **Extract code** -- `_extract_python_text()` strips markdown fences (`` ```python ... ``` ``) from the LLM response; falls back to raw text if no fences found
2. **Parse** -- `code_manipulation.text_to_function()` extracts the function body via Python AST
3. **Splice** -- `_sample_to_program()` replaces the seed function body with the new body and concatenates with `evaluate.py`
4. **Execute** -- run the spliced program in the sandbox with the training data
5. **Classify** -- map the sandbox 4-tuple into an `EvalResult` (see [Result Classification](#result-classification))

`analyse()` never raises -- it always returns an `EvalResult`, using `error_type` and `error_message` to report failures.

---

## Parse Error Path

If `text_to_function()` raises any exception, `analyse()` short-circuits immediately -- no sandbox execution occurs:

```python
EvalResult(
    function=self._seed_function,   # fallback to seed
    execution_result=None,
    evaluate_time=None,
    error_type="parse",
    error_message=str(e),
)
```

This is the cheapest failure path: no subprocess spawn, no timeout wait.

---

## Code Splicing

`_sample_to_program(sample_body, evaluate_code, seed_function)` creates a runnable script:

1. Replace the seed function's body with the LLM-generated body via `dataclasses.replace(seed_function, body=sample_body)`
2. Concatenate: `evaluate_code + "\n\n" + str(new_function)`

The `evaluate.py` harness calls `equation(...)` -- the spliced function provides that definition. The combined script is passed to `exec()` inside the sandbox subprocess.

**Returns:** `(new_function, program)` -- the `ParsedFunction` with the new body, and the full program string.

---

## Sandbox

The `Sandbox` class provides process isolation for executing untrusted LLM-generated code.

### Design

- Uses `multiprocessing.Pool(processes=1)` with `spawn` context
- The pool is created **lazily** on the first `run()` call via `_ensure_pool()`
- On timeout or crash, the pool is **killed and transparently recreated** on the next call

This avoids the ~1-3s overhead of spawning a new process for every evaluation.

### Execution flow

```{mermaid}
sequenceDiagram
    participant E as Evaluator
    participant S as Sandbox
    participant P as Subprocess (Pool)

    E->>S: run(program, data_dict, timeout)
    S->>S: _ensure_pool()
    S->>P: apply_async(_execute_in_subprocess, args)
    P->>P: exec(program) + evaluate(data_dict)
    alt success
        P-->>S: (result, True, None, eval_output)
        S-->>E: 4-tuple
    else timeout
        S->>S: clean() -- kill & join pool
        S-->>E: (None, False, "TimeoutError: ...", "")
    else exception / crash
        S->>S: clean() -- kill & join pool
        S-->>E: (None, False, "SandboxError: ...", "")
    end
```

### Return type

`run()` returns a 4-tuple: `(result, success: bool, error_str: str | None, eval_output: str)`

---

## Output Capture

`_capture_fd_output()` is a context manager that captures **all** stdout/stderr at the OS file-descriptor level. This catches output from C extensions (e.g., CMA-ES `disp()`, scipy BFGS) that bypass Python's `sys.stdout`/`sys.stderr`.

**How it works:**

1. Save original fds 1 and 2 (and Python `sys.stdout`/`sys.stderr`)
2. Create a temporary file and redirect OS fds 1 and 2 to it via `os.dup2()`
3. Replace Python streams with new file objects wrapping duplicated fds to the temp file
4. Yield a callable `_get_captured()` that flushes and reads the temp file
5. On exit, restore all original fds and Python streams

The captured text is stored in `EvalResult.eval_output`.

---

## Result Classification

`_evaluate_body()` classifies the sandbox 4-tuple `(run_result, runs_ok, error_str, eval_output)` into an `EvalResult`:

| Path | `runs_ok` | `run_result` | `error_str` | Result |
|---|---|---|---|---|
| `evaluate()` returns value | `True` | value | `None` | `ExecutionResult(score=value[0], optimized_params=value[1])` |
| `evaluate()` returns `None` | `True` | `None` | `None` | `error_type="execution"`, msg = "evaluate() returned None" |
| `evaluate()` raises exception | `False` | `None` | `"TypeError: ..."` | `error_type="execution"` |
| Missing `evaluate` function | `False` | `None` | `"NameError: ..."` | `error_type="execution"` |
| Timeout | `False` | `None` | `"TimeoutError: ..."` | `error_type="timeout"` |
| Sandbox crash | `False` | `None` | `"SandboxError: ..."` | `error_type="execution"` |

When `runs_ok=True` and `run_result` is not `None`, the value is unpacked as `(score, optimized_params)` into an `ExecutionResult`. All other paths produce an `EvalResult` with `execution_result=None` and an appropriate `error_type`/`error_message`.

The timeout vs. execution distinction is determined by checking whether `"TimeoutError"` appears in `error_str`.
