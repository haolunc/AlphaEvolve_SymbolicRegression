# Evaluator

```{contents}
:local:
:depth: 2
```

---

## Overview

The `Evaluator` takes an LLM-generated code sample, parses it into a function body, splices it into the user's evaluation harness, and executes the resulting program in a sandboxed subprocess. It returns an `EvalResult` containing the parsed function, execution result (score, optimized parameters, complexity), and timing.

```{mermaid}
flowchart LR
    SM["SampleMessage"] --> Parse["Parse\n(text_to_function)"]
    Parse --> Splice["Splice\n(_sample_to_program)"]
    Splice --> Sandbox["Sandbox\n(subprocess)"]
    Sandbox --> ER["EvalResult"]
```

---

## Constructor

```python
Evaluator(
    evaluate_code: str,      # Contents of evaluate.py
    seed_function: ParsedFunction,  # Parsed equation.py
    data_dict: dict,         # Training data (from train.csv)
    config: EvaluatorConfig, # timeout_seconds
)
```

The evaluator holds the evaluation harness code, the seed function template (for splicing), and the training data. These are loaded once at startup and reused for every sample.

---

## Initialization

Before the main loop starts, `initialize()` evaluates the **seed function** to establish a baseline score:

```python
initial_result = evaluator.initialize()
# → EvalResult with the seed function's score and complexity
```

This calls `_evaluate_body()` with the seed function's own body — the same path used for LLM samples, ensuring consistent scoring.

---

## Analysis Flow

`analyse(sample_message)` is the main entry point during the evolution loop:

1. **Extract code** — strip markdown fences (```` ```python ... ``` ````) from the LLM response
2. **Parse** — `code_manipulation.text_to_function()` extracts the function body via Python AST
3. **Splice** — `_sample_to_program()` replaces the seed function body with the new body and concatenates with `evaluate.py`
4. **Execute** — run the spliced program in the sandbox with the training data
5. **Score** — if execution succeeds, compute AST complexity and build an `ExecutionResult`
6. **Return** — wrap everything in an `EvalResult`

Returns `None` if parsing or execution fails.

---

## Code Splicing

`_sample_to_program()` creates a runnable script by:

1. Replacing the seed function's body with the LLM-generated body (via `dataclasses.replace`)
2. Concatenating: `evaluate_code + "\n\n" + str(new_function)`

The `evaluate.py` code references `equation(...)` — the spliced function provides that definition. The combined script is passed to `exec()` in the sandbox.

---

## Sandbox

The `Sandbox` class provides process isolation for executing untrusted LLM-generated code.

**Design:**
- Uses a `multiprocessing.Pool(processes=1)` with `spawn` context
- The pool is created **lazily** on the first `run()` call and reused across evaluations
- On timeout or crash, the pool is **killed and transparently recreated**

**Execution flow:**
1. Submit the program + data to the pool via `apply_async`
2. Wait for the result with `timeout_seconds`
3. If the program defines an `evaluate` function, call it with `data_dict`
4. Return `(result, success_bool)`

**Failure modes:**
- `TimeoutError` → kill pool, return `(None, False)`
- Exception during execution → kill pool, return `(None, False)`
- Missing `evaluate` function → return `(None, False)`
