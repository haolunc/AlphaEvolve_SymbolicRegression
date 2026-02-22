# Problem Specification Format

A problem is defined by a directory containing three files, plus a
separate data directory with training CSV(s).

## Directory Structure

```
specs/<problem_name>/
├── prompt.txt      # Task description for the LLM (free-form text)
├── equation.py     # Seed equation function (the starting point)
└── evaluate.py     # Evaluation harness (scoring + optimization)

data/<problem_name>/
└── train.csv       # Training data (columns become dict keys)
```

## `prompt.txt`

Free-form text describing the problem, background, inputs, constraints,
and philosophy. This becomes the first section of every LLM prompt. It
should be self-contained — the LLM receives no other context about the
problem domain.

## `equation.py` — Seed Equation

Defines exactly **one** top-level function named `equation`. The
signature must accept the data columns as positional arguments plus a
`params` array:

```python
def equation(rho: jnp.ndarray, s: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Computes exchange-correlation energy density (e_xc)."""
    # LDA exchange energy density: proportional to rho^(4/3)
    e_x_lda = params[0] * rho**(4/3)
    # ... enhancement factor, correlation, etc.
    return e_x, e_c
```

Rules:
- Exactly one `def` at the top level
- The function body is the seed that gets evolved
- Parameters are indexed from `params[0]`, `params[1]`, etc.
- Must be parseable by `code_manipulation.text_to_function()`

## `evaluate.py` — Evaluation Harness

Must define an `evaluate(data: dict)` function. This function:
- Receives `data_dict` (columns from `train.csv` as numpy arrays)
- Calls `equation(...)` with the data and a parameter vector
- Runs numerical optimization (e.g., BFGS + CMA-ES) to find optimal `params`
- Returns `(score, optimized_params)` on success, or `None` on failure

The `equation` function referenced inside `evaluate.py` is **replaced at
runtime** — the evaluator splices the evolved function body into the
template before execution. The harness code (imports, data preprocessing,
loss functions, optimizers) remains unchanged.

> **Important markers.** The `evaluate.py` file must import the same
> libraries used by `equation.py` (e.g., `jax.numpy as jnp`). The
> evaluator concatenates `evaluate.py` + the evolved `equation` function
> into a single script that is passed to `exec()`.
