# Input Files

```{contents}
:local:
:depth: 2
```

---

## Overview

The pipeline requires four user-provided files -- three in the **problem directory** and one in the **data directory**:

| File | Directory | Purpose |
|------|-----------|---------|
| `train.csv` | `data_folder` | Training data -- columns become `data_dict` keys |
| `prompt.txt` | `problem_dir` | Free-form task description for the LLM |
| `equation.py` | `problem_dir` | Seed function -- the starting point for evolution |
| `evaluate.py` | `problem_dir` | Evaluation harness -- scoring + parameter optimization |

These paths are set in the config YAML via `problem_dir` and `data_folder` (see {doc}`config`).

Loading is handled by `cli.load_problem()`:

```python
def load_problem(problem_dir, data_folder):
    prompt_text = open(problem_dir / "prompt.txt").read()
    evaluate_code = open(problem_dir / "evaluate.py").read()
    seed_function = text_to_function(open(problem_dir / "equation.py").read())
    df = pd.read_csv(data_folder / "train.csv")
    data_dict = {col: df[col].values for col in df.columns}
    return prompt_text, evaluate_code, seed_function, data_dict
```

---

## Training Data (`train.csv`)

`train.csv` is a standard CSV file. Each column becomes a key in the `data_dict` dictionary, with values as NumPy arrays:

```python
df = pd.read_csv("data_folder/train.csv")
data_dict = {col: df[col].values for col in df.columns}
```

For example, a DFT exchange-correlation problem might have columns:

| Column | Meaning |
|--------|---------|
| `rho` | Electron density at each grid point |
| `s` | Reduced density gradient |
| `exc` | Target exchange-correlation energy density |
| `vxc` | Target exchange-correlation potential |
| `weights` | Grid weights for numerical integration |
| `r` | Radial coordinate |
| `atom_index` | Which atom each grid point belongs to |
| `rho_gradient_sign` | Sign of density gradient |

---

## Prompt Text (`prompt.txt`)

`prompt.txt` is free-form text describing the problem, inputs, constraints, and philosophy. It becomes the first section of every LLM prompt -- the LLM receives no other context about the domain.

**Example** -- DFT exchange-correlation:

```{literalinclude} ../../examples/dft_xc/prompt.txt
```

---

## Seed Function (`equation.py`)

Defines exactly **one** top-level function named `equation`. This function body is the starting point that gets evolved by the LLM.

Rules:
- Exactly one `def` at the top level
- Must be parseable by `code_manipulation.text_to_function()`
- Parameters are indexed from `params[0]`, `params[1]`, etc.
- The function signature defines the contract with `evaluate.py`

**Example** -- DFT PBE exchange seed:

```{literalinclude} ../../examples/dft_xc/equation.py
:language: python
```

---

## Evaluate Code (`evaluate.py`)

Defines an `evaluate(data: dict)` function that:

1. Receives `data_dict` (columns from `train.csv` as arrays)
2. Calls `equation(...)` with the data and a parameter vector
3. Runs numerical optimization (e.g., BFGS + CMA-ES) to find optimal `params`
4. Returns `(score, optimized_params)` on success, or `None` on failure

### Code Splicing

The `equation` function referenced inside `evaluate.py` does **not** exist in that file -- it is **spliced in at runtime**. The evaluator concatenates the evaluate code with the evolved equation function into a single script:

```python
# evaluator.py: _sample_to_program()
program = evaluate_code + "\n\n" + str(new_function)
```

```
+-----------------------------------------+
|  evaluate.py  (imports, harness)        |
+-----------------------------------------+
|  def equation(...):                     |
|      <evolved body from LLM>            |
+-----------------------------------------+
         | passed to exec()
```

This means `evaluate.py` must import the same libraries used by `equation.py` (e.g., `jax.numpy as jnp`).

**Example** -- DFT evaluation harness (abbreviated):

```{literalinclude} ../../examples/dft_xc/evaluate.py
:language: python
:lines: 1-36
```
