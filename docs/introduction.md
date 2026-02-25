# Introduction

> How AlphaEvolve SR finds closed-form equations from data.

```{contents}
:local:
:depth: 2
```

---

## 1. What is Symbolic Regression?

Given data $(X, y)$, find a closed-form expression $f$ such that $f(X) \approx y$.
Unlike black-box models (neural networks, gradient-boosted trees), the result is a human-readable equation -- interpretable and portable.

The search optimizes two competing objectives:

| Objective | Measure | Direction |
|-----------|---------|-----------|
| **Accuracy** | Score (negative loss) | Higher is better |
| **Simplicity** | AST complexity | Lower is better |

---

## 2. High-Level Loop -- LLM-Guided Evolutionary Search

```{mermaid}
flowchart TB
    DB["Program<br/>Database"] -->|"1 · sample programs"| P["Prompt Builder"]
    P -->|"2 · prompt"| LLM["LLM"]
    LLM -->|"3 · candidate code"| E["Evaluator"]
    E -->|"4 · EvalResult"| M["Main Thread"]
    M -->|"5 · register program"| DB
```

1. The **database** samples a handful of previously seen programs and sends them to the **prompt builder**.
2. The prompt builder arranges them into a structured prompt (worst-to-best) and asks the **LLM** to write the next version.
3. The LLM's candidate is executed and scored by the **evaluator** (running in a thread-local `Sandbox`). The evaluator returns an `EvalResult` containing the score but *not* complexity.
4. The **main thread** receives the `EvalResult`, computes AST complexity via `_attach_complexity()`, then calls `register_program()` on the database. The loop repeats.

---

## 3. Prompt Construction

The prompt is assembled from three sources:

| Source | Content |
|--------|---------|
| **User-provided** | Task description (`prompt.txt`) |
| **From database** | Sampled programs -- selected via the island sampling process described in [Section 5](#the-program-database-island-based-evolution) |
| **Auto-generated** | Rules, versioned function names (`equation_v0` ... `equation_vN`), new function skeleton |

Programs are sorted by score in **ascending** order (worst-to-best), priming the LLM to improve on the best. Versioned names are assigned sequentially: `equation_v0` gets the seed function's original docstring; all subsequent versions (`v1`, `v2`, ...) get `"Improved version of equation_vN-1."`.

```
+-----------------------------------------------------------+
|  TASK DESCRIPTION (from prompt.txt)                       |
+-----------------------------------------------------------+
|  RULES:                                                   |
|  - Preserve function signature and docstring structure    |
|  - Only output the full definition of equation_vN         |
|  - Add inline comments explaining physical meaning        |
+-----------------------------------------------------------+
|  PREVIOUS VERSIONS:                                       |
|  ```python                                                |
|  def equation_v0(...):                                    |
|      """<seed docstring>"""                               |
|      # implementation (lowest score)                      |
|                                                           |
|  def equation_v1(...):                                    |
|      """Improved version of equation_v0."""               |
|      # implementation (medium score)                      |
|                                                           |
|  def equation_v2(...):                                    |
|      """Improved version of equation_v1."""               |
|      # implementation (highest score)                     |
|  ```                                                      |
+-----------------------------------------------------------+
|  NOW DEFINE:                                              |
|  ```python                                                |
|  def equation_v3(...):                                    |
|      """Improved version of equation_v2."""               |
|  ```                                                      |
+-----------------------------------------------------------+
```

---

## 4. Scoring -- The Evaluator

- **Score** = value returned by the user's `evaluate()` function (higher is better; typically negative loss).
  Learnable parameters inside the equation are optimized numerically -- gradient-based (BFGS) followed by an evolutionary strategy (CMA-ES) -- to minimize loss on the data before the score is recorded.
- **Complexity** = weighted AST node count, computed by the **main thread** (not the evaluator) via `_attach_complexity()`.
  Each operator (`+`, `*`, `**`, ...), function call (`sin`, `exp`, ...), variable reference, and numeric constant contributes a weight of 1 (customizable).

The evaluator also captures sandbox stdout/stderr (`eval_output`) and stores it in the `EvalResult`. This output is persisted to SQLite for debugging (e.g., CMA-ES progress logs, BFGS warnings).

---

(the-program-database-island-based-evolution)=
## 5. The Program Database -- Island-Based Evolution

The database maintains diversity through a three-level hierarchy:

```{mermaid}
graph TD
    PDB["ProgramsDatabase"] --> I0["Island 0"]
    PDB --> I1["Island 1"]
    PDB --> IN["Island N-1"]
    I0 --> B0["Bin 0<br/>(complexity 0-9)"]
    I0 --> B1["Bin 1<br/>(complexity 10-19)"]
    I0 --> BK["Bin K<br/>..."]
    B0 --> E0["(gsn, score)<br/>(gsn, score)<br/>..."]
```

- **Islands** evolve independently -- prevents premature convergence.
- **Bins** group programs by complexity (bin = `complexity // bin_size`).
- Each bin stores `(global_sample_num, score)` pairs, capped at `cluster_max_size` (lowest-scoring entries pruned).

### 5.1 Pareto Front

The database tracks a global Pareto front over (complexity bin, score).

- **Dominance**: $(c_1, s_1)$ dominates $(c_2, s_2)$ iff $c_1 \le c_2$ and $s_1 \ge s_2$ (strictly better in at least one dimension).
- Maintained sorted by complexity bin; updated on every new registration.
- When Pareto-aware sampling is enabled, the front biases bin selection toward under-performing complexity regions (see [Section 5.2](#two-stage-sampling)).

(two-stage-sampling)=
### 5.2 Two-Stage Sampling

Each prompt is built from programs sampled on a single, randomly chosen island.

**Stage 1 -- Bin selection**: pick $k$ bins ($k$ = `functions_per_prompt`, default 4).

- **Default mode**: uniform random over available bins.
- **Pareto-aware mode** (when $\ge 2$ Pareto entries exist): weight each bin by how far it lags behind the Pareto front:

$$w_i = 1 + \max\!\bigl(0,\; s^*_{\text{nearest}} - s_{\text{best},i}\bigr)$$

where $s^*_{\text{nearest}}$ is the Pareto score of the nearest complexity bin and $s_{\text{best},i}$ is the best score in bin $i$. Weights are then normalized to form a probability distribution. This biases selection toward under-performing complexity regions.

**Stage 2 -- Program selection** within each chosen bin: softmax over scores scaled by temperature $T$:

$$P(i) = \frac{\exp(s_i / T)}{\sum_j \exp(s_j / T)}$$

### 5.3 Temperature Schedule

Sampling temperature follows a cyclic linear decay:

$$T = T_{\text{init}} \times \left(1 - \frac{n \bmod P}{P}\right)$$

| Symbol | Meaning | Default |
|--------|---------|---------|
| $T_{\text{init}}$ | Initial temperature | 0.005 |
| $P$ | Period length (samples) | 200 |
| $n$ | Global sample counter | -- |

At the start of each period the temperature is high (more exploration); it decays to zero (greedy) by the end of the period, then resets.

### 5.4 Island Reset

Every `reset_period` samples (default 700):

1. Rank islands by best score (with small random noise to break ties).
2. Reset the bottom 50% (clear all bins).
3. Re-seed each reset island with the best program from a random surviving island.

This prevents stagnation while preserving the most promising lineages.
