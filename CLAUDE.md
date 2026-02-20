# AlphaEvolve Symbolic Regression

## Environment

- Python env: `alpha_evolve_sr` via miniforge (Python 3.11)
- Always use full path: `/opt/homebrew/Caskroom/miniforge/base/envs/alpha_evolve_sr/bin/<cmd>`
- Package installed in editable mode: `pip install -e ".[dev]"`

## Development Commands

- Run tests: `/opt/homebrew/Caskroom/miniforge/base/envs/alpha_evolve_sr/bin/pytest tests/ -v`
- Run lint: `/opt/homebrew/Caskroom/miniforge/base/envs/alpha_evolve_sr/bin/ruff check src/`
- Run single test: `/opt/homebrew/Caskroom/miniforge/base/envs/alpha_evolve_sr/bin/pytest tests/test_foo.py::TestClass::test_method -v`

## Project Structure

```
src/alpha_evolve_sr/       # Main package (src layout)
├── cli.py                 # Entry point: main(), main_distributed(), main_single()
├── code_manipulation.py   # AST parsing: Function, Program dataclasses
├── database.py            # Evolutionary algorithm: ProgramsDatabase, Island, Cluster
├── workers.py             # Multiprocessing workers for distributed mode
├── sampler.py             # LLM provider abstraction (OpenAI, Qwen, Gemini)
├── evaluator.py           # Sandbox code execution and evaluation
├── profiler.py            # TensorBoard + JSON logging
├── config.py              # Frozen config dataclasses
├── checkpoint.py          # Pickle-based persistence
├── complexity.py          # AST complexity scoring
├── exceptions.py          # Custom exception hierarchy
├── logging_config.py      # Logging setup
└── messages.py            # Queue message dataclasses

tests/                     # pytest tests
├── conftest.py            # Shared fixtures: sample_function, sample_program, db_config, SAMPLE_SPEC
├── test_code_manipulation.py
├── test_database.py
├── test_checkpoint.py
├── test_config.py
├── test_complexity.py
└── ...
```

## Test Conventions

- Use `tmp_path` pytest fixture for temporary directories (avoids teardown issues with file handles)
- Shared fixtures in `tests/conftest.py`
- Test classes named `Test<Component>`, test methods `test_<behavior>`

## Documentation Style (Markdown)

- Keep text concise — avoid long paragraphs; prefer bullet points and short sentences
- Use visual aids liberally: Mermaid diagrams (flowcharts, sequence diagrams, class diagrams), tables, or math formulas (LaTeX `$...$`) where they clarify better than prose
- Add a **Table of Contents** at the top if a document has more than 3 sections
