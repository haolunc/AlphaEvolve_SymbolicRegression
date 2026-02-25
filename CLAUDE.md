# AlphaEvolve Symbolic Regression

## Environment

- Python env: `alpha_evolve_sr` via miniforge (Python 3.11)
- Always use full path: `/opt/homebrew/Caskroom/miniforge/base/envs/alpha_evolve_sr/bin/<cmd>`
- Package installed in editable mode: `pip install -e ".[dev]"`

## Development Commands

- Run tests: `/opt/homebrew/Caskroom/miniforge/base/envs/alpha_evolve_sr/bin/pytest tests/ -v`
- Run lint: `/opt/homebrew/Caskroom/miniforge/base/envs/alpha_evolve_sr/bin/ruff check src/`
- Run single test: `/opt/homebrew/Caskroom/miniforge/base/envs/alpha_evolve_sr/bin/pytest tests/test_foo.py::TestClass::test_method -v`

## Test Conventions

- Use `tmp_path` pytest fixture for temporary directories (avoids teardown issues with file handles)
- Shared fixtures in `tests/conftest.py`
- Test classes named `Test<Component>`, test methods `test_<behavior>`

## Documentation Style (Markdown)

- Keep text concise — avoid long paragraphs; prefer bullet points and short sentences
- Use visual aids liberally: Mermaid diagrams (flowcharts, sequence diagrams, class diagrams), tables, or math formulas (LaTeX `$...$`) where they clarify better than prose
- Add a **Table of Contents** at the top if a document has more than 3 sections

### Documentation Principles

- **First-principles narrative** — lead with *why* before *what*; don't just list features
- **Diagrams first** — show the picture, then explain it
- **Dataclasses as documentation** — config and message dataclasses define contracts; reference them directly
- **Stay focused** — include only what serves understanding; omit trivia

No backward-compat needed!