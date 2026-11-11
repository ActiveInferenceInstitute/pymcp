# Testing

The pymcp test suite exercises `PyMDPInterface` directly **and** through the MCP
layer. It is the repository's single declared gate and the one CI runs.

## Running the suite

```bash
uv sync --frozen --python 3.12          # install the committed lock (incl. dev group)
uv run --frozen python -m pytest tests/ # canonical command
```

`pytest` collects **54 tests** across seven files:

| File | Collected tests | Focus |
| --- | --- | --- |
| `tests/test_pymdp_interface.py` | 15 | `PyMDPInterface` model construction, inference, action selection |
| `tests/test_mcp_tools.py` | 16 | MCP tool wrappers over the interface |
| `tests/test_additional_functions.py` | 7 | Helper functions and edge cases |
| `tests/test_mcp_full.py` | 5 | End-to-end MCP server flow |
| `tests/test_agent_visualization.py` | 4 | Visualization outputs (PNG + JSON companions) |
| `tests/test_template.py` | 4 | Test scaffolding template |
| `tests/test_belief_dynamics.py` | 3 | Belief-dynamics traces |

The count `54` is what `pytest --collect-only -q tests/` reports; a raw
`grep -c 'def test_'` over `tests/` reports 59 because five matches are not
pytest tests (MCP handler functions defined inside `tests/test_mcp_full.py`,
and one literal `def test_{specific_function}(self)` inside the template string
in `tests/test_template.py`).

> Offline note: the suite requires the resolved environment (`inferactively-pymdp`
> and the other declared dependencies). On a bare interpreter without those
> packages, `pytest` cannot import the modules and collection fails — install the
> lock first with `uv sync`.

## Legacy runners

Before pytest was standardised, two helpers ran and organised tests by category.
They still work when the dependencies are installed:

```bash
./run_and_organize.sh                       # all tests
./run_and_organize.sh --category visualization
./run_and_organize.sh --file test_belief_dynamics.py
./run_and_organize.sh --summary
uv run python run_all_tests.py --category belief_dynamics
```

`run_and_organize.sh` is a thin wrapper over `run_all_tests.py`; the categories
are defined in `run_all_tests.py`'s `category_patterns`.

## Test outputs

Every run writes under `tests/output/` (gitignored): `generative_models/`,
`simulations/`, `belief_dynamics/`, `free_energy/`, `logs/`, and `results/`.
See [`../tests/README.md`](../tests/README.md) and
[`../tests/DOCUMENTATION_TESTING.md`](../tests/DOCUMENTATION_TESTING.md) for the
full category and output reference.

## Adding tests

1. Add a `tests/test_*.py` file (pytest- or unittest-style; both are collected).
2. Reuse the modular code under `src/mcp/` rather than duplicating it.
3. Register the file in `run_all_tests.py`'s `category_patterns` if it should
   run under a named category.

## CI

`.github/workflows/ci.yml` runs two jobs on push and pull request to `main`:

- **`test suite (python 3.12, frozen lock)`** — `uv sync --frozen --python 3.12`
  then `uv run --frozen python -m pytest tests/`.
- **`gnn conformance (pinned)`** — installs the pinned GNN parser
  (`.github/gnn-pin.json`) and validates any repository `*.gnn.md` sketches. See
  [`TODO.md`](../TODO.md) for a known warning-sentinel caveat in this check.

## Lint

`ruff` is a declared dev dependency. Inspect without rewriting the tree:

```bash
uv run ruff check --no-fix
```

`pyproject.toml` sets `[tool.ruff] fix = true`, so plain `ruff check` rewrites
files. See [`TODO.md`](../TODO.md) for the current lint-debt status.
