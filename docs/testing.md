# Testing

The pymcp test suite exercises `PyMDPInterface` directly **and** through the MCP
layer. It is the repository's declared gate and the one CI runs.

## Running the suite

Use the gate, not remembered command lines (see the [Run Guide](RUN_GUIDE.md)):

```bash
./run.sh setup       # install the committed lock (Python 3.12) — once
./run.sh test        # fast arm: the suite minus tests marked `slow`
./run.sh test-full   # every collected test — what CI runs
```

Underneath, both arms are one pytest invocation over `tests/`; `test` adds
`-m "not slow"`. The exact commands live in `run.sh`, never in prose or YAML.

## The `slow` marker

`pyproject.toml` declares one marker, `slow`, and it is applied to the tests
that run a full PyMDP simulation or visualization pass — the eight that
`pytest --durations` shows dominating suite call time (gridworld simulation
visualization, both `visualize_simulation` and `run_simulation` pairs, the
multi-modality agent plot, and the belief-dynamics plot).

`./run.sh test` deselects exactly those; `./run.sh test-full` runs everything.
A benchmark or CI run that needs the visualization artifacts must use
`test-full`.

## How many tests, and in which files

Counts are generated, not maintained here: the collected total, the per-file
breakdown, and the `slow`-marked split are all in
[`_generated/gate-facts.md`](_generated/gate-facts.md), produced by
`scripts/gate_facts.py`. `./run.sh docs` fails if that page and its receipt
disagree, so there is never a second, hand-edited copy to drift.

A raw `grep -c 'def test_'` over `tests/` over-reports the collection: five
matches are not pytest tests (MCP handler functions defined inside
`tests/test_mcp_full.py`, and one literal `def test_{specific_function}(self)`
inside a template string in `tests/test_template.py`).

> Offline note: the suite requires the resolved environment (`inferactively-pymdp`
> and the other declared dependencies). On a bare interpreter without those
> packages, `pytest` cannot import the modules and collection fails — install the
> lock first with `./run.sh setup`.

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

`run_and_organize.sh` (inside `tests/`) is a thin wrapper over
`run_all_tests.py`; the categories are defined in `run_all_tests.py`'s
`category_patterns`. They are conveniences for browsing output by category, not
the gate.

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
4. Mark a test `@pytest.mark.slow` if it runs a full simulation or
   visualization pass, so the fast arm stays fast.

## CI

`.github/workflows/ci.yml` has two jobs, and every step is a `./run.sh` arm:

- **`tests`** — display name `test suite (python 3.12, frozen lock)`: one
  `setup-python-uv` step, one `./run.sh setup`, then `./run.sh lint`,
  `./run.sh docs` and `./run.sh test-full`.
- **`gnn-conformance`** — display name `gnn conformance (pinned)`: the same
  toolchain action, then `./run.sh gnn`. See
  [`../.github/GNN.md`](../.github/GNN.md).

GitHub Actions is billing-locked org-wide, so local runs are the only gate that
executes today; `./run.sh all` is the local equivalent of both jobs.

## Lint

`ruff` is a declared dev dependency and the `lint` arm is a real gate: it exits
non-zero on any finding, so CI fails rather than accumulating debt.

```bash
./run.sh lint     # uv run --frozen ruff check --no-fix
```

`pyproject.toml` sets `[tool.ruff] fix = true`, so plain `ruff check` rewrites
the tree; `--no-fix` is mandatory when judging. `tests/*` and `scripts/*` carry
an `E402` per-file ignore because those files put `src/` on `sys.path` (and pick
the matplotlib backend) before importing project modules.
