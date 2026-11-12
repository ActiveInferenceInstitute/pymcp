# Run guide — the gate

One entry point, one command per arm: `./run.sh <arm>`. Every arm is a single
command, and that exact command is what `.github/workflows/ci.yml` runs. No gate
command is spelled out a second time in YAML, in this page, or in a script.

## Arms

| Arm | Command | What it runs |
| --- | --- | --- |
| `setup` | `./run.sh setup` | `uv sync --frozen --python 3.12` — install the committed lock into `.venv` |
| `lint` | `./run.sh lint` | `uv run --frozen ruff check --no-fix` |
| `test` *(default)* | `./run.sh test` | `uv run --frozen python -m pytest tests/ -q -m "not slow"` |
| `test-full` | `./run.sh test-full` | `uv run --frozen python -m pytest tests/ -q` |
| `links` | `./run.sh links` | `scripts/check_docs.py` — Markdown relative links + `docs/` index coverage |
| `docs` | `./run.sh docs` | `links`, then `scripts/gate_facts.py --check` (generated facts are current) |
| `gnn` | `./run.sh gnn` | `scripts/gnn_conformance.py` — pinned GNN parser conformance |
| `all` | `./run.sh all` | `lint`, `test-full`, `docs`, `gnn` |

Run `./run.sh --help` for the same list from the script itself.

There is **no `types` arm**: the repository declares no type checker (no mypy,
pyright or pytype dependency, and no configuration or CI step for one). Adding
one is a code change, not a gate wiring change.

`ruff` is configured with `[tool.ruff] fix = true`, so plain `ruff check`
rewrites the tree. The `lint` arm passes `--no-fix`; a gate must not edit the
files it is judging.

## The fast arm

`test` is the loop arm, not a different suite: `pyproject.toml` declares one
marker, `slow`, applied to the eight tests that run a full PyMDP simulation or
visualization pass. `pytest --durations` attributes most of the suite's call
time to those eight; the collected and slow-marked counts are generated into
[`_generated/gate-facts.md`](_generated/gate-facts.md). `test-full` runs them;
CI runs `test-full`.

Generated counts and per-arm wall times live in
[`_generated/gate-facts.md`](_generated/gate-facts.md), produced by
`scripts/gate_facts.py --record` — see
[`_generated/README.md`](_generated/README.md).

## What CI runs

`.github/workflows/ci.yml` has two jobs and no inline gate commands:

| Job key | Display name | Steps |
| --- | --- | --- |
| `tests` | `test suite (python 3.12, frozen lock)` | `setup-python-uv` → `./run.sh setup` → `./run.sh lint` → `./run.sh docs` → `./run.sh test-full` |
| `gnn-conformance` | `gnn conformance (pinned)` | `setup-python-uv` → `./run.sh gnn` |

Both jobs install the toolchain with the shared org composite action
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`
(`python-version: "3.12"`; it installs a pinned uv plus the interpreter and
leaves the job name untouched). The `tests` job syncs the frozen lock exactly
once and runs lint, docs and the full suite in that one environment. The
`gnn-conformance` job installs no project dependency: `./run.sh gnn` builds its
own `.gnn-venv/` (gitignored) for the parser.

> GitHub Actions is billing-locked org-wide, so local runs are the only gate
> that currently executes. `./run.sh all` locally is the substitute for a green
> workflow.

## The GNN arm

`./run.sh gnn` runs `scripts/gnn_conformance.py`, which reads the canonical pin
file `.github/gnn-pin.json` — schema `gnn-importer-pin/v1`, repository
`ActiveInferenceInstitute/Generalized_Notation_Notation`, tag `v5.0.3`,
revision `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1` (recorded 2026-11-12). It
verifies the pin fields, installs that exact commit into `.gnn-venv/`, and
requires the installed `gnn` version to match the pin before parsing anything.

What it parses is documented in [`../.github/GNN.md`](../.github/GNN.md): every
`*.gnn.md` sketch in the repository (currently `models/gridworld_3x3.gnn.md`),
and the pinned upstream `input/gnn_files/basics/static_perception.md` exemplar
when the repository has no sketch. The type-check verdict is compared
explicitly against a clean result (`True` or `0`); the checker's warning
sentinel `2` fails the arm instead of passing it through truthiness.

The arm needs network access the first time it installs the pinned revision;
after that the `.gnn-venv/` is reused and the check is local.

## Refreshing the generated facts

```bash
./run.sh setup
uv run --frozen python scripts/gate_facts.py --record   # re-measures every arm
uv run --frozen python scripts/gate_facts.py --check    # what `./run.sh docs` runs
```

`--record` runs each arm through `./run.sh`, so the recorded wall time describes
the real entry point. It writes `docs/_generated/arms.json` and re-renders
`docs/_generated/gate-facts.md`; `--check` (part of the `docs` arm) fails when
the committed Markdown and the receipt disagree. Do not hand-edit either file.
