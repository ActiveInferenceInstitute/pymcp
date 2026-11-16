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

## Focused contracts and collection

The [test inventory](../tests/README.md) lists the native lifecycle, model,
environment, numerical, transport, renderer/output and owned-process modules.
Use actual pytest collection/outcomes; counting source declarations does not
measure executed parameter cases.

The `slow` marker is a selector, not a permanent count or performance claim.
A full gate runs every collected case. A focused proof names its selection,
retains skips/failures and records its source/runtime/process boundaries.
After a repair, prior passing evidence can be retained while only affected
contracts are rerun; that is not a newly executed full gate.

## Generated facts

Collected counts, per-file breakdowns and arm timings belong to
[`_generated/gate-facts.md`](_generated/gate-facts.md), produced by
`scripts/gate_facts.py`. A focused source receipt does not refresh those
historical producer records. Do not hand-edit generated metrics or infer a
current lint/coverage/GNN pass from earlier results.

## Prepared environment and outputs

Use the existing resolved project interpreter, explicit source selection and
owned off-checkout temporary/cache roots. Dependency provisioning is a
separate authorized step. Follow qualified GitVault execution requirements
where applicable.

Fixtures produce real JSON/images and bounded child processes in isolated
temporary paths. The legacy category organizers, shared-output arrangement,
inert test template and `src/main.py test` command were retired.
See [verification details](../tests/DOCUMENTATION_TESTING.md).

## Adding tests

Add deterministic `tests/test_*.py` consumer regressions, reuse actual project
implementations, and exercise real filesystem/process/renderer boundaries.
Do not copy algorithms, fabricate output files, suppress failures or treat
source-wording assertions as behavioral acceptance.

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
