# Agent guide — pymcp

An MCP server and FastAPI backend for PyMDP active-inference agents — expose the PyMDP library to AI agents over Model Context Protocol (SSE/stdio) or REST.

## Working notes

- Repository: `ActiveInferenceInstitute/pymcp` (default branch `main`).
- InstituteOS classification: category `research-software`, maturity `wip`, status `active`.
- License: `MIT`.
- Keep private content in private repositories; never commit credentials,
  personal data, or unverified claims.
- Read [`README.md`](README.md) first, then the nearest nested `AGENTS.md`.
- Prefer small, verified changes; run the declared tasks before handing work back.

## Layout

| Path | What it is |
| --- | --- |
| `src/main.py` | MCP server entry point (FastMCP over SSE / stdio). |
| `src/mcp/` | Core package: `utils.py` (`PyMDPInterface`), servers, clients, environments, visualization. See [`src/README.md`](src/README.md). |
| `tests/` | pytest suite; counts and the `slow` split are generated into [`docs/_generated/gate-facts.md`](docs/_generated/gate-facts.md). See [`tests/README.md`](tests/README.md). |
| `run.sh` | The gate: one arm per command, and the exact commands CI runs. See [`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md). |
| `scripts/` | Repo tooling (`run.sh` helpers, doc checks), examples, legacy runners. See [`scripts/README.md`](scripts/README.md). |
| `docs/` | Documentation tree; hub at [`docs/README.md`](docs/README.md), flat index at [`docs/documentation-index.md`](docs/documentation-index.md). |
| `.aii/` | InstituteOS sidecar metadata. |

## Tasks

| Task | Command | Working directory |
| --- | --- | --- |
| `setup` | `./run.sh setup` | `.` |
| `test` (fast) | `./run.sh test` | `.` |
| `test-full` | `./run.sh test-full` | `.` |
| `lint` | `./run.sh lint` | `.` |
| `docs` | `./run.sh docs` | `.` |
| `gnn` | `./run.sh gnn` | `.` |
| `all` | `./run.sh all` | `.` |

Install the environment first with `./run.sh setup`
(`uv sync --frozen --python 3.12`). Arms, CI job keys and the pinned GNN
conformance check are documented in [`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md);
measured test counts and per-arm wall times are generated into
[`docs/_generated/gate-facts.md`](docs/_generated/gate-facts.md) by
`scripts/gate_facts.py`. `ruff` is configured with `fix = true`, so the `lint`
arm passes `--no-fix` to avoid rewriting the tree.

## InstituteOS sidecar

This repository carries a committed [`.aii/`](.aii/README.md) sidecar with curated
InstituteOS metadata (`aii-sidecar/v1`): category, maturity, ecosystem relations,
key artifacts, provenance, and portable tasks. It is read by
[InstituteOS](https://github.com/ActiveInferenceInstitute/InstituteOS) tooling and
does not affect the project's build or runtime. Regenerating the sidecar docs
rewrites only `.aii/README.md`, `.aii/AGENTS.md`, `.aii/docs/README.md`,
`.aii/docs/AGENTS.md`, and `.aii/docs/INTEGRATION.md`.
