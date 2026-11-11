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
| `tests/` | pytest suite (54 tests). See [`tests/README.md`](tests/README.md). |
| `scripts/` | Example scripts and legacy launchers. See [`scripts/README.md`](scripts/README.md). |
| `docs/` | Documentation tree; index at [`docs/README.md`](docs/README.md). |
| `.aii/` | InstituteOS sidecar metadata. |

## Tasks

| Task | Command | Working directory |
| --- | --- | --- |
| `test` | `uv run python -m pytest tests/` | `.` |
| `lint` | `uv run ruff check --no-fix` | `.` |

Install the environment first with `uv sync --frozen --python 3.12`. The suite is
54 tests; see [`docs/testing.md`](docs/testing.md). `ruff` is configured with
`fix = true`, so use `--no-fix` to inspect without rewriting the tree.

## InstituteOS sidecar

This repository carries a committed [`.aii/`](.aii/README.md) sidecar with curated
InstituteOS metadata (`aii-sidecar/v1`): category, maturity, ecosystem relations,
key artifacts, provenance, and portable tasks. It is read by
[InstituteOS](https://github.com/ActiveInferenceInstitute/InstituteOS) tooling and
does not affect the project's build or runtime. Regenerating the sidecar docs
rewrites only `.aii/README.md`, `.aii/AGENTS.md`, `.aii/docs/README.md`,
`.aii/docs/AGENTS.md`, and `.aii/docs/INTEGRATION.md`.
