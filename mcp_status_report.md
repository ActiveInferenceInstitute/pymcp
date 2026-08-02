# PyMCP Implementation Status Report

> Status as of **2026-08-02** — package `pymdp-mcp` 1.0.3, suite **54 passed /
> 0 failed** against `inferactively-pymdp` 1.0.3. See [TODO.md](TODO.md) for the
> live backlog and [docs/README.md](docs/README.md) for the documentation index.

## Overview

PyMCP provides a standardized way for AI agents and applications to interact
with PyMDP's active-inference functionality: the same core
(`PyMDPInterface`) is exposed over the Model Context Protocol (MCP, SSE or
stdio) and as a plain FastAPI REST API. Every call goes to the **unmodified
PyMDP library** — there are no mock or simplified implementations, and the test
suite verifies that wrapper results match direct PyMDP calls.

## Architecture components

1. **MCP server** — `src/main.py` declares the MCP tools
   (`@mcp.tool()`) on the bundled FastMCP implementation in
   `src/mcp/server/fastmcp.py`; served over SSE (default, `HOST`/`PORT`) or
   stdio (`TRANSPORT=stdio`).
2. **FastAPI REST server** — `src/mcp/server/app.py` (`mcp.server.app:app`)
   exposes the same capability set as HTTP endpoints (`/ping`, `/tools`,
   `/agents`, `/environments`, `/sessions`, visualization and analysis routes).
3. **PyMDP interface** — `PyMDPInterface` in `src/mcp/utils.py` is the single
   bridge between the servers and PyMDP: it holds in-memory registries
   (agents, environments, simulation history, sessions), converts JSON ⇄
   NumPy/object arrays, and implements the batched/one-hot conventions of the
   current (immutable, equinox) PyMDP.
4. **Clients** — `src/mcp/client.py` (`MCPClient`, `DirectMCPClient`,
   `get_best_client`) and `src/mcp/client/` (`MCPToolKit`) for programmatic
   access.
5. **Example scripts** — `scripts/` (server launchers, examples, connectivity
   tests; see `scripts/README.md`).

## Functional status

### Working (verified 2026-08-02)

- Package installs and imports (`uv pip install -e .`); `uv run src/main.py`
  starts the MCP server; `uvicorn mcp.server.app:app` starts the REST server.
- Full active-inference loop: `create_agent` → `infer_states` →
  `infer_policies` → `sample_action` → `step_environment`, and end-to-end via
  `run_simulation` (returns `timesteps`, `history`, `free_energy_trace`,
  stable `simulation_id`).
- Gridworld agents/environments with rewards; reset; JSON-in/JSON-out
  conversion (`NumpyEncoder`, `_convert_to_obj_array` / `_convert_from_obj_array`).
- Visualization: belief dynamics, free-energy analysis, generative-model and
  simulation plots (PNG/JSON) via `src/mcp/visualization/`.
- Sessions on the REST server bind an agent to an environment and accumulate
  history for inspection/visualization.
- Security posture: FastAPI CORS restricted to local origins
  (env-configurable via `CORS_ORIGINS`), and the HTTP `visualize_simulation`
  `output_file` is confined to a fixed output directory
  (`MCP_OUTPUT_DIR`, default `./output`) — path traversal closed.

### Testing

- **54 tests / 0 failures** — `uv run python -m pytest tests/` (~35 s).
- The suite exercises `PyMDPInterface` directly and through the MCP layer and
  asserts that wrapper results equal direct PyMDP results (no mocks).
- Legacy organizers (`tests/run_and_organize.sh`, `tests/run_all_tests.py`)
  still run tests by category and organize outputs.

## Known open items

- **Server consolidation** — `scripts/start_mcp_server.py` (a large
  near-duplicate of the FastAPI server) and the hand-rolled FastMCP layer are
  candidates for unification onto the official `mcp`/FastMCP SDK.
- **Resource caps** — `define_generative_model`, `create_gridworld_agent`, and
  `run_simulation` accept unbounded dimensions/grid/steps from JSON input.
- **Lint debt** — cosmetic `ruff` findings (unused imports, whitespace) remain.
- **CI** — no GitHub Actions workflow exists yet.

Tracked with details in [TODO.md](TODO.md).

## References

- [README](README.md) — install, quickstart, server run modes.
- [Docs index](docs/README.md) — architecture, tools reference, integration
  details.
- [PyMDP](https://github.com/infer-actively/pymdp) — the underlying active
  inference library.
