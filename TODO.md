# PyMCP (PyMDP-MCP) — Backlog

Status:      OPEN — suite GREEN (54 passed / 0 failed against inferactively-pymdp 1.0.3)
Owner:       Daniel Ari Friedman (Active Inference Institute)
Last reviewed: 2026-08-01

## Current status (measured 2026-08-01, after the full fix-and-push pass)

- The package now imports, builds, and serves. The real runtime dependency is
  declared (`inferactively-pymdp>=1.0.3`) in `pyproject.toml`; all phantom
  `pymdp-clone` path-inserts were removed (src, tests, scripts).
- `PyMDPInterface` was rewritten for the current (immutable/equinox, batched)
  PyMDP API. The full suite passes: **54 passed / 0 failed** against
  `inferactively-pymdp` 1.0.3 (Python 3.14 venv).
- The documented entry point is real: `uv run src/main.py` starts the MCP
  server (SSE via `TRANSPORT`, `HOST`, `PORT`; stdio via `TRANSPORT=stdio`).
  The FastMCP `run_sse` was made loop-safe (`uvicorn.Server.serve()`).
- Docs were rewritten to a single coherent README; fabricated tool documentation
  (`enable_jax`, `run_tmaze_example`, …) was pruned and explicitly marked as
  non-existent.
- Security: FastAPI CORS defaulted to local origins (env-configurable) with no
  wildcard+credentials; HTTP `visualize_simulation` `output_file` is confined to
  a fixed output dir (path-traversal closed).
- Lint: `ruff check --fix src/ scripts/` reduced errors from 2366 to ~371
  (remaining are mostly unused imports F401, config-marked unfixable, and
  whitespace inside code — cosmetic, not a gate).

## Completed / Closed (2026-08-01)

- [x] **PyMDP dependency declared + phantom `pymdp-clone` removed (MAJOR-1).**
  Added `inferactively-pymdp>=1.0.3` to `pyproject.toml`; removed every
  `pymdp-clone` reference across `src/`, `tests/`, and `scripts/`
  (`setup_pymdp_path.py` now adds only `src/`).
- [x] **PyMDP API drift fixed — suite green (MAJOR-2).** Rewrote
  `PyMDPInterface` (create_agent construction params, batched one-hot
  observations, `infer_states(obs, prior)`, `infer_policies(qs)`,
  `sample_action(q_pi)`, B normalization over destination states) and adapted
  the tests that called the old mutable API. **54/54 pass.**
- [x] **Missing interface methods added (MAJOR-3).** Implemented
  `visualize_simulation`, `get_all_agents`, `get_all_environments`,
  `get_all_simulations`, `visualize_agent_model` (server endpoints/tools no
  longer 500).
- [x] **Working entry point (MAJOR-4, part 1).** `main()` now runs the MCP
  server; `run_sse` is loop-safe; `TRANSPORT`/`HOST`/`PORT` honored.
- [x] **Docs consolidation (MAJOR-5).** Single coherent README; fabricated tool
  docs pruned; version/port/file references reconciled; `.aii` DOI fixed.
- [x] **HTTP security (MAJOR-6, part 1).** CORS restricted; `output_file`
  path-traversal closed in `app.py`.
- [x] **Minor/Medium batch from the 2026-08-01 review:** shell-injection
  `os.system`→`subprocess` in `main.py`; `step_environment` dict-env branch;
  `create_agent` empty-A/B validation; `_normalize_A_matrix` degenerate
  columns; `reward_positions` bounds check; `.gitignore` hygiene; dead
  `agent[...]()` branches; `simulation_id` refactor; `pyproject.urls`; `.aii`
  DOI. All probe/regression verified.

## Open / remaining (scoped, still TODO)

- [ ] **Server consolidation (MAJOR-4, part 2).** Three parallel servers remain:
  `src/mcp/server/app.py` (FastAPI REST), `src/main.py` + hand-rolled
  `src/mcp/server/fastmcp.py` (MCP), and `scripts/start_mcp_server.py`
  (1412-line near-duplicate of app.py). Unify to one MCP server on the real
  `mcp`/FastMCP SDK and one thin FastAPI wrapper; delete or thin the duplicate.
  Affects `src/main.py`, `src/mcp/server/*`, `scripts/start_mcp_server.py`.
- [ ] **Resource caps (MAJOR-6, part 2).** `define_generative_model`,
  `create_gridworld_agent`, and `run_simulation` accept unbounded
  dimensions/grid/step-counts from JSON input → potential OOM on malicious or
  mistaken inputs. Add caps (e.g. max states, max steps) in `src/mcp/utils.py`
  and `src/mcp/server/app.py`.
- [ ] **License/attribution reconcile (MAJOR-7).** `LICENSE` = MIT © 2025 Cole
  Medin while README attributes development to Daniel Friedman / the Active
  Inference Institute. Confirm correct copyright holder and dedupe any
  conflicting attribution. Also `scripts/start_mcp_server.py` and
  `scripts/mcp_gridworld_examples.py` synthesize "free-energy" traces with
  `np.random` and plot them as analysis output — label/remove these so no
  synthetic trace is presented as measured.
- [ ] **Lint debt (cosmetic).** ~371 remaining `ruff` errors: unused imports
  (`F401`, config `unfixable`) + `W293` whitespace inside code. Enable F401
  autofix and re-run, or prune manually, when convenient.
- [ ] **Un-track committed artifacts / hygiene.** `tests/output/**` (21 files),
  `tests/tests/output`, `server.log`, and `src/pymdp_mcp.egg-info/` are still
  git-tracked; `git rm --cached` them (they are already in `.gitignore`).

## Open — non-negotiables / process notes

- Tests must stay green against a pinned `inferactively-pymdp` (currently
  `>=1.0.3`); re-verify with `uv run pytest tests/`.
- Run `ruff check --fix src/ scripts/` after edits; do not regress the suite.
