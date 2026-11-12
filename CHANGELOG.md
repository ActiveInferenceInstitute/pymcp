# Changelog

All notable changes to `pymcp` (package `pymdp-mcp`) are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `run.sh` — the single gate entry point: `setup`, `lint`, `test` (fast arm),
  `test-full`, `links`, `docs`, `gnn`, `all`. CI calls these arms, so no gate
  command is duplicated in the workflow.
- `scripts/gnn_conformance.py` — the pinned GNN check as a script instead of two
  inline heredocs in the workflow. It verifies `.github/gnn-pin.json`
  (`gnn-importer-pin/v1`), installs the pinned revision into an isolated
  `.gnn-venv/`, and treats the type checker's warning sentinel `2` as a failure
  rather than passing it through truthiness.
- GNN–FEP shared-reader adoption. `.github/gnn-pin.json` advances from GNN
  `v5.0.3` (`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`) to `v5.1.0`
  (`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`), the revision that ships
  `src/gnn/importers.py`. `./run.sh gnn` now runs
  `python -m gnn.importers check --repo . --require-pointer --workflow
  .github/workflows/ci.yml --job gnn-conformance` inside the materialized
  `.gnn-venv/`, so pin discovery and shape, the `.github/GNN.md` pointer and
  `*.gnn.md` syntax validation come from the shared reader; the only pin key
  this repository still reads is the revision it must install.
- `scripts/check_docs.py` — relative Markdown link check plus `docs/`
  documentation-index coverage, run by `./run.sh links`.
- `scripts/gate_facts.py` — records per-arm wall times and collected test counts
  into `docs/_generated/arms.json` and renders
  `docs/_generated/gate-facts.md`; `--check` is part of the `docs` arm.
- Documentation: `docs/RUN_GUIDE.md` (arms, CI jobs, pin facts),
  `docs/documentation-index.md` (flat index), `docs/_generated/README.md`.

### Changed
- GNN conformance: parser reuse in `.gnn-venv/` is bound to the pinned revision
  (installed `direct_url.json` → `vcs_info.commit_id`), not just the tag, so a
  same-version pin change re-installs instead of silently checking older code.
- GNN conformance: the local pin reader is deleted. Pin discovery and shape, the
  `.github/GNN.md` pointer and `*.gnn.md` syntax validation are the shared
  reader's, and the pin-shape regression tests went with the code they covered.
  The sketch-less upstream exemplar fallback is gone too: the repository carries
  `models/gridworld_3x3.gnn.md`, and the reader reports `spec-absent` if it ever
  disappears.
- CI (`.github/workflows/ci.yml`) now runs only `./run.sh` arms. Both jobs
  compose the same shared composite action
  `ActiveInferenceInstitute/.github/.github/actions/setup-python-uv`
  (pinned `@7159e67…`, `python-version: "3.12"`). The `tests` job syncs the
  frozen lock once and runs lint, docs and the full suite in that one
  environment; the `gnn-conformance` job installs no project dependency.
- Test suite: the eight simulation/visualization tests that dominate wall time
  are marked `slow`; `./run.sh test` deselects exactly those.
- Lint debt retired: 435 ruff findings (W293/W291 whitespace, F401 unused
  imports, F841 unused locals, E402, I001) fixed, with `E402` exempted for
  `tests/*` and `scripts/*` because those files put `src/` on `sys.path` before
  importing project modules. `./run.sh lint` is now a passing gate.
- Documentation counts and test commands replaced with the `./run.sh` arms and
  the generated facts page; `AGENTS.md`, `tests/README.md`,
  `tests/DOCUMENTATION_TESTING.md`, `scripts/README.md` and the InstituteOS
  sidecar task entry updated to match.
- PyMDP floor raised to `inferactively-pymdp>=1.0.5`; `uv.lock` re-resolved with
  `uv lock` (1.0.4 → 1.0.5, `sha256:c5846121…` sdist / `sha256:5e423a33…`
  wheel). The suite is green against the new pin with no test or assertion
  changed (74 passed, 14 subtests), and no live 1.0.5 behaviour differs for the
  surface this repository uses.

### Fixed
- Restored the sidecar's `docs/INTEGRATION.md` and `docs/AGENTS.md` entries:
  the pinned InstituteOS validator resolves `docs[]` relative to `.aii/`, not
  the repository root; the audit's proposed `.aii/docs/` prefix was incorrect.
- GNN–FEP adoption is no longer blocked. The blocker was the pin naming a
  revision that predates `gnn.importers` (`8f293d9b`, `v5.0.3`); the pin now
  names `75c834bd8519` (`v5.1.0`), so the reader runs in the pinned environment
  and this repository no longer duplicates pin, pointer or syntax validation.
- GNN conformance verdict: the type-check result is now an identity check on
  `True`. The previous membership test accepted `False` (`False == 0`) and `1`
  (`1 == True`) alongside the clean value, and `ConformanceError` exited with
  the message as its status code rather than a fixed status 1. `False`, `2`,
  `1`, `0` and non-booleans now fail with status 1; regression tests are in
  `tests/test_gnn_conformance.py`.

### Previously recorded
- CI workflow (`.github/workflows/ci.yml`): installs the frozen lock
  (`uv sync --frozen --python 3.12`) and runs the documented test gate
  (`uv run --frozen python -m pytest tests/`) on every push and pull request.
- Pinned GNN conformance job (`gnn conformance (pinned)`) in the same workflow:
  it verifies `.github/gnn-pin.json` (GNN v5.0.3) and, as the repository has no
  `.gnn.md` model file, validates the pinned upstream exemplar (`.github/GNN.md`).
- Raised the PyMDP floor to `inferactively-pymdp>=1.0.4` and re-resolved
  `uv.lock` (1.0.3 → 1.0.4). The suite is green against the new pin: 54 tests
  passed, also from a frozen lock under Python 3.12.
- Documented the role of each server entry point, and corrected the backlog
  wording that described `scripts/start_mcp_server.py` as a near-duplicate of
  `src/mcp/server/app.py`: the script is the legacy standalone REST server and
  shares little code with the current app.

## [1.0.3] - 2026-08-01

### Added
- Missing `PyMDPInterface` methods that the servers/tools depend on:
  `visualize_simulation`, `get_all_agents`, `get_all_environments`,
  `get_all_simulations`, `visualize_agent_model`. The corresponding MCP tools and
  FastAPI endpoints no longer error.
- `run_simulation` now returns a `timesteps` list (one entry per step with
  `observation`, `state`, `action`) plus a `history` and stable `simulation_id`.
- `step_environment` returns a tracked `position` for grid-world environments.
- Learning configuration (`learn_A`, `learn_B`, `pA`, `pB`, …) is now honored at
  agent construction.
- New docs: `docs/ARCHITECTURE.md`; visual/Mermaid diagrams in the README and
  docs.

### Changed
- **Rewritten for the current PyMDP (immutable/equinox, batched) API.** Agents
  are configured at construction time; observations and priors use the batched
  one-hot convention; `infer_policies`/`sample_action` take the posterior/policy
  posterior. **Full test suite is green (54 tests).**
- Declared the real runtime dependency `inferactively-pymdp>=1.0.3` in
  `pyproject.toml` and removed all phantom `pymdp-clone` references.
- `src/main.py` now actually starts the MCP server when run; the FastMCP SSE
  server is loop-safe (`uvicorn.Server.serve()`).
- Docs rewritten to a single coherent README; fabricated tool documentation
  (`enable_jax`, `run_tmaze_example`, …) pruned.
- Version bumped `0.1.0` → `1.0.3`.

### Security
- FastAPI CORS restricted (no wildcard + credentials); env-configurable.
- HTTP `visualize_simulation` `output_file` is confined to a fixed output dir
  (path-traversal closed).
- Replaced shell-string `os.system` test runner with `subprocess` arg-list.

### Fixed
- `step_environment` could never reach dict-gridworld `env["step"]` (unreachable
  dead branch) — inline stepping now works.
- `create_agent` returned a cryptic `NameError` for empty `A`/`B` — now a clear
  validation error.
- `_normalize_A_matrix` produced invalid all-zero columns for degenerate input.
- `reward_positions` out-of-range values silently wrapped — now bounds-checked.

### Removed
- Git-tracked build/test artifacts (`tests/output/**`, `server.log`,
  `src/pymdp_mcp.egg-info/`) — now gitignored.

## [0.1.0] - 2025

Initial release (template-era snapshot). Pre-1.0 development state; no
semantic-version compatibility guarantees.
