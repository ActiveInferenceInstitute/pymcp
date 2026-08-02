# PyMCP (PyMDP-MCP) — Backlog

Status:      OPEN — suite GREEN (54 passed / 0 failed against inferactively-pymdp 1.0.3)
Owner:       Daniel Ari Friedman (Active Inference Institute)
Last reviewed: 2026-08-02 (docs-deep pass — see REVIEW_LOG_2026-08-02.md)

## Sections

- **Minor** — typo, broken link, formatting, small metadata fix.
- **Medium** — stale section rewrite, doc restructure, added missing guide,
  broken helper repair.
- **Major** — large doc system overhaul, cross-cutting refactors, deferred
  code work.

---

## Minor — completed this pass (2026-08-02)

- [x] ✓ **Dead external link.** `https://pymdp.readthedocs.io/` returns 404;
  the PyMDP docs moved to `https://pymdp-rtd.readthedocs.io/en/latest/`
  (verified). Files: `docs/mcp_tools.md`, `docs/mcp_pymdp_integration.md`.
  Commit `3ffa020`.
- [x] ✓ **Project-layout path error.** `docs/README.md` placed `main.py`
  inside `src/mcp/`; it lives at `src/main.py` (sibling of `src/mcp/`).
  Commit `3ffa020`.
- [x] ✓ **Architecture-diagram path error.** `docs/ARCHITECTURE.md` module
  diagram showed `main.py` under `src/mcp/`; corrected to `src/main.py`.
  Commit `3ffa020`.
- [x] ✓ **Stale `__version__`.** `src/mcp/__init__.py` said `0.1.0` while the
  package is `1.0.3`. Commit `01124e7`.
- [x] ✓ **Stale terminology.** "Message-based Cognitive Protocol" → "Model
  Context Protocol" in `pyproject.toml` description and `src/mcp/__init__.py`
  docstring. Commit `01124e7`.
- [x] ✓ **`requires-python` mismatch.** `>=3.8` was inconsistent with the
  resolved `uv.lock` (`>=3.12`) and the `inferactively-pymdp>=1.0.3` floor;
  `uv run` could not resolve. Aligned to `>=3.12`; README Python badge and
  ruff `target-version` updated. Files: `pyproject.toml`, `README.md`.
  Commits `01124e7`, `72e1d88`.
- [x] ✓ **Wrong `uv.lock`.** The committed lock was a template-era artifact
  listing unrelated packages (`mem0ai`, `openai`, `qdrant-client`, …) and
  missing the real deps (`inferactively-pymdp`, `fastapi`, `matplotlib`,
  `aiohttp`, `jax`, `equinox`). Regenerated to match `pyproject.toml`; raised
  the `pydantic` floor to `>=2.13` so `pydantic-core` resolves to a build with
  Python 3.14 wheels (`uv sync` verified). Files: `uv.lock`, `pyproject.toml`.
  Commits `01124e7`, `27a982e`.
- [x] ✓ **Missing dev dependencies.** `uv sync` produced an environment
  without `pytest` — the documented test command could not run on a clean
  checkout. Added `[dependency-groups] dev = ["pytest", "ruff"]`. Files:
  `pyproject.toml`, `uv.lock`. Commit `27a982e`.
- [x] ✓ **README test command.** Bare `pytest` → canonical
  `uv run python -m pytest tests/`. Commit `72e1d88`.
- [x] ✓ **Gitignore hygiene.** `.ruff_cache/` (present at root) added to
  `.gitignore`. Commit `01124e7`.

## Medium — completed this pass (2026-08-02)

- [x] ✓ **`docs/pymdp_integration.md` rewrite.** Two stale claims removed:
  (a) "PyMDP is intentionally **not** pinned as a dependency" — false,
  `inferactively-pymdp>=1.0.3` is declared and installed automatically;
  (b) legacy mutable-API example (`infer_states(observation)`,
  `infer_policies()`, `sample_action()`) replaced with the current
  immutable/batched call convention (`infer_states(obs, prior)`,
  `infer_policies(qs)`, `sample_action(q_pi)`). Commit `ed06406`.
- [x] ✓ **`docs/implementation_details.md` snippets.** `create_agent` example
  updated to construction-time configuration with batched matrices; the
  environment description now reflects JSON-serializable env dicts with inline
  stepping (`_step_gridworld`). Commit `199308d`.
- [x] ✓ **`scripts/README.md` rewrite.** Removed instructions for
  `requirements.txt` and the `pymdp-clone` PYTHONPATH entry (neither exists);
  documented `pymcp_examples_fixed.py`; added a note that some example
  scripts plot synthetic (np.random) free-energy traces. Commit `176d190`.
- [x] ✓ **Synthetic traces labeled.** `scripts/start_mcp_server.py` and
  `scripts/mcp_gridworld_examples.py`: np.random free-energy branches marked
  `SYNTHETIC / ILLUSTRATIVE ONLY` in comments; the `start_mcp_server.py` plot
  title now reads "(illustrative sample)" — no synthetic trace is presented as
  measured output (legacy MAJOR-7, code part). Commit `176d190`.
- [x] ✓ **`tests/README.md`.** `run_tests.py` → `run_all_tests.py`; canonical
  `uv run python -m pytest tests/` documented; "adding new tests" steps point
  at `run_all_tests.py`'s `category_patterns`. Commit `da6300f`.
- [x] ✓ **`tests/DOCUMENTATION_TESTING.md`.** Runner references fixed; missing
  test files added (`test_mcp_full.py`, `test_belief_dynamics.py`,
  `test_template.py`); learning section corrected (construction-time
  configuration, not post-hoc Dirichlet updates); 54-test green state noted.
  Commit `da6300f`.
- [x] ✓ **Broken test wrapper.** `tests/run_and_organize.sh` invoked
  non-existent `run_tests.py` and passed unsupported `--organize`; now calls
  `run_all_tests.py`. Verified end-to-end (`run_all_tests.py --category
  belief_dynamics` → "All tests completed successfully"). Commit `680bf63`.
- [x] ✓ **`mcp_status_report.md` rewritten.** The root status report described
  the pre-1.0.3 failing state ("most MCP tool tests are failing", open NumPy
  conversion issue, legacy snippets). Rewritten as an accurate, dated snapshot
  (verified capabilities, security posture, testing, open items) and linked
  from `docs/README.md` and the root README. Commit `676c615`.

## Major — completed / open

- [x] ✓ **Docs-pass infrastructure.** `REVIEW_LOG_2026-08-02.md` (dated
  audit log) + this TODO restructured into Minor/Medium/Major with completed
  markers and commit references. Commit: the final commit of the 2026-08-02
  docs-deep pass (all pass commits are listed in `REVIEW_LOG_2026-08-02.md`).
- [x] ✓ **Legacy "un-track committed artifacts" item** — resolved by
  verification: nothing is git-tracked anymore (`tests/output/**`,
  `server.log`, `src/pymcp_mdp.egg-info/` are all untracked and gitignored).
- [x] ✓ **Legacy MAJOR-7 license part** — resolved by verification: `LICENSE`
  and `README.md` both state "Copyright (c) 2025 Cole Medin"; the attributed
  developer in the backlog header is the maintainer, not a license conflict.
  No dedupe needed.
- [ ] **OPEN — Server consolidation (legacy MAJOR-4 part 2).** Three parallel
  servers remain: `src/mcp/server/app.py` (FastAPI REST), `src/main.py` +
  hand-rolled `src/mcp/server/fastmcp.py` (MCP), and
  `scripts/start_mcp_server.py` (1412-line near-duplicate of app.py). Deferred:
  code refactor (unify onto the official `mcp`/FastMCP SDK); out of scope for
  a docs pass. Files: `src/main.py`, `src/mcp/server/*`,
  `scripts/start_mcp_server.py`.
- [ ] **OPEN — Resource caps (legacy MAJOR-6 part 2).**
  `define_generative_model`, `create_gridworld_agent`, and `run_simulation`
  accept unbounded dimensions/grid/step-counts from JSON input → potential OOM
  on malicious or mistaken inputs. Deferred: code change in `src/mcp/utils.py`
  and `src/mcp/server/app.py`.
- [ ] **OPEN — Lint debt (cosmetic).** ~371 remaining ruff findings (F401
  unused imports, W293 whitespace). NOTE: `[tool.ruff] fix = true` means plain
  `ruff check` auto-rewrites the tree — inspect with `ruff check --no-fix`;
  a dedicated cleanup pass is required.
- [ ] **OPEN — CI.** No GitHub Actions workflow exists; the suite runs locally
  only (`uv run python -m pytest tests/`).

---

## Open — non-negotiables / process notes

- Tests must stay green against the pinned `inferactively-pymdp` (currently
  `>=1.0.3`); re-verify with `uv run python -m pytest tests/`.
- Inspect lint with `ruff check --no-fix` — plain `ruff check` self-modifies
  the tree (`fix = true` in `pyproject.toml`).
- Every doc claim must be verified against the repo before writing (this pass
  fixed several stale claims, including the PyMDP dependency status and the
  legacy mutable-API examples).
