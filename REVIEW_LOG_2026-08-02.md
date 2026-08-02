# pymcp — Docs-Deep Review Log (2026-08-02)

Review pass: mega-deep documentation review + implementation of all scoped
improvements. Runs entirely inside this repo (no parent/sibling repos touched).

## Phase 0 — Preflight

- Branch: `main` (default per `origin/HEAD`), fast-forward pulled; HEAD before
  work: `43e3b42` ("Polish repo for 1.0.3: visual README + Mermaid, ARCHITECTURE
  docs, CHANGELOG, About").
- Working tree clean at start.
- Inventory: README.md, CHANGELOG.md, TODO.md, docs/ (7 files: README,
  ARCHITECTURE, advanced_features, implementation_details, mcp_pymdp_integration,
  mcp_tools, pymdp_integration), scripts/README.md, tests/README.md,
  tests/DOCUMENTATION_TESTING.md, mcp_status_report.md (root, orphaned), .aii/
  sidecar, Dockerfile, pyproject.toml, LICENSE. No AGENTS.md/CLAUDE.md,
  no .github/ CI, no CONTRIBUTING.md / SECURITY.md / CITATION.cff.
- Baseline suite: `uv run --no-sync python -m pytest tests/` →
  **54 passed in 35.28s** (matches the "54 passed / 0 failed" claim).
- README quickstart snippet executed end-to-end (5 timesteps) — works.
- Link check: no broken relative links or anchors in any markdown; external
  links OK except `https://pymdp.readthedocs.io/` → 404 (docs moved to
  `https://pymdp-rtd.readthedocs.io/en/latest/`, verified 200).

## Phase 1 — Review findings (severity)

### Minor
1. Dead external link `pymdp.readthedocs.io` in `docs/mcp_tools.md:175` and
   `docs/mcp_pymdp_integration.md:68` → replace with `pymdp-rtd.readthedocs.io`.
2. `docs/README.md` project layout shows `main.py` inside `src/mcp/`; the file
   is actually `src/main.py` (sibling of `src/mcp/`).
3. `docs/ARCHITECTURE.md` module-layout diagram places `main.py` inside the
   `src["src/mcp"]` subgraph; same path correction needed.
4. `src/mcp/__init__.py` `__version__ = "0.1.0"` vs pyproject `1.0.3` — stale.
5. Stale terminology "Message-based Cognitive Protocol" in `src/mcp/__init__.py`
   docstring and `pyproject.toml` description; the project implements the
   Model Context Protocol.
6. `pyproject.toml` `requires-python = ">=3.8"` inconsistent with its own
   `uv.lock` (`requires-python = ">=3.12"`); `uv run` fails to resolve for the
   declared range. README Python badge says `>=3.8`. Fix metadata + badge.
7. README "Testing" section uses bare `pytest`; canonical repo command is
   `uv run python -m pytest tests/` (avoids system-pytest drift).
8. `.ruff_cache/` not in `.gitignore` (present at root).

### Medium
1. `docs/pymdp_integration.md` is stale/wrong in two ways: (a) claims PyMDP is
   "intentionally **not** pinned as a dependency" — false, `pyproject.toml`
   declares `inferactively-pymdp>=1.0.3`; (b) code example uses the legacy
   mutable PyMDP API (`agent.infer_states(observation)`, `infer_policies()`,
   `sample_action()`) — current PyMDP is immutable and requires
   `infer_states(obs, prior)`, `infer_policies(qs)`, `sample_action(q_pi)`.
2. `scripts/README.md` references non-existent `requirements.txt` and
   `pymdp-clone` directory; omits `pymcp_examples_fixed.py` (the script the
   runner actually uses); no honesty note about illustrative synthetic
   free-energy traces in example outputs.
3. `tests/README.md` references `run_tests.py` (does not exist; real runner is
   `run_all_tests.py`); test-adding instructions reference `run_tests.py`.
4. `tests/DOCUMENTATION_TESTING.md` references `run_tests.py` throughout;
   missing the newer test files (`test_mcp_full.py`, `test_belief_dynamics.py`,
   `test_template.py`) and the 54-test count.
5. `tests/run_and_organize.sh` is broken: checks for and invokes `run_tests.py`
   (missing) and passes `--organize` (not accepted by `run_all_tests.py`).
6. `mcp_status_report.md` (root, orphaned) describes a pre-1.0.3 state that
   contradicts reality: "most MCP tool tests are failing", a still-open
   `A matrix must be a numpy array` issue, legacy code snippets. Rewrite to
   reflect the current green state and link it from the docs index.
7. Synthetic "free-energy" traces generated with `np.random` in
   `scripts/mcp_gridworld_examples.py` and `scripts/start_mcp_server.py` are
   plotted as analysis output (TODO MAJOR-7). `start_mcp_server.py` plot title
   does not label them as illustrative; strengthen labeling in code + docs.

### Major
1. TODO.md restructure into Minor/Medium/Major with completed markers and an
   explicit open/deferred list (task requirement).
2. Server consolidation (legacy TODO MAJOR-4 part 2): three parallel servers
   (`app.py`, `main.py`+`fastmcp.py`, 1412-line `scripts/start_mcp_server.py`).
   **Deferred** — code refactor, outside this docs pass.
3. Resource caps (legacy TODO MAJOR-6 part 2). **Deferred** — code change.
4. License/attribution: verified LICENSE and README both state
   "Copyright (c) 2025 Cole Medin" — consistent, no conflict; legacy TODO
   item marked resolved. Synthetic-trace part handled as Medium-7 above.
5. Lint debt (~371 ruff errors, mostly F401 + W293). **Deferred** — cosmetic;
   `ruff` is configured with `fix = true` so bulk autofix self-modifies the
   tree (would be a separate cleanup pass, not a docs pass).

## Phase 2/3 — Scope & implementation

See the restructured `TODO.md` (root) for the full scoped list, per-item
completed markers with commit refs, and the open/deferred list.

Commits in this pass (all on `main`):

| Commit | Scope |
| --- | --- |
| `01124e7` | chore: metadata alignment (requires-python, `__version__`, MCP terminology, `uv.lock` regeneration, `.gitignore`) |
| `3ffa020` | docs: dead PyMDP docs links + `main.py` path corrections |
| `ed06406` | docs: `pymdp_integration.md` rewrite (declared dep + current API) |
| `176d190` | docs+fix: `scripts/README.md` rewrite + synthetic-trace labels |
| `680bf63` | fix: `tests/run_and_organize.sh` → `run_all_tests.py` |
| `da6300f` | docs: tests README + DOCUMENTATION_TESTING fixes |
| `676c615` | docs: `mcp_status_report.md` rewrite + index link |
| `72e1d88` | docs: README badge / test command / docs list |
| `199308d` | docs: `implementation_details.md` snippets |
| `27a982e` | chore: pydantic floor + dev dependency group (`uv sync` works) |
| (final) | docs: TODO.md restructure + REVIEW_LOG |

## Phase 4 — Verification & push

- Full suite re-run after all edits: `uv run python -m pytest tests/` →
  **54 passed in 22.54s** (fresh `uv sync` environment, Python 3.14).
- `uv sync` on the corrected lockfile: clean (pydantic-core 2.46.4 wheels).
- `run_all_tests.py --category belief_dynamics` (repaired legacy runner):
  "All tests completed successfully".
- Link check re-run on all markdown: no broken relative links/anchors; the two
  previously-dead `pymdp.readthedocs.io` links replaced with the verified live
  docs URL.
- Python edits compile-checked; `ruff check --no-fix` on touched files:
  `src/mcp/__init__.py` clean; `scripts/start_mcp_server.py` +
  `scripts/mcp_gridworld_examples.py` 21 pre-existing errors (identical
  before/after this pass — see the lint-debt item in `TODO.md`).
- Pushed to `origin/main` (the final commit of this pass; see `git log`).

## Notes

- Heavy suites (Julia/LaTeX) — N/A, this is a Python repo; the full pytest
  suite is fast (~25–35 s) and was run, not skipped.
- No secrets touched; no private paths referenced in any committed content.
- Deferred items (server consolidation, resource caps, lint debt, CI) are
  code-scoped and tracked as OPEN in `TODO.md` with reasons.
