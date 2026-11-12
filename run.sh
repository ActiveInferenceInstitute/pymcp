#!/usr/bin/env bash
#
# pymcp gate — the one documented entry point.
#
# Every arm below is one command, and that exact command is what
# `.github/workflows/ci.yml` runs. There is no second copy of a gate command in
# YAML, docs, or scripts: CI invokes `./run.sh <arm>`.
#
#   ./run.sh setup       install the committed lock (Python 3.12) into .venv
#   ./run.sh lint        ruff, read-only (`pyproject.toml` sets fix = true)
#   ./run.sh test        fast arm: the suite minus tests marked `slow` (default)
#   ./run.sh test-full   every collected test — the arm CI requires
#   ./run.sh links       Markdown relative links + docs/ index coverage
#   ./run.sh docs        links + generated-facts drift (docs/_generated/)
#   ./run.sh gnn         pinned GNN conformance check
#   ./run.sh all         lint, test-full, docs, gnn
#
# There is no `types` arm: the repository declares no type checker (no mypy,
# pyright, or pytype dependency or configuration).
#
# Arms assume the environment is already synced (`./run.sh setup`, or the
# `setup-python-uv` composite action plus one `uv sync --frozen` in CI).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_VERSION="3.12"

run_uv() { uv run --frozen "$@"; }

arm_setup() { uv sync --frozen --python "$PYTHON_VERSION"; }
arm_lint() { run_uv ruff check --no-fix; }
arm_test() { run_uv python -m pytest tests/ -q -m "not slow"; }
arm_test_full() { run_uv python -m pytest tests/ -q; }
arm_links() { run_uv python scripts/check_docs.py; }
arm_facts_check() { run_uv python scripts/gate_facts.py --check; }
arm_gnn() {
  # The pinned parser lives in its own .gnn-venv/, so the arm never needs the
  # project environment: no second lock, no duplicated install.
  uv run --no-project --python "$PYTHON_VERSION" python scripts/gnn_conformance.py
}

arm_docs() {
  arm_links
  arm_facts_check
}

arm_all() {
  arm_lint
  arm_test_full
  arm_docs
  arm_gnn
}

usage() {
  # The header comment block above `set -euo pipefail` is the usage text.
  awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"
}

main() {
  local arm="${1:-test}"
  case "$arm" in
    setup) arm_setup ;;
    lint) arm_lint ;;
    test) arm_test ;;
    test-full) arm_test_full ;;
    links) arm_links ;;
    docs) arm_docs ;;
    gnn) arm_gnn ;;
    all) arm_all ;;
    -h | --help | help) usage; return 0 ;;
    *)
      echo "unknown arm: $arm" >&2
      echo >&2
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
