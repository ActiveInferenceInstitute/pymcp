#!/bin/bash
# The Python owner handles readiness, signals, exact-child reaping and exit status.
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/run_all_examples.py" "$@"