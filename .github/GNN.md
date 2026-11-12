# GNN conformance

`gnn conformance (pinned)` is the display name of the `gnn-conformance` job in
`.github/workflows/ci.yml`. The job runs one command, `./run.sh gnn`, which runs
`scripts/gnn_conformance.py` — the check itself lives there, not in YAML.

It reads `.github/gnn-pin.json`, the canonical pin file (schema
`gnn-importer-pin/v1`), which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, recorded 2026-11-12). The script
verifies the pin fields, installs that exact revision into an isolated
`.gnn-venv/`, and requires the installed `gnn` version to match the pin tag
before parsing anything.

What the job parses:

- every `*.gnn.md` sketch in the repository (currently
  `models/gridworld_3x3.gnn.md`), each parsed, syntax-validated, then
  type-checked together;
- when there is no repository sketch, the pinned upstream
  `input/gnn_files/basics/static_perception.md` exemplar — a pipeline and pin
  conformance baseline, not a claim that the repository already contains a
  model.

The type-check verdict is compared explicitly against a clean result (`True` or
`0`). `GNNTypeChecker.validate_gnn_files` returns `2` for its warning sentinel,
and every nonzero integer is truthy, so a truthiness test would pass on a
warning; this check fails instead.

The first run needs network access to install the pinned revision; after that
`.gnn-venv/` is reused and the check is local. See the
[Run Guide](../docs/RUN_GUIDE.md) for the full gate.
