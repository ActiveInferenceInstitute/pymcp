# GNN conformance

`gnn conformance (pinned)` is the display name of the `gnn-conformance` job in
`.github/workflows/ci.yml`. The job runs one command, `./run.sh gnn`, which runs
`scripts/gnn_conformance.py` — the check itself lives there, not in YAML.

It reads `.github/gnn-pin.json`, the canonical pin file (schema
`gnn-importer-pin/v1`), which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, recorded 2026-11-12). The script
verifies the pin fields and installs that exact revision into an isolated
`.gnn-venv/`. The environment is reused only when the installed parser's
recorded git commit (`direct_url.json` → `vcs_info.commit_id`) equals the pin
revision, so a same-version pin change re-installs rather than testing the old
parser.

What the job parses:

- every `*.gnn.md` sketch in the repository (currently
  `models/gridworld_3x3.gnn.md`), each parsed, syntax-validated, then
  type-checked together;
- when there is no repository sketch, the pinned upstream
  `input/gnn_files/basics/static_perception.md` exemplar — a pipeline and pin
  conformance baseline, not a claim that the repository already contains a
  model.

The type-check verdict is an identity check on the boolean `True`, the value
`GNNTypeChecker.validate_gnn_files` returns for a clean model. Equality tests
are unsound here: `GNNTypeChecker` returns `2` for its warning sentinel, `not 2`
is `False`, and `False == 0`, so a truthiness or membership test would pass on a
warning or a failure. `2`, `False`, `1` and `0` all fail the job;
`tests/test_gnn_conformance.py` pins those cases.

The first run needs network access to install the pinned revision; after that
`.gnn-venv/` is reused and the check is local. See the
[Run Guide](../docs/RUN_GUIDE.md) for the full gate.
