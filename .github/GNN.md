# GNN conformance

`gnn conformance (pinned)` is the display name of the `gnn-conformance` job in
`.github/workflows/ci.yml`. The job runs one command, `./run.sh gnn`, which runs
`scripts/gnn_conformance.py` — the check itself lives there, not in YAML.

It reads `.github/gnn-pin.json`, the canonical pin file (schema
`gnn-importer-pin/v1`), which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.1.0`
(`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, recorded 2026-11-12). The script
materializes that exact revision into an isolated `.gnn-venv/`, reused only when
the installed parser's recorded git commit (`direct_url.json` →
`vcs_info.commit_id`) equals the pin revision, so a same-version pin change
re-installs rather than testing the old parser.

## The shared reader

This member has **adopted** the
[GNN–FEP shared-reader contract](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/workflows/gnn-fep.md).
Inside `.gnn-venv/` the arm runs the reader the pin names:

```bash
python -m gnn.importers check --repo . --require-pointer \
    --workflow .github/workflows/ci.yml --job gnn-conformance
```

The reader owns what this repository used to implement locally: pin discovery
and shape, the pin's declared repository/tag/revision, `.github/GNN.md` (this
file) naming the pin path, the revision, the workflow and the job key, and the
syntax validation of every `*.gnn.md` specification with the pinned parser.
There is no second pin reader in this checkout, and no copy of
`src/gnn/importers.py`.

The pin was `v5.0.3` at `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1` until
2026-11-12; that revision predates `src/gnn/importers.py` (added in `e6a9d3917`,
first released in `v5.1.0`), which is why adoption was blocked rather than
skipped. The pin now names `v5.1.0`, so the reader and the parser come from the
same revision.

## The type check

The reader validates syntax, not model semantics, so the arm also type-checks
the specifications the reader discovers (`gnn.importers.discover_specs`) with
the pinned `GNNTypeChecker`. Currently that is `models/gridworld_3x3.gnn.md`.

The type-check verdict is an identity check on the boolean `True`, the value
`GNNTypeChecker.validate_gnn_files` returns for a clean model. Equality tests
are unsound here: `GNNTypeChecker` returns `2` for its warning sentinel, `not 2`
is `False`, and `False == 0`, so a truthiness or membership test would pass on a
warning or a failure. `2`, `False`, `1` and `0` all fail the job;
`tests/test_gnn_conformance.py` pins those cases.

A corpus with no specifications is also a failure, not a skip:
`validate_gnn_files` reports "no GNN files found" as that same `2` sentinel, so
the arm refuses to print a verified verdict with nothing checked. A green GNN
job means at least one `*.gnn.md` specification passed the identity check.

The first run needs network access to install the pinned revision; after that
`.gnn-venv/` is reused and the check is local. See the
[Run Guide](../docs/RUN_GUIDE.md) for the full gate.
