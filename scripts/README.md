# MCP-PyMDP Example Scripts

This directory contains scripts for testing and demonstrating the PyMCP
integration (MCP server + REST API over the PyMDP active inference library), plus
the tooling behind the `./run.sh` gate arms.

## Gate tooling

These scripts are not examples; they back the arms of `./run.sh` (see
[`../docs/RUN_GUIDE.md`](../docs/RUN_GUIDE.md)):

| Script | Called by | Purpose |
| --- | --- | --- |
| `gnn_conformance.py` | `./run.sh gnn` | Installs the revision `.github/gnn-pin.json` records into `.gnn-venv/`, runs the shared `python -m gnn.importers check` reader from that environment (pin, pointer and spec validation), and type-checks the specifications the reader discovers. Prints a JSON verdict. |
| `check_docs.py` | `./run.sh links` | Checks every relative Markdown link in tracked `*.md` files and that each page under `docs/` appears in `docs/documentation-index.md`. |
| `gate_facts.py` | `./run.sh docs` (`--check`) and `--record` | Re-runs the arms, writes `docs/_generated/arms.json`, and renders `docs/_generated/gate-facts.md`. |

## Prepared runtime

Run from the repository root with `PROJECT_PY` set to an existing prepared
environment's absolute Python executable and `PYTHONPATH="$PWD/src"`.
Dependency provisioning is a separate authorized step; see
[setup](../docs/setup.md). No example command should implicitly install or sync.

## Supported entry points

| Script | Role |
| --- | --- |
| `start_fastapi_server.py` | Loopback, single-worker, no-reload REST launcher |
| `mcp_pymdp_example.py` | Actual agent/environment example |
| `mcp_gridworld_examples.py` | Matching-grid examples and measured plots |
| `pymcp_examples_fixed.py` | Four maintained cases: numerical inference, manual transitions and real model/environment operations |
| `test_server.py`, `test_mcp_pymdp.py` | Readiness/integration consumers with failure-visible status |
| `run_all_examples.py` | Owned server lifecycle, deadlines and per-case result summary |
| `run_with_server.sh` | Delegates to `run_all_examples.py` |
| `example_support.py` | Shared real operations and completed/failed/not-run reporting |

`start_mcp_server.py` and `pymcp_examples.py` are retired. The bundled MCP
entry point is `src/main.py`, not another REST launcher.

## Run examples

```bash
PYTHONPATH="$PWD/src" "$PROJECT_PY" -B scripts/run_all_examples.py --port 8090 --output-dir /path/to/owned-output
```

The runner owns its new REST process, verifies PID/instance-nonce readiness,
preserves each case's exit status and reaps only that process. A collision,
timeout, failed case or not-run required case cannot become success.
It does not terminate an unrelated process occupying a port.

For an already running authorized REST instance:

```bash
PYTHONPATH="$PWD/src" "$PROJECT_PY" -B scripts/mcp_gridworld_examples.py --host 127.0.0.1 --port 8080 --output-dir /path/to/owned-output
```

The REST launcher and bundled MCP commands are in [setup](../docs/setup.md).
They create independent in-memory state; running an MCP server does not
populate a REST server or a direct client.

## Results and measurements

Each case writes into a unique owned run directory. JSON and images come from
actual returned model/run records, with real IDs, executed/requested counts,
rewards and measured-status entries. No random fallback energy trace or
placeholder plot is generated after a required operation fails.

The fixed numerical example has posterior `[1/17, 9/17, 7/17]`, evidence
`17/30`, and VFE `-log(17/30)`. Its manual loop reports actual transitions,
not an invented registered simulation ID. PNG/SVG downloads are actual bytes;
an SVG destination requires an explicit SVG request.

Grid reset remains origin by default; custom reset remains first-state.
Configured starts and stopping on terminal state are explicit opt-ins.
See [the lifecycle and measurement contract](../docs/mcp_tools.md); these
examples do not decide new scientific reset/terminal defaults.
