# MCP-PyMDP Example Scripts

This directory contains scripts for testing and demonstrating the PyMCP
integration (MCP server + REST API over the PyMDP active inference library), plus
the tooling behind the `./run.sh` gate arms.

## Gate tooling

These scripts are not examples; they back the arms of `./run.sh` (see
[`../docs/RUN_GUIDE.md`](../docs/RUN_GUIDE.md)):

| Script | Called by | Purpose |
| --- | --- | --- |
| `gnn_conformance.py` | `./run.sh gnn` | Verifies `.github/gnn-pin.json` (schema `gnn-importer-pin/v1`), installs the pinned GNN revision into `.gnn-venv/`, and parses/type-checks the repository sketches. Prints a JSON verdict. |
| `check_docs.py` | `./run.sh links` | Checks every relative Markdown link in tracked `*.md` files and that each page under `docs/` appears in `docs/documentation-index.md`. |
| `gate_facts.py` | `./run.sh docs` (`--check`) and `--record` | Re-runs the arms, writes `docs/_generated/arms.json`, and renders `docs/_generated/gate-facts.md`. |

The rest of this page covers the example scripts.

## Setup

Install the package — including the PyMDP dependency, which is declared in
`pyproject.toml` — from the repository root:

```bash
cd ..
uv pip install -e .
```

When running a script directly, make sure the `src` directory is on your
Python path:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/../src
```

`scripts/setup_pymdp_path.py` does the same thing programmatically: import it
at the top of your own script, or run it to print the path that will be added.

## Scripts Overview

### Server Scripts

These scripts start a server that exposes PyMDP functionality:

- `start_mcp_server.py`: Standalone server script (FastAPI-based; provides the
  REST/tool surface for the example scripts). Note: it predates the current
  `src/mcp/server/app.py` + `src/main.py` servers and overlaps with them.
- `start_fastapi_server.py`: Alternative FastAPI server implementation.

### Example Scripts

These scripts demonstrate various aspects of the PyMDP active inference
framework:

- `mcp_pymdp_example.py`: Basic PyMDP examples with simple agent-environment
  interactions.
- `mcp_gridworld_examples.py`: Detailed gridworld environment examples with
  visualizations.
- `pymcp_examples.py`: Comprehensive examples covering the core PyMDP
  functionality.
- `pymcp_examples_fixed.py`: Maintained variant of `pymcp_examples.py`
  (updated client usage); this is the script `run_all_examples.py` runs.

### Test Scripts

These scripts are for testing the MCP server and PyMDP integration:

- `test_server.py`: Basic connectivity tests for the server.
- `test_agent.py`: Direct PyMDP agent testing without the MCP interface.
- `test_mcp_pymdp.py`: Tests for the MCP-PyMDP integration layer.

### Utility Scripts

- `setup_pymdp_path.py`: Utility to set up the correct Python path.
- `run_with_server.sh`: Shell script to run an example with automatic server
  management.
- `run_all_examples.py`: Python script to run all examples in sequence with
  server management.

## Running the Examples

**Important: run the scripts with the repository's virtualenv Python
(`uv run python ...`) or `python3` (not `python`).**

### Method 1: Using the run_with_server.sh script

This is the simplest way to run a single example:

```bash
./run_with_server.sh --example mcp_gridworld_examples.py
```

You can specify a different port and output directory:

```bash
./run_with_server.sh --port 8090 --output-dir ./my_outputs --example mcp_pymdp_example.py
```

### Method 2: Using the run_all_examples.py script

To run all examples in sequence:

```bash
python3 run_all_examples.py
```

### Method 3: Manual server and example execution

1. Start the server:

```bash
python3 start_mcp_server.py --port 8080
```

2. In another terminal, run an example:

```bash
python3 mcp_gridworld_examples.py --port 8080
```

## Output Organization

All scripts create timestamped output directories that contain:

- JSON files with all experiment data
- PNG visualizations of agents, environments, and simulations
- Log files with detailed execution information

## A note on example outputs

Some example scripts (`mcp_gridworld_examples.py`, `start_mcp_server.py`) plot
free-energy traces that are **generated with `np.random`** when no measured
trace is available in the session history. These traces are **synthetic /
illustrative** — they are labeled as such in the code and on the plots, and
must not be presented as measured analysis output.

## Recommended Execution Order

For first-time users, we recommend running scripts in this order:

1. `test_server.py` — Verify server functionality
2. `mcp_pymdp_example.py` — Understand basic agent-environment interaction
3. `mcp_gridworld_examples.py` — Explore gridworld environments in detail
4. `pymcp_examples.py` / `pymcp_examples_fixed.py` — Explore advanced PyMDP
   functionality

Or simply run `run_all_examples.py` to execute them all in the correct
sequence.
