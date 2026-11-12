# pymcp Documentation

Welcome to the documentation for **pymcp** (`pymdp-mcp`) — an MCP server and
FastAPI HTTP API for the PyMDP active inference library.

**Quick Reference:** [Documentation Index](documentation-index.md) |
[Run Guide (`./run.sh`)](RUN_GUIDE.md) | [Generated gate facts](_generated/gate-facts.md) |
[GNN conformance](../.github/GNN.md) | [Project overview](../README.md)

## Contents

1. **[Setup and Usage](setup.md)**
   Install, run either server, configure environment variables, and drive
   `PyMDPInterface` directly.

2. **[Testing](testing.md)**
   The suite layout, the `slow` marker and the fast arm, the legacy category
   runners, test outputs, and adding tests.

3. **[Run Guide](RUN_GUIDE.md)**
   The gate entry point: every `./run.sh` arm, what CI runs, the pinned GNN
   conformance arm, and how the measured numbers are generated.

4. **[Documentation Index](documentation-index.md)**
   Flat list of every long-lived page in this directory, plus the repository
   records worth reading.

5. **[Architecture](ARCHITECTURE.md)**
   Module layout, request lifecycle, and design decisions — with diagrams.

6. **[MCP & PyMDP Integration](mcp_pymdp_integration.md)**
   High-level overview of the project: what it is, its components, data flow,
   and how MCP tools map onto the `PyMDPInterface`.

7. **[MCP Tools Reference](mcp_tools.md)**
   The complete set of real tools exposed by the servers — every method on
   `PyMDPInterface`, with signatures, parameters, and return values.

8. **[Implementation Details](implementation_details.md)**
   Technical detail on the layered architecture: `PyMDPInterface`, the FastMCP
   server, the FastAPI server, JSON/NumPy serialization, and session handling.

9. **[Advanced Features](advanced_features.md)**
   The capabilities that actually exist beyond the core loop: gridworld and
   custom environments, visualization/belief-dynamics, free-energy analysis,
   sessions, and direct vs. network clients.

10. **[PyMDP Integration](pymdp_integration.md)**
    How this project integrates the PyMDP pip package (`inferactively-pymdp`)
    as an unmodified dependency, and how compatibility is verified.

11. **[Implementation Status Report](../mcp_status_report.md)**
   A dated snapshot of the current state: what works, test status, and known
   open items.

## Getting Started

1. Install the package and PyMDP (see the root [`README.md`](../README.md)).
2. Use `PyMDPInterface` directly for scripting, or run one of the two servers.
3. See [mcp_tools.md](mcp_tools.md) for the full method reference.

## Verifying a change

One entry point, no remembered command lines:

```bash
./run.sh setup      # install the committed lock (Python 3.12)
./run.sh test       # fast arm: suite minus the eight `slow` tests
./run.sh all        # lint, full suite, docs, pinned GNN conformance
```

Arms, the two CI jobs (`tests`, `gnn-conformance`), the canonical pin
`.github/gnn-pin.json` (schema `gnn-importer-pin/v1`, `v5.0.3`,
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`) and what the conformance job parses
are documented in the [Run Guide](RUN_GUIDE.md). Per-arm wall times and the
collected test counts are generated into
[`_generated/gate-facts.md`](_generated/gate-facts.md) by
`scripts/gate_facts.py`; they are not maintained by hand.

## Project Layout

```
src/
  main.py                  # MCP tools over FastMCP (SSE / stdio) — entry point
  mcp/
    utils.py               # PyMDPInterface — the core integration layer
    server/app.py          # FastAPI HTTP server (uvicorn)
    server/fastmcp.py      # Bundled FastMCP implementation
    client.py, client/     # Network (MCPClient) and direct (DirectMCPClient) clients
    environments/          # Grid-world and custom environment helpers
    visualization/         # Model / free-energy / belief-dynamics visualization
tests/                     # Test suite (run with pytest)
```
