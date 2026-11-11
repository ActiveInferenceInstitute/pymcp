# pymcp Documentation

Welcome to the documentation for **pymcp** (`pymdp-mcp`) — an MCP server and
FastAPI HTTP API for the PyMDP active inference library.

## Contents

1. **[Setup and Usage](setup.md)**
   Install, run either server, configure environment variables, and drive
   `PyMDPInterface` directly.

2. **[Testing](testing.md)**
   The 54-test suite, the legacy category runners, test outputs, adding tests,
   the CI gate, and lint.

3. **[Architecture](ARCHITECTURE.md)**
   Module layout, request lifecycle, and design decisions — with diagrams.

4. **[MCP & PyMDP Integration](mcp_pymdp_integration.md)**
   High-level overview of the project: what it is, its components, data flow,
   and how MCP tools map onto the `PyMDPInterface`.

5. **[MCP Tools Reference](mcp_tools.md)**
   The complete set of real tools exposed by the servers — every method on
   `PyMDPInterface`, with signatures, parameters, and return values.

6. **[Implementation Details](implementation_details.md)**
   Technical detail on the layered architecture: `PyMDPInterface`, the FastMCP
   server, the FastAPI server, JSON/NumPy serialization, and session handling.

7. **[Advanced Features](advanced_features.md)**
   The capabilities that actually exist beyond the core loop: gridworld and
   custom environments, visualization/belief-dynamics, free-energy analysis,
   sessions, and direct vs. network clients.

8. **[PyMDP Integration](pymdp_integration.md)**
   How this project integrates the PyMDP pip package (`inferactively-pymdp`)
   as an unmodified dependency, and how compatibility is verified.

9. **[Implementation Status Report](../mcp_status_report.md)**
   A dated snapshot of the current state: what works, test status, and known
   open items.

## Getting Started

1. Install the package and PyMDP (see the root [`README.md`](../README.md)).
2. Use `PyMDPInterface` directly for scripting, or run one of the two servers.
3. See [mcp_tools.md](mcp_tools.md) for the full method reference.

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
