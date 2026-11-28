# Documentation index

Flat index of the long-lived documentation in this repository. `README.md` in
this directory is the hub (what to read first); this page is the complete list.
`./run.sh links` fails if a page directly under `docs/` is missing from this
index, so add the entry here when you add a page.

## Start here

- [`../README.md`](../README.md) — project overview, install, server modes
- [`README.md`](README.md) — documentation hub
- [`setup.md`](guides/setup.md) — install, run either server, `PyMDPInterface` basics
- [`RUN_GUIDE.md`](guides/RUN_GUIDE.md) — the gate entry point (`./run.sh`): arms, what
  CI runs, how measured numbers are produced
- [`testing.md`](development/testing.md) — test suite layout, `slow` marker, legacy runners,
  outputs

## Reference

- [`ARCHITECTURE.md`](architecture/ARCHITECTURE.md) — module layout, request lifecycle, design
  decisions
- [`implementation_details.md`](architecture/implementation_details.md) — `PyMDPInterface`,
  FastMCP server, FastAPI server, serialization, sessions
- [`mcp_tools.md`](reference/mcp_tools.md) — every tool/endpoint with signatures and returns
- [`mcp_pymdp_integration.md`](reference/mcp_pymdp_integration.md) — components and data
  flow from MCP to PyMDP
- [`pymdp_integration.md`](reference/pymdp_integration.md) — how the PyMDP dependency
  is declared and how compatibility is verified

## Guides

- [`advanced_features.md`](guides/advanced_features.md) — gridworld/custom
  environments, visualization, free-energy analysis, sessions, clients

## Generated facts

- [`_generated/README.md`](_generated/README.md) — what is generated here and
  which script produces it
- [`_generated/gate-facts.md`](_generated/gate-facts.md) — per-arm wall times,
  collected test counts, `slow`-marked split

## Repository-level records

- [`../CHANGELOG.md`](../CHANGELOG.md) — release notes
- [`../TODO.md`](../TODO.md) — open backlog
- [`../mcp_status_report.md`](../mcp_status_report.md) — dated implementation
  status snapshot
- [`../REVIEW_LOG_2026-08-02.md`](../REVIEW_LOG_2026-08-02.md) — dated review log
- [`../tests/README.md`](../tests/README.md) — test directory guide
- [`../tests/DOCUMENTATION_TESTING.md`](../tests/DOCUMENTATION_TESTING.md) — test
  categories and outputs
- [`../scripts/README.md`](../scripts/README.md) — example scripts and launchers
- [`../.github/GNN.md`](../.github/GNN.md) — what the pinned GNN conformance job
  parses

## Additional pages

- [Architecture](architecture/README.md)
- [Development](development/README.md)
- [Guides](guides/README.md)
- [Abstract](manuscript/00_abstract.md)
- [Model and interface contract](manuscript/01_contract.md)
- [Lifecycle and measured run records](manuscript/02_lifecycle.md)
- [Numerical and protocol evidence](manuscript/03_measurement.md)
- [Discussion and limits](manuscript/04_limits.md)
- [References](manuscript/99_references.md)
- [Manuscript guidance](manuscript/AGENTS.md)
- [A Transport-Independent Contract for PyMDP Active Inference](manuscript/README.md)
- [LaTeX preamble](manuscript/preamble.md)
- [Reference](reference/README.md)
