# MCP-PyMDP Tests

The canonical gate remains `./run.sh`; its arms and measured facts are in
the [run guide](../docs/guides/RUN_GUIDE.md) and
[generated gate facts](../docs/_generated/gate-facts.md).
A focused receipt is not a replacement for the full gate or its measurements.

## Focused contracts

| Module | Consumer-visible contract |
| --- | --- |
| `test_core_contracts.py` | Actual observation/commit/reset lifecycle, prior ownership and bounded retention |
| `test_environment_contract.py` | Rectangular grid actions, custom modalities, reset and terminal options |
| `test_pymdp_interface.py` | Native model/support admission, immutable exports and observation validation |
| `test_additional_functions.py` | Stable numerical Bayes/VFE and admitted native configuration |
| `test_belief_dynamics.py` | Actual renderer arrays, state labels and pre/next-state separation |
| `test_agent_visualization.py` | Measured histories/logs, real GIF frames and primary/companion confinement |
| `test_mcp_tools.py` | REST/direct consumers, strict schemas, actual JSON/image bytes and errors |
| `test_mcp_full.py` | Real stdio JSON-RPC and ASGI SSE lifecycle, budgets and disconnect cleanup |
| `test_example_processes.py` | Real process ownership, readiness, deadlines, failures and measured examples |

`contract_support.py` supplies shared small inputs and real ASGI exchanges.
`test_gnn_conformance.py` protects existing local GNN verdict handling; those
unit controls alone are not a pinned-parser conformance run.

## Running

With dependencies already prepared, a focused payload is an ordinary pytest
selection, for example:

```bash
"$PROJECT_PY" -B -m pytest -q -p no:cacheprovider tests/test_core_contracts.py
```

Select this checkout's `src`, use an owned off-checkout temporary/cache root,
and use the qualified GitVault test runner where repository governance
requires it. Preserve real stdout/stderr, selected runtime/source identities,
case outcomes and child settlement. Do not silently deselect failed cases.

The legacy category organizers and inert `test_template.py` were retired.
There is no `src/main.py test` command. New regressions exercise the actual
consumer boundary; copied renderers, fabricated image files, source-wording
assertions and suppressed/skipped failures are not substitute proof.

## Output and limits

Fixtures use isolated temporary paths, actual images/JSON, and bounded real
subprocesses. No shared `tests/output` organizer is required. Inspect an actual
rendered surface as well as its arrays. These tests do not certify an image
build, installed wheel, distributed service, scientific result, full
coverage/lint gate or merge/release permission.

