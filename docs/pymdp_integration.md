# PyMDP Integration

The ordinary `inferactively-pymdp` dependency is declared as `>=1.0.5` and
locked at 1.0.5. No vendored PyMDP or replacement scientific engine is used.
Do not upgrade it independently of the reviewed lock or install a conflicting
official `mcp` namespace over this project's bundled package.

## Native boundary

`PyMDPInterface.create_agent` admits an unbatched native model, makes owned
copies, adds the batch dimension and constructs the real immutable
`pymdp.agent.Agent`. The native constructor derives controllable factors from
their action dimensions; passive one-action factors remain valid.
Configuration is passed at construction, not assigned to the agent afterward.

State inference receives actual one-hot observations and the current predictive
prior. Policy inference consumes that posterior; selection consumes the full
policy posterior. A successful, token-matched transition or explicit external
commit advances the prior once. Selection and inspection do not advance it.

Filtering, model interchange, signed preferences, normalization, unsupported
method/horizon rejection and reset semantics are defined in the
[tool contract](mcp_tools.md). The wrapper intentionally exposes only that
subset, not every upstream algorithm or learning operation.

## Environments and measurements

Grid and custom environments are real objects implementing step/reset.
Custom observation modalities are fixed by the declared model; rewards do
not create extra observation modalities. Serialized environment metadata is
not the live environment implementation.

Numerical Bayes/VFE helpers use stable log-space arithmetic with explicit
support admission. VFE uses the actual prior and observation; plots consume
stored measurement records and full policy vectors. Neither an illustrative
curve nor a summary scalar substitutes for those records.

## Verification boundary

Focused native, transport, filesystem, process and renderer contracts are
listed in [the test inventory](../tests/README.md). They establish the named
engineering paths, not scientific validity, every PyMDP capability, installed
artifact behavior or the full GNN/release gate. Historical gate facts remain
dated producer output until the canonical gate records new measurements.

