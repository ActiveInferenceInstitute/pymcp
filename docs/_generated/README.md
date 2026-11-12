# Generated documentation

Everything in this directory is produced by a script in `scripts/` and must not
be hand-edited. `./run.sh docs` fails when a committed file no longer matches
what its script renders.

| File | Produced by | Contents |
| --- | --- | --- |
| `arms.json` | `uv run --frozen python scripts/gate_facts.py --record` | The raw receipt: one measured wall time per `./run.sh` arm, the host, uv/Python versions, commit, and the collected test counts. |
| `gate-facts.md` | same (`--record` renders it, `--check` verifies it) | Human-readable arm table and test-suite counts rendered from `arms.json` |

Why a receipt instead of prose: wall times and collected counts change with the
code and the host. Putting them in a generated table keeps one source of truth
and makes a stale number a gate failure instead of a documentation bug.

Wall times measure one run per arm on one host. They describe that host and are
not a benchmark claim.
