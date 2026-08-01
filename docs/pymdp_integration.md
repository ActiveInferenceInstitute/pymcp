# PyMDP Integration

## How this project uses PyMDP

This project integrates the **official PyMDP library** as an ordinary pip
dependency (`inferactively-pymdp`). There is **no vendored `pymdp-clone`
directory** in this repository and no modified copy of PyMDP. The code imports
PyMDP directly through its public API:

```python
from pymdp.agent import Agent        # src/mcp/utils.py
from pymdp.envs import Env
import pymdp.utils as utils
from pymdp.maths import get_joint_likelihood
```

## Why install PyMDP separately

`pymdp` is intentionally **not** pinned as a dependency in `pyproject.toml`, so
you must install it alongside this package:

```bash
uv pip install inferactively-pymdp
```

This keeps pymcp usable against whatever PyMDP version you already have in your
environment, and avoids resolving/conflicting PyMDP versions.

## Key PyMDP components used

### Agent class

The core of the integration. `PyMDPInterface.create_agent` constructs a PyMDP
`Agent` from JSON-decoded `A`, `B`, `C`, `D` matrices, normalizing them into the
object arrays PyMDP expects, and stores the instance for later inference.

```python
agent = Agent(A=A_obj, B=B_obj, C=C_obj, D=D_obj)
agent.infer_states(observation)
agent.infer_policies()
agent.sample_action()
```

### Environments

`create_grid_world_env` builds environments compatible with PyMDP's
`Env`-style step/reset interface, storing them in `PyMDPInterface.environments`.

### Maths and utils

`PyMDPInterface.calculate_free_energy` and
`infer_states_from_observation` use `pymdp.maths.get_joint_likelihood`;
agent creation reuses normalizing helpers from `pymdp.utils`.

## Data structure compatibility

PyMDP uses NumPy **object arrays** for its `A`/`B`/`C`/`D` matrices (one array
entry per observation modality / state factor). The interface converts between
these and plain JSON lists via `_convert_to_obj_array`, `_convert_from_obj_array`,
`_ensure_native_types`, and the `NumpyEncoder` used by the servers. This keeps
every MCP/HTTP tool JSON-in/JSON-out while remaining faithful to PyMDP's internal
representation.

## Verification

The test suite (under `tests/`, run with `pytest`) exercises the interface
directly and through the MCP layer, and confirms the wrapper reproduces the
results of direct PyMDP calls — so the integration is a thin, behavior-preserving
layer rather than a re-implementation. See the root
[`README.md`](../README.md) and [implementation_details.md](implementation_details.md)
for more.
