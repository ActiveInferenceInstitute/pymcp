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

## PyMDP is a declared dependency

`inferactively-pymdp>=1.0.4` is declared in `pyproject.toml`, so installing this
package (e.g. `uv pip install -e .`) pulls PyMDP in automatically. Installing
PyMDP explicitly (`uv pip install inferactively-pymdp`) is only needed if you
want a version other than the declared floor, or an upgrade, in your
environment.

## Key PyMDP components used

### Agent class

The core of the integration. `PyMDPInterface.create_agent` normalizes the
JSON-decoded `A`, `B`, `C`, `D` matrices into the batched NumPy form the current
PyMDP expects, constructs the `Agent` with all configuration passed **at
construction time** (PyMDP agents are immutable equinox dataclasses — nothing
can be assigned afterwards), and stores the instance for later inference:

```python
agent = Agent(A=A_b, B=B_b, C=C_b, D=D_b, policy_len=1, num_controls=[...])
qs = agent.infer_states(obs_onehot, prior)   # one-hot, batched observation
q_pi, efe = agent.infer_policies(qs)         # posterior over hidden states
action = agent.sample_action(q_pi)           # policy posterior
```

The interface wraps these calls so callers never see the batched/one-hot
convention: it converts plain observation indices with `_onehot_observations`,
adds/removes the leading batch dimension (`_to_batch` / `_unwrap`), and keeps
each agent's prior (`D`) so `infer_states`, `infer_policies`, and
`sample_action` can be chained without extra bookkeeping.

### Environments

`create_grid_world_env` builds environments compatible with PyMDP's
`Env`-style step/reset interface, storing them in `PyMDPInterface.environments`.
Grid-world stepping is performed inline against the environment dict (no
attached functions), so environments stay JSON-serializable.

### Maths and utils

`PyMDPInterface.calculate_free_energy` and `infer_states_from_observation` use
`pymdp.maths.get_joint_likelihood`. Matrix normalization (columns of `A`, and
each distribution over destination states of `B`) is performed in-interface
before the `Agent` is constructed.

## Data structure compatibility

PyMDP uses NumPy **object arrays** for its `A`/`B`/`C`/`D` matrices (one array
entry per observation modality / state factor), and the current release adds a
leading **batch dimension** to every matrix. The interface converts between
these and plain JSON lists via `_convert_to_obj_array`, `_convert_from_obj_array`,
`_ensure_native_types`, `_to_batch`, `_unwrap`, and the `NumpyEncoder` used by
the servers. This keeps every MCP/HTTP tool JSON-in/JSON-out while remaining
faithful to PyMDP's internal representation.

## Verification

The test suite (under `tests/`, run with `./run.sh test-full`) exercises the
interface directly and through the MCP layer, and confirms the wrapper
reproduces the results of direct PyMDP calls — so the integration is a thin,
behavior-preserving layer rather than a re-implementation. See the root
[`README.md`](../README.md) and [implementation_details.md](implementation_details.md)
for more.
