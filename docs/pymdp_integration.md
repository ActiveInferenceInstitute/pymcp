# PyMDP Integration

## About the `pymdp-clone` Directory

The `pymdp-clone` directory contains a complete clone of the [PyMDP GitHub repository](https://github.com/infer-actively/pymdp). This is **not** a mock or simplified version - it is the full, unmodified PyMDP library that provides all functionality for active inference in discrete state spaces.

## Why Include the Full PyMDP Codebase?

Including the complete PyMDP codebase serves several important purposes:

1. **Direct Integration**: The MCP server calls PyMDP functions directly, without any abstraction layers that might introduce differences in behavior.

2. **Complete Feature Support**: By including the entire PyMDP library, the MCP server supports all features of PyMDP, including the most advanced and experimental features.

3. **Consistency with Original Implementation**: Users familiar with PyMDP can be confident that the MCP server will produce identical results to using PyMDP directly.

4. **Access to Examples**: The PyMDP examples directory contains valuable reference implementations that the MCP server makes available through dedicated tools.

## How the MCP Server Uses PyMDP

The MCP server interacts with the PyMDP library in the following ways:

1. **Direct Imports**: The server imports PyMDP classes and functions directly from the pymdp-clone directory.

2. **Unmodified API Calls**: All calls to PyMDP functions use the original API without modification.

3. **Data Structure Compatibility**: The server maintains compatibility with PyMDP's data structures, using the same object arrays and NumPy arrays.

## Key PyMDP Components Used

The MCP server uses several key components from PyMDP:

### Agent Class

The PyMDP `Agent` class is the core component used by the MCP server. It provides the main interface for creating active inference agents, defining generative models, and performing inference and planning.

```python
from pymdp.agent import Agent

# The MCP server creates PyMDP Agent instances directly
agent = Agent(A=A, B=B, C=C, D=D)
```

### Inference Algorithms

The MCP server uses PyMDP's inference algorithms:

```python
from pymdp.algos import inference

# The MCP server uses PyMDP's inference functions directly
qs = inference.update_posterior_states(
    A, B, C, D, 
    observation, 
    method=inference_method
)
```

### Utility Functions

The MCP server uses PyMDP's utility functions for array manipulation, distribution operations, and information-theoretic calculations:

```python
from pymdp.utils import random_A_matrix, random_B_matrix
from pymdp.maths import spm_log, softmax

# The MCP server uses PyMDP's utility functions directly
A = random_A_matrix(num_obs, num_states)
```

## Versioning and Updates

The MCP server is kept in sync with the latest stable version of PyMDP. When PyMDP is updated, the pymdp-clone directory is updated accordingly to ensure that the MCP server always provides access to the most current features and bug fixes.

## Verification of Integration

To verify that the MCP server correctly integrates with PyMDP, comprehensive tests compare the results of operations performed through the MCP server with the results of the same operations performed directly using PyMDP.

These tests ensure that the MCP server faithfully reproduces PyMDP's behavior and provides users with a reliable interface to active inference functionality. 