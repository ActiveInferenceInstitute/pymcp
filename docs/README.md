# MCP-PyMDP Documentation

Welcome to the documentation for MCP-PyMDP, a comprehensive implementation of the Message-based Cognitive Protocol (MCP) for the PyMDP active inference framework.

## Contents

This documentation is organized into the following sections:

1. **[MCP-PyMDP Integration](mcp_pymdp_integration.md)**
   - Detailed overview of how MCP and PyMDP are integrated
   - Architecture and component breakdown
   - Data flow and session management

2. **[MCP Tools](mcp_tools.md)**
   - Complete reference for all MCP tools provided by the server
   - Parameter descriptions and expected return values
   - Code examples for each tool

3. **[Implementation Details](implementation_details.md)**
   - Technical details of the implementation
   - Core components and their functionality
   - Client implementation options

4. **[Advanced Features](advanced_features.md)**
   - Specialized visualization capabilities
   - Session management and computation logging
   - Asynchronous operation and performance optimizations

5. **[PyMDP Integration](pymdp_integration.md)**
   - Information about the PyMDP library integration
   - Direct access to PyMDP functionality
   - Versioning and compatibility information

## Getting Started

If you're new to MCP-PyMDP, we recommend starting with the following steps:

1. Read the [MCP-PyMDP Integration](mcp_pymdp_integration.md) document for a high-level overview
2. Explore the [MCP Tools](mcp_tools.md) reference to understand available functionality
3. Check the [Implementation Details](implementation_details.md) for technical information

## Running Examples

The `scripts` directory contains several example scripts that demonstrate the use of MCP-PyMDP:

- **Agent Demo**: Basic agent creation and inference
- **Grid World Tutorial**: Complete example of an agent in a grid world environment
- **Free Energy Calculation**: Detailed explanation of free energy computation
- **Agent Loop**: Example of a full perception-action cycle

All scripts should be run using `python3` (not `python`). You can run them as follows:

```bash
# Start the MCP server in background
cd scripts
python3 start_mcp_server.py --port 8090 --output-dir /tmp/mcp_test_output > /tmp/mcp_server.log 2>&1 &

# Run individual examples
python3 mcp_gridworld_examples.py --port 8090 --output-dir /tmp/gridworld_examples
python3 mcp_pymdp_example.py --port 8090 --output-dir /tmp/pymdp_example
python3 pymcp_examples_fixed.py --port 8090 --output-dir /tmp/pymcp_examples_fixed

# Or run all examples at once
python3 run_all_examples.py --port 8090 --output-dir /tmp/all_examples
```

Examples output will be saved to timestamped directories under `scripts/output/` by default, or to the directory specified with the `--output-dir` parameter.

## Server and Client

The MCP server can be run using the main script:

```
python src/main.py
```

Clients can connect to the server using either the network-based client or the direct client:

```python
from mcp.client import MCPClient, DirectMCPClient

# Network client
client = MCPClient(host="localhost", port=8050)

# Direct client (for testing without network overhead)
direct_client = DirectMCPClient(output_dir="./output")
```

## Development

For developers looking to extend MCP-PyMDP, please refer to the [Implementation Details](implementation_details.md) document for guidance on architecture and design patterns.

## Code Assessment and Recommendations

Based on a comprehensive assessment of the source code, here are key observations and recommendations:

### Current Strengths

1. **Well-structured Architecture**: Clear separation between server, interface, and client layers
2. **Comprehensive Error Handling**: Most functions have try/except blocks with detailed error messages
3. **Type Annotations**: Consistent use of type hints throughout the codebase
4. **Asynchronous Design**: Properly implemented async/await patterns for efficient operation

### Improvement Areas

1. **Documentation Coverage**:
   - Additional inline documentation needed in `src/mcp/utils.py` for complex algorithms
   - More comprehensive docstrings for visualization functions

2. **Code Organization**:
   - Consider breaking up the large `utils.py` file (1135 lines) into smaller modules
   - Create separate modules for state inference, environment management, and agent creation

3. **Redundancy**:
   - Some duplicate code for array conversion in utils.py could be consolidated
   - Multiple similar visualization functions could use common helper functions

4. **Test Coverage**:
   - More comprehensive unit tests needed for the PyMDPInterface class
   - Additional integration tests for end-to-end workflows

### High-Priority Recommendations

1. **Refactor `utils.py`**: Split into logical modules for better maintainability
2. **Improve API Documentation**: Add more examples to function docstrings
3. **Add Error Handling**: Better handling of edge cases in matrix operations
4. **Enhance Logging**: More detailed logging throughout the codebase

For developers working on the codebase, these improvements would enhance maintainability and extensibility while preserving the current functionality. 