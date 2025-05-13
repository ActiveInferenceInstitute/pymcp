# MCP-PyMDP Example Scripts

This directory contains various scripts for testing and demonstrating the MCP-PyMDP integration.

## Setup

Before running any scripts, ensure you have installed the required dependencies:

```bash
pip install -r ../requirements.txt
```

Also make sure that both the `src` directory and the `pymdp-clone` directory are in your Python path:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/../src:$(pwd)/../pymdp-clone
```

## Scripts Overview

### Server Scripts

These scripts start the MCP server that provides PyMDP functionality:

- `start_mcp_server.py`: Primary server script that implements the MCP protocol
- `start_fastapi_server.py`: Alternative server implementation using FastAPI directly

### Example Scripts

These scripts demonstrate various aspects of the PyMDP active inference framework:

- `mcp_pymdp_example.py`: Basic PyMDP examples with simple agent-environment interactions
- `mcp_gridworld_examples.py`: Detailed gridworld environment examples with visualizations
- `pymcp_examples.py`: Comprehensive examples covering all core PyMDP functionalities

### Test Scripts

These scripts are for testing the MCP server and PyMDP integration:

- `test_server.py`: Basic connectivity tests for the MCP server
- `test_agent.py`: Direct PyMDP agent testing without MCP interface
- `test_mcp_pymdp.py`: Tests for the MCP-PyMDP integration layer

### Utility Scripts

These scripts provide utilities and configuration:

- `setup_pymdp_path.py`: Utility to set up the correct Python paths
- `run_with_server.sh`: Shell script to run an example with automatic server management
- `run_all_examples.py`: Python script to run all examples in sequence with server management

## Running the Examples

**Important: All scripts must be run using `python3` (not `python`).**

### Method 1: Using the run_with_server.sh script

This is the simplest way to run a single example:

```bash
./run_with_server.sh --example mcp_gridworld_examples.py
```

You can specify a different port and output directory:

```bash
./run_with_server.sh --port 8090 --output-dir ./my_outputs --example mcp_pymdp_example.py
```

### Method 2: Using the run_all_examples.py script

To run all examples in sequence:

```bash
python3 run_all_examples.py
```

### Method 3: Manual server and example execution

1. Start the server:
```bash
python3 start_mcp_server.py --port 8080
```

2. In another terminal, run an example:
```bash
python3 mcp_gridworld_examples.py --port 8080
```

## Output Organization

All scripts create timestamped output directories that contain:

- JSON files with all experiment data
- PNG visualizations of agents, environments, and simulations
- Log files with detailed execution information

## Recommended Execution Order

For first-time users, we recommend running scripts in this order:

1. `test_server.py` - Verify server functionality
2. `mcp_pymdp_example.py` - Understand basic agent-environment interaction
3. `mcp_gridworld_examples.py` - Explore gridworld environments in detail
4. `pymcp_examples.py` - Explore advanced PyMDP functionality

Or simply run `run_all_examples.py` to execute them all in the correct sequence. 