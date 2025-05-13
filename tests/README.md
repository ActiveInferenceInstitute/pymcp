# MCP-PyMDP Tests

This directory contains tests for the MCP-PyMDP project. The tests are organized by category and can be run using the `run_and_organize.sh` script.

## Running Tests

The easiest way to run tests is using the `run_and_organize.sh` script. By default, this script will run all tests and organize the outputs into appropriate directories.

```bash
# Run all tests
./run_and_organize.sh

# Run a specific category of tests
./run_and_organize.sh --category visualization

# Run tests for belief dynamics
./run_and_organize.sh --category belief_dynamics

# Run a specific test file
./run_and_organize.sh --file test_belief_dynamics.py

# Generate a detailed summary report
./run_and_organize.sh --summary

# Increase verbosity
./run_and_organize.sh --verbose
```

You can also run tests using the main application:

```bash
# From the project root
python src/main.py test --category belief_dynamics
```

## Test Categories

The tests are organized into the following categories:

- `interface`: Tests for the PyMDP interface
- `mcp`: Tests for MCP server and tools
- `advanced`: Tests for additional functionality
- `visualization`: Tests for visualization capabilities
- `belief_dynamics`: Tests for belief dynamics functionality
- `core`: Core tests (interface + mcp)
- `all`: All tests

## Test Outputs

Test outputs are organized in the `tests/output` directory, with subdirectories for different types of outputs:

- `generative_models`: Visualizations of generative models
- `simulations`: Simulation visualizations
- `belief_dynamics`: Belief dynamics visualizations
- `free_energy`: Free energy component visualizations
- `logs`: Log files
- `results`: Test result data

## Adding New Tests

When adding new tests:

1. Create a new test file in the appropriate directory
2. Use the appropriate test class or create a new one
3. Add the test file to the `TEST_CATEGORIES` in `run_tests.py`
4. Ensure tests use the modular code from `src/mcp/` rather than duplicating functionality

## Notes on Belief Dynamics

If the belief_dynamics directory is empty after running tests, you may need to run tests that specifically generate belief dynamics visualizations:

```bash
./run_and_organize.sh --category visualization
``` 