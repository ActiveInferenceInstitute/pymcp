# PyMDP-MCP Test Suite Documentation

## Table of Contents
1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Test Coverage](#test-coverage)
5. [Validation Approach](#validation-approach)
6. [Output Organization](#output-organization)
7. [Recent Improvements](#recent-improvements)
8. [Future Development](#future-development)

## Overview

This test suite verifies that the PyMDP-MCP server is a complete, authentic implementation providing access to the real PyMDP library through the Model Context Protocol (MCP). The tests confirm proper integration with PyMDP by validating that:

1. The server uses the real PyMDP library, not a mock implementation
2. Results from the MCP server match results from direct PyMDP API calls
3. All key PyMDP features are accessible through the MCP interface
4. The system correctly handles complex active inference scenarios

## Test Structure

The test suite consists of several focused test files:

- **test_pymdp_interface.py**: Tests the PyMDPInterface class that connects the MCP server to the PyMDP library
- **test_mcp_tools.py**: Tests the MCP tools that expose PyMDP functionality through the MCP protocol
- **test_additional_functions.py**: Tests advanced features and edge cases of the PyMDP library integration
- **test_agent_visualization.py**: Tests visualization capabilities for active inference agents and their generative models

Each test file contains multiple test classes and methods that validate specific aspects of the integration:

| Test File | Focus Areas | Key Test Classes |
|-----------|-------------|------------------|
| test_pymdp_interface.py | Core functionality, API | TestPyMDPInterface |
| test_mcp_tools.py | MCP protocol, tools | TestMCPTools |
| test_additional_functions.py | Advanced features, learning | TestAdditionalFunctions |
| test_agent_visualization.py | Visualization, generative models | TestAgentVisualization |

## Running Tests

### Running All Tests

To run all tests:

```bash
python3 run_tests.py
```

This script will run all tests and generate a summary of the results, including information about test outputs.

### Running Specific Test Categories

You can run specific test categories:

```bash
python3 run_tests.py --category interface
python3 run_tests.py --category mcp
python3 run_tests.py --category advanced
python3 run_tests.py --category visualization
```

### Running Specific Test Files

You can run individual test files:

```bash
python3 -m unittest tests/test_pymdp_interface.py
python3 -m unittest tests/test_mcp_tools.py
python3 -m unittest tests/test_additional_functions.py
python3 -m unittest tests/test_agent_visualization.py
```

### Running Specific Test Classes or Methods

To run a specific test class:

```bash
python3 -m unittest tests.test_pymdp_interface.TestPyMDPInterface
```

To run a specific test method:

```bash
python3 -m unittest tests.test_pymdp_interface.TestPyMDPInterface.test_create_agent
```

### Advanced Test Options

The `run_tests.py` script provides several options for controlling test execution:

```bash
# Run tests with increased verbosity
python3 run_tests.py --verbose

# Run tests and organize outputs into subdirectories
python3 run_tests.py --organize

# Run tests and generate a detailed summary
python3 run_tests.py --summary

# Run tests without removing previous output files
python3 run_tests.py --keep-old
```

## Test Coverage

The test suite provides comprehensive coverage of PyMDP's functionality:

| PyMDP Feature | Test Coverage | Test Files | Key Test Methods |
|--------------|--------------|------------|------------------|
| Agent Creation | ✅ High | test_pymdp_interface.py, test_additional_functions.py | test_create_agent, test_complex_generative_model |
| Generative Models | ✅ High | test_pymdp_interface.py, test_agent_visualization.py | test_define_generative_model, test_simple_agent_visualization |
| Observation Models (A) | ✅ High | test_pymdp_interface.py, test_agent_visualization.py | test_define_generative_model, test_multi_modality_agent_visualization |
| Transition Models (B) | ✅ High | test_pymdp_interface.py, test_agent_visualization.py | test_define_generative_model, test_grid_world_simulation_visualization |
| Preference Distribution (C) | ✅ High | test_pymdp_interface.py, test_agent_visualization.py | test_define_generative_model, test_grid_world_simulation_visualization |
| Prior Beliefs (D) | ✅ High | test_agent_visualization.py | test_simple_agent_visualization, test_multi_modality_agent_visualization |
| State Inference | ✅ High | test_pymdp_interface.py, test_additional_functions.py | test_infer_states, test_inference_methods_comparison |
| Policy Inference | ✅ High | test_pymdp_interface.py | test_infer_policies |
| Action Selection | ✅ High | test_pymdp_interface.py | test_sample_action |
| Parameter Learning | ✅ Medium | test_additional_functions.py | test_learning_parameters |
| Grid World Environment | ✅ High | test_pymdp_interface.py, test_agent_visualization.py | test_create_grid_world_env, test_grid_world_simulation_visualization |
| Custom Environments | ⚠️ Medium | test_additional_functions.py | test_custom_environment |
| Temporal Planning | ✅ High | test_additional_functions.py | test_temporal_planning |
| Multi-factor Models | ⚠️ Medium | test_additional_functions.py | test_complex_generative_model |
| MCP Integration | ✅ High | test_mcp_tools.py | Various methods |
| Visualization | ✅ High | test_pymdp_interface.py, test_agent_visualization.py | test_visualize_simulation, test_grid_world_simulation_visualization |

### Active Inference Core Components

#### Perception (State Inference)

Tests cover the following aspects of perception in active inference:

- **Variational Inference Methods**: Both Variational Message Passing (VMP) and Fixed-Point Iteration (FPI) methods are tested.
- **Multiple Observation Modalities**: Tests with agents having multiple sensory channels.
- **Hidden State Inference**: Tests verify correct posterior beliefs over hidden states.
- **Hierarchical Processing**: Tests for multi-factor state spaces.

#### Action (Policy Selection)

Tests cover the following aspects of action selection in active inference:

- **Expected Free Energy Calculation**: Tests verify correct expected free energy calculation.
- **Policy Posterior Computation**: Tests verify correct probability distribution over policies.
- **Action Sampling**: Tests verify actions are sampled from the correct distribution.
- **Multi-step Policies**: Tests verify planning over multiple time steps.

#### Learning (Parameter Updates)

Tests cover the following aspects of learning in active inference:

- **Dirichlet Parameter Updates**: Tests verify correct updating of Dirichlet parameters.
- **A Matrix Learning**: Tests verify learning of observation model parameters.
- **B Matrix Learning**: Tests verify learning of transition model parameters.

## Validation Approach

The tests use a multi-layered validation approach to ensure correct integration with PyMDP:

1. **Direct API Comparison**: Compare results obtained via the MCP server with results obtained by calling PyMDP directly
2. **Instance Verification**: Verify that objects created by the MCP server are instances of the corresponding PyMDP classes
3. **Functional Validation**: Test end-to-end functionality to ensure the server behaves like a proper PyMDP implementation
4. **Visual Validation**: Generate visualizations that confirm the generative model structure and agent behavior

## Output Organization

All test outputs are saved to the `tests/output` directory, which contains the following subdirectories:

- **generative_models/**: Visualizations of agent A, B, C, and D matrices
- **simulations/**: Visualizations of agent trajectories, policies, and actions
- **belief_dynamics/**: Visualizations of belief updates over time
- **free_energy/**: Free energy calculations and expected free energy components
- **logs/**: Detailed computation logs and JSON data
- **results/**: Test results and summary files

Example output files include:

| File Type | Example Files | Purpose |
|-----------|---------------|---------|
| Generative Model Visualizations | `simple_agent_generative_model.png` | Visualize A, B, C, D matrices |
| Simulation Visualizations | `grid_agent_simulation.png` | Show agent trajectories and environment |
| Belief Dynamics | `grid_agent_belief_dynamics.png` | Track belief updates over time |
| Free Energy Components | `grid_agent_expected_free_energy.png` | Show risk, ambiguity, and total EFE |
| Computation Logs | `grid_agent_simulation_computation_logs.json` | Detailed logs of inference steps |
| Test Results | `simulation_test_results.json` | Test input, output, and validation data |

## Recent Improvements

The test suite has undergone several recent improvements:

### Documentation Improvements

1. ✅ **Enhanced Documentation Structure**
   - Consolidated multiple documentation files
   - Added detailed coverage information
   - Included comprehensive running instructions

2. ✅ **Added Test Coverage Documentation**
   - Created a detailed mapping of PyMDP features to test coverage
   - Highlighted which aspects of Active Inference are tested
   - Identified areas for future test coverage expansion

### Test Organization Improvements

1. ✅ **Output Directory Structure**
   - Created subdirectories for different output types
   - Added automatic file organization
   - Implemented output file tracking and validation

2. ✅ **Test Categorization**
   - Added test categories in run_tests.py
   - Implemented selective test running
   - Added verbosity controls

### Process Improvements

1. ✅ **Enhanced run_tests.py**
   - Added command-line arguments for specific test categories
   - Implemented verbosity controls
   - Added options to preserve existing output files
   - Added automatic output file organization

2. ✅ **Added Summary Generation**
   - Enhanced summary outputs with file statistics
   - Added support for detailed JSON test summaries
   - Included categorization of output files

### Visualization Tests

1. ✅ **Added Comprehensive Agent Visualization**
   - Created tests for visualizing agent generative models
   - Added visualization of A, B, C, and D matrices
   - Included information panels explaining the models

2. ✅ **Added Belief Dynamics Visualization**
   - Created tests for visualizing beliefs over time
   - Added visualization of multiple state factors
   - Included heatmap representations of belief changes

3. ✅ **Enhanced Simulation Visualization**
   - Improved visualization of agent trajectories
   - Added visualization of policy selection
   - Added visualization of expected free energy components

## Future Development

The following areas could benefit from expanded test coverage and improvements:

1. **More Complex Hierarchical Models**: Additional tests for deeper hierarchical models.
2. **Continuous State Space Models**: Tests for continuous state representations.
3. **Custom Inference Schemes**: Tests for custom message passing algorithms.
4. **Precision Parameters**: More thorough testing of precision parameters and their effects.
5. **Advanced Learning Mechanisms**: Tests for more complex learning scenarios and parameter sharing.
6. **Alternative Environment Types**: Tests for different environment classes beyond grid worlds.

### Priority Tasks

If continuing with improvements, the following tasks are recommended in order of priority:

1. **Update Existing Test Docstrings**: Ensure all test methods have standardized docstrings.
2. **Add More Edge Case Tests**: Improve robustness with additional test cases.
3. **Set Up CI Integration**: Automate testing through GitHub Actions.
4. **Enhance Visualization Tests**: Add more comprehensive visualization capabilities.
5. **Implement Test Database**: Track and compare test results over time.

## Conclusion

The PyMDP-MCP test suite provides strong evidence that this is a real MCP server for PyMDP, offering complete and accurate access to PyMDP's functionality through the Model Context Protocol. The comprehensive testing, detailed visualizations, and thorough validation ensure that clients can rely on this server for authentic active inference capabilities. 