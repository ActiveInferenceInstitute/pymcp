"""
Testing Utilities for PyMDP with MCP.

This package provides utility functions for testing PyMDP with MCP.
"""

from .setup import (
    setup_test_environment,
    setup_output_directories,
    clean_output_directory
)

from .fixtures import (
    create_simple_generative_model,
    create_multimodal_generative_model,
    create_gridworld_generative_model,
    create_grid_world_environment,
    create_custom_environment
)

from .helpers import (
    compare_matrices,
    compare_generative_models,
    save_test_results,
    generate_test_report,
    plot_test_results,
    memory_profile
)

__all__ = [
    # Setup
    'setup_test_environment',
    'setup_output_directories',
    'clean_output_directory',
    
    # Fixtures
    'create_simple_generative_model',
    'create_multimodal_generative_model',
    'create_gridworld_generative_model',
    'create_grid_world_environment',
    'create_custom_environment',
    
    # Helpers
    'compare_matrices',
    'compare_generative_models',
    'save_test_results',
    'generate_test_report',
    'plot_test_results',
    'memory_profile'
] 