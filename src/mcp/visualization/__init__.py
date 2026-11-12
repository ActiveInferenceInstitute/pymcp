"""
Visualization Module for PyMDP with MCP.

This package provides visualization functions for Active Inference models.
"""

# Importing `extensions` is the effect: extensions.py calls
# `_add_visualization_methods()` at import time, which attaches
# `plot_belief_dynamics` and `analyze_free_energy` to `PyMDPInterface`.
from .extensions import _add_visualization_methods as _add_visualization_methods  # noqa: F401
from .free_energy import (
    visualize_free_energy_components,
    visualize_variational_free_energy,
)
from .generative_model import visualize_generative_model
from .simulation import (
    create_belief_heatmap,
    create_simulation_animation,
    visualize_free_energy,
    visualize_policy_evolution,
    visualize_simulation,
)

__all__ = [
    'visualize_generative_model',
    'visualize_free_energy_components',
    'visualize_variational_free_energy',
    'visualize_simulation',
    'create_belief_heatmap',
    'visualize_policy_evolution',
    'visualize_free_energy',
    'create_simulation_animation'
]
