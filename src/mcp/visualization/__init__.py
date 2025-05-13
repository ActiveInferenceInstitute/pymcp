"""
Visualization Module for PyMDP with MCP.

This package provides visualization functions for Active Inference models.
"""

from .generative_model import visualize_generative_model
from .free_energy import (
    visualize_free_energy_components,
    visualize_variational_free_energy
)
from .simulation import (
    visualize_simulation,
    create_belief_heatmap,
    visualize_policy_evolution,
    visualize_free_energy,
    create_simulation_animation
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

"""
Visualization module for the PyMDPInterface class.
"""

from .extensions import _add_visualization_methods 