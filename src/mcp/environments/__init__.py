"""
Environments Module for PyMDP with MCP.

This package provides environment implementations for Active Inference agents.
"""

from .gridworld import GridWorldEnvironment, create_grid_world_env
from .custom import CustomEnvironment, create_custom_env

__all__ = [
    'GridWorldEnvironment',
    'create_grid_world_env',
    'CustomEnvironment',
    'create_custom_env'
] 