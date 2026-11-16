"""
MCP Client Module for PyMDP with MCP.

This package provides client utilities for interacting with the MCP server.
"""

from .config import DEFAULT_CONFIG, get_server_url, load_config, save_config
from .core import ClientError, MCPClient, RemoteError
from .direct import DirectMCPClient
from .tools import MCPToolKit

__all__ = [
    # Config
    'load_config',
    'save_config',
    'get_server_url',
    'DEFAULT_CONFIG',

    # Core
    'MCPClient',
    'DirectMCPClient',
    'ClientError',
    'RemoteError',

    # Tools
    'MCPToolKit'
]
