"""
MCP Client Configuration.

This module provides configuration utilities for MCP client connection.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_CONFIG = {
    "server": {
        "host": "localhost",
        "port": 8080,
        "use_ssl": False
    },
    "auth": {
        "enabled": False,
        "token": None,
        "username": None,
        "password": None
    },
    "options": {
        "timeout": 30,
        "retry_attempts": 3,
        "retry_delay": 2
    }
}

def get_config_path() -> Path:
    """Get the path to the configuration file.
    
    Looks for configuration in the following places (in order):
    1. Path specified in MCP_CONFIG_PATH environment variable
    2. .mcp/config.json in the user's home directory
    3. .mcp.json in the current working directory
    
    Returns
    -------
    Path
        Path to the configuration file
    """
    # Check environment variable
    env_path = os.environ.get("MCP_CONFIG_PATH")
    if env_path and os.path.exists(env_path):
        return Path(env_path)
    
    # Check home directory
    home_path = Path.home() / ".mcp" / "config.json"
    if home_path.exists():
        return home_path
    
    # Check current directory
    cwd_path = Path.cwd() / ".mcp.json"
    if cwd_path.exists():
        return cwd_path
    
    # Default to home directory (even if it doesn't exist yet)
    return home_path

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the MCP client configuration.
    
    Parameters
    ----------
    config_path : Path, optional
        Path to the configuration file, by default None (auto-detect)
    
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    if config_path is None:
        config_path = get_config_path()
    
    # If configuration file exists, load it
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
                
            # Merge with default configuration
            config = merge_configs(DEFAULT_CONFIG.copy(), user_config)
            return config
        
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading configuration from {config_path}: {str(e)}")
            print("Using default configuration")
            return DEFAULT_CONFIG.copy()
    
    # If configuration file doesn't exist, use default
    return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> bool:
    """Save the MCP client configuration.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    config_path : Path, optional
        Path to the configuration file, by default None (auto-detect)
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if config_path is None:
        config_path = get_config_path()
    
    # Create parent directory if it doesn't exist
    if not config_path.parent.exists():
        try:
            config_path.parent.mkdir(parents=True)
        except OSError as e:
            print(f"Error creating configuration directory: {str(e)}")
            return False
    
    # Save configuration
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return True
    
    except IOError as e:
        print(f"Error saving configuration to {config_path}: {str(e)}")
        return False

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.
    
    Parameters
    ----------
    base_config : Dict[str, Any]
        Base configuration dictionary
    override_config : Dict[str, Any]
        Override configuration dictionary
    
    Returns
    -------
    Dict[str, Any]
        Merged configuration dictionary
    """
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merge_configs(base_config[key], value)
        else:
            # Override base value
            base_config[key] = value
    
    return base_config

def get_server_url(config: Optional[Dict[str, Any]] = None) -> str:
    """Get the server URL from configuration.
    
    Parameters
    ----------
    config : Dict[str, Any], optional
        Configuration dictionary, by default None (load from file)
    
    Returns
    -------
    str
        Server URL
    """
    if config is None:
        config = load_config()
    
    server_config = config.get("server", {})
    host = server_config.get("host", "localhost")
    port = server_config.get("port", 8080)
    use_ssl = server_config.get("use_ssl", False)
    
    protocol = "https" if use_ssl else "http"
    return f"{protocol}://{host}:{port}" 