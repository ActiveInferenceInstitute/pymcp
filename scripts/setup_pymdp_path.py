#!/usr/bin/env python3
"""
Helper script to set up the Python path for the MCP package.

This script adds the `src` directory to sys.path so that MCP modules can be
imported. PyMDP itself is an installed package (see pyproject.toml:
`inferactively-pymdp`), so no local clone path is needed.

Import this script at the beginning of any script that needs to use
MCP modules.
"""

import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.absolute()

# Add the src directory to the Python path
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Print a message if this script is run directly
if __name__ == "__main__":
    print("Python path set up for MCP modules.")
    print(f"Project root: {project_root}")
    print("Added to Python path:")
    print(f"  - {src_path}")
