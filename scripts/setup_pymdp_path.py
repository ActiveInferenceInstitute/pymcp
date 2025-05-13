#!/usr/bin/env python3
"""
Helper script to set up the Python path for PyMDP and MCP modules.

This script adds the necessary paths to sys.path to ensure that both
the local pymdp-clone directory and the src directory are in the Python path.

Import this script at the beginning of any script that needs to use
PyMDP or MCP modules.
"""

import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.absolute()

# Add the src directory to the Python path
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add the pymdp-clone directory to the Python path
pymdp_path = project_root / "pymdp-clone"
if str(pymdp_path) not in sys.path:
    sys.path.insert(0, str(pymdp_path))

# Print a message if this script is run directly
if __name__ == "__main__":
    print(f"Python path set up for PyMDP and MCP modules.")
    print(f"Project root: {project_root}")
    print(f"Added to Python path:")
    print(f"  - {src_path}")
    print(f"  - {pymdp_path}") 