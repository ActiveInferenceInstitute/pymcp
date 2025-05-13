"""
PyMDP-MCP Test Suite

This package contains tests for the PyMDP-MCP server implementation.
All tests verify that the MCP server correctly integrates with the full PyMDP library.

Test outputs are saved to the tests/output directory.
"""

import os

# Define output directory for test results
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True) 