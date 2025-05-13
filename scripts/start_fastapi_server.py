#!/usr/bin/env python3
"""
Script to start the FastAPI server for MCP-PyMDP.

This script initializes and runs the FastAPI application defined in src/mcp/server/app.py.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Ensure the project root and PyMDP clone are in the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "pymdp-clone"))

# Modify Python module path to include src
src_path = project_root / "src"
if src_path not in sys.path:
    sys.path.insert(0, str(src_path))

# Try to import directly
try:
    from mcp.server.app import create_app
    print("Successfully imported create_app from mcp.server.app")
except ImportError as e:
    print(f"Error importing create_app: {e}")
    sys.exit(1)

def main():
    """Run the FastAPI server."""
    # Create the FastAPI app
    app = create_app()
    
    # Get host and port from environment variables or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    
    print(f"Starting MCP-PyMDP FastAPI server on {host}:{port}")
    
    # Run the server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main() 