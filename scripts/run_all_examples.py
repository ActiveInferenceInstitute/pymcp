#!/usr/bin/env python3
"""
Run all MCP-PyMDP tests and examples.

This script will start the MCP server and then run all the example scripts
to test the functionality in the correct sequence.
"""

import os
import sys
import time
import asyncio
import subprocess
import requests
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "pymdp-clone"))

def start_server(output_dir, port=8080):
    """Start the MCP server in a subprocess."""
    print("Starting MCP server...")
    server_process = subprocess.Popen(
        [sys.executable, "scripts/start_mcp_server.py", "--port", str(port), "--output-dir", output_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(project_root)
    )
    
    # Wait for the server to start by checking the endpoint
    max_retries = 10
    retry_interval = 1
    server_url = f"http://localhost:{port}/ping"
    
    print(f"Waiting for server to start on {server_url}...")
    for i in range(max_retries):
        try:
            response = requests.get(server_url, timeout=1)
            if response.status_code == 200:
                print(f"Server started successfully after {i+1} retries!")
                return server_process
        except requests.RequestException:
            pass
        
        # Check if process is still running
        if server_process.poll() is not None:
            print("Error: Server process terminated unexpectedly")
            stdout, stderr = server_process.communicate()
            print(f"Server stdout: {stdout.decode('utf-8')}")
            print(f"Server stderr: {stderr.decode('utf-8')}")
            return None
            
        print(f"Server not ready, retrying in {retry_interval} seconds... ({i+1}/{max_retries})")
        time.sleep(retry_interval)
    
    print("Error: Server failed to start within the timeout period")
    return None

def run_test_script(script_name, output_dir, port=8080):
    """Run a test script and return the result."""
    script_path = Path(__file__).parent / script_name
    print(f"\nRunning {script_name}...")
    
    # Create command with appropriate arguments
    cmd = [sys.executable, str(script_path), "--port", str(port), "--output-dir", output_dir]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(project_root)
    )
    
    print(f"--- {script_name} output ---")
    print(result.stdout)
    
    if result.stderr:
        print(f"--- {script_name} errors ---")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    """Run all tests and examples in the correct sequence."""
    print("Starting test suite for MCP-PyMDP")
    
    # Create a timestamp-based directory for all outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = project_root / "scripts" / "outputs" / f"run_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"All outputs will be saved to: {base_output_dir}")
    
    # Use a non-default port to avoid conflicts
    port = 8090
    
    # Create subdirectories for each test script
    script_dirs = {
        "server": str(base_output_dir / "server"),
        "test_server.py": str(base_output_dir / "test_server"),
        "mcp_pymdp_example.py": str(base_output_dir / "mcp_pymdp_example"),
        "mcp_gridworld_examples.py": str(base_output_dir / "mcp_gridworld_examples"),
        "pymcp_examples_fixed.py": str(base_output_dir / "pymcp_examples_fixed")
    }
    
    # Create all subdirectories
    for directory in script_dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    # Start the server
    server_process = start_server(script_dirs["server"], port=port)
    
    if server_process is None:
        print("Failed to start server. Exiting.")
        return 1
    
    try:
        # Define the scripts to run in the correct order
        scripts = [
            "test_server.py",             # Basic server functionality test
            "mcp_pymdp_example.py",       # Simple PyMDP examples 
            "mcp_gridworld_examples.py",  # Grid world examples
            "pymcp_examples_fixed.py"     # More advanced PyMDP examples
        ]
        
        # Run each script in sequence
        results = {}
        for script in scripts:
            results[script] = run_test_script(script, script_dirs[script], port=port)
            
            # If a critical script fails, we might want to stop
            if not results[script] and script == "test_server.py":
                print(f"Critical test {script} failed. Stopping further tests.")
                break
        
        # Print summary
        print("\n--- Test Results Summary ---")
        for script, success in results.items():
            status = "PASS" if success else "FAIL"
            print(f"{script}: {status}")
        
        # Print location of outputs
        print(f"\nAll test outputs are saved in: {base_output_dir}")
    
    finally:
        # Terminate the server
        print("\nStopping MCP server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Server didn't terminate gracefully, forcing...")
            server_process.kill()
    
    print("\nTest suite completed.")
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main()) 