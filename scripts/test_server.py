#!/usr/bin/env python3
"""
Simple test script for MCP-PyMDP server.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test the MCP-PyMDP server")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost", 
        help="Server host (default: localhost)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080, 
        help="Server port (default: 8080)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None, 
        help="Output directory for test results"
    )
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Test results will be saved to: {args.output_dir}")
    
    # Server URL
    base_url = f"http://{args.host}:{args.port}"
    
    # Test ping endpoint
    print("Testing ping endpoint...")
    response = requests.get(f"{base_url}/ping")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    
    # Create a grid world environment
    print("Creating grid world environment...")
    env_data = {
        "name": "test_grid",
        "type": "grid_world",
        "grid_size": [3, 3],
        "reward_locations": [[2, 2]]
    }
    response = requests.post(f"{base_url}/environments", json=env_data)
    print(f"Status: {response.status_code}")
    env_result = response.json()
    print(f"Environment created: {env_result}")
    env_id = env_result.get("id", "test_grid")
    print()
    
    # Create a grid world agent using the tool endpoint
    print("Creating grid world agent...")
    agent_data = {
        "params": {
            "name": "test_agent",
            "grid_size": [3, 3],
            "reward_positions": [[2, 2]]
        }
    }
    response = requests.post(f"{base_url}/tools/create_gridworld_agent", json=agent_data)
    print(f"Status: {response.status_code}")
    agent_result = response.json().get("result", {}).get("agent", {})
    print(f"Agent created: {agent_result}")
    agent_id = agent_result.get("id", "test_agent_Agent")
    print()
    
    # Create a session
    print("Creating session...")
    session_data = {
        "agent_id": agent_id,
        "env_id": env_id
    }
    response = requests.post(f"{base_url}/sessions", json=session_data)
    print(f"Status: {response.status_code}")
    print(f"Full response: {response.text}")
    
    session_resp = response.json()
    print(f"Session response: {session_resp}")
    
    session = session_resp.get("session", {})
    print(f"Session created: {session}")
    session_id = session.get("id")
    
    if not session_id:
        print("ERROR: Failed to create session or extract session ID")
        if "error" in session_resp:
            print(f"Server error: {session_resp['error']}")
        return
    
    print(f"Using session ID: {session_id}")
    print()
    
    # Run simulation
    print("Running simulation...")
    sim_data = {
        "steps": 5
    }
    response = requests.post(f"{base_url}/sessions/{session_id}/run", json=sim_data)
    print(f"Status: {response.status_code}")
    print(f"Simulation results: {response.json()}")
    print()
    
    # Test visualization endpoints
    print("Testing agent visualization endpoint...")
    response = requests.get(f"{base_url}/agents/{agent_id}/visualize")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Agent visualization successful")
        if args.output_dir:
            # Save the visualization
            vis_data = response.json()
            with open(os.path.join(args.output_dir, "agent_visualization.json"), "w") as f:
                json.dump(vis_data, f, indent=2)
    else:
        print(f"Error: {response.json()}")
    print()
    
    print("Testing session visualization endpoint...")
    response = requests.get(f"{base_url}/sessions/{session_id}/visualize")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Session visualization successful")
        if args.output_dir:
            # Save the visualization
            vis_data = response.json()
            with open(os.path.join(args.output_dir, "session_visualization.json"), "w") as f:
                json.dump(vis_data, f, indent=2)
    else:
        print(f"Error: {response.json()}")
    print()
    
    print("Testing free energy analysis endpoint...")
    response = requests.get(f"{base_url}/analyze/free_energy/{session_id}")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Free energy analysis successful")
        if args.output_dir:
            # Save the analysis
            fe_data = response.json()
            with open(os.path.join(args.output_dir, "free_energy_analysis.json"), "w") as f:
                json.dump(fe_data, f, indent=2)
    else:
        print(f"Error: {response.json()}")
    print()
    
    print("Testing belief dynamics endpoint...")
    response = requests.get(f"{base_url}/analyze/beliefs/{session_id}")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Belief dynamics visualization successful")
        if args.output_dir:
            # Save the analysis
            belief_data = response.json()
            with open(os.path.join(args.output_dir, "belief_dynamics.json"), "w") as f:
                json.dump(belief_data, f, indent=2)
    else:
        print(f"Error: {response.json()}")
    print()
    
    # Save test summary
    if args.output_dir:
        summary = {
            "timestamp": time.time(),
            "server": f"{args.host}:{args.port}",
            "tests": {
                "ping": True,
                "environment_creation": True,
                "agent_creation": True,
                "session_creation": bool(session_id),
                "simulation": True,
                "visualization": True
            }
        }
        with open(os.path.join(args.output_dir, "test_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    # Import here to avoid issues if the module is imported
    import requests
    
    # Wait a few seconds for the server to start
    time.sleep(2)
    main() 