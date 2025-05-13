#!/usr/bin/env python3
"""
MCP-PyMDP Example Script.

This script demonstrates how to use the MCP client to interact with the MCP server
for running active inference simulations with PyMDP.
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add the pymdp-clone directory to the path for direct access to PyMDP if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "pymdp-clone"))

from mcp.client import MCPClient, MCPToolKit
from mcp.client.config import load_config, save_config

async def run_example(host="localhost", port=8080, use_ssl=False, output_dir=None):
    """Run the example simulation."""
    # Create output directory with timestamp if not provided
    if output_dir is None:
        # Use scripts/output as the base directory instead of the current directory
        scripts_dir = Path(__file__).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(scripts_dir / "output" / f"example_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")
    
    # Set server URL
    protocol = "https" if use_ssl else "http"
    server_url = f"{protocol}://{host}:{port}"
    print(f"Connecting to MCP server at {server_url}")
    
    # Create client and toolkit
    async with MCPClient(server_url=server_url) as client:
        toolkit = MCPToolKit(client)
        
        # Check server connection
        try:
            ping_result = await client.ping()
            print(f"Server connection: {ping_result.get('status', 'unknown')}")
        except Exception as e:
            print(f"Error connecting to server: {str(e)}")
            print("Attempting to continue with direct client...")
        
        # Get available tools
        try:
            tools = await client.get_tools()
            print(f"Available tools: {len(tools)}")
            
            # Save the tool list to output directory for reference
            with open(os.path.join(output_dir, "available_tools.json"), "w") as f:
                json.dump(tools, f, indent=2)
        except Exception as e:
            print(f"Error fetching tools: {str(e)}")
        
        # STEP 1: Define a simple generative model manually
        print("\nDefining a custom generative model...")
        
        # Define A dimensions: [[obs_dim1, state_dim1], [obs_dim2, state_dim2]]
        A_dims = [[9, 9], [2, 9]]
        
        # Define B dimensions: [[state_dim1, state_dim1, control_dim1], ...]
        B_dims = [[9, 9, 4]]
        
        # Define the generative model
        try:
            generative_model = await toolkit.client.call_tool(
                "define_generative_model", 
                {"A_dims": json.dumps(A_dims), "B_dims": json.dumps(B_dims)}
            )
            
            if "error" in generative_model:
                print(f"Error defining generative model: {generative_model.get('error')}")
            else:
                # Save the model for reference
                model_file = os.path.join(output_dir, "generative_model.json")
                with open(model_file, "w") as f:
                    json.dump(generative_model, f, indent=2)
                print(f"Saved generative model to: {model_file}")
        except Exception as e:
            print(f"Error defining generative model: {str(e)}")
        
        # STEP 2: Create a grid world agent using the toolkit
        print("\nCreating grid world agent...")
        agent = await toolkit.create_gridworld_agent(
            name="GridWorldAgent",
            grid_size=[3, 3],
            reward_positions=[[2, 2]]
        )
        # Extract agent ID, handling both formats returned by the server
        if isinstance(agent, dict):
            agent_id = agent.get("id", "GridWorldAgent")
            # If id is not in the top level, check for nested structure
            if not agent_id and "agent" in agent and isinstance(agent["agent"], dict):
                agent_id = agent["agent"].get("id", "GridWorldAgent")
            # If still no ID, use the name
            if not agent_id:
                agent_id = "GridWorldAgent"
        else:
            agent_id = "GridWorldAgent"
            
        print(f"Agent created with ID: {agent_id}")
        
        # STEP 3: Create a grid world environment
        print("\nCreating grid world environment...")
        env = await toolkit.create_gridworld_environment(
            name="GridWorldEnv",
            grid_size=[3, 3],
            reward_positions=[[2, 2]]
        )
        # Extract environment ID, handling different response formats
        if isinstance(env, dict):
            env_id = env.get("id", "GridWorldEnv")
            # If id is not in the top level, check for nested structure
            if not env_id and "environment" in env and isinstance(env["environment"], dict):
                env_id = env["environment"].get("id", "GridWorldEnv")
            # If still no ID, use the name
            if not env_id:
                env_id = "GridWorldEnv"
        else:
            env_id = "GridWorldEnv"
            
        print(f"Environment created with ID: {env_id}")
        
        # STEP 4: Get agent and environment details
        print("\nRetrieving agent and environment details...")
        try:
            agent_details = await client.get_agent(agent_id)
            env_details = await client.get_environment(env_id)
            
            # Save details
            with open(os.path.join(output_dir, "agent_details.json"), "w") as f:
                json.dump(agent_details, f, indent=2)
                
            with open(os.path.join(output_dir, "environment_details.json"), "w") as f:
                json.dump(env_details, f, indent=2)
                
            print(f"Agent and environment details saved to output directory")
        except Exception as e:
            print(f"Error retrieving details: {str(e)}")
        
        # STEP 5: Run inference on the agent with a sample observation
        print("\nTesting inference on agent with sample observation...")
        try:
            # Create an observation (position 0, no reward)
            observation = [0, 0]
            
            # Infer states
            infer_result = await toolkit.client.call_tool(
                "infer_states", 
                {
                    "agent_id": agent_id, 
                    "observation": json.dumps(observation),
                    "method": "FPI"  # Fixed-point iteration method
                }
            )
            
            # Save inference results
            with open(os.path.join(output_dir, "inference_result.json"), "w") as f:
                json.dump(infer_result, f, indent=2)
                
            print(f"Inference results saved to output directory")
            
            # Infer policies
            policies_result = await toolkit.client.call_tool(
                "infer_policies", 
                {"agent_id": agent_id}
            )
            
            # Save policy inference results
            with open(os.path.join(output_dir, "policy_result.json"), "w") as f:
                json.dump(policies_result, f, indent=2)
                
            print(f"Policy inference results saved to output directory")
            
            # Sample an action
            action_result = await toolkit.client.call_tool(
                "sample_action", 
                {"agent_id": agent_id}
            )
            
            # Save action sampling results
            with open(os.path.join(output_dir, "action_result.json"), "w") as f:
                json.dump(action_result, f, indent=2)
                
            print(f"Action sampling results saved to output directory")
        except Exception as e:
            print(f"Error during inference: {str(e)}")
        
        # STEP 6: Run the simulation
        print("\nRunning simulation...")
        try:
            # Get session details
            session_response = await client.call_tool(
                "create_session",
                {
                    "agent_id": agent_id,
                    "environment_id": env_id
                }
            )
            session = session_response.get("session", {})
            session_id = session.get("id")
            
            # Run simulation
            simulation_response = await client.call_tool(
                "run_simulation",
                {
                    "agent_id": agent_id,
                    "environment_id": env_id,
                    "steps": 10,
                    "save_history": True
                }
            )
            
            # Check if simulation has timesteps
            timesteps = len(simulation_response.get("history", {}).get("timesteps", []))
            
            # If simulation has no timesteps, run manual simulation
            if timesteps == 0:
                print("Server simulation returned no timesteps, running manual simulation...")
                
                # Reset environment
                reset_result = await client.call_tool(
                    "reset_environment",
                    {"environment_id": env_id}
                )
                observation = reset_result.get("observation", [0])
                
                # Initialize history with timesteps structure
                history = {
                    "agent_id": agent_id,
                    "env_id": env_id,
                    "timesteps": []
                }
                
                # Run manual simulation loop
                for step in range(10):  # Run 10 steps
                    # Step data
                    step_data = {"step": step}
                    
                    # Infer states
                    infer_result = await client.call_tool(
                        "infer_states",
                        {
                            "agent_id": agent_id,
                            "observation": json.dumps(observation),
                            "method": "FPI"
                        }
                    )
                    posterior_states = infer_result.get("posterior_states", [])
                    step_data["belief"] = [posterior_states]
                    
                    # Infer policies
                    policy_result = await client.call_tool(
                        "infer_policies",
                        {"agent_id": agent_id}
                    )
                    policy_posterior = policy_result.get("policy_posterior", [])
                    step_data["policy"] = policy_posterior
                    
                    # Sample action
                    action_result = await client.call_tool(
                        "sample_action",
                        {"agent_id": agent_id}
                    )
                    action = action_result.get("action", 0)
                    step_data["action"] = action
                    
                    # Step environment
                    step_result = await client.call_tool(
                        "step_environment",
                        {"environment_id": env_id, "action": action}
                    )
                    observation = step_result.get("observation", [0])
                    reward = step_result.get("reward", 0.0)
                    done = step_result.get("done", False)
                    
                    # Update step data
                    step_data["observation"] = observation
                    step_data["reward"] = reward
                    step_data["done"] = done
                    
                    # Add to history
                    history["timesteps"].append(step_data)
                    
                    # Check if done
                    if done:
                        break
                
                # Update simulation response with new history
                simulation_response["history"] = history
                simulation_response["id"] = session_id or f"{agent_id}_{env_id}_session"
                
                # Get new timestep count
                timesteps = len(history["timesteps"])
            
            # Save simulation results
            with open(os.path.join(output_dir, "simulation.json"), "w") as f:
                json.dump(simulation_response, f, indent=2)
            
            print(f"Simulation completed with {timesteps} timesteps")
            
            # Generate visualizations
            print("\nGenerating visualizations...")
            
            try:
                session_id = simulation_response.get("id")
                if session_id:
                    # Try to visualize the simulation trajectory with server
                    try:
                        vis_result = await toolkit.visualize_simulation(
                            session_id=session_id,
                            output_file=os.path.join(output_dir, "simulation.png")
                        )
                    except Exception as e:
                        print(f"Server visualization failed: {str(e)}")
                        print("Generating local visualization...")
                        
                        # Create custom visualization showing agent trajectory on grid
                        try:
                            history = simulation_response.get("history", {}).get("timesteps", [])
                            if history:
                                # Generate local visualization
                                from matplotlib import pyplot as plt
                                import numpy as np
                                
                                # Create grid
                                grid_size = [3, 3]  # Default grid size
                                grid = np.zeros(grid_size)
                                
                                # Mark reward position
                                reward_position = [2, 2]  # Default reward at bottom right
                                grid[reward_position[0], reward_position[1]] = 1
                                
                                # Create figure
                                plt.figure(figsize=(8, 6))
                                plt.imshow(grid, cmap='viridis', alpha=0.3)
                                plt.colorbar(label='Reward')
                                
                                # Extract agent positions
                                positions = []
                                for step_data in history:
                                    observation = step_data.get("observation", [0])
                                    if observation and len(observation) > 0:
                                        pos = observation[0]
                                        row = pos // grid_size[1]
                                        col = pos % grid_size[1]
                                        positions.append((row, col))
                                
                                # Plot trajectory
                                if positions:
                                    x_coords = [p[1] for p in positions]
                                    y_coords = [p[0] for p in positions]
                                    plt.plot(x_coords, y_coords, 'r-', linewidth=2)
                                    plt.plot(x_coords[0], y_coords[0], 'go', markersize=12)  # Start
                                    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=12)  # End
                                
                                # Add grid and labels
                                plt.grid(True)
                                plt.title('Agent Trajectory in Grid World')
                                plt.xlabel('Column')
                                plt.ylabel('Row')
                                
                                # Save figure
                                plt.savefig(os.path.join(output_dir, "simulation_local.png"))
                                plt.close()
                                print("Local visualization saved to simulation_local.png")
                        except Exception as vis_e:
                            print(f"Local visualization also failed: {str(vis_e)}")
            except Exception as e:
                print(f"Visualization error: {str(e)}")
        except Exception as e:
            print(f"Error during simulation: {str(e)}")
        
        # STEP 7: Visualize the simulation results
        print("\nGenerating visualizations...")
        try:
            session_id = simulation_response.get("id")
            if session_id:
                # Try to visualize the simulation trajectory with server
                try:
                    vis_result = await toolkit.visualize_simulation(
                        session_id=session_id,
                        output_file=os.path.join(output_dir, "simulation.png")
                    )
                except Exception as e:
                    print(f"Server visualization failed: {str(e)}")
                    print("Generating local visualization...")
                    
                    # Create custom visualization showing agent trajectory on grid
                    try:
                        history = simulation_response.get("history", {}).get("timesteps", [])
                        if history:
                            # Generate local visualization
                            from matplotlib import pyplot as plt
                            import numpy as np
                            
                            # Create grid
                            grid_size = [3, 3]  # Default grid size
                            grid = np.zeros(grid_size)
                            
                            # Mark reward position
                            reward_position = [2, 2]  # Default reward at bottom right
                            grid[reward_position[0], reward_position[1]] = 1
                            
                            # Create figure
                            plt.figure(figsize=(8, 6))
                            plt.imshow(grid, cmap='viridis', alpha=0.3)
                            plt.colorbar(label='Reward')
                            
                            # Extract agent positions
                            positions = []
                            for step_data in history:
                                observation = step_data.get("observation", [0])
                                if observation and len(observation) > 0:
                                    pos = observation[0]
                                    row = pos // grid_size[1]
                                    col = pos % grid_size[1]
                                    positions.append((row, col))
                            
                            # Plot trajectory
                            if positions:
                                x_coords = [p[1] for p in positions]
                                y_coords = [p[0] for p in positions]
                                plt.plot(x_coords, y_coords, 'r-', linewidth=2)
                                plt.plot(x_coords[0], y_coords[0], 'go', markersize=12)  # Start
                                plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=12)  # End
                            
                            # Add grid and labels
                            plt.grid(True)
                            plt.title('Agent Trajectory in Grid World')
                            plt.xlabel('Column')
                            plt.ylabel('Row')
                            
                            # Save figure
                            plt.savefig(os.path.join(output_dir, "simulation_local.png"))
                            plt.close()
                            print("Local visualization saved to simulation_local.png")
                    except Exception as vis_e:
                        print(f"Local visualization also failed: {str(vis_e)}")
        except Exception as e:
            print(f"Visualization error: {str(e)}")
        
        print("\nExample completed successfully!")
        print(f"All outputs saved to: {output_dir}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run MCP-PyMDP example")
    
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
        "--use-ssl", 
        action="store_true", 
        help="Use HTTPS instead of HTTP"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None, 
        help="Output directory for generated files (default: scripts/output/example_TIMESTAMP)"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the example."""
    args = parse_args()
    asyncio.run(run_example(
        host=args.host,
        port=args.port,
        use_ssl=args.use_ssl,
        output_dir=args.output_dir
    ))

if __name__ == "__main__":
    main() 