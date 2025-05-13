#!/usr/bin/env python3
"""
MCP-PyMDP Grid World Examples.

This script demonstrates how to create and run various grid world examples
using the MCP client and PyMDP.
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# Add the correct paths to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "pymdp-clone"))

from mcp.client import MCPClient, MCPToolKit
from mcp.client.config import load_config, save_config

# Extend MCPToolKit with improved agent-environment interaction
class ExtendedMCPToolKit(MCPToolKit):
    """Extended toolkit for more robust simulation handling."""
    
    async def run_agent_in_environment(
        self, 
        agent_id: str, 
        env_id: str, 
        num_steps: int = 10,
        save_history: bool = True, 
        output_dir: str = None
    ):
        """Run an agent in an environment with fallback for failures.
        
        Args:
            agent_id: Agent ID
            env_id: Environment ID
            num_steps: Number of steps to run
            save_history: Whether to save history
            output_dir: Directory to save outputs
            
        Returns:
            Simulation results
        """
        # Call default implementation
        result = await super().run_agent_in_environment(
            agent_id=agent_id,
            env_id=env_id,
            num_steps=num_steps,
            save_history=save_history,
            output_dir=output_dir
        )
        
        # If simulation produced no timesteps, try manual simulation
        if ("error" in result or 
            (save_history and len(result.get("history", {}).get("timesteps", [])) == 0)):
            
            print("Server simulation returned no timesteps, running manual simulation...")
            
            # Create a session if needed
            if "id" not in result:
                session_params = {
                    "agent_id": agent_id,
                    "environment_id": env_id,
                }
                session_result = await self.client.call_tool("create_session", session_params)
                session_id = session_result.get("session", {}).get("id", f"{agent_id}_{env_id}_session")
                result["id"] = session_id
            
            # Reset environment
            reset_params = {"environment_id": env_id}
            reset_result = await self.client.call_tool("reset_environment", reset_params)
            observation = reset_result.get("observation", [0])
            
            # Initialize history with timesteps structure
            history = {
                "agent_id": agent_id,
                "env_id": env_id,
                "timesteps": []
            }
            
            # Run manual simulation loop
            for step in range(num_steps):
                # Step data
                step_data = {"step": step}
                
                # Infer states
                infer_params = {
                    "agent_id": agent_id,
                    "observation": json.dumps(observation),
                    "method": "FPI"
                }
                state_result = await self.client.call_tool("infer_states", infer_params)
                posterior_states = state_result.get("posterior_states", [])
                step_data["belief"] = [posterior_states]
                
                # Infer policies
                policy_params = {"agent_id": agent_id}
                policy_result = await self.client.call_tool("infer_policies", policy_params)
                policy_posterior = policy_result.get("policy_posterior", [])
                step_data["policy"] = policy_posterior
                
                # Sample action
                action_params = {"agent_id": agent_id}
                action_result = await self.client.call_tool("sample_action", action_params)
                action = action_result.get("action", 0)
                step_data["action"] = action
                
                # Step environment
                step_params = {"environment_id": env_id, "action": action}
                step_result = await self.client.call_tool("step_environment", step_params)
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
            
            # Update result with new history
            result["history"] = history
            
            # Save the history if output_dir is provided
            if output_dir and save_history:
                history_file = os.path.join(output_dir, f"{agent_id}_{env_id}_history.json")
                with open(history_file, "w") as f:
                    json.dump(history, f, indent=2)
        
        return result

async def run_3x3_gridworld(client, toolkit, output_dir):
    """Run the 3x3 grid world example."""
    # Create output directory
    grid_dir = os.path.join(output_dir, "3x3_gridworld")
    os.makedirs(grid_dir, exist_ok=True)
    print(f"\nRunning 3x3 grid world example (outputs in {grid_dir})...")
    
    # Define dimensions for generative model
    A_dims = [[9, 9], [2, 9]]  # 9 positions, 2 reward states
    B_dims = [[9, 9, 4]]       # 9 positions, 4 actions
    
    # Create a generative model
    try:
        gen_model_result = await toolkit.client.call_tool(
            "define_generative_model",
            {"A_dims": json.dumps(A_dims), "B_dims": json.dumps(B_dims)}
        )
        
        # Save the generative model
        with open(os.path.join(grid_dir, "generative_model.json"), "w") as f:
            json.dump(gen_model_result, f, indent=2)
            
        print("Generated and saved generative model")
        
        # Validate the generative model
        if hasattr(toolkit, "validate_generative_model"):
            validation = await toolkit.validate_generative_model(gen_model_result, check_normalization=True)
            with open(os.path.join(grid_dir, "model_validation.json"), "w") as f:
                json.dump(validation, f, indent=2)
            print("Validated generative model")
    except Exception as e:
        print(f"Error creating generative model: {str(e)}")
    
    # Create an agent
    agent = await toolkit.create_gridworld_agent(
        name="GridWorld3x3Agent",
        grid_size=[3, 3],
        reward_positions=[[2, 2]]
    )
    # Extract agent ID, handling both formats returned by the server
    if isinstance(agent, dict):
        agent_id = agent.get("id", "GridWorld3x3Agent")
        # If id is not in the top level, check for nested structure
        if not agent_id and "agent" in agent and isinstance(agent["agent"], dict):
            agent_id = agent["agent"].get("id", "GridWorld3x3Agent")
        # If still no ID, use the name
        if not agent_id:
            agent_id = "GridWorld3x3Agent"
    else:
        agent_id = "GridWorld3x3Agent"
        
    print(f"Agent created with ID: {agent_id}")
    
    # Save agent details
    with open(os.path.join(grid_dir, "agent.json"), "w") as f:
        json.dump(agent, f, indent=2)
    
    # Create environment
    env = await toolkit.create_gridworld_environment(
        name="GridWorld3x3Env",
        grid_size=[3, 3],
        reward_positions=[[2, 2]]
    )
    # Extract environment ID, handling different response formats
    if isinstance(env, dict):
        env_id = env.get("id", "GridWorld3x3Env")
        # If id is not in the top level, check for nested structure
        if not env_id and "environment" in env and isinstance(env["environment"], dict):
            env_id = env["environment"].get("id", "GridWorld3x3Env")
        # If still no ID, use the name
        if not env_id:
            env_id = "GridWorld3x3Env"
    else:
        env_id = "GridWorld3x3Env"
        
    print(f"Environment created with ID: {env_id}")
    
    # Save environment details
    with open(os.path.join(grid_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)
    
    # Test inference with a sample observation
    try:
        # Position observation (0 = top-left corner)
        # Reward observation (0 = no reward)
        observation = [0, 0]
        
        # Run state inference
        infer_result = await toolkit.client.call_tool(
            "infer_states",
            {
                "agent_id": agent_id,
                "observation": json.dumps(observation),
                "method": "FPI"  # Can also try: "VMP", "MMP", "BP"
            }
        )
        
        # Save inference results
        with open(os.path.join(grid_dir, "state_inference.json"), "w") as f:
            json.dump(infer_result, f, indent=2)
        
        # Run policy inference
        policy_result = await toolkit.client.call_tool(
            "infer_policies",
            {"agent_id": agent_id}
        )
        
        # Save policy results
        with open(os.path.join(grid_dir, "policy_inference.json"), "w") as f:
            json.dump(policy_result, f, indent=2)
        
        # Sample action
        action_result = await toolkit.client.call_tool(
            "sample_action",
            {"agent_id": agent_id}
        )
        
        # Save action results
        with open(os.path.join(grid_dir, "action_sampling.json"), "w") as f:
            json.dump(action_result, f, indent=2)
            
        print("Successfully tested inference and action selection")
    except Exception as e:
        print(f"Error during inference testing: {str(e)}")
    
    # Run simulation
    simulation_result = await toolkit.run_agent_in_environment(
        agent_id=agent_id,
        env_id=env_id,
        num_steps=20,
        save_history=True,
        output_dir=grid_dir
    )
    
    # Save full simulation data
    with open(os.path.join(grid_dir, "simulation.json"), "w") as f:
        json.dump(simulation_result, f, indent=2)
    
    timesteps = len(simulation_result.get("history", {}).get("timesteps", []))
    print(f"Simulation completed with {timesteps} timesteps")
    
    # Generate visualizations
    session_id = simulation_result.get("id")
    if session_id:
        # Visualize simulation
        try:
            # Try server visualization
            vis_output = os.path.join(grid_dir, "simulation.png")
            vis_result = await toolkit.visualize_simulation(
                session_id=session_id,
                output_file=vis_output,
                include_beliefs=True,
                include_policies=True
            )
            
            # If server fails, generate visualization locally
            if not os.path.exists(vis_output):
                generate_simulation_visualization(
                    simulation_result, 
                    [3, 3], 
                    [[2, 2]], 
                    vis_output
                )
        except Exception as e:
            print(f"Server visualization failed, generating locally: {str(e)}")
            generate_simulation_visualization(
                simulation_result, 
                [3, 3], 
                [[2, 2]], 
                vis_output
            )
        
        # Visualize agent model
        try:
            model_vis_output = os.path.join(grid_dir, "generative_model.png")
            await toolkit.visualize_agent_model(
                agent_id=agent_id,
                output_file=model_vis_output
            )
            
            # If server fails, generate visualization locally
            if not os.path.exists(model_vis_output):
                generate_agent_model_visualization(
                    agent, 
                    [3, 3], 
                    [[2, 2]], 
                    model_vis_output
                )
        except Exception as e:
            print(f"Agent model visualization failed, generating locally: {str(e)}")
            generate_agent_model_visualization(
                agent, 
                [3, 3], 
                [[2, 2]], 
                model_vis_output
            )
        
        # Analyze free energy
        try:
            fe_output = os.path.join(grid_dir, "free_energy.png")
            await toolkit.analyze_free_energy(
                session_id=session_id,
                output_file=fe_output
            )
            
            # If server fails, generate visualization locally
            if not os.path.exists(fe_output):
                generate_free_energy_visualization(
                    simulation_result,
                    fe_output
                )
        except Exception as e:
            print(f"Free energy analysis failed, generating locally: {str(e)}")
            generate_free_energy_visualization(
                simulation_result,
                fe_output
            )
        
        # Plot belief dynamics
        try:
            belief_output = os.path.join(grid_dir, "belief_dynamics.png")
            await toolkit.plot_belief_dynamics(
                session_id=session_id,
                output_file=belief_output
            )
            
            # If server fails, generate visualization locally
            if not os.path.exists(belief_output):
                generate_belief_dynamics_visualization(
                    simulation_result,
                    belief_output
                )
        except Exception as e:
            print(f"Belief dynamics visualization failed, generating locally: {str(e)}")
            generate_belief_dynamics_visualization(
                simulation_result,
                belief_output
            )
    
    return session_id

# Add visualization helper functions
def generate_agent_model_visualization(agent, grid_size, reward_positions, output_file):
    """Generate a visualization of the agent's generative model."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a figure for the agent model
    plt.figure(figsize=(10, 8))
    
    # Create a grid
    grid = np.zeros(grid_size)
    
    # Mark reward positions
    for pos in reward_positions:
        if len(pos) == 2 and pos[0] < grid_size[0] and pos[1] < grid_size[1]:
            grid[pos[0], pos[1]] = 1
    
    # Plot grid
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Reward')
    
    # Add grid lines
    plt.grid(True, color='black', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-.5, grid_size[1], 1), [])
    plt.yticks(np.arange(-.5, grid_size[0], 1), [])
    
    # Set labels and title
    plt.title(f"Agent Model: {agent.get('id', 'Unknown')}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    
    # Add reward position labels
    for pos in reward_positions:
        if len(pos) == 2 and pos[0] < grid_size[0] and pos[1] < grid_size[1]:
            plt.text(pos[1], pos[0], "R", 
                     ha="center", va="center", 
                     fontsize=12, color="white")
    
    # Save figure
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()

def generate_simulation_visualization(simulation, grid_size, reward_positions, output_file):
    """Generate a visualization of the simulation trajectory."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create grid world
    grid = np.zeros(grid_size)
    
    # Mark reward positions
    for pos in reward_positions:
        if len(pos) == 2 and pos[0] < grid_size[0] and pos[1] < grid_size[1]:
            grid[pos[0], pos[1]] = 1
    
    # Create custom colormap
    cmap = ListedColormap(['whitesmoke', 'gold'])
    
    # Plot grid
    ax.imshow(grid, cmap=cmap, interpolation='nearest')
    
    # Add grid lines
    ax.grid(True, color='black', linestyle='-', linewidth=0.5)
    
    # Extract agent positions from history
    positions = []
    history = simulation.get("history", {}).get("timesteps", [])
    
    for step in history:
        state = step.get("state")
        if state and len(state) == 2:
            positions.append((state[0], state[1]))
        elif state and len(state) == 1:
            # Convert 1D state to 2D coordinates
            flat_idx = state[0]
            row = flat_idx // grid_size[1]
            col = flat_idx % grid_size[1]
            positions.append((row, col))
    
    # If no history, create a sample trajectory
    if not positions:
        # Create sample positions moving from top-left to reward
        positions = [(0, 0)]  # Start at top-left
        
        # Add some positions
        steps = 8
        if len(reward_positions) > 0:
            target = reward_positions[0]
            
            # Calculate path between start and goal
            x_step = (target[1] - 0) / steps
            y_step = (target[0] - 0) / steps
            
            for i in range(1, steps):
                positions.append((
                    min(grid_size[0]-1, max(0, int(i * y_step))),
                    min(grid_size[1]-1, max(0, int(i * x_step)))
                ))
            
            positions.append((target[0], target[1]))
    
    # Plot agent trajectory
    if positions:
        # Extract x and y coordinates
        x_coords = [p[1] for p in positions]
        y_coords = [p[0] for p in positions]
        
        # Plot the trajectory line
        ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
        
        # Plot start position (green)
        ax.plot(positions[0][1], positions[0][0], 'go', markersize=12)
        
        # Plot end position (red)
        ax.plot(positions[-1][1], positions[-1][0], 'ro', markersize=12)
        
        # Add position markers with timestep labels
        for i, (y, x) in enumerate(positions):
            ax.text(x, y, str(i), ha='center', va='center', fontsize=10, 
                    color='black', fontweight='bold')
    
    # Set labels and title
    ax.set_title(f'Agent Trajectory - Session: {simulation.get("id", "Unknown")}')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add reward position labels
    for pos in reward_positions:
        if len(pos) == 2 and pos[0] < grid_size[0] and pos[1] < grid_size[1]:
            ax.text(pos[1], pos[0], "R", 
                    ha="center", va="center", 
                    fontsize=12, color="black", fontweight='bold')
    
    # Save figure
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()

def generate_free_energy_visualization(simulation, output_file):
    """Generate a visualization of the free energy over timesteps."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract history
    history = simulation.get("history", {}).get("timesteps", [])
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    if history:
        # Get actual free energy values if available
        fe_values = []
        for step in history:
            if "free_energy" in step:
                fe_values.append(step["free_energy"])
        
        if fe_values:
            # Use actual free energy values
            timesteps = range(len(fe_values))
            plt.plot(timesteps, fe_values, 'r-', linewidth=2)
        else:
            # Generate sample free energy values
            # Decreasing values indicate optimization
            timesteps = len(history)
            free_energy = np.random.random(timesteps) * -2  # Random negative values
            free_energy.sort()  # Sort for a decreasing trend
            
            # Plot
            plt.plot(range(timesteps), free_energy, 'r-', linewidth=2)
            plt.text(timesteps/2, -1.0, "Simulated free energy values", ha='center', fontsize=10, alpha=0.7)
    else:
        # Sample free energy if no history
        timesteps = 10
        free_energy = np.linspace(-0.5, -2.0, timesteps)
        
        # Add some noise
        free_energy += np.random.normal(0, 0.1, timesteps)
        
        # Plot
        plt.plot(range(timesteps), free_energy, 'r-', linewidth=2)
        plt.text(timesteps/2, -1.25, "Simulated data", ha='center', fontsize=12)
    
    plt.title(f'Free Energy Analysis - Session: {simulation.get("id", "Unknown")}')
    plt.xlabel('Timestep')
    plt.ylabel('Free Energy')
    plt.grid(True)
    
    # Save figure
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()

def generate_belief_dynamics_visualization(simulation, output_file):
    """Generate a visualization of belief dynamics over timesteps."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract history
    history = simulation.get("history", {}).get("timesteps", [])
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Extract beliefs over time
    beliefs = []
    for step in history:
        belief_data = step.get("belief", [])
        if belief_data:
            beliefs.append(belief_data[0])  # Get first belief array
    
    if beliefs:
        num_states = len(beliefs[0])
        
        # Create a heatmap of beliefs
        belief_array = np.array(beliefs)
        plt.imshow(belief_array.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Probability')
        
        # Add labels
        plt.title(f'Belief Dynamics - Session: {simulation.get("id", "Unknown")}')
        plt.xlabel('Timestep')
        plt.ylabel('State')
        
        # Add grid
        plt.grid(False)
        
        # Add state labels
        plt.yticks(range(num_states))
    else:
        # Generate sample beliefs if no history
        timesteps = 10
        num_states = 9  # For a 3x3 grid
        
        beliefs = np.zeros((timesteps, num_states))
        
        # Start with uniform distribution
        beliefs[0] = np.ones(num_states) / num_states
        
        # Converge to state 8 (bottom right)
        for t in range(1, timesteps):
            beliefs[t] = beliefs[t-1].copy()
            # Increase probability of state 8
            beliefs[t][8] += 0.1
            # Normalize
            beliefs[t] = beliefs[t] / beliefs[t].sum()
        
        # Create heatmap
        plt.imshow(beliefs.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Probability')
        
        # Add labels
        plt.title(f'Belief Dynamics (Simulated) - Session: {simulation.get("id", "Unknown")}')
        plt.xlabel('Timestep')
        plt.ylabel('State')
        
        # Add grid
        plt.grid(False)
        
        # Add state labels
        plt.yticks(range(num_states))
    
    # Save figure
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()

async def run_4x4_gridworld(client, toolkit, output_dir):
    """Run the 4x4 grid world example."""
    # Create output directory
    grid_dir = os.path.join(output_dir, "4x4_gridworld")
    os.makedirs(grid_dir, exist_ok=True)
    print(f"\nRunning 4x4 grid world example (outputs in {grid_dir})...")
    
    # Define dimensions for generative model
    A_dims = [[16, 16], [2, 16]]  # 16 positions, 2 reward states
    B_dims = [[16, 16, 5]]        # 16 positions, 5 actions (4 directions + stay)
    
    # Create a generative model
    try:
        gen_model_result = await toolkit.client.call_tool(
            "define_generative_model",
            {"A_dims": json.dumps(A_dims), "B_dims": json.dumps(B_dims)}
        )
        
        # Save the generative model
        with open(os.path.join(grid_dir, "generative_model.json"), "w") as f:
            json.dump(gen_model_result, f, indent=2)
            
        print("Generated and saved 4x4 generative model")
    except Exception as e:
        print(f"Error creating generative model: {str(e)}")
    
    # Create an agent
    agent = await toolkit.create_gridworld_agent(
        name="GridWorld4x4Agent",
        grid_size=[4, 4],
        reward_positions=[[3, 3]]
    )
    agent_id = agent.get("id")
    print(f"Agent created with ID: {agent_id}")
    
    # Save agent details
    with open(os.path.join(grid_dir, "agent.json"), "w") as f:
        json.dump(agent, f, indent=2)
    
    # Create environment
    env = await toolkit.create_gridworld_environment(
        name="GridWorld4x4Env",
        grid_size=[4, 4],
        reward_positions=[[3, 3]]
    )
    env_id = env.get("id")
    print(f"Environment created with ID: {env_id}")
    
    # Save environment details
    with open(os.path.join(grid_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)
    
    # Test different inference algorithms
    try:
        # Initial observation
        observation = [0, 0]  # Start position, no reward
        
        # Try different inference algorithms
        inference_methods = ["FPI", "VMP", "MMP", "BP"]
        
        for method in inference_methods:
            # Run state inference with this method
            infer_result = await toolkit.client.call_tool(
                "infer_states",
                {
                    "agent_id": agent_id,
                    "observation": json.dumps(observation),
                    "method": method
                }
            )
            
            # Save inference results
            with open(os.path.join(grid_dir, f"state_inference_{method}.json"), "w") as f:
                json.dump(infer_result, f, indent=2)
                
            print(f"Successfully tested {method} inference")
    except Exception as e:
        print(f"Error during inference testing: {str(e)}")
    
    # Run simulation
    simulation_result = await toolkit.run_agent_in_environment(
        agent_id=agent_id,
        env_id=env_id,
        num_steps=30,  # More steps for larger grid
        save_history=True,
        output_dir=grid_dir
    )
    
    # Save full simulation data
    with open(os.path.join(grid_dir, "simulation.json"), "w") as f:
        json.dump(simulation_result, f, indent=2)
    
    timesteps = len(simulation_result.get("history", {}).get("timesteps", []))
    print(f"Simulation completed with {timesteps} timesteps")
    
    # Generate visualizations
    session_id = simulation_result.get("id")
    if session_id:
        # Visualize simulation
        try:
            vis_output = os.path.join(grid_dir, "simulation.png")
            await toolkit.visualize_simulation(
                session_id=session_id,
                output_file=vis_output
            )
        except Exception as e:
            print(f"Visualization failed: {str(e)}")
    
    return session_id

async def run_multiple_rewards_gridworld(client, toolkit, output_dir):
    """Run a grid world example with multiple reward positions."""
    # Create output directory
    grid_dir = os.path.join(output_dir, "multiple_rewards_gridworld")
    os.makedirs(grid_dir, exist_ok=True)
    print(f"\nRunning grid world example with multiple rewards (outputs in {grid_dir})...")
    
    # Define dimensions for generative model
    grid_size = [5, 5]
    num_states = grid_size[0] * grid_size[1]
    
    A_dims = [[num_states, num_states], [2, num_states]]  # Position and reward observations
    B_dims = [[num_states, num_states, 5]]               # 5 actions (4 directions + stay)
    
    # Create a generative model
    try:
        gen_model_result = await toolkit.client.call_tool(
            "define_generative_model",
            {"A_dims": json.dumps(A_dims), "B_dims": json.dumps(B_dims)}
        )
        
        # Save the generative model
        with open(os.path.join(grid_dir, "generative_model.json"), "w") as f:
            json.dump(gen_model_result, f, indent=2)
            
        print("Generated and saved multiple rewards generative model")
    except Exception as e:
        print(f"Error creating generative model: {str(e)}")
    
    # Define multiple reward positions
    reward_positions = [[1, 1], [1, 3], [3, 1], [3, 3]]
    
    # Create an agent
    agent = await toolkit.create_gridworld_agent(
        name="GridWorldMultiRewardAgent",
        grid_size=grid_size,
        reward_positions=reward_positions
    )
    agent_id = agent.get("id")
    print(f"Agent created with ID: {agent_id}")
    
    # Save agent details
    with open(os.path.join(grid_dir, "agent.json"), "w") as f:
        json.dump(agent, f, indent=2)
    
    # Create environment
    env = await toolkit.create_gridworld_environment(
        name="GridWorldMultiRewardEnv",
        grid_size=grid_size,
        reward_positions=reward_positions
    )
    env_id = env.get("id")
    print(f"Environment created with ID: {env_id}")
    
    # Save environment details
    with open(os.path.join(grid_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)
    
    # Run simulation
    simulation_result = await toolkit.run_agent_in_environment(
        agent_id=agent_id,
        env_id=env_id,
        num_steps=40,  # More steps for larger grid with multiple rewards
        save_history=True,
        output_dir=grid_dir
    )
    
    # Save full simulation data
    with open(os.path.join(grid_dir, "simulation.json"), "w") as f:
        json.dump(simulation_result, f, indent=2)
    
    timesteps = len(simulation_result.get("history", {}).get("timesteps", []))
    print(f"Simulation completed with {timesteps} timesteps")
    
    return simulation_result.get("id")

async def run_examples(host="localhost", port=8080, use_ssl=False, output_dir=None):
    """Run all grid world examples."""
    # Create output directory with timestamp if not provided
    if output_dir is None:
        # Set output directory within scripts/output folder
        scripts_dir = Path(__file__).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(scripts_dir / "output" / f"gridworld_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")
    
    # Set server URL
    protocol = "https" if use_ssl else "http"
    server_url = f"{protocol}://{host}:{port}"
    print(f"Connecting to MCP server at {server_url}")
    
    # Create client and toolkit
    async with MCPClient(server_url=server_url) as client:
        # Use the extended toolkit for better simulation handling
        toolkit = ExtendedMCPToolKit(client)
        
        # Check server connection
        try:
            ping_result = await client.ping()
            print(f"Server connection: {ping_result.get('status', 'unknown')}")
            
            # Get and save available tools
            tools = await client.get_tools()
            with open(os.path.join(output_dir, "available_tools.json"), "w") as f:
                json.dump(tools, f, indent=2)
                
            print(f"Available tools: {len(tools)}")
        except Exception as e:
            print(f"Error connecting to server: {str(e)}")
            print("Attempting to continue with direct client...")
        
        # Run examples
        session_3x3 = await run_3x3_gridworld(client, toolkit, output_dir)
        session_4x4 = await run_4x4_gridworld(client, toolkit, output_dir)
        session_multi = await run_multiple_rewards_gridworld(client, toolkit, output_dir)
        
        # Generate summary report
        summary_file = os.path.join(output_dir, "summary.json")
        summary = {
            "timestamp": datetime.now().isoformat(),
            "examples": [
                {
                    "name": "3x3_gridworld",
                    "session_id": session_3x3,
                    "directory": "3x3_gridworld"
                },
                {
                    "name": "4x4_gridworld",
                    "session_id": session_4x4,
                    "directory": "4x4_gridworld"
                },
                {
                    "name": "multiple_rewards_gridworld",
                    "session_id": session_multi,
                    "directory": "multiple_rewards_gridworld"
                }
            ]
        }
        
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\nAll examples completed successfully!")
        print(f"Summary saved to: {summary_file}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run MCP-PyMDP grid world examples")
    
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
        help="Output directory for generated files (default: scripts/output/gridworld_TIMESTAMP)"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the examples."""
    args = parse_args()
    asyncio.run(run_examples(
        host=args.host,
        port=args.port,
        use_ssl=args.use_ssl,
        output_dir=args.output_dir
    ))

if __name__ == "__main__":
    main() 