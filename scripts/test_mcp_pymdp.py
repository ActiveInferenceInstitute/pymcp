#!/usr/bin/env python3

import sys
import os
import json
import numpy as np
import time
import matplotlib.pyplot as plt

# Add the proper paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'pymdp-clone'))
sys.path.insert(0, os.path.join(root_dir, 'src'))

# Setup paths for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

print(f"Python path: {sys.path}")

# Import our MCP utilities
from mcp.utils import PyMDPInterface, NumpyEncoder


def test_mcp_implementation():
    """Test the full MCP-PyMDP implementation."""
    print("Testing MCP-PyMDP implementation...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", 
                             f"pymcp_examples_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to {output_dir}")
    
    # Create the PyMDP interface
    interface = PyMDPInterface()
    
    # Step 1: Create a generative model
    print("\n1. Creating a generative model...")
    
    # Define A and B matrix dimensions
    A_dims = [[3, 2], [4, 2]]  # Two observation modalities
    B_dims = [[2, 2, 3]]      # One state factor with 3 possible actions
    
    # Define the generative model
    model_result = interface.define_generative_model(A_dims, B_dims)
    
    if "error" in model_result:
        print(f"Error defining model: {model_result['error']}")
        return
    
    # Save model to file
    model_file = os.path.join(output_dir, "generative_model.json")
    with open(model_file, "w") as f:
        json.dump(model_result, f, indent=2, cls=NumpyEncoder)
    print(f"Generative model saved to {model_file}")
        
    # Step 2: Create an agent
    print("\n2. Creating an agent...")
    agent_name = "mcp_test_agent"
    agent_result = interface.create_agent(agent_name, model_result)
    
    if "error" in agent_result:
        print(f"Error creating agent: {agent_result['error']}")
        return
    
    print(f"Agent created: {agent_result}")
    
    # Step 3: Create a simple GridWorld environment
    print("\n3. Creating a GridWorld environment...")
    env_name = "mcp_test_env"
    grid_size = [3, 3]
    reward_positions = [[2, 2]]  # Bottom-right corner
    
    env_result = interface.create_grid_world_env(env_name, grid_size, reward_positions)
    
    if "error" in env_result:
        print(f"Error creating environment: {env_result['error']}")
        return
    
    print(f"Environment created: {env_result}")
    
    # Step 4: Test state inference
    print("\n4. Testing state inference...")
    observation = [0]  # Starting observation
    
    inference_result = interface.infer_states(agent_name, observation)
    
    if "error" in inference_result:
        print(f"Error inferring states: {inference_result['error']}")
        return
    
    print(f"State inference: {inference_result}")
    
    # Step 5: Test policy inference 
    print("\n5. Testing policy inference...")
    policy_result = interface.infer_policies(agent_name)
    
    if "error" in policy_result:
        print(f"Error inferring policies: {policy_result['error']}")
        return
    
    print(f"Policy inference: {policy_result}")
    
    # Step 6: Test action sampling
    print("\n6. Testing action sampling...")
    action_result = interface.sample_action(agent_name)
    
    if "error" in action_result:
        print(f"Error sampling action: {action_result['error']}")
        return
    
    print(f"Action sampled: {action_result}")
    
    # Step 7: Run a simulation
    print("\n7. Running a simulation...")
    simulation_result = interface.run_simulation(agent_name, env_name, num_steps=10)
    
    if "error" in simulation_result:
        print(f"Error running simulation: {simulation_result['error']}")
        return
    
    # Save simulation results to file
    sim_file = os.path.join(output_dir, "simulation_results.json")
    with open(sim_file, "w") as f:
        json.dump(simulation_result, f, indent=2, cls=NumpyEncoder)
    print(f"Simulation results saved to {sim_file}")
    
    # Step 8: Visualize the simulation
    print("\n8. Visualizing the simulation...")
    
    # Extract trajectory data
    observations = simulation_result["history"]["observations"]
    actions = simulation_result["history"]["actions"]
    rewards = simulation_result["history"]["rewards"]
    
    # Create a simple visualization
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot observations
    axs[0].plot(observations, 'o-')
    axs[0].set_title('Observations')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Observation')
    
    # Plot actions
    axs[1].plot(actions, 'o-')
    axs[1].set_title('Actions')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Action')
    
    # Plot rewards
    axs[2].plot(rewards, 'o-')
    axs[2].set_title('Rewards')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Reward')
    
    plt.tight_layout()
    
    # Save the figure
    fig_file = os.path.join(output_dir, "simulation_visualization.png")
    plt.savefig(fig_file)
    plt.close()
    print(f"Visualization saved to {fig_file}")
    
    print("\nMCP-PyMDP implementation test completed successfully!")
    
    return interface, agent_name, env_name

if __name__ == "__main__":
    test_mcp_implementation() 