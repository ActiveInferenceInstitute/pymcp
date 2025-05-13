#!/usr/bin/env python3
"""
PyMDP-MCP Examples

This script demonstrates the core functionality of PyMDP using the MCP (Message Passing Protocol) interface.
It replicates key examples from PyMDP tutorials:
- agent_demo.py: Basic agent setup and inference
- gridworld_tutorial_2.ipynb: Inference and planning in gridworld
- free_energy_calculation.ipynb: Variational free energy computation
- building_up_agent_loop.ipynb: Agent loop construction

Usage:
    python pymcp_examples.py [--host HOST] [--port PORT] [--output-dir DIR]
"""

import os
import sys
import json
import asyncio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add the pymdp-clone directory to the path for direct access to PyMDP if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "pymdp-clone"))

from mcp.client import MCPClient

class PyMDPToolkit:
    """Helper class to wrap MCP client calls for PyMDP functionality following Model Context Protocol."""
    
    def __init__(self, client):
        """Initialize with an MCP client."""
        self.client = client
    
    async def create_agent(self, name, observation_dimensions, state_dimensions, control_factor_idx=None):
        """Create a basic agent using MCP protocol."""
        response = await self.client.call_tool(
            "create_agent",
            {
                "name": name,
                "observation_dimensions": json.dumps(observation_dimensions),
                "state_dimensions": json.dumps(state_dimensions),
                "control_factor_idx": json.dumps(control_factor_idx) if control_factor_idx else None
            }
        )
        return response.get("result", {})
    
    async def create_gridworld_agent(self, name, grid_size, reward_positions, action_precision=1.0, 
                                    inference_horizon=5, inference_algorithm="FPI"):
        """Create a gridworld agent using MCP protocol."""
        response = await self.client.call_tool(
            "create_gridworld_agent",
            {
                "name": name,
                "grid_size": json.dumps(grid_size),
                "reward_positions": json.dumps(reward_positions),
                "action_precision": action_precision,
                "inference_horizon": inference_horizon,
                "inference_algorithm": inference_algorithm
            }
        )
        return response.get("result", {})
    
    async def create_gridworld_environment(self, name, grid_size, reward_positions):
        """Create a gridworld environment using MCP protocol."""
        response = await self.client.call_tool(
            "create_environment",
            {
                "name": name,
                "type": "gridworld",
                "grid_size": json.dumps(grid_size),
                "reward_positions": json.dumps(reward_positions)
            }
        )
        return response.get("result", {})
    
    async def infer_states(self, agent_id, observation, method="FPI"):
        """Run state inference for an agent."""
        response = await self.client.call_tool(
            "infer_states",
            {
                "agent_id": agent_id,
                "observation": json.dumps(observation),
                "method": method
            }
        )
        return response.get("result", {})
    
    async def infer_policies(self, agent_id, planning_horizon=None):
        """Run policy inference for an agent."""
        params = {"agent_id": agent_id}
        if planning_horizon is not None:
            params["planning_horizon"] = planning_horizon
            
        response = await self.client.call_tool("infer_policies", params)
        return response.get("result", {})
    
    async def sample_action(self, agent_id, planning_horizon=None):
        """Sample an action from the agent's policy posterior."""
        params = {"agent_id": agent_id}
        if planning_horizon is not None:
            params["planning_horizon"] = planning_horizon
            
        response = await self.client.call_tool("sample_action", params)
        return response.get("result", {})
    
    async def step_environment(self, env_id, action):
        """Take a step in the environment with the specified action."""
        # If action is a list with one integer, extract the integer
        if isinstance(action, list) and len(action) == 1 and isinstance(action[0], int):
            action = action[0]
            
        response = await self.client.call_tool(
            "step_environment",
            {
                "env_id": env_id,
                "action": action
            }
        )
        return response.get("result", {})
    
    async def reset_environment(self, env_id):
        """Reset an environment to its initial state."""
        response = await self.client.call_tool(
            "reset_environment",
            {"env_id": env_id}
        )
        return response.get("result", {})
    
    async def run_simulation(self, agent_id, env_id, num_steps=10, save_history=True, planning_horizon=None):
        """Run a simulation with an agent in an environment.
        
        Args:
            agent_id: Agent ID
            env_id: Environment ID
            num_steps: Number of steps to run
            save_history: Whether to save history
            planning_horizon: Planning horizon for policy inference
            
        Returns:
            Simulation results
        """
        # Call run_simulation tool
        params = {
            "agent_id": agent_id,
            "environment_id": env_id,
            "steps": num_steps,
            "save_history": save_history
        }
        
        if planning_horizon is not None:
            params["planning_horizon"] = planning_horizon
        
        # Call the tool
        result = await self.client.call_tool("run_simulation", params)
        
        # If that fails, try manual simulation
        if "error" in result or "timesteps" not in result.get("history", {}):
            # Reset environment
            reset_result = await self.reset_environment(env_id)
            observation = reset_result.get("observation", [0])
            
            # Initialize history
            history = {
                "observations": [observation],
                "actions": [],
                "rewards": [0.0],
                "states": [],
                "policies": []
            }
            
            total_reward = 0.0
            
            # Run manual simulation loop
            for step in range(num_steps):
                # Infer states
                state_result = await self.infer_states(agent_id, observation)
                posterior_states = state_result.get("posterior_states", [])
                
                # Infer policies
                policy_result = await self.infer_policies(agent_id, planning_horizon)
                policy_posterior = policy_result.get("policy_posterior", [])
                
                # Sample action
                action_result = await self.sample_action(agent_id)
                action = action_result.get("action", 0)
                
                # Step environment
                step_result = await self.step_environment(env_id, action)
                next_observation = step_result.get("observation", [0])
                reward = step_result.get("reward", 0.0)
                done = step_result.get("done", False)
                
                # Update history
                if save_history:
                    history["observations"].append(next_observation)
                    history["actions"].append(action)
                    history["rewards"].append(reward)
                    history["states"].append(posterior_states)
                    history["policies"].append(policy_posterior)
                
                # Update total reward
                total_reward += reward
                
                # Update observation
                observation = next_observation
                
                # Check if done
                if done:
                    break
            
            # Create result dictionary
            result = {
                "agent_id": agent_id,
                "env_id": env_id,
                "num_steps": len(history["actions"]),
                "total_reward": total_reward,
                "history": history
            }
        
        return result
    
    async def calculate_free_energy(self, A_matrix, prior, observation, q_states):
        """Calculate variational free energy."""
        response = await self.client.call_tool(
            "calculate_free_energy",
            {
                "A_matrix": json.dumps(A_matrix),
                "prior": json.dumps(prior),
                "observation": json.dumps(observation),
                "q_states": json.dumps(q_states)
            }
        )
        return response.get("result", {})
    
    async def infer_states_from_observation(self, A_matrix, prior, observation):
        """Infer states from observation using Bayes rule."""
        response = await self.client.call_tool(
            "infer_states_from_observation",
            {
                "A_matrix": json.dumps(A_matrix),
                "prior": json.dumps(prior),
                "observation": json.dumps(observation)
            }
        )
        return response.get("result", {})

async def run_agent_demo(client, toolkit, output_dir):
    """
    Replicates core functionality from agent_demo.py using the MCP interface.
    
    Creates a simple agent with observation and action modalities, and runs inference.
    """
    print("\n=== Running Agent Demo ===")

    example_dir = os.path.join(output_dir, "agent_demo")
    os.makedirs(example_dir, exist_ok=True)
    
    # Define the model dimensions
    obs_dims = [3, 3, 3]  # Three observation modalities with 3 states each
    state_dims = [2, 3]  # Two hidden state factors with 2 and 3 states respectively
    
    # Create agent using the toolkit
    agent_result = await toolkit.create_agent(
        name="AgentDemo",
        observation_dimensions=obs_dims,
        state_dimensions=state_dims,
        control_factor_idx=[1]  # Second state factor is controllable
    )
    
    # Check for errors
    if "error" in agent_result:
        print(f"Error creating agent: {agent_result['error']}")
        agent_id = "AgentDemo"  # Fallback ID
    else:
        agent_id = agent_result.get("id", "AgentDemo")
    
    print(f"Created agent with ID: {agent_id}")
    
    # Save agent details
    with open(os.path.join(example_dir, "agent_details.json"), "w") as f:
        json.dump(agent_result, f, indent=2)
    
    # Define observation
    observation = [2, 2, 0]  # Initial observation for the three modalities
    
    # Run state inference
    infer_result = await toolkit.infer_states(
        agent_id=agent_id,
        observation=observation,
        method="FPI"  # Fixed-point iteration
    )
    
    # Save inference results
    with open(os.path.join(example_dir, "inference_result.json"), "w") as f:
        json.dump(infer_result, f, indent=2)
    
    print(f"Agent demo completed. Results saved to {example_dir}")
    return agent_id

async def run_gridworld_tutorial(client, toolkit, output_dir):
    """
    Replicates core functionality from gridworld_tutorial_2.ipynb using the MCP interface.
    
    Creates a gridworld environment and agent, then runs inference and planning.
    """
    print("\n=== Running Gridworld Tutorial ===")
    
    example_dir = os.path.join(output_dir, "gridworld_tutorial")
    os.makedirs(example_dir, exist_ok=True)
    
    # Create a 3x3 gridworld agent
    agent_result = await toolkit.create_gridworld_agent(
        name="GridWorldAgent",
        grid_size=[3, 3],
        reward_positions=[[2, 2]],  # Bottom-right corner is rewarding
        action_precision=1.0
    )
    
    # Check for errors
    if "error" in agent_result:
        print(f"Error creating gridworld agent: {agent_result['error']}")
        agent_id = "GridWorldAgent"  # Fallback ID
    else:
        agent_id = agent_result.get("id", "GridWorldAgent")
    
    print(f"Created gridworld agent with ID: {agent_id}")
    
    # Create a gridworld environment
    env_result = await toolkit.create_gridworld_environment(
        name="GridWorldEnv",
        grid_size=[3, 3],
        reward_positions=[[2, 2]]
    )
    
    # Check for errors
    if "error" in env_result:
        print(f"Error creating gridworld environment: {env_result['error']}")
        env_id = "GridWorldEnv"  # Fallback ID
    else:
        env_id = env_result.get("id", "GridWorldEnv")
    
    print(f"Created gridworld environment with ID: {env_id}")
    
    # Save agent and environment details
    with open(os.path.join(example_dir, "agent_details.json"), "w") as f:
        json.dump(agent_result, f, indent=2)
    
    with open(os.path.join(example_dir, "environment_details.json"), "w") as f:
        json.dump(env_result, f, indent=2)
    
    # Run simulation
    print("Running gridworld simulation...")
    simulation_response = await client.call_tool(
        "run_simulation",
        {
            "agent_id": agent_id,
            "environment_id": env_id,
            "num_steps": 10,
            "save_history": True,
            "output_dir": example_dir
        }
    )
    
    # Extract result from response
    simulation_result = simulation_response.get("result", {})
    
    # Save simulation results
    with open(os.path.join(example_dir, "simulation_result.json"), "w") as f:
        json.dump(simulation_result, f, indent=2)
    
    print(f"Gridworld tutorial completed. Results saved to {example_dir}")
    return agent_id, env_id

async def run_free_energy_calculation(client, toolkit, output_dir):
    """
    Replicates core functionality from free_energy_calculation.ipynb using the MCP interface.
    
    Demonstrates calculation of variational free energy and its components.
    """
    print("\n=== Running Free Energy Calculation Example ===")
    
    example_dir = os.path.join(output_dir, "free_energy_calculation")
    os.makedirs(example_dir, exist_ok=True)
    
    # Define a simple generative model
    # A matrix: p(o|s) - observation likelihood
    # 2 observations x 3 states
    A_matrix = [
        [0.9, 0.1, 0.3],  # p(o=1|s)
        [0.1, 0.9, 0.7]   # p(o=2|s)
    ]
    
    # Prior belief over states: uniform
    prior = [1/3, 1/3, 1/3]
    
    # Define the true observation
    observation = [1]  # Second observation (o=2)
    
    # Calculate posterior belief using Bayes' rule via the API
    posterior_result = await toolkit.infer_states_from_observation(A_matrix, prior, observation)
    
    if "error" in posterior_result:
        print(f"Error in state inference: {posterior_result['error']}")
        posterior = [0.1, 0.6, 0.3]  # Fallback posterior
    else:
        posterior = posterior_result.get("posterior", [0.1, 0.6, 0.3])
    
    print(f"Prior beliefs: {[round(p, 2) for p in prior]}")
    print(f"Posterior beliefs: {[round(p, 2) for p in posterior]}")
    
    # Calculate variational free energy using the posterior
    fe_result = await toolkit.calculate_free_energy(A_matrix, prior, observation, posterior)
    
    if "error" in fe_result:
        print(f"Error calculating free energy: {fe_result['error']}")
    else:
        # Extract components
        free_energy = fe_result.get("free_energy", 0.0)
        energy = fe_result.get("energy", 0.0)
        entropy = fe_result.get("entropy", 0.0)
        divergence = fe_result.get("divergence", 0.0)
        accuracy = fe_result.get("accuracy", 0.0)
        
        print("\nVariational Free Energy Components:")
        print(f"  Free Energy: {free_energy:.4f}")
        print(f"  = Energy: {energy:.4f}")
        print(f"  + Entropy: {entropy:.4f}")
        print(f"\nAlternative Decomposition:")
        print(f"  Free Energy: {free_energy:.4f}")
        print(f"  = Divergence: {divergence:.4f}")
        print(f"  - Accuracy: {accuracy:.4f}")
    
    # Save results
    results = {
        "A_matrix": A_matrix,
        "prior": prior,
        "observation": observation,
        "posterior": posterior,
        "free_energy_components": fe_result
    }
    
    with open(os.path.join(example_dir, "free_energy_calculation.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Visualize posterior vs. prior
    plt.figure(figsize=(8, 5))
    states = [f"State {i+1}" for i in range(len(prior))]
    
    width = 0.35
    x = np.arange(len(states))
    
    plt.bar(x - width/2, prior, width, label='Prior', color='skyblue')
    plt.bar(x + width/2, posterior, width, label='Posterior', color='coral')
    
    plt.xlabel('States')
    plt.ylabel('Probability')
    plt.title('Belief Update: Prior vs. Posterior')
    plt.xticks(x, states)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.savefig(os.path.join(example_dir, "belief_update.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Free energy calculation completed. Results saved to {example_dir}")
    return results

async def run_agent_loop_example(client, toolkit, output_dir):
    """
    Replicates core functionality from building_up_agent_loop.ipynb using the MCP interface.
    
    Creates a gridworld environment and agent, then runs a full agent-environment loop.
    """
    print("\n=== Running Agent Loop Example ===")
    
    example_dir = os.path.join(output_dir, "agent_loop")
    os.makedirs(example_dir, exist_ok=True)
    
    # Create a 3x3 gridworld agent
    agent_result = await toolkit.create_gridworld_agent(
        name="LoopAgent",
        grid_size=[3, 3],
        reward_positions=[[2, 2]],  # Bottom-right corner is rewarding
        action_precision=2.0,  # Higher precision for more focused action selection
        inference_horizon=3     # Plan 3 steps ahead
    )
    
    # Check for errors
    if "error" in agent_result:
        print(f"Error creating gridworld agent: {agent_result['error']}")
        agent_id = "LoopAgent"  # Fallback ID
    else:
        agent_id = agent_result.get("id", "LoopAgent")
    
    print(f"Created gridworld agent with ID: {agent_id}")
    
    # Create a gridworld environment
    env_result = await toolkit.create_gridworld_environment(
        name="LoopEnv",
        grid_size=[3, 3],
        reward_positions=[[2, 2]]
    )
    
    # Check for errors
    if "error" in env_result:
        print(f"Error creating gridworld environment: {env_result['error']}")
        env_id = "LoopEnv"  # Fallback ID
    else:
        env_id = env_result.get("id", "LoopEnv")
    
    print(f"Created gridworld environment with ID: {env_id}")
    
    # Reset the environment to get initial observation
    reset_result = await toolkit.reset_environment(env_id)
    
    # Check for errors
    if "error" in reset_result:
        print(f"Error resetting environment: {reset_result['error']}")
        observation = [0]  # Fallback observation
    else:
        observation = reset_result.get("observation", [0])
    
    print(f"Initial observation: {observation}")
    
    # Run agent loop for 10 steps
    history = {
        "observations": [observation],
        "actions": [],
        "states": [],
        "rewards": [0.0]
    }
    
    for step in range(10):
        print(f"Step {step + 1}/10:")
        
        # 1. State inference
        infer_result = await toolkit.infer_states(
            agent_id=agent_id,
            observation=observation,
            method="FPI"
        )
        
        posterior_states = infer_result.get("state_beliefs", [[0.11] * 9])
        print(f"  - Posterior states: {[round(p, 2) for p in posterior_states[0][:3]]}...")
        
        # 2. Policy inference
        policy_result = await toolkit.infer_policies(agent_id=agent_id)
        
        policy_posterior = policy_result.get("policy_posterior", [0.2] * 5)
        expected_free_energy = policy_result.get("expected_free_energy", [1.0] * 5)
        print(f"  - Policy posterior: {[round(p, 2) for p in policy_posterior]}")
        
        # 3. Action selection (default to action 0 - no movement)
        action_result = await toolkit.sample_action(agent_id=agent_id)
        action = action_result.get("action", 0)
        
        print(f"  - Selected action: {action}")
        
        # 4. Environment step
        env_result = await toolkit.step_environment(
            env_id=env_id,
            action=action
        )
        
        observation = env_result.get("observation", [0])
        reward = env_result.get("reward", 0.0)
        done = env_result.get("done", False)
        
        print(f"  - New observation: {observation}, Reward: {reward}")
        
        # Update history
        history["observations"].append(observation)
        history["actions"].append(action)
        history["states"].append(posterior_states)
        history["rewards"].append(reward)
        
        # Check if done
        if done:
            print(f"  - Environment signaled completion at step {step + 1}")
            break
    
    # Calculate total reward
    total_reward = sum(history["rewards"])
    print(f"\nAgent loop completed with total reward: {total_reward}")
    
    # Save history
    with open(os.path.join(example_dir, "agent_loop_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    # Try running a full simulation using the run_simulation method
    print("\nRunning full simulation...")
    
    sim_result = await toolkit.run_simulation(
        agent_id=agent_id,
        env_id=env_id,
        num_steps=10,
        save_history=True
    )
    
    # Check for errors
    if "error" in sim_result:
        print(f"Error running simulation: {sim_result['error']}")
    else:
        print(f"Simulation completed with total reward: {sim_result.get('total_reward', 0.0)}")
        
        # Save simulation result
        with open(os.path.join(example_dir, "simulation_result.json"), "w") as f:
            json.dump(sim_result, f, indent=2)
    
    print(f"Agent loop example completed. Results saved to {example_dir}")
    return agent_id, env_id

async def run_examples(host="localhost", port=8080, use_ssl=False, output_dir=None):
    """Run all examples."""
    
    # Create timestamp-based output directory if not specified
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("scripts/output", f"pymcp_examples_{timestamp}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")
    
    # Create MCP client
    try:
        # Standard MCP client initialization following the protocol
        protocol = "https" if use_ssl else "http"
        server_url = f"{protocol}://{host}:{port}"
        
        # Create client using the MCP protocol standard and context manager
        async with MCPClient(server_url=server_url) as client:
            # Test connection
            print("Testing connection to MCP server...")
            ping_result = await client.ping()
            
            if not ping_result or ping_result.get("status") != "ok":
                print(f"Failed to connect to MCP server at {host}:{port}")
                print(f"Response: {ping_result}")
                return False
            
            print(f"Successfully connected to MCP server at {host}:{port}")
            
            # Create toolkit
            toolkit = PyMDPToolkit(client)
            
            # Run examples
            results = {}
            
            try:
                # Example 1: Agent Demo
                agent_id = await run_agent_demo(client, toolkit, output_dir)
                results["agent_demo"] = {"success": True, "agent_id": agent_id}
            except Exception as e:
                print(f"Error running agent demo: {str(e)}")
                results["agent_demo"] = {"success": False, "error": str(e)}
            
            try:
                # Example 2: Gridworld Tutorial
                await run_gridworld_tutorial(client, toolkit, output_dir)
                results["gridworld_tutorial"] = {"success": True}
            except Exception as e:
                print(f"Error running gridworld tutorial: {str(e)}")
                results["gridworld_tutorial"] = {"success": False, "error": str(e)}
            
            try:
                # Example 3: Free Energy Calculation
                fe_results = await run_free_energy_calculation(client, toolkit, output_dir)
                results["free_energy_calculation"] = {"success": True, "free_energy": fe_results.get("free_energy_components", {}).get("free_energy")}
            except Exception as e:
                print(f"Error running free energy calculation: {str(e)}")
                results["free_energy_calculation"] = {"success": False, "error": str(e)}
            
            try:
                # Example 4: Agent Loop
                agent_id, env_id = await run_agent_loop_example(client, toolkit, output_dir)
                results["agent_loop"] = {"success": True, "agent_id": agent_id, "env_id": env_id}
            except Exception as e:
                print(f"Error running agent loop example: {str(e)}")
                results["agent_loop"] = {"success": False, "error": str(e)}
            
            # Save summary results
            with open(os.path.join(output_dir, "results_summary.json"), "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "server": f"{host}:{port}",
                    "results": results
                }, f, indent=2)
            
            print("\n===== Example Runs Complete =====")
            print(f"All results saved to: {output_dir}")
            
            return True
    
    except Exception as e:
        print(f"Error connecting to MCP server: {str(e)}")
        return False

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run PyMDP examples via MCP")
    
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
        help="Output directory for generated files (default: scripts/output/pymcp_examples_TIMESTAMP)"
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