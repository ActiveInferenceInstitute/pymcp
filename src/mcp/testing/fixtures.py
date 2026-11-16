"""Synthetic inputs following the same unbatched model/environment contracts."""
import math

import numpy as np

from ..environments import CustomEnvironment, GridWorldEnvironment
from ..environments.common import grid_shape, grid_transition, integer, position
from ..models import Limits


def _likelihood(num_obs, num_states):
    result = np.full((num_obs, num_states), 1 / num_obs)
    if num_obs > 1:
        for state in range(min(num_obs, num_states)):
            result[:, state] = 0.1 / (num_obs - 1)
            result[state, state] = 0.9
    return result


def _transition(num_states, num_actions):
    result = np.ones((num_states, num_states, num_actions)) if num_states == 1 else np.empty((num_states, num_states, num_actions))
    if num_states > 1:
        for action in range(num_actions):
            preferred = 0.9 if action == 0 else 0.8
            result[:, :, action] = (1 - preferred) / (num_states - 1)
            for state in range(num_states):
                destination = state if action == 0 else (state + 1) % num_states
                result[destination, state, action] = preferred
    return result


def create_simple_generative_model(num_obs=2, num_states=2, num_actions=2):
    limits = Limits()
    num_obs = integer(num_obs, "num_obs", 1)
    num_states = integer(num_states, "num_states", 1)
    num_actions = integer(num_actions, "num_actions", 1, limits.max_policies)
    if num_obs * num_states + num_states * num_states * num_actions + num_obs + num_states > limits.max_elements:
        raise ValueError("fixture exceeds max_elements")
    A, B = _likelihood(num_obs, num_states), _transition(num_states, num_actions)
    C = np.zeros(num_obs)
    C[0] = 1
    return {"A": [A.tolist()], "B": [B.tolist()], "C": [C.tolist()],
            "D": [(np.ones(num_states) / num_states).tolist()]}


def create_multimodal_generative_model(num_modalities=2, state_dims=None, control_dims=None):
    state_dims = [2] if state_dims is None else list(state_dims)
    control_dims = [2] if control_dims is None else list(control_dims)
    num_modalities = integer(num_modalities, "num_modalities", 1, Limits().max_entities)
    if not state_dims or len(state_dims) != len(control_dims) or len(state_dims) > Limits().max_entities:
        raise ValueError("state/control factor counts must match")
    states = [integer(n, "state dimension", 1) for n in state_dims]
    controls = [integer(n, "control dimension", 1) for n in control_dims]
    observations = [states[m % len(states)] + m % 2 for m in range(num_modalities)]
    cost = sum(s * s * c + s for s, c in zip(states, controls))
    cost += sum(o * states[m % len(states)] + o for m, o in enumerate(observations))
    if cost > Limits().max_elements or math.prod(controls) > Limits().max_policies:
        raise ValueError("multimodal fixture exceeds resource budget")
    return {"A": [_likelihood(o, states[m % len(states)]).tolist() for m, o in enumerate(observations)],
            "B": [_transition(s, c).tolist() for s, c in zip(states, controls)],
            "C": [[1] + [0] * (o - 1) for o in observations],
            "D": [(np.ones(s) / s).tolist() for s in states],
            "A_dependencies": [[m % len(states)] for m in range(num_modalities)]}


def create_gridworld_generative_model(grid_size=None, reward_position=None):
    shape = grid_shape([3, 3] if grid_size is None else grid_size)
    reward = position([shape[0] - 1, shape[1] - 1] if reward_position is None else reward_position, shape)
    n = math.prod(shape)
    if n * n * 6 + n * 4 + 7 > Limits().max_elements:
        raise ValueError("grid fixture exceeds max_elements")
    B = np.zeros((n, n, 5))
    for state in range(n):
        for action in range(5):
            B[grid_transition(state, action, shape), state, action] = 1
    A_reward = np.zeros((2, n))
    A_reward[0] = 1
    A_reward[:, reward[0] * shape[1] + reward[1]] = [0, 1]
    D = np.zeros(n)
    D[0] = 1
    return {"A": [np.eye(n).tolist(), A_reward.tolist()], "B": [B.tolist()],
            "C": [np.zeros(n).tolist(), [0, 4]], "D": [D.tolist()]}


def create_grid_world_environment(grid_size=None, reward_positions=None):
    shape = grid_shape([3, 3] if grid_size is None else grid_size)
    rewards = [[shape[0] - 1, shape[1] - 1]] if reward_positions is None else reward_positions
    env = GridWorldEnvironment(shape, rewards)
    return {"type": "grid_world", "grid_size": env.grid_size.copy(),
            "reward_positions": [p.copy() for p in env.reward_positions]}


def create_custom_environment(name="custom_env", states=None, observations=None):
    states = ["state1", "state2", "state3"] if states is None else list(states)
    observations = ["obs1", "obs2", "obs3", "reward", "no_reward"] if observations is None else list(observations)
    if not states:
        raise ValueError("states cannot be empty")
    transitions = {s: {"forward": states[(i + 1) % len(states)],
                       "backward": states[(i - 1) % len(states)], "stay": s}
                   for i, s in enumerate(states)}
    rewards = {s: float(i == len(states) - 1) for i, s in enumerate(states)}
    return CustomEnvironment(name, states, observations, transitions, rewards).to_dict()
