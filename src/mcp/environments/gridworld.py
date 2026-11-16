"""Validated grid world using the shared up/right/down/left/stay controls."""
import copy

import numpy as np

from .common import GRID_ACTION_NAMES, action_index, grid_shape, grid_transition, integer, position as validate_position


class GridWorldEnvironment:
    def __init__(self, grid_size, reward_positions, start_position=None,
                 reset_mode="origin", terminate_on_reward=True, seed=None):
        self.grid_size = grid_shape(grid_size)
        if not isinstance(reward_positions, (list, tuple)) or len(reward_positions) > 10000:
            raise ValueError("reward_positions must be a bounded list")
        self.reward_positions = [validate_position(p, self.grid_size) for p in reward_positions]
        self.start_position = validate_position([0, 0] if start_position is None else start_position, self.grid_size)
        if reset_mode not in ("origin", "configured"):
            raise ValueError("reset_mode must be origin or configured")
        if type(terminate_on_reward) is not bool:
            raise ValueError("terminate_on_reward must be boolean")
        self.reset_mode = reset_mode
        self.terminate_on_reward = terminate_on_reward
        self.agent_pos = self.start_position.copy()
        self.num_states = self.grid_size[0] * self.grid_size[1]
        self.actions = dict(enumerate(GRID_ACTION_NAMES))
        self.rng = np.random.default_rng(seed)

    def reset(self, start_position=None):
        """Default origin reset is retained; configured reset is explicit opt-in."""
        target = self.start_position if self.reset_mode == "configured" else [0, 0]
        self.agent_pos = validate_position(target if start_position is None else start_position, self.grid_size)
        return self.get_observation()

    def step(self, action):
        index = grid_transition(self.get_state_index(), action, self.grid_size)
        self.agent_pos = list(divmod(index, self.grid_size[1]))
        return self.get_observation()

    def get_observation(self):
        reward = int(self.has_reached_reward())
        return {
            "observation": [self.get_state_index(), reward],
            "state": [self.get_state_index()],
            "position": self.agent_pos.copy(),
            "reward": reward,
            "done": self.is_terminal(),
            "info": {},
        }

    def validate_action(self, action):
        return action_index(action, len(self.actions))

    def get_state_index(self, position=None):
        row, col = self.agent_pos if position is None else validate_position(position, self.grid_size)
        return row * self.grid_size[1] + col

    def get_position_from_index(self, index):
        index = integer(index, "state", 0, self.num_states - 1)
        return list(divmod(index, self.grid_size[1]))

    def get_valid_actions(self, position=None):
        if position is not None:
            validate_position(position, self.grid_size)
        return list(self.actions)  # Boundary-clamped movement and stay are valid.

    def get_random_action(self):
        return int(self.rng.integers(len(self.actions)))

    def has_reached_reward(self):
        return self.agent_pos in self.reward_positions

    def is_terminal(self):
        return self.terminate_on_reward and self.has_reached_reward()

    def render(self, mode="ascii"):
        if mode != "ascii":
            raise NotImplementedError(mode)
        grid = [["."] * self.grid_size[1] for _ in range(self.grid_size[0])]
        for row, col in self.reward_positions:
            grid[row][col] = "R"
        row, col = self.agent_pos
        grid[row][col] = "A"
        return "".join(" ".join(row) + "\n" for row in grid)

    def to_dict(self):
        return copy.deepcopy({
            "type": "grid_world", "grid_size": self.grid_size,
            "reward_positions": self.reward_positions, "agent_pos": self.agent_pos,
            "num_states": self.num_states, "current_state": self.get_state_index(),
            "start_position": self.start_position, "reset_mode": self.reset_mode,
            "terminate_on_reward": self.terminate_on_reward,
            "actions": list(GRID_ACTION_NAMES),
        })


def create_grid_world_env(name, grid_size, reward_positions, start_position=None,
                          reset_mode="origin", terminate_on_reward=True, seed=None):
    env = GridWorldEnvironment(grid_size, reward_positions, start_position,
                               reset_mode, terminate_on_reward, seed)
    description = env.to_dict()
    description["name"] = name
    return description, env
