"""Deterministic custom environments with fixed observation modalities."""
import copy
from numbers import Real

import numpy as np

from .common import action_index, integer


class CustomEnvironment:
    def __init__(self, name, states, observations, transitions, rewards,
                 current_state=None, state_observations=None, actions=None,
                 reset_mode="first", terminate_on_reward=True, seed=None):
        self.name = name
        for label, values in (("states", states), ("observations", observations)):
            if (not isinstance(values, (list, tuple)) or not values
                    or any(not isinstance(v, str) or not v for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{label} must contain unique nonempty names")
            if len(values) > 10000:
                raise ValueError(f"too many {label}")
        self.states, self.observations = list(states), list(observations)
        if not isinstance(transitions, dict) or not isinstance(rewards, dict):
            raise ValueError("transitions and rewards must be dictionaries")
        self.transitions, self.rewards = copy.deepcopy(transitions), copy.deepcopy(rewards)
        encountered = []
        for state, edges in self.transitions.items():
            if state not in self.states or not isinstance(edges, dict):
                raise ValueError("invalid transition source")
            for action, target in edges.items():
                if not isinstance(action, str) or not action or target not in self.states:
                    raise ValueError("invalid transition action or target")
                if action not in encountered:
                    encountered.append(action)
        self.actions = encountered if actions is None else list(actions)
        if (any(not isinstance(a, str) or not a for a in self.actions)
                or len(set(self.actions)) != len(self.actions)
                or set(self.actions) != set(encountered)):
            raise ValueError("actions must enumerate every transition action exactly once")
        for state, reward in self.rewards.items():
            if (state not in self.states or isinstance(reward, (bool, np.bool_))
                    or not isinstance(reward, Real) or not np.isfinite(reward)):
                raise ValueError("invalid reward")
        self.start_state = self.states[0] if current_state is None else current_state
        if self.start_state not in self.states:
            raise ValueError("unknown starting state")
        if reset_mode not in ("first", "configured") or type(terminate_on_reward) is not bool:
            raise ValueError("invalid reset/termination setting")
        self.reset_mode, self.terminate_on_reward = reset_mode, terminate_on_reward
        self.current_state = self.start_state
        self.state_to_index = {s: i for i, s in enumerate(self.states)}
        self.index_to_state = dict(enumerate(self.states))
        self.obs_to_index = {s: i for i, s in enumerate(self.observations)}
        self.index_to_obs = dict(enumerate(self.observations))
        self.action_to_index = {a: i for i, a in enumerate(self.actions)}
        self.index_to_action = dict(enumerate(self.actions))
        self.rng = np.random.default_rng(seed)
        if state_observations is None:
            if len(observations) < len(states):
                raise ValueError("explicit state_observations required; no outcome-zero fallback")
            state_observations = {
                s: [self.obs_to_index[s] if s in self.obs_to_index else i]
                for i, s in enumerate(self.states)
            }
        if not isinstance(state_observations, dict) or set(state_observations) != set(self.states):
            raise ValueError("state_observations must cover every state")
        lengths = set()
        self.state_observations = {}
        for state, values in state_observations.items():
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError("state observations must be nonempty lists")
            self.state_observations[state] = [integer(v, "observation", 0, len(observations) - 1) for v in values]
            lengths.add(len(values))
        if len(lengths) != 1:
            raise ValueError("observation modality count cannot change with state or reward")

    def reset(self, state=None):
        target = self.start_state if self.reset_mode == "configured" else self.states[0]
        target = target if state is None else state
        if target not in self.state_to_index:
            raise ValueError("unknown reset state")
        self.current_state = target
        return self.get_observation()

    def validate_action(self, action):
        if isinstance(action, (list, tuple)):
            if len(action) != 1:
                raise ValueError("environment requires one control factor")
            action = action[0]
        if not isinstance(action, str):
            action = self.index_to_action[action_index(action, len(self.actions))]
        if action not in self.transitions.get(self.current_state, {}):
            raise ValueError("action has no transition from the current state")
        return self.action_to_index[action]

    def step(self, action):
        action = self.actions[self.validate_action(action)]
        self.current_state = self.transitions[self.current_state][action]
        return self.get_observation()

    def get_observation(self):
        return {
            "observation": self.state_observations[self.current_state].copy(),
            "state": [self.state_to_index[self.current_state]],
            "position": None, "reward": self.rewards.get(self.current_state, 0),
            "done": self.is_terminal(), "info": {},
            "current_state": self.current_state,
        }

    def get_valid_actions(self, state=None):
        state = self.current_state if state is None else state
        if state not in self.state_to_index:
            raise ValueError("unknown state")
        return [a for a in self.actions if a in self.transitions.get(state, {})]

    def get_valid_action_indices(self, state=None):
        return [self.action_to_index[a] for a in self.get_valid_actions(state)]

    def get_random_action(self, state=None):
        actions = self.get_valid_actions(state)
        return actions[int(self.rng.integers(len(actions)))] if actions else None

    def has_reward(self, state=None):
        return bool(self.rewards.get(self.current_state if state is None else state, 0) > 0)

    def is_terminal(self, state=None):
        state = self.current_state if state is None else state
        return not self.transitions.get(state) or (self.terminate_on_reward and self.has_reward(state))

    def render(self, mode="text"):
        if mode != "text":
            raise NotImplementedError(mode)
        return (f"Environment: {self.name}\nCurrent state: {self.current_state}\n"
                f"Valid actions: {', '.join(self.get_valid_actions())}\n"
                f"Reward: {self.rewards.get(self.current_state, 0)}\n")

    def to_dict(self):
        return copy.deepcopy({
            "type": "custom", "name": self.name, "states": self.states,
            "observations": self.observations, "transitions": self.transitions,
            "rewards": self.rewards, "current_state": self.current_state,
            "start_state": self.start_state, "state_observations": self.state_observations,
            "actions": self.actions, "reset_mode": self.reset_mode,
            "terminate_on_reward": self.terminate_on_reward,
        })


def create_custom_env(name, states, observations, transitions=None, rewards=None,
                      current_state=None, state_observations=None, actions=None,
                      reset_mode="first", terminate_on_reward=True, seed=None):
    if not states:
        raise ValueError("states cannot be empty")
    if transitions is None:
        transitions = {s: {"forward": states[(i + 1) % len(states)],
                           "backward": states[(i - 1) % len(states)], "stay": s}
                       for i, s in enumerate(states)}
    if rewards is None:
        rewards = {s: float(i == len(states) - 1) for i, s in enumerate(states)}
    env = CustomEnvironment(name, states, observations, transitions, rewards, current_state,
                            state_observations, actions, reset_mode, terminate_on_reward, seed)
    return env.to_dict(), env
