"""Shared discrete environment contract; five grid controls in fixed order."""
from numbers import Integral, Real

import numpy as np

GRID_ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1), (0, 0))
GRID_ACTION_NAMES = ("up", "right", "down", "left", "stay")


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the supported range")
    return value


def grid_shape(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("grid_size must contain height and width")
    shape = [integer(v, "grid dimension", 1, 10000) for v in value]
    if shape[0] * shape[1] > 10000:
        raise ValueError("grid exceeds 10000 states")
    return shape


def position(value, shape):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("position must contain row and column")
    return [integer(value[i], "position", 0, shape[i] - 1) for i in range(2)]


def action_index(action, count):
    if isinstance(action, (list, tuple)):
        if len(action) != 1:
            raise ValueError("environment requires exactly one control factor")
        action = action[0]
    return integer(action, "action", 0, count - 1)


def grid_transition(state, action, shape):
    state = integer(state, "state", 0, shape[0] * shape[1] - 1)
    action = action_index(action, len(GRID_ACTIONS))
    row, col = divmod(state, shape[1])
    dr, dc = GRID_ACTIONS[action]
    row = min(max(row + dr, 0), shape[0] - 1)
    col = min(max(col + dc, 0), shape[1] - 1)
    return row * shape[1] + col


def environment_record(value):
    """Validate a record without synthesizing absent evidence or modalities."""
    if not isinstance(value, dict):
        raise ValueError("environment must return an observation record")
    required = {"observation", "state", "position", "reward", "done", "info"}
    if not required <= value.keys():
        raise ValueError("environment record is missing required fields")
    obs = value["observation"]
    if not isinstance(obs, (list, tuple)) or not obs:
        raise ValueError("environment observation must be a nonempty list")
    for index in obs:
        integer(index, "observation")
    state = value["state"]
    if not isinstance(state, (list, tuple)) or not state:
        raise ValueError("environment state must be a nonempty list of physical indices")
    for index in state:
        integer(index, "state")
    if value["position"] is not None:
        for index in value["position"]:
            integer(index, "position")
        if len(value["position"]) != 2:
            raise ValueError("position must contain row and column")
    reward = value["reward"]
    if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, Real) or not np.isfinite(reward):
        raise ValueError("reward must be finite")
    if type(value["done"]) is not bool or not isinstance(value["info"], dict):
        raise ValueError("environment done/info are invalid")
    return value
