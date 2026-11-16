"""Production environment schemas, stable controls and preserved episode defaults."""
import copy

import pytest

from mcp.environments import (
    CustomEnvironment,
    GridWorldEnvironment,
    create_custom_env,
    create_grid_world_env,
)
from mcp.utils import PyMDPInterface
from tests.contract_support import ok


@pytest.mark.parametrize("shape,rewards", [([0, 2], []), ([2.5, 3], []), ([True, 2], []),
                                           ([2, 3], [[-1, 0]]), ([2, 3], [[1, 3]]),
                                           ([2, 3], [[0.5, 0]]), ([2, 3], [[False, 0]])])
def test_grid_admission_rejects_before_registration(shape, rewards):
    interface = PyMDPInterface()
    assert interface.create_grid_world_env("bad", shape, rewards)["error_code"] == "invalid_argument"
    assert interface.environments == {}


@pytest.mark.parametrize("action", [-1, 5, 1.5, True, [], [0, 1], [False]])
def test_invalid_actions_do_not_move(action):
    env = GridWorldEnvironment([2, 3], [[0, 0]])
    before = env.get_observation()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.get_observation() == before


def test_grid_reset_reward_and_serialization_share_one_record():
    description, env = create_grid_world_env("g", [2, 3], [[0, 0]])
    assert description["current_state"] == 0
    assert description["reward_positions"] == [[0, 0]]
    reset = env.reset()
    assert reset == {"observation": [0, 1], "state": [0], "position": [0, 0],
                     "reward": 1, "done": True, "info": {}}
    following = env.step([1])
    assert following["observation"] == [1, 0] and following["position"] == [0, 1]
    assert following["state"] == [1] and following["reward"] == 0 and following["done"] is False
    description["reward_positions"][0][0] = 1
    assert env.reward_positions == [[0, 0]]


def test_configured_reset_is_explicit_without_changing_origin_default():
    original = GridWorldEnvironment([2, 3], [], [1, 2])
    assert original.get_observation()["position"] == [1, 2]
    assert original.reset()["position"] == [0, 0]
    configured = GridWorldEnvironment([2, 3], [], [1, 2], reset_mode="configured")
    assert configured.reset()["position"] == [1, 2]
    assert configured.reset([0, 1])["position"] == [0, 1]
    assert configured.reset()["position"] == [1, 2]
    with pytest.raises(ValueError):
        configured.reset([2, 0])
    assert configured.get_observation()["position"] == [1, 2]


def test_custom_controls_are_input_ordered_and_modalities_do_not_depend_on_reward():
    env = CustomEnvironment("c", ["s0", "s1"], ["o0", "o1", "reward"],
                            {"s0": {"right": "s1", "left": "s0"}, "s1": {"left": "s0"}}, {"s1": 2})
    assert env.actions == ["right", "left"]
    assert env.get_observation()["observation"] == [0]
    following = env.step([0])
    assert following["observation"] == [1] and following["reward"] == 2
    assert following["position"] is None and following["state"] == [1] and following["done"] is True
    before = env.get_observation()
    with pytest.raises(ValueError):
        env.step([0])  # No right edge at s1; never silently stay.
    assert env.get_observation() == before
    assert env.step([1])["state"] == [0]


def test_custom_reset_and_reward_termination_defaults_remain_explicit():
    args = ("c", ["s0", "s1"], ["s0", "s1"], {"s0": {"go": "s1"}, "s1": {"go": "s1"}}, {"s1": 1})
    original = CustomEnvironment(*args, current_state="s1")
    assert original.is_terminal() is True
    assert original.reset()["current_state"] == "s0"
    configured = CustomEnvironment(*args, current_state="s1", reset_mode="configured", terminate_on_reward=False)
    assert configured.reset()["current_state"] == "s1" and configured.is_terminal() is False
    terminal = CustomEnvironment("terminal", ["s0"], ["o0"], {}, {}, terminate_on_reward=False)
    assert terminal.get_observation()["done"] is True


@pytest.mark.parametrize("changes", [
    {"states": []}, {"states": ["s0", "s0"]}, {"observations": []},
    {"transitions": {"unknown": {"go": "s0"}}}, {"transitions": {"s0": {"go": "missing"}}},
    {"rewards": {"s0": float("nan")}}, {"current_state": "missing"},
    {"state_observations": {"s0": [0], "s1": [0, 1]}},
    {"state_observations": {"s0": [True], "s1": [0]}}, {"actions": ["go", "go"]},
])
def test_custom_admission_never_fabricates_support(changes):
    arguments = {"name": "c", "states": ["s0", "s1"], "observations": ["o0", "o1"],
                 "transitions": {"s0": {"go": "s1"}}, "rewards": {}}
    with pytest.raises(ValueError):
        CustomEnvironment(**{**arguments, **changes})


def test_custom_mapping_requires_actual_declared_support_and_copies_callers():
    with pytest.raises(ValueError):
        CustomEnvironment("c", ["s0", "s1"], ["o0"], {"s0": {"go": "s1"}}, {})
    transitions = {"s0": {"go": "s1"}, "s1": {"go": "s1"}}
    original = copy.deepcopy(transitions)
    description, env = create_custom_env("c", ["s0", "s1"], ["o0", "o1"], transitions, {},
                                         state_observations={"s0": [1], "s1": [0]})
    transitions["s0"]["go"] = "s0"
    description["transitions"]["s0"]["go"] = "s0"
    assert env.transitions == original
    assert env.reset()["observation"] == [1]
    assert env.step(0)["observation"] == [0]


def test_malformed_legacy_dictionary_has_no_synthetic_reset_or_step():
    interface = PyMDPInterface()
    interface.environments["bad"] = {"type": "custom"}
    assert "error" in interface.reset_environment("bad")
    assert "error" in interface.step_environment("bad", 0)
    assert interface.environments["bad"] == {"type": "custom"}


def test_fixed_steps_remain_default_and_stop_on_done_is_opt_in():
    interface = PyMDPInterface()
    model = {"A": [[[1]]], "B": [[[[1]]]], "D": [[1]]}
    ok(interface.create_agent("a", model))
    interface.environments["c"] = CustomEnvironment("c", ["s0"], ["o0"], {"s0": {"stay": "s0"}}, {"s0": 1})
    fixed = ok(interface.run_simulation("a", "c", 2))
    stopped = ok(interface.run_simulation("a", "c", 2, stop_on_done=True))
    assert [row["reward"] for row in fixed["timesteps"]] == [1, 1]
    assert all(row["done"] for row in fixed["timesteps"])
    assert stopped["num_steps"] == 0 and stopped["requested_steps"] == 2 and stopped["timesteps"] == []


def test_fixed_custom_modalities_survive_real_core_steps_without_reward_modalities():
    interface = PyMDPInterface()
    model = {"A": [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
             "B": [[[[0], [1]], [[1], [0]]]], "D": [[1, 0]]}
    ok(interface.create_agent("a", model))
    interface.environments["e"] = CustomEnvironment(
        "e", ["left", "right"], ["zero", "one"],
        {"left": {"swap": "right"}, "right": {"swap": "left"}}, {},
        state_observations={"left": [0, 1], "right": [1, 0]})
    run = ok(interface.run_simulation("a", "e", 2))
    assert [step["observation"] for step in run["timesteps"]] == [[0, 1], [1, 0]]
    assert [step["next_observation"] for step in run["timesteps"]] == [[1, 0], [0, 1]]
    assert [step["state"] for step in run["timesteps"]] == [[0], [1]]
