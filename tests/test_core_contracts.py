"""Real-engine synthetic regressions for temporal, model and retention contracts."""
import copy
from unittest.mock import patch

import numpy as np
import pytest
from pymdp.agent import Agent

from mcp.environments import GridWorldEnvironment
from mcp.models import Limits
from mcp.testing.fixtures import create_gridworld_generative_model
from mcp.utils import PyMDPInterface
from tests.contract_support import ONE_CELL, one_cell, ok


def temporal_interface():
    interface = PyMDPInterface()
    model = {"A": [[[0.8, 0.2], [0, 0.6], [0.2, 0.2]]],
             "B": [np.array([[0, 1], [1, 0]])[:, :, None].tolist()], "D": [[0.5, 0.5]]}
    ok(interface.create_agent("a", model))
    return interface


def test_predictive_prior_advances_once_after_explicit_commit():
    interface = temporal_interface()
    first = ok(interface.infer_states("a", [0]))
    np.testing.assert_allclose(first["posterior_states"], [[0.8, 0.2]], atol=1e-6)
    selected = ok(interface.sample_action("a"))
    repeated = ok(interface.sample_action("a"))
    assert selected == repeated
    np.testing.assert_allclose(interface.priors["a"][0], [[0.5, 0.5]])
    ok(interface.commit_action("a", selected["action_id"]))
    assert interface.commit_action("a", selected["action_id"])["error_code"] == "invalid_argument"
    following = ok(interface.infer_states("a", [2]))
    np.testing.assert_allclose(following["posterior_states"], [[0.2, 0.8]], atol=1e-6)


def test_new_evidence_invalidates_policy_and_selection():
    interface = PyMDPInterface()
    B = np.stack((np.eye(2), np.array([[0, 1], [1, 0]])), axis=2)
    ok(interface.create_agent("a", {"A": [[[0.9, 0.1], [0.1, 0.9]]],
                                    "B": [B.tolist()], "C": [[4, 0]]}))
    ok(interface.infer_states("a", [0]))
    selected = ok(interface.sample_action("a"))
    assert selected["action"] == [0]
    ok(interface.infer_states("a", [1]))
    assert "error" in interface.commit_action("a", selected["action_id"])
    next_selection = ok(interface.sample_action("a"))
    assert next_selection["action"] == [1]
    assert next_selection["action_id"] != selected["action_id"]


def test_planning_never_invents_an_observation():
    interface = PyMDPInterface()
    ok(interface.create_agent("a", {"A": [[[0.9, 0.1], [0.1, 0.9]]],
                                    "B": [np.eye(2)[:, :, None].tolist()], "D": [[0.25, 0.75]]}))
    assert "error" in interface.infer_policies("a")
    assert "error" in interface.sample_action("a")
    np.testing.assert_allclose(interface.priors["a"][0], [[0.25, 0.75]])
    assert interface.computation_logs == {}


def test_failed_or_mismatched_environment_step_cannot_commit():
    interface = one_cell()
    ok(interface.infer_states("a", [0, 0]))
    selected = ok(interface.sample_action("a"))
    before = copy.deepcopy(interface.priors["a"])
    for action, token in (([5], selected["action_id"]), ([1], selected["action_id"]), ([0], "stale")):
        assert "error" in interface.step_environment("e", action, "a", token)
        np.testing.assert_array_equal(interface.priors["a"], before)
        assert ok(interface.sample_action("a"))["action_id"] == selected["action_id"]
    with patch.object(interface.environments["e"], "step", side_effect=RuntimeError("synthetic step failure")):
        assert "error" in interface.step_environment("e", [0], "a", selected["action_id"])
    assert interface.sample_action("a") == selected
    transition = ok(interface.step_environment("e", [0], "a", selected["action_id"]))
    assert transition["commit"]["action_id"] == selected["action_id"]
    assert "error" in interface.step_environment("e", [0], "a", selected["action_id"])


def test_episode_reset_restores_D_and_invalidates_old_tokens():
    interface = temporal_interface()
    ok(interface.infer_states("a", [0]))
    selected = ok(interface.sample_action("a"))
    ok(interface.commit_action("a", selected["action_id"]))
    ok(interface.reset_agent("a"))
    np.testing.assert_allclose(interface.priors["a"][0], [[0.5, 0.5]])
    assert "error" in interface.commit_action("a", selected["action_id"])
    assert "error" in interface.infer_policies("a")


def test_every_rectangular_grid_column_matches_shared_environment():
    interface = PyMDPInterface()
    ok(interface.create_gridworld_agent("g", [2, 3], [[1, 2]]))
    assert isinstance(interface.agents["g"], Agent)
    B = np.asarray(ok(interface.get_agent("g"))["B"][0])
    fixture_B = np.asarray(create_gridworld_generative_model([2, 3], [1, 2])["B"][0])
    np.testing.assert_array_equal(B, fixture_B)
    for state in range(6):
        for action in range(5):
            env = GridWorldEnvironment([2, 3], [[1, 2]], list(divmod(state, 3)))
            following = env.step(action)
            column = B[:, state, action]
            assert column.sum() == 1
            assert column[following["state"][0]] == 1
    assert B[:, 0, :].argmax(axis=0).tolist() == [0, 1, 3, 0, 0]
    assert B[:, 5, :].argmax(axis=0).tolist() == [2, 5, 5, 4, 5]


def test_deterministic_factory_does_not_consume_or_depend_on_global_rng():
    interface = PyMDPInterface()
    before = np.random.get_state()
    ok(interface.create_gridworld_agent("g1", [2, 3], []))
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    ok(interface.create_gridworld_agent("g2", [2, 3], []))
    for key in ("A", "B", "C", "D"):
        assert interface.get_agent("g1")[key] == interface.get_agent("g2")[key]


def test_policy_budget_is_independent_of_episode_step_budget():
    interface = PyMDPInterface(limits=Limits(max_policies=16, max_steps=3))
    model = {"A": [[[0.5, 0.5], [0.5, 0.5]]],
             "B": [np.repeat(np.eye(2)[:, :, None], 2, axis=2).tolist()], "policy_len": 4}
    ok(interface.create_agent("fits", model))
    ok(interface.infer_states("fits", [0]))
    assert len(ok(interface.infer_policies("fits"))["policy_posterior"]) == 16
    assert "error" in interface.create_agent("too_many", {**model, "policy_len": 5})
    assert "too_many" not in interface.agents


@pytest.mark.parametrize("steps", [4, -1, 1.5, True])
def test_invalid_steps_refuse_before_reset_or_state_mutation(steps):
    interface = one_cell(limits=Limits(max_steps=3))
    ok(interface.infer_states("a", [0, 0]))
    selected = ok(interface.sample_action("a"))
    before = ok(interface.get_environment("e"))
    assert "error" in interface.run_simulation("a", "e", steps)
    assert interface.get_environment("e") == before
    assert interface.sample_action("a") == selected
    assert interface.get_all_simulations() == {"simulations": []}


def test_direct_tensor_concentration_and_derived_costs_are_bounded():
    interface = PyMDPInterface(limits=Limits(max_elements=32))
    huge = {"A": [np.ones((20, 1)) / 20], "B": [np.ones((1, 1, 1))], "C": [np.zeros(20)]}
    assert "error" in interface.create_agent("large", huge)
    assert "error" in interface.create_gridworld_agent("grid", [3, 3], [])
    model = copy.deepcopy(ONE_CELL)
    model["pA"] = [[[1]] * 40, [[1], [1]]]
    assert "error" in interface.create_agent("concentration", model)
    assert "error" in interface.define_generative_model([[1, 3]], [[2, 2, 1]])
    assert interface.agents == {}


def test_one_hot_has_linear_cost_with_a_bounded_allocator():
    interface = PyMDPInterface()
    ok(interface.create_agent("a", {"A": [(np.ones((64, 1)) / 64).tolist()], "B": [[[[1]]]]}))
    original_eye = np.eye

    def bounded_eye(n, *args, **kwargs):
        if n * n > 256:
            raise MemoryError("synthetic allocator budget")
        return original_eye(n, *args, **kwargs)

    with patch("numpy.eye", bounded_eye):
        vectors = interface._onehot_observations(interface.agents["a"], [63])
    assert vectors[0].shape == (1, 64)
    assert vectors[0][0, 63] == 1 and vectors[0].sum() == 1


def test_retention_ids_fifo_and_no_history_contract():
    interface = one_cell(limits=Limits(max_runs=2, max_logs=2))
    with patch("mcp.utils.time.time", return_value=1):
        runs = [ok(interface.run_simulation("a", "e", 1)) for _ in range(3)]
    ids = [run["simulation_id"] for run in runs]
    assert len(set(ids)) == 3
    listed = interface.get_all_simulations()["simulations"]
    assert [row["simulation_id"] for row in listed] == ids[1:]
    assert "error" in interface.get_simulation(ids[0])
    assert interface.get_computation_logs(ids[-1])["error_code"] == "gone"
    not_retained = ok(interface.run_simulation("a", "e", 1, save_history=False))
    assert not_retained["history"] is None
    assert "error" in interface.get_simulation(not_retained["simulation_id"])
    assert len(interface._logs) <= 2


def test_complete_unbatched_roundtrip_preserves_effective_settings_and_custody():
    interface = PyMDPInterface()
    B = np.zeros((3, 3, 2))
    for state in range(3):
        B[state, state, 0] = B[(state + 1) % 3, state, 1] = 1
    model = {"A": [[[0.8, 0.2, 0.4], [0.2, 0.8, 0.6]]], "B": [B],
             "C": [[-2, 3]], "D": [[0.6, 0.3, 0.1]], "policy_len": 2,
             "inference_horizon": 2, "num_iter": 8, "alpha": 3, "gamma": 2, "seed": 7}
    ok(interface.create_agent("original", model))
    exported = ok(interface.get_agent("original"))
    assert np.asarray(exported["A"][0]).shape == (2, 3)
    assert np.asarray(exported["B"][0]).shape == (3, 3, 2)
    np.testing.assert_allclose(exported["D"], [[0.6, 0.3, 0.1]])
    ok(interface.create_agent("restored", exported))
    restored = ok(interface.get_agent("restored"))
    for key in ("A", "B", "C", "D", "policy_len", "inference_horizon", "num_iter", "alpha", "gamma", "seed"):
        assert restored[key] == exported[key]
    exported["D"][0][0] = 0
    assert interface.get_agent("original")["D"][0][0] == 0.6
