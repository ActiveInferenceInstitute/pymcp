"""Consumer-visible admission, settings, ownership and full numerical evidence."""
import copy

import numpy as np
import pytest

from mcp.models import Limits
from mcp.testing.fixtures import create_simple_generative_model
from mcp.utils import PyMDPInterface
from tests.contract_support import ok, one_cell


@pytest.mark.parametrize("patch", [
    {"A": [[[0.8, 0.2], [0.1, 0.7]]]},
    {"A": [[[0, 0], [0, 0]]]},
    {"A": [[[float("inf"), 0], [0, 1]]]},
    {"B": [[[[1], [0], [0]], [[0], [1], [1]]]]},
    {"C": [[1]]}, {"D": [[0, 0]]}, {"D": [[-1, 2]]},
    {"pA": [[[1, 0], [0, 1]]]}, {"pB": [[[1, 1], [1, 1]]]},
    {"learn_A": True}, {"num_controls": [3]}, {"num_controls": [True]},
    {"num_state_factors": 2}, {"num_observation_modalities": 3},
    {"policy_len": True}, {"inference_horizon": 0}, {"num_iter": 100000},
    {"inference_algo": "vmp"}, {"action_precision": float("nan")},
    {"alpha": 1, "action_precision": 1}, {"unsupported_setting": 3},
])
def test_bad_model_has_a_real_invalid_result_and_cannot_be_used(patch):
    interface = PyMDPInterface()
    model = {**create_simple_generative_model(), **patch}
    assert interface.validate_generative_model(model)["valid"] is False
    result = interface.create_agent("bad", model)
    assert result["error_code"] == "invalid_argument"
    assert interface.get_agent("bad")["error_code"] == "not_found"


def test_explicit_weight_normalization_never_changes_the_callers_arrays():
    interface = PyMDPInterface()
    A = np.array([[4., 1.], [1., 4.]])
    B = np.full((2, 2, 1), 2.)
    D = np.array([3., 1.])
    model = {"A": [A], "B": [B], "D": [D], "C": [[-5, 2]]}
    before = [value.copy() for value in (A, B, D)]
    assert not interface.validate_generative_model(model)["valid"]
    assert "error" in interface.create_agent("strict", model)
    assert interface.validate_generative_model(model, check_normalization=False)["valid"]
    ok(interface.create_agent("weighted", model, normalize=True))
    for actual, original in zip((A, B, D), before):
        np.testing.assert_array_equal(actual, original)
    exported = ok(interface.get_agent("weighted"))
    np.testing.assert_allclose(exported["A"], [[[0.8, 0.2], [0.2, 0.8]]])
    np.testing.assert_allclose(exported["B"], [np.full((2, 2, 1), 0.5)])
    np.testing.assert_allclose(exported["D"], [[0.75, 0.25]])
    assert exported["C"] == [[-5, 2]]


def test_signed_utilities_are_not_probabilities_or_divided_by_their_sum():
    interface = PyMDPInterface()
    B = np.stack((np.eye(2), np.array([[0, 1], [1, 0]])), axis=2)
    ok(interface.create_agent("a", {"A": [np.eye(2).tolist()], "B": [B.tolist()],
                                    "D": [[1, 0]], "C": [[-5, 2]]}))
    ok(interface.infer_states("a", [0]))
    assert ok(interface.sample_action("a"))["action"] == [1]
    assert ok(interface.get_agent("a"))["C"] == [[-5, 2]]


def test_policy_outputs_preserve_every_real_pymdp_value_and_explicit_sign():
    interface = PyMDPInterface()
    model = create_simple_generative_model()
    model.update(policy_len=2, gamma=3, alpha=2, inference_horizon=4)
    ok(interface.create_agent("a", model))
    ok(interface.infer_states("a", [0]))
    actual = ok(interface.infer_policies("a"))
    q_pi, negative_efe = interface.agents["a"].infer_policies(interface.last_qs["a"])
    assert np.shape(actual["policy_posterior"]) == (4,)
    np.testing.assert_allclose(actual["policy_posterior"], np.asarray(q_pi)[0])
    np.testing.assert_allclose(actual["negative_expected_free_energy"], np.asarray(negative_efe)[0])
    np.testing.assert_allclose(actual["expected_free_energy"], -np.asarray(negative_efe)[0])
    assert "free_energy" not in actual and "ambiguity" not in actual and "risk" not in actual
    exported = ok(interface.get_agent("a"))
    assert (exported["policy_len"], exported["inference_horizon"], exported["gamma"], exported["alpha"]) == (2, 4, 3, 2)


def test_policy_precision_has_a_measurable_effect_on_the_policy_distribution():
    interface = PyMDPInterface()
    B = np.stack((np.eye(2), np.array([[0, 1], [1, 0]])), axis=2).tolist()
    model = {"A": [[[0.9, 0.1], [0.1, 0.9]]], "B": [B], "C": [[4, 0]]}
    probabilities = []
    for name, gamma in (("low", 0.1), ("high", 8)):
        ok(interface.create_agent(name, {**model, "gamma": gamma}))
        ok(interface.infer_states(name, [0]))
        probabilities.append(ok(interface.infer_policies(name))["policy_posterior"][0])
    assert 0.5 < probabilities[0] < probabilities[1] < 1.000001


@pytest.mark.parametrize("observation", [[-1, 0], [1, 0], [0], [True, 0], [0, 0, 0], [0.5, 0]])
def test_observation_admission_rejects_instead_of_casting_or_dropping(observation):
    interface = one_cell()
    assert interface.infer_states("a", observation)["error_code"] == "invalid_argument"
    assert "error" in interface.sample_action("a")


def test_structural_zero_support_is_reported_instead_of_claiming_finite_vfe():
    interface = PyMDPInterface()
    ok(interface.create_agent("a", {"A": [np.eye(2).tolist()],
                                    "B": [np.eye(2)[:, :, None].tolist()], "D": [[1, 0]]}))
    result = ok(interface.infer_states("a", [0]))
    record = result["variational_free_energy"]
    q = np.asarray(result["posterior_states"][0])
    if q[1] > 0:
        assert record["status"] == "infinite" and record["free_energy"] is None
    else:
        assert record["status"] == "measured" and record["free_energy"] == pytest.approx(0)


def test_singleton_models_and_exported_copies_preserve_all_dimensions():
    interface = one_cell()
    exported = ok(interface.get_agent("a"))
    assert [np.shape(a) for a in exported["A"]] == [(1, 1), (2, 1)]
    assert [np.shape(b) for b in exported["B"]] == [(1, 1, 1)]
    original = copy.deepcopy(exported)
    exported["A"][0][0][0] = 0
    assert ok(interface.get_agent("a")) == original
    ok(interface.create_agent("roundtrip", original))
    ok(interface.infer_states("roundtrip", [0, 0]))


def test_entity_limits_reject_new_bindings_without_replacing_existing_objects():
    interface = PyMDPInterface(limits=Limits(max_entities=1))
    ok(interface.create_agent("a", {"A": [[[1]]], "B": [[[[1]]]]}))
    original = interface.agents["a"]
    assert interface.create_agent("a", {"A": [[[1]]], "B": [[[[1]]]]})["error_code"] == "conflict"
    assert "error" in interface.create_agent("b", {"A": [[[1]]], "B": [[[[1]]]]})
    assert interface.agents["a"] is original


def test_finite_concentrations_with_overflowed_mass_are_not_admitted():
    core = PyMDPInterface()
    model = create_simple_generative_model()
    model["pA"] = [[[1e308, 1e308], [1e308, 1e308]]]
    assert core.create_agent("overflow", model)["error_code"] == "invalid_argument"
    assert core.get_agent("overflow")["error_code"] == "not_found"
