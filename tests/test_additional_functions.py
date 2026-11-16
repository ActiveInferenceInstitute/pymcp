"""Exact numerical helpers and supported construction settings, not fake methods."""
import math

import numpy as np
import pytest

from mcp.models import Limits
from mcp.testing.fixtures import (
    create_multimodal_generative_model,
    create_simple_generative_model,
)
from mcp.utils import PyMDPInterface
from tests.contract_support import ok


def test_vfe_uniform_oracle_has_both_identical_real_decompositions():
    interface = PyMDPInterface()
    result = ok(interface.calculate_free_energy([[0.5, 0.5], [0.5, 0.5]], [0.5, 0.5], [0], [0.5, 0.5]))
    expected = {"free_energy": math.log(2), "energy": math.log(4), "entropy": -math.log(2),
                "divergence": 0, "accuracy": -math.log(2), "free_energy_alt": math.log(2)}
    for key, value in expected.items():
        assert result[key] == pytest.approx(value)


def test_vfe_is_posterior_expectation_not_negative_log_evidence_for_arbitrary_q():
    interface = PyMDPInterface()
    A, prior = [[0.9, 0.1], [0.1, 0.9]], [0.5, 0.5]
    posterior = ok(interface.infer_states_from_observation(A, prior, [0]))["posterior"]
    np.testing.assert_allclose(posterior, [0.9, 0.1])
    assert ok(interface.calculate_free_energy(A, prior, [0], posterior))["free_energy"] == pytest.approx(math.log(2))
    uniform = ok(interface.calculate_free_energy(A, prior, [0], [0.5, 0.5]))
    assert uniform["free_energy"] == pytest.approx(-math.log(0.3))
    assert uniform["free_energy_alt"] == pytest.approx(uniform["free_energy"])


def test_bayes_uses_elementwise_joint_with_the_actual_evidence():
    result = ok(PyMDPInterface().infer_states_from_observation(
        [[0.2, 0.8], [0.8, 0.2]], [0.8, 0.2], [0]))
    np.testing.assert_allclose(result["posterior"], [0.5, 0.5])
    assert result["evidence"] == pytest.approx(0.32)
    assert result["log_evidence"] == pytest.approx(math.log(0.32))


def test_tiny_positive_likelihood_is_not_replaced_by_epsilon():
    interface = PyMDPInterface()
    A = [[1e-250, 2e-250], [1.0, 1.0]]
    result = ok(interface.infer_states_from_observation(A, [0.5, 0.5], [0]))
    np.testing.assert_allclose(result["posterior"], [1 / 3, 2 / 3])
    assert result["log_evidence"] == pytest.approx(math.log(1.5e-250))
    vfe = ok(interface.calculate_free_energy(A, [0.5, 0.5], [0], result["posterior"]))
    assert vfe["free_energy"] == pytest.approx(-math.log(1.5e-250))


@pytest.mark.parametrize("A,prior,observation", [
    ([[1, 0], [0, 1]], [1, 0], [1]),
    ([[0.5, 0.5], [0.5, 0.5]], [1], [0]),
    ([[0.5, 0.5], [0.5, 0.5]], [[0.5], [0.5]], [0]),
    ([[2, -1], [-1, 2]], [0.5, 0.5], [0]),
    ([[float("nan"), 0.5], [0.5, 0.5]], [0.5, 0.5], [0]),
    ([[0.5, 0.5], [0.5, 0.5]], [0.5, 0.5], [True]),
])
def test_invalid_or_zero_evidence_has_no_fabricated_posterior(A, prior, observation):
    assert "error" in PyMDPInterface().infer_states_from_observation(A, prior, observation)


def test_zero_mass_terms_are_exact_and_unsupported_q_is_explicit():
    interface = PyMDPInterface()
    A = [[1, 0], [0, 1]]
    result = ok(interface.calculate_free_energy(A, [1, 0], [0], [1, 0]))
    assert all(value == 0 for value in result.values())
    assert "error" in interface.calculate_free_energy(A, [1, 0], [0], [0.5, 0.5])
    assert "error" in interface.calculate_free_energy(A, [1, 0], [0], [[1], [0]])


def test_multifactor_fixture_is_real_admitted_factor_local_inference():
    interface = PyMDPInterface()
    model = create_multimodal_generative_model(3, [2, 3], [1, 2])
    ok(interface.create_agent("multi", model))
    result = ok(interface.infer_states("multi", [0, 0, 0]))
    assert [len(q) for q in result["posterior_states"]] == [2, 3]
    for q in result["posterior_states"]:
        assert sum(q) == pytest.approx(1)
    policies = ok(interface.infer_policies("multi"))
    assert len(policies["expected_free_energy"]) == 2


def test_learning_configuration_and_concentrations_survive_export_without_mutation():
    interface = PyMDPInterface()
    model = create_simple_generative_model()
    model.update(learn_A=True, learn_B=True, pA=[np.ones((2, 2))], pB=[np.ones((2, 2, 2))])
    ok(interface.create_agent("learning", model))
    agent = interface.agents["learning"]
    assert agent.learn_A and agent.learn_B
    exported = ok(interface.get_agent("learning"))
    np.testing.assert_array_equal(exported["pA"], [np.ones((2, 2))])
    assert exported["learn_A"] is True
    ok(interface.infer_states("learning", [0]))


def test_call_time_controls_reject_instead_of_echoing_unused_settings():
    interface = PyMDPInterface()
    ok(interface.create_agent("a", create_simple_generative_model()))
    ok(interface.infer_states("a", [0]))
    assert "error" in interface.infer_states("a", [0], method="VMP")
    assert "error" in interface.infer_policies("a", planning_horizon=2)
    assert "error" in interface.sample_action("a", planning_horizon=2)
    supported = ok(interface.infer_states("a", [0], method="FPI", save_intermediate_results=True))
    assert supported["method"] == supported["computation_details"]["method"] == "fpi"
    hidden = ok(interface.infer_states("a", [1], save_intermediate_results=False))
    assert "computation_details" not in hidden
    policy = ok(interface.infer_policies("a", save_computation=False))
    assert "log_key" not in policy


@pytest.mark.parametrize("limits", [False, 0, "", {"max_runs": True}, {"max_steps": 0}])
def test_invalid_trusted_limits_do_not_silently_select_defaults(limits):
    with pytest.raises((ValueError, TypeError)):
        PyMDPInterface(limits=limits)


def test_exact_filtering_is_supported_and_smoothing_is_not_silently_claimed():
    interface = PyMDPInterface(limits=Limits())
    model = create_simple_generative_model()
    ok(interface.create_agent("exact", {**model, "inference_algo": "exact"}))
    result = ok(interface.infer_states("exact", [0]))
    np.testing.assert_allclose(result["posterior_states"], [[0.9, 0.1]], atol=1e-6)
    assert result["method"] == "exact"
    assert "error" in interface.create_agent("sequence", {**model, "inference_algo": "mmp"})
