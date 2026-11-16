"""Real producer history fed unchanged into the production belief readers."""
import copy

import numpy as np
import pytest

from mcp.belief_dynamics import (
    calculate_belief_statistics,
    create_belief_dynamics_visualization,
    extract_belief_history,
)
from mcp.outputs import OutputPolicy
from tests.contract_support import chain


def test_chain_history_keeps_pretransition_beliefs_separate_from_physical_state(tmp_path):
    _, run = chain()
    history = run["history"]
    original = copy.deepcopy(history)
    assert [step["observation"] for step in history["timesteps"]] == [[0], [1]]
    assert [step["state"] for step in history["timesteps"]] == [[0], [1]]
    assert [step["next_observation"] for step in history["timesteps"]] == [[1], [2]]
    assert [step["next_state"] for step in history["timesteps"]] == [[1], [2]]
    np.testing.assert_allclose([step["beliefs"][0] for step in history["timesteps"]],
                               [[1, 0, 0], [0, 1, 0]], atol=1e-6)
    beliefs, count, sizes = extract_belief_history(history)
    assert count == 2 and sizes == [3]
    statistics = calculate_belief_statistics(beliefs)
    factor = statistics["factors"][0]
    assert factor["max_belief_states"] == [0, 1]
    assert factor["state_changes"] == 1 and factor["state_change_rate"] == 1
    assert factor["mean_entropy"] == pytest.approx(0, abs=1e-5)
    output = tmp_path / "actual-chain.png"
    create_belief_dynamics_visualization(history, output)
    assert output.read_bytes().startswith(b"\x89PNG")
    assert history == original


def test_heatmap_centers_labels_and_probabilities_are_real(tmp_path, monkeypatch):
    _, run = chain()
    captured = {}
    original = OutputPolicy.save_figure
    def capture(policy, figure, destination, **options):
        ax = figure.axes[0]
        captured["extent"] = list(ax.images[0].get_extent())
        captured["values"] = np.asarray(ax.images[0].get_array()).copy()
        captured["labels"] = [label.get_text() for label in ax.get_yticklabels()]
        captured["origin"] = ax.images[0].origin
        return original(policy, figure, destination, **options)
    monkeypatch.setattr(OutputPolicy, "save_figure", capture)
    create_belief_dynamics_visualization(run["history"], tmp_path / "heatmap.png",
                                        custom_state_labels=[["zero", "one", "two"]])
    assert captured["origin"] == "lower"
    assert captured["extent"] == [-0.5, 1.5, -0.5, 2.5]
    assert captured["labels"] == ["zero", "one", "two"]
    np.testing.assert_allclose(captured["values"], [[1, 0], [0, 1], [0, 0]], atol=1e-6)


@pytest.mark.parametrize("beliefs", [
    [], [[[0.5, 0.5]], []], [[[0.5, 0.5]], [[1]]],
    [[[0.2, 0.2]]], [[[float("nan"), 1]]], [[[-1, 2]]],
])
def test_missing_ragged_or_invalid_history_is_not_filled_or_compressed(beliefs):
    with pytest.raises(ValueError):
        calculate_belief_statistics(beliefs)


def test_zero_probability_entropy_and_single_transition_denominator():
    statistics = calculate_belief_statistics([[[1, 0]]])
    assert statistics["factors"][0]["entropy_over_time"] == [0]
    assert statistics["factors"][0]["state_change_rate"] == 0
    assert calculate_belief_statistics([[[1, 0]], [[0, 1]]])["factors"][0]["state_change_rate"] == 1


def test_incomplete_labels_fail_instead_of_mislabeling_state_rows(tmp_path):
    _, run = chain()
    with pytest.raises(ValueError):
        create_belief_dynamics_visualization(run["history"], tmp_path / "bad.png",
                                            custom_state_labels=[["not", "enough"]])
    assert not (tmp_path / "bad.png").exists()
