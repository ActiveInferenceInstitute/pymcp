"""Production renderers, measured evidence, geometry and all output boundaries."""
import asyncio
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from mcp.api import ToolRegistry
from mcp.models import Limits
from mcp.outputs import OutputPolicy
from mcp.server.app import create_app
from mcp.testing.helpers import (
    generate_test_report,
    plot_test_results,
    save_test_results,
)
from mcp.visualization import (
    create_simulation_animation,
    visualize_expected_free_energy,
    visualize_free_energy_components,
    visualize_generative_model,
    visualize_simulation,
)
from tests.contract_support import asgi_request, document, ok, one_cell, rectangular


def test_rectangular_trajectory_uses_actual_positions_columns_and_final_endpoint(tmp_path, monkeypatch):
    _, run = rectangular()
    history = run["history"]
    assert [row["action"] for row in history["timesteps"]] == [[1], [1], [2]]
    captured = {}
    original = OutputPolicy.save_figure
    def capture(policy, figure, destination, **options):
        ax = figure.axes[0]
        captured["x"] = list(ax.lines[0].get_xdata())
        captured["y"] = list(ax.lines[0].get_ydata())
        captured["xlim"], captured["ylim"] = ax.get_xlim(), ax.get_ylim()
        return original(policy, figure, destination, **options)
    monkeypatch.setattr(OutputPolicy, "save_figure", capture)
    unchanged = copy.deepcopy(history)
    visualize_simulation(history, tmp_path / "rectangle.png")
    assert captured["x"] == [0, 1, 2, 2]
    assert captured["y"] == [0, 0, 0, 1]
    np.testing.assert_allclose(captured["xlim"], [-0.5, 2.5])
    np.testing.assert_allclose(captured["ylim"], [1.5, -0.5])
    assert history == unchanged
    with Image.open(tmp_path / "rectangle.png") as image:
        image.verify()


def test_model_roundtrip_renderer_keeps_unbatched_asymmetric_axes_and_signed_C(tmp_path, monkeypatch):
    from mcp.utils import PyMDPInterface
    core = PyMDPInterface()
    model = {"A": [[[0.8, 0.3, 0.1], [0.2, 0.7, 0.9]]],
             "B": [np.stack((np.eye(3), np.roll(np.eye(3), 1, axis=0)), axis=2).tolist()],
             "C": [[-4, 2]], "D": [[0.6, 0.3, 0.1]], "policy_len": 2, "inference_horizon": 4}
    ok(core.create_agent("first", model))
    exported = ok(core.get_agent("first"))
    ok(core.create_agent("second", exported))
    assert ok(core.get_agent("second"))["D"] == model["D"]
    captured = {}
    original = OutputPolicy.save_figure
    def capture(policy, figure, destination, **options):
        captured["A"] = np.asarray(figure.axes[0].images[0].get_array()).copy()
        return original(policy, figure, destination, **options)
    monkeypatch.setattr(OutputPolicy, "save_figure", capture)
    visualize_generative_model("second", ok(core.get_agent("second")), tmp_path / "model.svg")
    np.testing.assert_array_equal(captured["A"], model["A"][0])
    assert b"<svg" in (tmp_path / "model.svg").read_bytes()


def test_real_run_logs_never_include_a_later_run_and_expiry_is_unavailable(tmp_path):
    core = one_cell(output_root=tmp_path)
    first = ok(core.run_simulation("a", "e", 2))
    second = ok(core.run_simulation("a", "e", 1))
    result = ok(core.visualize_simulation(first["simulation_id"], "first.png"))
    log_file = next(Path(path) for path in result["additional_visualizations"] if path.endswith("_computation_logs.json"))
    logs = json.loads(log_file.read_text())
    assert logs["simulation_id"] == first["simulation_id"]
    assert {entry["simulation_id"] for entry in logs["logs"]} == {first["simulation_id"]}
    assert [entry["step"] for entry in logs["logs"]] == [0] * 4 + [1] * 4
    assert not any(entry["simulation_id"] == second["simulation_id"] for entry in logs["logs"])
    small = one_cell(output_root=tmp_path / "expired", limits=Limits(max_logs=4))
    old = ok(small.run_simulation("a", "e", 1))
    ok(small.run_simulation("a", "e", 1))
    assert small.visualize_simulation(old["simulation_id"], "must-not-fake.png")["error_code"] == "gone"
    assert not (tmp_path / "expired" / "must-not-fake.png").exists()


def test_measured_vfe_and_full_efe_are_not_scalar_or_invented_components(tmp_path, monkeypatch):
    core = one_cell(output_root=tmp_path)
    run = ok(core.run_simulation("a", "e", 2))
    visualize_free_energy_components(run["history"], tmp_path / "vfe.png")
    actual = json.loads((tmp_path / "vfe_data.json").read_text())["records"]
    assert actual == run["free_energy_trace"]
    for entry in actual:
        assert entry["status"] == "measured" and entry["free_energy"] == 0
        assert entry["energy"] + entry["entropy"] == pytest.approx(entry["divergence"] - entry["accuracy"])
        assert not {"ambiguity", "risk"} & entry.keys()
    captured = {}
    original = OutputPolicy.save_figure
    def capture(policy, figure, destination, **options):
        captured["matrix"] = np.asarray(figure.axes[0].images[0].get_array()).copy()
        return original(policy, figure, destination, **options)
    monkeypatch.setattr(OutputPolicy, "save_figure", capture)
    visualize_expected_free_energy([[1, -2, 3], [4, 5, -6]], tmp_path / "efe.png")
    np.testing.assert_array_equal(captured["matrix"], [[1, 4], [-2, 5], [3, -6]])
    unavailable = copy.deepcopy(run["history"])
    unavailable["free_energy_trace"] = [{"status": "unavailable", "free_energy": None, "timestep": 0}]
    with pytest.raises(ValueError, match="no finite measured"):
        visualize_free_energy_components(unavailable, tmp_path / "unavailable.png")
    assert not (tmp_path / "unavailable.png").exists()


def test_actual_gif_contains_final_endpoint_and_requested_duration(tmp_path):
    figures_before = set(plt.get_fignums())
    _, run = rectangular()
    create_simulation_animation(run["history"], tmp_path / "trajectory.gif", interval=120)
    with Image.open(tmp_path / "trajectory.gif") as image:
        assert image.format == "GIF" and image.n_frames == 4
        durations = []
        for frame in range(image.n_frames):
            image.seek(frame)
            durations.append(image.info["duration"])
        assert durations == [120] * 4
    with pytest.raises(ValueError):
        create_simulation_animation(run["history"], tmp_path / "bad.gif", interval=123)
    with pytest.raises(ValueError):
        create_simulation_animation(run["history"], tmp_path / "oversize.gif", max_frames=2)
    assert not (tmp_path / "bad.gif").exists() and not (tmp_path / "oversize.gif").exists()
    assert set(plt.get_fignums()) == figures_before


def test_basename_only_helpers_create_real_report_and_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = {"accuracy": 0.75, "passed": True}
    save_test_results(results, str(tmp_path / "results"), "case", include_timestamp=False)
    generate_test_report(str(tmp_path / "results"), "report.md")
    assert "0.75" in (tmp_path / "report.md").read_text()
    plot_test_results(results, "metrics.png")
    with Image.open("metrics.png") as image:
        assert image.format == "PNG"
        image.verify()


PLOTS = [
    ("visualize_agent_model", "agent_id", "a"),
    ("visualize_simulation", "simulation_id", None),
    ("visualize_belief_dynamics", "session_id", "s"),
    ("plot_belief_dynamics", "session_id", "s"),
    ("analyze_free_energy", "session_id", "s"),
]


def confined_case(root):
    core = one_cell(output_root=root)
    ok(core.create_session("s", "a", "e"))
    run = ok(core.run_session("s", 2))
    return core, run["simulation_id"]


@pytest.mark.parametrize("name,key,identity", PLOTS)
@pytest.mark.parametrize("transport", ["registry", "rest"])
def test_every_plot_and_every_reported_companion_rejects_outside_symlinks(tmp_path, name, key, identity, transport):
    core, simulation_id = confined_case(tmp_path / "server")
    root = core.output_policy.root
    registry = ToolRegistry(core)
    app = create_app(interface=core, output_root=root)
    arguments = {key: simulation_id if identity is None else identity, "output_file": "render.png"}
    def call(params):
        if transport == "registry":
            return registry.call(name, params)
        status, value = document(asyncio.run(asgi_request(app, "POST", "/tools/" + name, {"params": params})))
        if status == 200:
            return value["result"]
        assert status == 422
        return {"error": value["error"]["message"], "error_code": value["error"]["code"]}
    manifest = ok(call(arguments))
    outputs = [Path(manifest["figure_path"]), *map(Path, manifest.get("additional_visualizations", []))]
    sentinel = tmp_path / "outside"
    sentinel.write_bytes(b"outside-must-not-change")
    for victim in outputs:
        for output in outputs:
            if output.exists() or output.is_symlink():
                output.unlink()
        victim.symlink_to(sentinel)
        try:
            assert "error" in call(arguments), (name, victim)
            assert sentinel.read_bytes() == b"outside-must-not-change"
            assert victim.is_symlink()
            assert all(not output.exists() for output in outputs if output != victim)
        finally:
            victim.unlink()


@pytest.mark.parametrize("name,key,identity", PLOTS)
@pytest.mark.parametrize("transport", ["registry", "rest"])
def test_all_plot_primary_paths_reject_traversal_absolute_and_symlink_parents(tmp_path, name, key, identity, transport):
    core, simulation_id = confined_case(tmp_path / "server")
    root = core.output_policy.root
    root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "external"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    registry, app = ToolRegistry(core), create_app(interface=core, output_root=root)
    for destination in ("../external/traversed.png", str(outside / "absolute.png"), "linked/nested.png"):
        arguments = {key: simulation_id if identity is None else identity, "output_file": destination}
        if transport == "registry":
            assert "error" in registry.call(name, arguments)
        else:
            status, body = document(asyncio.run(asgi_request(app, "POST", "/tools/" + name, {"params": arguments})))
            assert status == 422 and set(body) == {"error"}
    assert list(outside.iterdir()) == []
