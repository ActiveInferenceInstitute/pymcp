"""Production renderers for the canonical pre-/post-transition history."""
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np

from ..belief_dynamics import belief_arrays, create_belief_dynamics_visualization
from ..environments.common import grid_shape, integer, position
from ..models import Limits, probability
from ..outputs import OutputPolicy


def _matrix(samples, probabilities=False):
    limits = Limits()
    if not isinstance(samples, (list, tuple)) or not samples or not samples[0]:
        raise ValueError("series is unavailable")
    width = len(samples[0])
    if len(samples) > limits.max_steps or len(samples) * width > limits.max_elements:
        raise ValueError("series exceeds the plotting budget")
    if any(len(sample) != width for sample in samples):
        raise ValueError("series cardinality changes across timesteps")
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("malformed/nonfinite series")
    if probabilities:
        for row in values:
            probability(row)
    return values


def spatial_segments(history):
    steps = history["timesteps"]
    if not steps or len(steps) > Limits().max_steps:
        raise ValueError("empty or oversized history")
    points = [(step["position"], step["next_position"]) for step in steps]
    if all(before is None and after is None for before, after in points):
        return []
    segments, previous_run = [], None
    shape = history.get("environment", {}).get("grid_size")
    if shape is not None:
        shape = grid_shape(shape)
    for step, (before, after) in zip(steps, points):
        clean = []
        for point in (before, after):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("spatial records require both row/column positions")
            clean.append(position(point, shape) if shape is not None else [integer(v, "position") for v in point])
        run = step["simulation_id"]
        if run != previous_run:
            segments.append([clean[0]])
        elif segments[-1][-1] != clean[0]:
            raise ValueError("trajectory is discontinuous within a run")
        segments[-1].append(clean[1])
        previous_run = run
    return segments


def _spatial_axes(ax, history, segments):
    shape = history.get("environment", {}).get("grid_size")
    if shape is not None:
        height, width = grid_shape(shape)
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
    else:
        points = [p for segment in segments for p in segment]
        ax.set_xlim(min(p[1] for p in points) - 0.5, max(p[1] for p in points) + 0.5)
        ax.set_ylim(max(p[0] for p in points) + 0.5, min(p[0] for p in points) - 0.5)
    ax.set(xlabel="Column", ylabel="Row")


def plot_beliefs(ax, belief_data):
    values = belief_arrays(belief_data)[0]
    ax.imshow(values.T, origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.set(title="Beliefs — factor 0", xlabel="Stored transition index", ylabel="State")


def plot_policies(ax, policy_data, action_data=None):
    values = _matrix(policy_data, probabilities=True)
    if action_data is not None and len(action_data) != len(values):
        raise ValueError("action/policy time axes differ")
    ax.imshow(values.T, origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.set(title="Full policy posterior", xlabel="Stored transition index", ylabel="Policy")


def plot_actions(ax, action_data):
    values = _matrix(action_data)
    for row in action_data:
        for action in row:
            integer(action, "action")
    for factor in range(values.shape[1]):
        ax.step(range(len(values)), values[:, factor], where="post", label=f"Control {factor}")
    ax.set(title="Selected actions (not policy identities)", xlabel="Stored transition index", ylabel="Action")
    ax.legend()


def visualize_simulation(simulation_history, output_file, include_beliefs=True, include_policies=True, *, policy=None):
    """Render the primary figure only; core owns its complete companion manifest."""
    if type(include_beliefs) is not bool or type(include_policies) is not bool:
        raise ValueError("include controls must be boolean")
    steps = simulation_history["timesteps"]
    segments = spatial_segments(simulation_history)
    beliefs = [step["beliefs"] for step in steps]
    policies = [step["policy_posterior"] for step in steps]
    efe = _matrix([step["expected_free_energy"] for step in steps])
    if include_beliefs:
        belief_arrays(beliefs)
    if include_policies:
        _matrix(policies, probabilities=True)
    rewards = [step["reward"] for step in steps]
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not np.isfinite(v) for v in rewards):
        raise ValueError("invalid transition reward")
    authority = policy or OutputPolicy()
    target = authority.path(output_file)
    panels = 2 + bool(segments) + include_beliefs + include_policies
    fig, axes = plt.subplots(panels, 1, figsize=(10, 3 * panels), squeeze=False)
    try:
        axes = iter(axes[:, 0])
        if segments:
            ax = next(axes)
            for index, segment in enumerate(segments):
                ax.plot([p[1] for p in segment], [p[0] for p in segment], "o-", label=f"Episode {index}")
            _spatial_axes(ax, simulation_history, segments)
            ax.set_title("Actual trajectory, including final endpoints")
        if include_beliefs:
            plot_beliefs(next(axes), beliefs)
        if include_policies:
            plot_policies(next(axes), policies)
        ax = next(axes)
        im = ax.imshow(efe.T, origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set(title="Full expected free energy by policy", xlabel="Stored transition index", ylabel="Policy")
        ax = next(axes)
        ax.bar(range(len(rewards)), rewards, label="Transition reward")
        ax.plot(np.cumsum(rewards), label="Cumulative reward")
        ax.set(xlabel="Stored transition index", ylabel="Reward")
        ax.legend()
        fig.suptitle(f"{simulation_history['agent_id']} in {simulation_history['env_id']}")
        fig.tight_layout()
        saved = authority.save_figure(fig, target, dpi=100)
        return {"figure_path": saved, "additional_visualizations": []}
    finally:
        plt.close(fig)


def create_belief_heatmap(belief_data, output_file, *, policy=None):
    return create_belief_dynamics_visualization(
        {"timesteps": [{"beliefs": sample} for sample in belief_data]}, output_file, policy=policy)


def plot_belief_trajectories(belief_data, output_file, *, policy=None):
    arrays = belief_arrays(belief_data)
    authority = policy or OutputPolicy()
    target = authority.path(output_file)
    fig, axes = plt.subplots(len(arrays), 1, figsize=(10, 3 * len(arrays)), squeeze=False)
    try:
        for factor, values in enumerate(arrays):
            axes[factor, 0].plot(values)
            axes[factor, 0].set(title=f"Belief trajectories — factor {factor}",
                                xlabel="Stored transition index", ylabel="Probability", ylim=(0, 1))
        fig.tight_layout()
        return authority.save_figure(fig, target, dpi=100)
    finally:
        plt.close(fig)


def _heatmap(samples, output_file, title, probabilities, policy):
    values = _matrix(samples, probabilities=probabilities)
    authority = policy or OutputPolicy()
    target = authority.path(output_file)
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        options = {"vmin": 0, "vmax": 1} if probabilities else {}
        im = ax.imshow(values.T, origin="lower", aspect="auto", **options)
        fig.colorbar(im, ax=ax)
        ax.set(title=title, xlabel="Stored transition index", ylabel="Policy")
        fig.tight_layout()
        return authority.save_figure(fig, target, dpi=100)
    finally:
        plt.close(fig)


def visualize_policy_evolution(policy_data, output_file, *, policy=None):
    return _heatmap(policy_data, output_file, "Full policy posterior", True, policy)


def visualize_expected_free_energy(efe_data, output_file, *, policy=None):
    return _heatmap(efe_data, output_file, "Full expected free energy (no scalar reduction)", False, policy)


def visualize_free_energy(fe_trace, output_file, *, policy=None):
    from .free_energy import visualize_variational_free_energy
    return visualize_variational_free_energy({"free_energy_trace": fe_trace}, output_file, policy=policy)


def create_simulation_animation(simulation_history, output_file, interval=200, *,
                                max_frames=2000, max_pixels=64_000_000, policy=None):
    """Real GIF; supported durations are 10..60000 ms in 10 ms increments."""
    interval = integer(interval, "interval", 10, 60000)
    if interval % 10:
        raise ValueError("GIF interval must be a multiple of 10 ms")
    max_frames = integer(max_frames, "max_frames", 1)
    max_pixels = integer(max_pixels, "max_pixels", 1)
    segments = spatial_segments(simulation_history)
    if not segments:
        raise ValueError("spatial animation unavailable for nonspatial history")
    count = sum(len(segment) for segment in segments)
    if count > max_frames or count * 600 * 400 > max_pixels:
        raise ValueError("animation exceeds frame/pixel budget")
    authority = policy or OutputPolicy()
    target = authority.path(output_file)
    if target.suffix.lower() != ".gif":
        raise ValueError("animation output must be GIF")
    from PIL import Image
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    images = []
    try:
        _spatial_axes(ax, simulation_history, segments)
        line, = ax.plot([], [], "o-")
        point, = ax.plot([], [], "o")
        for episode, segment in enumerate(segments):
            for frame, (row, col) in enumerate(segment):
                line.set_data([p[1] for p in segment[:frame + 1]], [p[0] for p in segment[:frame + 1]])
                point.set_data([col], [row])
                ax.set_title(f"Episode {episode}, position {frame}")
                fig.canvas.draw()
                with Image.fromarray(np.asarray(fig.canvas.buffer_rgba())) as rgba:
                    images.append(rgba.convert("RGB"))
        with authority.writer(target) as stream:
            images[0].save(stream, format="GIF", save_all=True, append_images=images[1:],
                           duration=interval, loop=0)
        return str(target)
    finally:
        for image in images:
            image.close()
        plt.close(fig)
