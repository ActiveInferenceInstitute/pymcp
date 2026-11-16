"""Validated, step-aligned belief readers and production heatmaps."""
import matplotlib.pyplot as plt
import numpy as np

from .models import Limits, probability
from .outputs import OutputPolicy


def belief_arrays(beliefs):
    if not isinstance(beliefs, (list, tuple)) or not beliefs or not beliefs[0]:
        raise ValueError("belief history is unavailable")
    limits = Limits()
    if len(beliefs) > limits.max_steps or len(beliefs[0]) > limits.max_entities:
        raise ValueError("belief history exceeds the plotting budget")
    sizes = [len(q) for q in beliefs[0]]
    if not all(sizes) or len(beliefs) * sum(sizes) > limits.max_elements:
        raise ValueError("belief history exceeds the element budget")
    for sample in beliefs:
        if len(sample) != len(sizes):
            raise ValueError("belief factor count changes across timesteps")
        for q, size in zip(sample, sizes):
            values = np.asarray(q, dtype=float)
            if values.shape != (size,) or not np.all(np.isfinite(values)):
                raise ValueError("malformed belief history")
            probability(values)
    return [np.asarray([sample[f] for sample in beliefs], dtype=float) for f in range(len(sizes))]


def extract_belief_history(history):
    timesteps = history["timesteps"]
    if not timesteps:
        return [], 0, []
    beliefs = [step["beliefs"] for step in timesteps]
    arrays = belief_arrays(beliefs)
    return beliefs, len(beliefs), [values.shape[1] for values in arrays]


def create_belief_dynamics_visualization(history, output_file,
                                        title="Belief Dynamics Over Time",
                                        show_state_labels=True, custom_state_labels=None, *, policy=None):
    beliefs, count, sizes = extract_belief_history(history)
    arrays = belief_arrays(beliefs)
    if type(show_state_labels) is not bool:
        raise ValueError("show_state_labels must be boolean")
    if custom_state_labels is not None:
        if len(custom_state_labels) != len(sizes) or any(len(labels) != n for labels, n in zip(custom_state_labels, sizes)):
            raise ValueError("custom labels must identify every factor/state exactly")
    authority = policy or OutputPolicy()
    target = authority.path(output_file)
    fig, axes = plt.subplots(len(arrays), 1, figsize=(10, 3 * len(arrays)), squeeze=False)
    try:
        for factor, values in enumerate(arrays):
            ax = axes[factor, 0]
            im = ax.imshow(values.T, origin="lower", aspect="auto", vmin=0, vmax=1,
                           extent=[-0.5, count - 0.5, -0.5, sizes[factor] - 0.5])
            fig.colorbar(im, ax=ax, label="Belief probability")
            ax.set(title=f"Factor {factor}", xlabel="Stored transition index", ylabel="State")
            if show_state_labels and sizes[factor] <= 20:
                ax.set_yticks(range(sizes[factor]))
                if custom_state_labels is not None:
                    ax.set_yticklabels(custom_state_labels[factor])
        fig.suptitle(title)
        fig.tight_layout()
        return authority.save_figure(fig, target, dpi=100)
    finally:
        plt.close(fig)


def calculate_belief_statistics(beliefs):
    arrays = belief_arrays(beliefs)
    result = {"num_factors": len(arrays), "num_steps": len(beliefs), "factors": []}
    for values in arrays:
        logs = np.zeros_like(values)
        np.log(values, out=logs, where=values > 0)
        entropy = -(values * logs).sum(axis=1)
        states = values.argmax(axis=1)
        changes = int(np.count_nonzero(np.diff(states)))
        result["factors"].append({
            "num_states": values.shape[1], "mean_entropy": float(entropy.mean()),
            "max_entropy": float(entropy.max()), "min_entropy": float(entropy.min()),
            "entropy_over_time": entropy.tolist(), "max_belief_states": states.tolist(),
            "state_changes": changes, "state_change_rate": changes / (len(values) - 1) if len(values) > 1 else 0,
        })
    return result
