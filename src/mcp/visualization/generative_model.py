"""Render the same validated, unbatched model accepted by the interface."""
import matplotlib.pyplot as plt

from ..models import Limits, admit_model
from ..outputs import OutputPolicy


def visualize_generative_model(agent_name, matrices, output_file, *, policy=None):
    clean, settings = admit_model(matrices, Limits())
    authority = policy or OutputPolicy()
    target = authority.path(output_file)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    try:
        for column, key in enumerate(("A", "B", "C", "D")):
            for row in range(2):
                ax = axes[row, column]
                if row >= len(clean[key]):
                    ax.set_axis_off()
                    continue
                value = clean[key][row]
                if key == "B":
                    value = value[:, :, 0]
                elif value.ndim == 1:
                    value = value[:, None]
                im = ax.imshow(value, aspect="auto", origin="lower")
                fig.colorbar(im, ax=ax)
                ax.set_title(f"{key}[{row}]" + (" control 0" if key == "B" else ""))
                if key == "A":
                    ax.set(xlabel="State", ylabel="Observation")
                elif key == "B":
                    ax.set(xlabel="Current state", ylabel="Next state")
                elif key == "C":
                    ax.set_ylabel("Finite preference utilities (not probabilities)")
                else:
                    ax.set_ylabel("Initial-state probabilities")
        fig.suptitle(f"{agent_name}: unbatched model, policy_len={settings['policy_len']}\n"
                     "Display: first two entries per component, B control 0; export remains complete")
        fig.tight_layout()
        return authority.save_figure(fig, target, dpi=100)
    finally:
        plt.close(fig)
