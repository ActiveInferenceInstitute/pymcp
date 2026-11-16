"""Measured variational free energy; policy EFE is never a VFE substitute."""
import copy
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np

from ..environments.common import integer
from ..models import Limits
from ..outputs import OutputPolicy


def measured_records(history, require=True):
    trace = history["free_energy_trace"]
    if not isinstance(trace, list) or len(trace) > Limits().max_steps:
        raise ValueError("invalid or oversized VFE trace")
    measured = []
    for record in trace:
        if not isinstance(record, dict) or record.get("status") not in ("measured", "infinite", "unavailable"):
            raise ValueError("VFE trace requires explicit measurement status")
        integer(record["timestep"], "VFE timestep")
        value = record["free_energy"]
        if record["status"] == "measured":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError("measured VFE must be finite")
            measured.append(copy.deepcopy(record))
        elif value is not None:
            raise ValueError("unavailable VFE cannot carry a numeric substitute")
    if require and not measured:
        raise ValueError("no finite measured variational free energy is available")
    return measured


def _render(history, output_file, components, policy):
    records = measured_records(history)
    keys = ("free_energy", "energy", "entropy", "divergence", "accuracy") if components else ("free_energy",)
    for record in records:
        for key in keys:
            value = record.get(key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"measured {key} component is unavailable")
    authority = policy or OutputPolicy()
    target = authority.path(output_file)
    data_file = str(target.with_suffix("")) + "_data.json"
    authority.admit(target, data_file)
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        trace = history["free_energy_trace"]
        for key in keys:
            values = [entry[key] if entry["status"] == "measured" else np.nan for entry in trace]
            ax.plot(range(len(trace)), values, marker="o", label=key)
        ax.set(title="Measured VFE components" if components else "Measured variational free energy",
               xlabel="Stored transition index (independent episodes)", ylabel="Nats")
        ax.legend()
        fig.tight_layout()
        authority.save_figure(fig, target, dpi=100)
        authority.write_json(data_file, {"records": copy.deepcopy(trace)})
        return str(target)
    finally:
        plt.close(fig)


def visualize_free_energy_components(history, output_file, *, policy=None):
    """Plot genuine VFE components, not fabricated EFE risk/ambiguity terms."""
    return _render(history, output_file, True, policy)


def visualize_variational_free_energy(history, output_file, *, policy=None):
    return _render(history, output_file, False, policy)
