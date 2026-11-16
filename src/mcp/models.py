"""Unbatched model admission and exact finite-support numerical helpers."""
import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from .environments.common import integer


@dataclass(frozen=True)
class Limits:
    max_elements: int = 4_000_000
    max_policies: int = 4096
    max_policy_len: int = 32
    max_steps: int = 10000
    max_runs: int = 128
    max_logs: int = 1024
    max_entities: int = 128
    max_iterations: int = 128
    max_frames: int = 2000

    def __post_init__(self):
        for name, value in vars(self).items():
            integer(value, name, 1)


def shape_of(value, limit):
    """Inspect shape and cost before converting/copying a caller tensor."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "fiu" or value.size > limit or not value.size:
            raise ValueError("array dtype/element budget is unsupported")
        return value.shape
    remaining = limit

    def visit(node):
        nonlocal remaining
        if isinstance(node, (list, tuple)):
            if not node or len(node) > limit:
                raise ValueError("empty or oversized tensor dimension")
            first = visit(node[0])
            for item in node[1:]:
                if visit(item) != first:
                    raise ValueError("ragged tensor")
            return (len(node),) + first
        if isinstance(node, (bool, np.bool_)) or not isinstance(node, Real) or not math.isfinite(node):
            raise ValueError("tensor entries must be finite real numbers, not booleans")
        remaining -= 1
        if remaining < 0:
            raise ValueError("tensor exceeds the element budget")
        return ()

    return visit(value)


def array(value, limit):
    shape_of(value, limit)
    result = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("tensor entries must be finite")
    return result


def probability(value, normalize=False):
    if np.any(value < 0):
        raise ValueError("probabilities cannot be negative")
    mass = value.sum(axis=0, keepdims=True)
    if np.any(mass <= 0) or not np.all(np.isfinite(mass)):
        raise ValueError("probability columns need positive finite support")
    if normalize:
        return value / mass
    if not np.allclose(mass, 1, rtol=1e-6, atol=1e-8):
        raise ValueError("probability columns must sum to one; request normalize=True explicitly")
    return value


def policy_cost(controls, horizon, limits):
    horizon = integer(horizon, "policy_len", 1, limits.max_policy_len)
    combinations = math.prod(controls)
    policies = 1
    for _ in range(horizon):
        if policies > limits.max_policies // combinations:
            raise ValueError("policy count exceeds max_policies")
        policies *= combinations
    if policies * horizon * len(controls) > limits.max_elements:
        raise ValueError("policy tensor exceeds max_elements")
    return policies


SETTINGS = {
    "policy_len", "inference_horizon", "inference_algo", "num_iter", "alpha", "gamma",
    "action_precision", "action_selection", "sampling_mode", "seed", "A_dependencies",
    "learn_A", "learn_B", "learn_C", "learn_D", "learn_E", "learning_mode",
}
METADATA = {"id", "name", "num_observation_modalities", "num_state_factors", "num_controls"}


def admit_model(model, limits, normalize=False):
    if not isinstance(model, dict) or type(normalize) is not bool:
        raise ValueError("generative_model must be an object; normalize must be boolean")
    unknown = set(model) - SETTINGS - METADATA - {"A", "B", "C", "D", "pA", "pB"}
    if unknown:
        raise ValueError("unsupported model settings: " + ", ".join(sorted(unknown)))
    shapes, total = {}, 0
    for key in ("A", "B", "C", "D", "pA", "pB"):
        values = model.get(key)
        if values is None and key not in ("A", "B"):
            continue
        if not isinstance(values, (list, tuple)) or not values or len(values) > limits.max_entities:
            raise ValueError(f"{key} must be a nonempty list of tensors")
        shapes[key] = []
        for value in values:
            shape = shape_of(value, limits.max_elements - total)
            total += math.prod(shape)
            if total > limits.max_elements:
                raise ValueError("model tensors exceed max_elements")
            shapes[key].append(shape)
    if any(len(s) != 3 or s[0] != s[1] for s in shapes["B"]):
        raise ValueError("B requires square unbatched [destination, source, control] tensors")
    states = [s[0] for s in shapes["B"]]
    controls = [s[2] for s in shapes["B"]]
    for key, count in (("num_state_factors", len(states)), ("num_observation_modalities", len(shapes["A"]))):
        if key in model and integer(model[key], key, 1) != count:
            raise ValueError(f"{key} metadata disagrees with the model")
    if "num_controls" in model:
        supplied = model["num_controls"]
        if (not isinstance(supplied, (list, tuple)) or len(supplied) != len(controls)
                or [integer(v, "num_controls", 1) for v in supplied] != controls):
            raise ValueError("num_controls metadata disagrees with B")
    if any(len(s) != 2 for s in shapes["A"]):
        raise ValueError("A requires unbatched 2D modalities (Check modality dimension)")
    deps = model.get("A_dependencies")
    if deps is None:
        if len(states) == 1:
            deps = [[0] for _ in shapes["A"]]
        elif len(shapes["A"]) == len(states):
            deps = [[i] for i in range(len(states))]
        else:
            raise ValueError("explicit A_dependencies required for ambiguous modalities")
    if not isinstance(deps, (list, tuple)) or len(deps) != len(shapes["A"]):
        raise ValueError("A_dependencies must cover each modality")
    clean_deps = []
    for shape, dependency in zip(shapes["A"], deps):
        if not isinstance(dependency, (list, tuple)) or len(dependency) != 1:
            raise ValueError("only one-factor observation dependencies are supported")
        factor = integer(dependency[0], "A dependency", 0, len(states) - 1)
        if shape[1] != states[factor]:
            raise ValueError("A/B state dimensions disagree")
        clean_deps.append([factor])
    observations = [s[0] for s in shapes["A"]]
    for key, expected in (("C", [(n,) for n in observations]), ("D", [(n,) for n in states])):
        if key in shapes and shapes[key] != expected:
            raise ValueError(f"{key} dimensions disagree with the model")
    for key, original in (("pA", "A"), ("pB", "B")):
        if key in shapes and shapes[key] != shapes[original]:
            raise ValueError(f"{key} dimensions must match {original}")
    settings = {
        "policy_len": integer(model.get("policy_len", 1), "policy_len", 1, limits.max_policy_len),
        "inference_algo": model.get("inference_algo", "fpi"),
        "inference_horizon": model.get("inference_horizon"),
        "num_iter": integer(model.get("num_iter", 16), "num_iter", 1, limits.max_iterations),
        "action_selection": model.get("action_selection", "deterministic"),
        "sampling_mode": model.get("sampling_mode", "full"),
        "learning_mode": model.get("learning_mode", "online"),
        "A_dependencies": clean_deps,
        "seed": integer(model.get("seed", 0), "seed", 0, 2**32 - 1),
    }
    if not isinstance(settings["inference_algo"], str):
        raise ValueError("inference_algo must be a string")
    settings["inference_algo"] = settings["inference_algo"].lower()
    # This interface is a filtering API, not a sequence/smoothing API.
    if settings["inference_algo"] not in ("fpi", "exact"):
        raise ValueError("this sequential interface supports fpi or single-factor exact filtering, not smoothing")
    if settings["inference_algo"] == "exact" and len(states) != 1:
        raise ValueError("exact filtering requires one state factor")
    if settings["inference_horizon"] is not None:
        settings["inference_horizon"] = integer(settings["inference_horizon"], "inference_horizon", 1, limits.max_policy_len)
    if settings["action_selection"] not in ("deterministic", "stochastic") or settings["sampling_mode"] not in ("full", "marginal"):
        raise ValueError("unsupported action selection/sampling mode")
    if settings["learning_mode"] not in ("online", "offline"):
        raise ValueError("unsupported learning_mode")
    if "action_precision" in model and "alpha" in model:
        raise ValueError("specify alpha or action_precision, not both")
    for key in ("alpha", "gamma"):
        value = model.get(key, model.get("action_precision", 1.0) if key == "alpha" else 1.0)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
        settings[key] = float(value)
    for key in ("learn_A", "learn_B", "learn_C", "learn_D", "learn_E"):
        settings[key] = model.get(key, False)
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for learn, concentration in (("learn_A", "pA"), ("learn_B", "pB")):
        if settings[learn] and concentration not in shapes:
            raise ValueError(f"{concentration} required for {learn}")
    policies = policy_cost(controls, settings["policy_len"], limits)
    derived = policies * settings["policy_len"] * len(controls)
    derived += 0 if "C" in shapes else sum(observations)
    derived += 0 if "D" in shapes else sum(states)
    if total + derived > limits.max_elements:
        raise ValueError("model plus derived tensors exceeds max_elements")
    # All shape/policy/cumulative tensor admission precedes tensor copies/Agent construction.
    clean = {key: [array(a, limits.max_elements) for a in model[key]] for key in shapes}
    for key in ("A", "B", "D"):
        if key in clean:
            clean[key] = [probability(a, normalize) for a in clean[key]]
    if "C" not in clean:
        clean["C"] = [np.ones(n) / n for n in observations]  # PyMDP's existing flat default.
    if "D" not in clean:
        clean["D"] = [np.ones(n) / n for n in states]
    for key, original in (("pA", "A"), ("pB", "B")):
        if key in clean:
            for concentration, probabilities in zip(clean[key], clean[original]):
                with np.errstate(over="ignore"):
                    mass = concentration.sum(axis=0)
                if np.any(concentration < 0) or np.any(mass <= 0) or not np.all(np.isfinite(mass)):
                    raise ValueError(f"{key} requires nonnegative, finite, nonempty concentration support")
                if np.any((concentration == 0) & (probabilities > 0)):
                    raise ValueError(f"{key} excludes positive model support")
    return clean, settings


def numerical_inputs(A, prior, observation, limit):
    A, prior = array(A, limit), array(prior, limit)
    if A.ndim != 2 or prior.ndim != 1 or A.shape[1] != prior.size:
        raise ValueError("expected A[outcome,state] and a matching 1D prior")
    probability(A)
    probability(prior)
    if not isinstance(observation, (list, tuple)) or len(observation) != 1:
        raise ValueError("one discrete outcome is required")
    index = integer(observation[0], "observation", 0, A.shape[0] - 1)
    return A[index], prior


def vfe_components(likelihood, prior, q):
    if q.ndim != 1 or q.shape != prior.shape or likelihood.shape != prior.shape:
        raise ValueError("likelihood, prior and q must be matching 1D vectors")
    probability(q)
    support = q > 0
    if np.any(prior[support] <= 0) or np.any(likelihood[support] <= 0):
        raise ValueError("posterior has positive mass outside prior/likelihood support; VFE is infinite")
    entropy = float(np.dot(q[support], np.log(q[support])))
    prior_term = float(np.dot(q[support], np.log(prior[support])))
    accuracy = float(np.dot(q[support], np.log(likelihood[support])))
    energy = -prior_term - accuracy
    divergence = entropy - prior_term
    return {"free_energy": energy + entropy, "energy": energy, "entropy": entropy,
            "divergence": divergence, "accuracy": accuracy,
            "free_energy_alt": divergence - accuracy}
