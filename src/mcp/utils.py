"""Bounded PyMDP filtering, explicit action commits, and run-owned evidence."""
import copy
import json
import logging
import math
import re
import time
import uuid
from collections import OrderedDict
from functools import wraps

import numpy as np

from .environments import CustomEnvironment, GridWorldEnvironment
from .environments.common import (
    environment_record,
    grid_shape,
    grid_transition,
    integer,
    position,
)
from .models import (
    Limits,
    admit_model,
    array,
    numerical_inputs,
    policy_cost,
    probability,
    vfe_components,
)
from .outputs import OutputPolicy

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if hasattr(obj, "__array__"):
            return np.asarray(obj).tolist()
        return super().default(obj)


class DomainError(ValueError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def public(function):
    """Keep direct/domain errors explicit; transports decide their envelope/status."""
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except DomainError as error:
            return {"error": str(error), "error_code": error.code}
        except KeyError as error:
            return {"error": str(error), "error_code": "not_found"}
        except (ValueError, TypeError) as error:
            return {"error": str(error), "error_code": "invalid_argument"}
        except FileNotFoundError as error:
            return {"error": str(error), "error_code": "not_found"}
        except Exception as error:
            logger.exception("%s failed", function.__name__)
            return {"error": str(error), "error_code": "operation_failed"}
    return call


class PyMDPInterface:
    def __init__(self, limits=None, output_root=None):
        self.limits = Limits(**limits) if isinstance(limits, dict) else (Limits() if limits is None else limits)
        if not isinstance(self.limits, Limits):
            raise ValueError("limits must be trusted Limits or a settings dictionary")
        self.output_policy = OutputPolicy(output_root)
        self.agents, self.environments, self.models = {}, {}, {}
        self.priors, self.last_qs, self.last_q_pi = {}, {}, {}
        self.pending_actions, self._random_keys = {}, {}
        self.simulation_history = OrderedDict()
        self.sessions = {}
        self.computation_logs, self._logs = {}, OrderedDict()
        self.log_counter = 0

    @staticmethod
    def _name(name):
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None:
            raise ValueError("resource name must be a bounded alphanumeric identifier")
        return name

    def _admit_new(self, registry, name):
        self._name(name)
        if name in registry:
            raise DomainError(f"resource {name} already exists", "conflict")
        if len(registry) >= self.limits.max_entities:
            raise ValueError("entity budget exhausted")

    @staticmethod
    def _to_batch(value):
        return np.array(value, dtype=float, copy=True)[None]

    @staticmethod
    def _unwrap(value):
        value = np.asarray(value)
        while value.ndim > 1 and value.shape[0] == 1:
            value = value[0]
        return value.tolist()

    def _onehot_observations(self, agent, observation):
        if not isinstance(observation, (list, tuple)) or len(observation) != agent.num_modalities:
            raise ValueError("observation must contain exactly one index per modality")
        result = []
        for value, count in zip(observation, agent.num_obs):
            index = integer(value, "observation", 0, count - 1)
            vector = np.zeros((1, count))
            vector[0, index] = 1
            result.append(vector)
        return result

    def _log(self, owner, operation, **details):
        self.log_counter += 1
        key = f"log_{self.log_counter}"
        entry = {"log_key": key, "owner": owner, "operation": operation,
                 "timestamp": time.time(), **copy.deepcopy(details)}
        self._logs[key] = entry
        self.computation_logs.setdefault(owner, []).append(entry)
        while len(self._logs) > self.limits.max_logs:
            _, expired = self._logs.popitem(last=False)
            entries = self.computation_logs[expired["owner"]]
            entries.pop(0)
            if not entries:
                del self.computation_logs[expired["owner"]]
        return key, copy.deepcopy(entry)

    @staticmethod
    def _require(result):
        if "error" in result:
            raise DomainError(result["error"], result.get("error_code", "operation_failed"))
        return result

    @public
    def create_agent(self, name: str, generative_model: dict, normalize: bool = False) -> dict:
        self._admit_new(self.agents, name)
        clean, settings = admit_model(generative_model, self.limits, normalize)
        from pymdp.agent import Agent
        kwargs = {key: value for key, value in settings.items() if key != "seed"}
        kwargs["categorical_obs"] = True
        kwargs["num_controls"] = [b.shape[-1] for b in clean["B"]]
        kwargs["B_dependencies"] = [[i] for i in range(len(clean["B"]))]
        agent = Agent(**{key: [self._to_batch(a) for a in values] for key, values in clean.items()}, **kwargs)
        model = {key: [a.tolist() for a in values] for key, values in clean.items()}
        model.update(copy.deepcopy(settings))
        self.agents[name], self.models[name] = agent, model
        self.priors[name] = [np.array(d, copy=True) for d in agent.D]
        self.last_qs[name] = self.last_q_pi[name] = None
        self.pending_actions[name] = None
        return {"id": name, "name": name, "num_observation_modalities": len(clean["A"]),
                "num_state_factors": len(clean["B"]), "num_controls": kwargs["num_controls"]}

    @public
    def define_generative_model(self, A_dims: list[list[int]], B_dims: list[list[int]], seed: int = None) -> dict:
        if not isinstance(A_dims, list) or not isinstance(B_dims, list) or not A_dims or not B_dims:
            raise ValueError("nonempty A_dims and B_dims required")
        if len(A_dims) + len(B_dims) > self.limits.max_entities:
            raise ValueError("too many model factors/modalities")
        dimensions, total = [], 0
        for dims, rank in [(d, 2) for d in A_dims] + [(d, 3) for d in B_dims]:
            if not isinstance(dims, (list, tuple)) or len(dims) != rank:
                raise ValueError("invalid model dimension entry")
            shape = [integer(d, "model dimension", 1, self.limits.max_elements) for d in dims]
            total += math.prod(shape)
            dimensions.append(shape)
        if any(d[0] != d[1] for d in B_dims):
            raise ValueError("B must be square")
        if len(B_dims) == 1:
            compatible = all(d[1] == B_dims[0][0] for d in A_dims)
        else:
            compatible = len(A_dims) == len(B_dims) and all(a[1] == b[0] for a, b in zip(A_dims, B_dims))
        if not compatible:
            raise ValueError("factory modality/state dimensions disagree")
        total += sum(d[0] for d in A_dims) + sum(d[0] for d in B_dims)
        if seed is not None:
            seed = integer(seed, "seed", 0, 2**32 - 1)
        total += policy_cost([d[2] for d in B_dims], 1, self.limits) * len(B_dims)
        if total > self.limits.max_elements:
            raise ValueError("factory model plus derived tensors exceeds max_elements")
        rng = np.random.default_rng(seed)
        matrices = [probability(rng.random(shape), normalize=True) for shape in dimensions]
        candidate = {"A": matrices[:len(A_dims)], "B": matrices[len(A_dims):],
                     "C": [np.zeros(d[0]) for d in A_dims],
                     "D": [np.ones(d[0]) / d[0] for d in B_dims]}
        clean, settings = admit_model(candidate, self.limits)
        return {**{key: [a.tolist() for a in values] for key, values in clean.items()}, **settings}

    @public
    def create_gridworld_agent(self, name: str, grid_size: list[int],
                               reward_positions: list[list[int]], action_precision: float = 1.0,
                               inference_horizon: int = 5, policy_len: int = 1) -> dict:
        self._admit_new(self.agents, name)
        shape = grid_shape(grid_size)
        if not isinstance(reward_positions, list):
            raise ValueError("reward_positions must be a list")
        rewards = [position(p, shape) for p in reward_positions]
        n = math.prod(shape)
        policies = policy_cost([5], policy_len, self.limits)
        if n * n * 6 + n * 4 + 2 + policies * policy_len > self.limits.max_elements:
            raise ValueError("grid model plus derived tensors exceeds max_elements before allocation")
        B = np.zeros((n, n, 5))
        for state in range(n):
            for action in range(5):
                B[grid_transition(state, action, shape), state, action] = 1
        A_reward = np.zeros((2, n))
        A_reward[0] = 1
        C_location = np.zeros(n)
        for row, col in rewards:
            index = row * shape[1] + col
            A_reward[:, index] = [0, 1]
            C_location[index] = 1
        model = {"A": [np.eye(n), A_reward], "B": [B], "C": [C_location, [0, 4]],
                 "action_precision": action_precision, "inference_horizon": inference_horizon,
                 "policy_len": policy_len}
        return self.create_agent(name, model)

    def _normalize_A_matrix(self, value):
        return probability(array(value, self.limits.max_elements), normalize=True)

    def validate_generative_model(self, generative_model: dict, check_normalization: bool = True) -> dict:
        try:
            if type(check_normalization) is not bool:
                raise ValueError("check_normalization must be boolean")
            admit_model(generative_model, self.limits, normalize=not check_normalization)
            return {"valid": True, "issues": []}
        except (ValueError, TypeError) as error:
            return {"valid": False, "issues": [str(error)]}

    @staticmethod
    def _log_prob(value):
        value = np.asarray(value, dtype=float)
        out = np.full(value.shape, -np.inf)
        np.log(value, out=out, where=value > 0)
        return out

    def _factor_likelihoods(self, agent_id, observation):
        model = self.models[agent_id]
        log_likelihoods = [np.zeros(len(d)) for d in model["D"]]
        for A, dependency, outcome in zip(model["A"], model["A_dependencies"], observation):
            log_likelihoods[dependency[0]] += self._log_prob(np.asarray(A)[outcome])
        return log_likelihoods

    def _measure_vfe(self, agent_id, observation, posterior, prior):
        log_likelihoods = self._factor_likelihoods(agent_id, observation)
        energy = entropy = divergence = accuracy = 0.0
        for q, p, ll in zip(posterior, prior, log_likelihoods):
            q, p = np.asarray(q), np.asarray(p)
            mask = q > 0
            lp = self._log_prob(p)
            if np.any(~np.isfinite(lp[mask])) or np.any(~np.isfinite(ll[mask])):
                return {"status": "infinite", "free_energy": None,
                        "reason": "PyMDP posterior has mass outside exact prior/likelihood support"}
            signed_entropy = float(np.dot(q[mask], np.log(q[mask])))
            prior_expectation = float(np.dot(q[mask], lp[mask]))
            likelihood_expectation = float(np.dot(q[mask], ll[mask]))
            entropy += signed_entropy
            energy -= prior_expectation + likelihood_expectation
            divergence += signed_entropy - prior_expectation
            accuracy += likelihood_expectation
        return {"status": "measured", "free_energy": energy + entropy,
                "energy": energy, "entropy": entropy, "divergence": divergence,
                "accuracy": accuracy, "free_energy_alt": divergence - accuracy}

    @public
    def infer_states(self, agent_id: str, observation: list[int], method: str = None,
                     save_intermediate_results: bool = False) -> dict:
        agent = self.agents[agent_id]
        if method is not None and (not isinstance(method, str) or method.lower() != agent.inference_algo):
            raise ValueError("method is construction-time; call-time override is unsupported")
        if type(save_intermediate_results) is not bool:
            raise ValueError("save_intermediate_results must be boolean")
        vectors = self._onehot_observations(agent, observation)
        prior = [self._unwrap(p) for p in self.priors[agent_id]]
        for p, ll in zip(prior, self._factor_likelihoods(agent_id, observation)):
            if not np.any(np.isfinite(self._log_prob(p) + ll)):
                raise ValueError("observation has zero evidence under the predictive prior")
        start = time.perf_counter()
        qs = agent.infer_states(vectors, self.priors[agent_id])
        posterior = [self._unwrap(q) for q in qs]
        for q, count in zip(posterior, agent.num_states):
            q = np.asarray(q)
            if q.shape != (count,) or not np.all(np.isfinite(q)):
                raise ValueError("PyMDP returned malformed posterior beliefs")
            probability(q)
        vfe = self._measure_vfe(agent_id, observation, posterior, prior)
        key, detail = self._log(agent_id, "state_inference", observation=list(observation),
                                method=agent.inference_algo, final_beliefs=posterior,
                                computation_time=time.perf_counter() - start,
                                variational_free_energy=vfe)
        self.last_qs[agent_id] = qs
        self.last_q_pi[agent_id] = None
        self.pending_actions[agent_id] = None
        result = {"posterior_states": posterior, "method": agent.inference_algo,
                  "log_key": key, "variational_free_energy": vfe}
        if save_intermediate_results:
            result["computation_details"] = detail
        return result

    def _horizon(self, agent_id, planning_horizon):
        if planning_horizon is not None:
            integer(planning_horizon, "planning_horizon", 1)
            raise ValueError("call-time planning_horizon is unsupported; set policy_len at construction")
        return self.agents[agent_id]

    @public
    def infer_policies(self, agent_id: str, save_computation: bool = True,
                       planning_horizon: int = None) -> dict:
        agent = self._horizon(agent_id, planning_horizon)
        if type(save_computation) is not bool:
            raise ValueError("save_computation must be boolean")
        qs = self.last_qs[agent_id]
        if qs is None:
            raise ValueError("infer_states is required before planning; no observation is fabricated")
        q_pi, negative_efe = agent.infer_policies(qs)
        posterior = self._unwrap(q_pi)
        negative = self._unwrap(negative_efe)
        if len(posterior) != len(negative) or not np.all(np.isfinite(negative)):
            raise ValueError("PyMDP returned malformed policy evidence")
        probability(np.asarray(posterior))
        result = {"policy_posterior": posterior,
                  "negative_expected_free_energy": negative,
                  "expected_free_energy": [-v for v in negative],
                  "efe_convention": "expected_free_energy = -PyMDP negative_expected_free_energy"}
        self.last_q_pi[agent_id] = q_pi
        self.pending_actions[agent_id] = None
        if save_computation:
            key, _ = self._log(agent_id, "policy_inference", **result)
            result["log_key"] = key
        return result

    @public
    def sample_action(self, agent_id: str, planning_horizon: int = None) -> dict:
        agent = self._horizon(agent_id, planning_horizon)
        pending = self.pending_actions[agent_id]
        if pending is not None:
            return copy.deepcopy(pending)
        if self.last_q_pi[agent_id] is None:
            self._require(self.infer_policies(agent_id))
        kwargs = {}
        if agent.action_selection == "stochastic":
            from jax import random
            key = self._random_keys.get(agent_id, random.PRNGKey(self.models[agent_id]["seed"]))
            next_key, selected_key = random.split(key)
            kwargs["rng_key"] = selected_key[None]
        action = [integer(a, "action", 0, agent.num_controls[i] - 1)
                  for i, a in enumerate(self._unwrap(agent.sample_action(self.last_q_pi[agent_id], **kwargs)))]
        if len(action) != len(agent.num_controls):
            raise ValueError("PyMDP returned malformed action cardinality")
        pending = {"action": action, "action_id": uuid.uuid4().hex}
        if kwargs:
            self._random_keys[agent_id] = next_key
        self.pending_actions[agent_id] = pending
        return copy.deepcopy(pending)

    def _selected(self, agent_id, action_id, action=None):
        agent = self.agents[agent_id]
        pending = self.pending_actions[agent_id]
        if pending is None or action_id != pending["action_id"] or self.last_qs[agent_id] is None:
            raise ValueError("action selection is absent, stale or already committed")
        if action is not None:
            values = action if isinstance(action, (list, tuple)) else [action]
            if len(values) != len(agent.num_controls):
                raise ValueError("action must cover every control factor")
            values = [integer(v, "action", 0, agent.num_controls[i] - 1) for i, v in enumerate(values)]
            if values != pending["action"]:
                raise ValueError("environment action differs from selected action")
        return pending

    def _commit(self, agent_id, action_id):
        pending = self._selected(agent_id, action_id)
        prior = self.agents[agent_id].update_empirical_prior(
            np.asarray([pending["action"]]), self.last_qs[agent_id])
        for p in prior:
            probability(np.asarray(self._unwrap(p)))
        self.priors[agent_id] = prior
        self.last_qs[agent_id] = self.last_q_pi[agent_id] = None
        self.pending_actions[agent_id] = None
        key, _ = self._log(agent_id, "action_commit", action=pending["action"], action_id=action_id)
        return {"action": pending["action"], "action_id": action_id, "log_key": key}

    @public
    def commit_action(self, agent_id: str, action_id: str) -> dict:
        """Explicit commit for a caller-owned successful external transition."""
        return self._commit(agent_id, action_id)

    @public
    def reset_agent(self, agent_id: str) -> dict:
        agent = self.agents[agent_id]
        self.priors[agent_id] = [np.array(d, copy=True) for d in agent.D]
        self.last_qs[agent_id] = self.last_q_pi[agent_id] = None
        self.pending_actions[agent_id] = None
        self._random_keys.pop(agent_id, None)
        return {"id": agent_id}

    @public
    def create_grid_world_env(self, name: str, grid_size: list[int],
                              reward_positions: list[list[int]], start_position: list[int] = None,
                              reset_mode: str = "origin", terminate_on_reward: bool = True) -> dict:
        self._admit_new(self.environments, name)
        env = GridWorldEnvironment(grid_size, reward_positions, start_position, reset_mode, terminate_on_reward)
        self.environments[name] = env
        return self.get_environment(name)

    def _environment(self, env_id):
        env = self.environments[env_id]
        if not isinstance(env, (GridWorldEnvironment, CustomEnvironment)):
            raise ValueError("environment must implement the validated environment/action record contract")
        return env

    @public
    def step_environment(self, env_id: str, action: list[int] | int,
                          agent_id: str = None, action_id: str = None) -> dict:
        env = self._environment(env_id)
        env.validate_action(action)
        if agent_id is None and action_id is not None:
            raise ValueError("action_id requires agent_id")
        if agent_id is not None:
            self._selected(agent_id, action_id, action)
        result = copy.deepcopy(environment_record(env.step(action)))
        if agent_id is not None:
            result["commit"] = self._commit(agent_id, action_id)
        key, _ = self._log(env_id, "environment_step", action=action, **result)
        result["log_key"] = key
        return result

    @public
    def reset_environment(self, env_id: str) -> dict:
        return copy.deepcopy(environment_record(self._environment(env_id).reset()))

    def _compatible(self, agent_id, env_id):
        agent, env = self.agents[agent_id], self._environment(env_id)
        if isinstance(env, GridWorldEnvironment):
            cardinalities = [env.num_states, 2]
        else:
            cardinalities = [len(env.observations)] * len(next(iter(env.state_observations.values())))
        if list(agent.num_obs) != cardinalities:
            raise ValueError("agent/environment observation modalities or cardinalities differ")
        if len(agent.num_controls) != 1 or agent.num_controls[0] > len(env.actions):
            raise ValueError("agent controls are incompatible with the environment action order")
        environment_record(env.get_observation())
        return agent, env

    @public
    def run_simulation(self, agent_id: str, env_id: str, num_steps: int = 10,
                       save_history: bool = True, planning_horizon: int = None,
                       stop_on_done: bool = False) -> dict:
        self._horizon(agent_id, planning_horizon)
        steps = integer(num_steps, "num_steps", 0, self.limits.max_steps)
        if type(save_history) is not bool or type(stop_on_done) is not bool:
            raise ValueError("save_history and stop_on_done must be boolean")
        agent, env = self._compatible(agent_id, env_id)
        policies = policy_cost(list(agent.num_controls), agent.policy_len, self.limits)
        cost = steps * (sum(agent.num_states) * 2 + policies * 3 + agent.num_modalities * 2 + 16)
        if cost > self.limits.max_elements:
            raise ValueError("derived simulation history exceeds max_elements")
        simulation_id = "run_" + uuid.uuid4().hex
        current = self._require(self.reset_environment(env_id))
        self._require(self.reset_agent(agent_id))
        start = time.time()
        timesteps, log_keys = [], []
        for index in range(steps):
            if stop_on_done and current["done"]:
                break
            state = self._require(self.infer_states(agent_id, current["observation"]))
            policy = self._require(self.infer_policies(agent_id))
            selected = self._require(self.sample_action(agent_id))
            transition = self._require(self.step_environment(env_id, selected["action"], agent_id, selected["action_id"]))
            keys = [state["log_key"], policy["log_key"], transition["commit"]["log_key"], transition["log_key"]]
            for key in keys:
                if key in self._logs:
                    self._logs[key].update(simulation_id=simulation_id, step=index)
            state["variational_free_energy"] = {
                **state["variational_free_energy"], "simulation_id": simulation_id, "timestep": index}
            log_keys.extend(keys)
            timesteps.append({
                "simulation_id": simulation_id, "step": index,
                "observation": copy.deepcopy(current["observation"]),
                "beliefs": state["posterior_states"],
                "state": copy.deepcopy(current["state"]), "position": copy.deepcopy(current["position"]),
                "action": selected["action"], "action_id": selected["action_id"],
                "policy_posterior": policy["policy_posterior"],
                "expected_free_energy": policy["expected_free_energy"],
                "negative_expected_free_energy": policy["negative_expected_free_energy"],
                "variational_free_energy": state["variational_free_energy"],
                "next_observation": transition["observation"], "next_state": transition["state"],
                "next_position": transition["position"], "reward": transition["reward"],
                "done": transition["done"], "info": transition["info"], "log_keys": keys,
            })
            current = transition
        history = {"simulation_id": simulation_id, "agent_id": agent_id, "env_id": env_id,
                   "requested_steps": steps, "num_steps": len(timesteps), "timesteps": timesteps,
                   "free_energy_trace": [t["variational_free_energy"] for t in timesteps],
                   "efe_convention": "expected_free_energy = -PyMDP negative_expected_free_energy",
                   "stop_on_done": stop_on_done, "start_time": start, "end_time": time.time(),
                   "environment": self.get_environment(env_id), "log_keys": log_keys}
        if save_history:
            self.simulation_history[simulation_id] = history
            while len(self.simulation_history) > self.limits.max_runs:
                self.simulation_history.popitem(last=False)
        return {"id": simulation_id, "simulation_id": simulation_id, "agent_id": agent_id,
                "env_id": env_id, "timesteps": copy.deepcopy(timesteps),
                "free_energy_trace": copy.deepcopy(history["free_energy_trace"]),
                "requested_steps": steps, "num_steps": len(timesteps),
                "total_reward": sum(t["reward"] for t in timesteps),
                "history": copy.deepcopy(history) if save_history else None}

    @public
    def get_agent(self, agent_id: str) -> dict:
        agent = self.agents[agent_id]
        return {"id": agent_id, "name": agent_id, **copy.deepcopy(self.models[agent_id]),
                "num_observation_modalities": agent.num_modalities,
                "num_state_factors": agent.num_factors, "num_controls": list(agent.num_controls)}

    @public
    def get_environment(self, env_id: str) -> dict:
        return {"id": env_id, "name": env_id, **self._environment(env_id).to_dict()}

    def get_all_functions(self) -> dict:
        from .api import TOOL_NAMES
        return {"functions": list(TOOL_NAMES)}

    def get_all_agents(self) -> dict:
        return {"agents": [self.get_agent(name) for name in self.agents]}

    def get_all_environments(self) -> dict:
        return {"environments": [self.get_environment(name) for name in self.environments]}

    def get_all_simulations(self) -> dict:
        return {"simulations": [
            {"id": run, "simulation_id": run, "agent_id": h["agent_id"], "env_id": h["env_id"],
             "num_steps": h["num_steps"]} for run, h in self.simulation_history.items()]}

    @public
    def get_simulation(self, simulation_id: str) -> dict:
        return copy.deepcopy(self.simulation_history[simulation_id])

    @public
    def get_computation_logs(self, simulation_id: str) -> dict:
        history = self.simulation_history[simulation_id]
        if any(key not in self._logs for key in history["log_keys"]):
            return {"error": "run-bound computation detail expired", "error_code": "gone"}
        return {"simulation_id": simulation_id,
                "logs": [copy.deepcopy(self._logs[key]) for key in history["log_keys"]]}

    @public
    def create_session(self, session_id: str, agent_id: str, env_id: str) -> dict:
        if session_id in self.sessions:
            existing = self.sessions[session_id]
            if (existing["agent_id"], existing["env_id"]) != (agent_id, env_id):
                raise DomainError("session identity is already bound to different resources", "conflict")
            return copy.deepcopy(existing)
        self._admit_new(self.sessions, session_id)
        self._compatible(agent_id, env_id)
        self.sessions[session_id] = {"id": session_id, "agent_id": agent_id,
                                     "env_id": env_id, "simulation_ids": []}
        return copy.deepcopy(self.sessions[session_id])

    @public
    def get_session(self, session_id: str) -> dict:
        return copy.deepcopy(self.sessions[session_id])

    def get_all_sessions(self) -> dict:
        return {"sessions": copy.deepcopy(list(self.sessions.values()))}

    @public
    def run_session(self, session_id: str, num_steps: int = 10, stop_on_done: bool = False) -> dict:
        session = self.sessions[session_id]
        steps = integer(num_steps, "num_steps", 0, self.limits.max_steps)
        existing = self._require(self.get_session_history(session_id))["num_steps"]
        if existing + steps > self.limits.max_steps or len(session["simulation_ids"]) >= self.limits.max_runs:
            raise ValueError("session retention/step budget exhausted")
        result = self._require(self.run_simulation(session["agent_id"], session["env_id"], steps,
                                                  stop_on_done=stop_on_done))
        session["simulation_ids"].append(result["simulation_id"])
        return {**result, "session_id": session_id}

    @public
    def get_session_history(self, session_id: str) -> dict:
        session = self.sessions[session_id]
        runs = [self.simulation_history[run] for run in session["simulation_ids"]]
        timesteps = [copy.deepcopy(t) for run in runs for t in run["timesteps"]]
        return {"session_id": session_id, "simulation_ids": session["simulation_ids"].copy(),
                "agent_id": session["agent_id"], "env_id": session["env_id"],
                "environment": self._require(self.get_environment(session["env_id"])),
                "num_steps": len(timesteps), "timesteps": timesteps,
                "free_energy_trace": [t["variational_free_energy"] for t in timesteps]}

    def _history(self, identity):
        if identity in self.simulation_history:
            return copy.deepcopy(self.simulation_history[identity])
        if identity in self.sessions:
            return self._require(self.get_session_history(identity))
        raise KeyError(f"history {identity} not retained")

    @public
    def visualize_agent_model(self, agent_id: str, output_file: str = "agent_model.png") -> dict:
        from .visualization.generative_model import visualize_generative_model
        model = self._require(self.get_agent(agent_id))
        target = self.output_policy.path(output_file)
        visualize_generative_model(agent_id, model, target, policy=self.output_policy)
        return {"figure_path": str(target), "agent_id": agent_id}

    @public
    def visualize_simulation(self, simulation_id: str, output_file: str = "simulation.png") -> dict:
        from .visualization.free_energy import (
            measured_records,
            visualize_variational_free_energy,
        )
        from .visualization.simulation import (
            create_belief_heatmap,
            visualize_expected_free_energy,
            visualize_policy_evolution,
            visualize_simulation,
        )
        history = self._history(simulation_id)
        logs = self._require(self.get_computation_logs(simulation_id))
        target = self.output_policy.path(output_file)
        base = str(target.with_suffix(""))
        companions = [base + suffix for suffix in (
            "_belief_heatmap.png", "_policy_posterior.png", "_expected_free_energy.png",
            "_computation_logs.json")]
        if measured_records(history, require=False):
            companions += [base + "_free_energy.png", base + "_free_energy_data.json"]
        self.output_policy.admit(target, *companions)
        visualize_simulation(history, target, policy=self.output_policy)
        create_belief_heatmap([t["beliefs"] for t in history["timesteps"]], companions[0], policy=self.output_policy)
        visualize_policy_evolution([t["policy_posterior"] for t in history["timesteps"]], companions[1], policy=self.output_policy)
        visualize_expected_free_energy([t["expected_free_energy"] for t in history["timesteps"]], companions[2], policy=self.output_policy)
        self.output_policy.write_json(companions[3], logs)
        if len(companions) > 4:
            visualize_variational_free_energy(history, companions[4], policy=self.output_policy)
        return {"figure_path": str(target), "additional_visualizations": companions}

    @public
    def visualize_belief_dynamics(self, session_id: str, output_file: str = "belief_heatmap.png") -> dict:
        from .belief_dynamics import create_belief_dynamics_visualization
        target = self.output_policy.path(output_file)
        create_belief_dynamics_visualization(self._history(session_id), target, policy=self.output_policy)
        return {"figure_path": str(target)}

    @public
    def plot_belief_dynamics(self, session_id: str, output_file: str = "belief_dynamics.png") -> dict:
        from .belief_dynamics import calculate_belief_statistics
        from .visualization.simulation import plot_belief_trajectories
        history = self._history(session_id)
        target = self.output_policy.path(output_file)
        summary = str(target.with_suffix("")) + "_statistics.json"
        self.output_policy.admit(target, summary)
        beliefs = [t["beliefs"] for t in history["timesteps"]]
        statistics = calculate_belief_statistics(beliefs)
        plot_belief_trajectories(beliefs, target, policy=self.output_policy)
        self.output_policy.write_json(summary, statistics)
        return {"figure_path": str(target), "statistics": statistics, "additional_visualizations": [summary]}

    @public
    def analyze_free_energy(self, session_id: str, output_file: str = "free_energy.png") -> dict:
        from .visualization.free_energy import visualize_free_energy_components
        target = self.output_policy.path(output_file)
        visualize_free_energy_components(self._history(session_id), target, policy=self.output_policy)
        return {"figure_path": str(target), "additional_visualizations": [str(target.with_suffix("")) + "_data.json"]}

    @public
    def calculate_free_energy(self, A_matrix: list, prior: list,
                              observation: list[int], q_states: list) -> dict:
        likelihood, prior = numerical_inputs(A_matrix, prior, observation, self.limits.max_elements)
        return vfe_components(likelihood, prior, array(q_states, self.limits.max_elements))

    @public
    def infer_states_from_observation(self, A_matrix: list, prior: list, observation: list[int]) -> dict:
        likelihood, prior = numerical_inputs(A_matrix, prior, observation, self.limits.max_elements)
        log_joint = self._log_prob(likelihood) + self._log_prob(prior)
        maximum = float(log_joint.max())
        if not math.isfinite(maximum):
            raise ValueError("observation has zero evidence; no posterior exists")
        relative = np.exp(log_joint - maximum)
        mass = float(relative.sum())
        log_evidence = maximum + math.log(mass)
        return {"posterior": (relative / mass).tolist(), "likelihood": likelihood.tolist(),
                "prior": prior.tolist(), "evidence": math.exp(log_evidence),
                "log_evidence": log_evidence}


def get_pymdp_interface():
    return PyMDPInterface()
