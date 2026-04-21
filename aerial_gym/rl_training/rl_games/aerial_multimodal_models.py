import numpy as np
import torch
import torch.nn as nn

from rl_games.algos_torch import models
from rl_games.algos_torch.running_mean_std import RunningMeanStd


class SelectiveRunningMeanStd(nn.Module):
    def __init__(self, vector_shape=None, candidate_keys=None):
        super().__init__()
        self.candidate_keys = candidate_keys or []
        self.vector_running_mean_std = (
            torch.jit.script(RunningMeanStd(vector_shape)) if vector_shape is not None else None
        )

    def _resolve_vector_key_from_obs(self, obs):
        for candidate in self.candidate_keys:
            if candidate in obs:
                return candidate
        for key, value in obs.items():
            if key not in ("img_observation", "image_observation") and len(value.shape) == 2:
                return key
        return None

    def forward(self, observation, denorm: bool = False):
        if self.vector_running_mean_std is None:
            return observation

        if not isinstance(observation, dict):
            return self.vector_running_mean_std(observation, denorm=denorm)

        normed_observation = dict(observation)
        vector_key = self._resolve_vector_key_from_obs(normed_observation)
        if vector_key is not None:
            normed_observation[vector_key] = self.vector_running_mean_std(
                normed_observation[vector_key], denorm=denorm
            )
        return normed_observation


class ModelA2CContinuousLogStdMultimodal(models.BaseModel):
    def __init__(self, network):
        super().__init__("aerial_multimodal")
        self.network_builder = network

    class Network(models.BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            obs_shape = kwargs.get("obs_shape")
            normalize_input = kwargs.pop("normalize_input", False)

            super().__init__(normalize_input=False, **kwargs)
            self.normalize_input = normalize_input
            self.a2c_network = a2c_network
            self.vector_obs_key = getattr(self.a2c_network, "vector_obs_key", "state")
            self.state_key = getattr(self.a2c_network, "state_key", "state")
            self.image_obs_key = getattr(self.a2c_network, "image_obs_key", "img_observation")

            vector_shape = self._resolve_vector_shape(obs_shape)
            candidate_keys = [
                self.vector_obs_key,
                self.state_key,
                "observations",
                "observation",
                "state",
            ]
            self.running_mean_std = SelectiveRunningMeanStd(
                vector_shape=vector_shape if self.normalize_input else None,
                candidate_keys=candidate_keys,
            )

        def _resolve_vector_shape(self, obs_shape):
            if not isinstance(obs_shape, dict):
                return obs_shape

            candidates = [
                self.vector_obs_key,
                self.state_key,
                "observations",
                "observation",
                "state",
            ]
            for candidate in candidates:
                if candidate in obs_shape:
                    return obs_shape[candidate]

            for key, shape in obs_shape.items():
                if key not in ("img_observation", "image_observation") and len(shape) == 1:
                    return shape
            return None

        def _resolve_vector_key_from_obs(self, obs):
            candidates = [
                self.vector_obs_key,
                self.state_key,
                "observations",
                "observation",
                "state",
            ]
            for candidate in candidates:
                if candidate in obs:
                    return candidate
            for key, value in obs.items():
                if key not in ("img_observation", "image_observation") and len(value.shape) == 2:
                    return key
            return None

        def norm_obs(self, observation):
            if not self.normalize_input:
                return observation

            with torch.no_grad():
                return self.running_mean_std(observation)

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                return {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }

            selected_action = distr.sample()
            neglogp = self.neglogp(selected_action, mu, sigma, logstd)
            return {
                "neglogpacs": torch.squeeze(neglogp),
                "values": self.denorm_value(value),
                "actions": selected_action,
                "rnn_states": states,
                "mus": mu,
                "sigmas": sigma,
            }

        def neglogp(self, x, mean, std, logstd):
            return (
                0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
                + 0.5 * np.log(2.0 * np.pi) * x.size(-1)
                + logstd.sum(dim=-1)
            )
