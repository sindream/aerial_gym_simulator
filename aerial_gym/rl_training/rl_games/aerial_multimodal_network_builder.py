import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.network_builder import NetworkBuilder


class AerialMultimodalA2CBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return AerialMultimodalA2CBuilder.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            self.value_size = kwargs.pop("value_size", 1)
            self.num_seqs = kwargs.pop("num_seqs", 1)

            super().__init__()
            self.load(params)

            if self.separate:
                raise NotImplementedError(
                    "AerialMultimodalA2CBuilder currently supports only shared actor/critic trunks."
                )

            self.actions_num = actions_num
            self.vector_obs_key = self._resolve_vector_key(input_shape)
            self.image_obs_key = self._resolve_image_key(input_shape)
            self.critic_obs_key = self._resolve_optional_key(
                input_shape, self.critic_observation_key
            )
            self.gain_target_obs_key = self._resolve_optional_key(
                input_shape, self.gain_target_key
            )
            self.sysid_target_obs_key = self._resolve_optional_key(
                input_shape, self.sysid_target_key
            )
            if self.sysid_target_obs_key is not None or self.sysid_target_dim > 0:
                self.estimator_target_obs_key = self.sysid_target_obs_key
                self.estimator_target_size = (
                    input_shape[self.sysid_target_obs_key][0]
                    if self.sysid_target_obs_key is not None
                    else self.sysid_target_dim
                )
                self.estimator_loss_name = "sysid_estimation_loss"
                self.estimation_coef = self.sysid_estimation_coef
            elif self.gain_target_obs_key is not None or self.gain_target_dim > 0:
                self.estimator_target_obs_key = self.gain_target_obs_key
                self.estimator_target_size = (
                    input_shape[self.gain_target_obs_key][0]
                    if self.gain_target_obs_key is not None
                    else self.gain_target_dim
                )
                self.estimator_loss_name = "rate_gain_estimation_loss"
                self.estimation_coef = self.gain_estimation_coef
            else:
                self.estimator_target_obs_key = None
                self.estimator_target_size = 0
                self.estimator_loss_name = "estimation_loss"
                self.estimation_coef = 0.0
            self._aux_loss = None

            vector_input_shape = input_shape[self.vector_obs_key]
            vector_input_size = vector_input_shape[0]

            self.state_mlp, state_out_size = self._build_optional_mlp(
                vector_input_size,
                self.state_mlp_cfg,
            )

            image_out_size = 0
            self.image_cnn = None
            self.image_mlp = nn.Identity()
            self.image_activation = self.activations_factory.create(self.image_activation_name)
            if self.image_obs_key is not None:
                image_input_shape = input_shape[self.image_obs_key]
                cnn_input_shape = image_input_shape
                if self.permute_input:
                    cnn_input_shape = torch_ext.shape_whc_to_cwh(cnn_input_shape)

                cnn_args = {
                    "ctype": self.cnn["type"],
                    "input_shape": cnn_input_shape,
                    "convs": self.cnn["convs"],
                    "activation": self.cnn["activation"],
                    "norm_func_name": self.normalization,
                }
                self.image_cnn = self._build_conv(**cnn_args)
                image_cnn_output = self._calc_input_size(cnn_input_shape, self.image_cnn)
                self.image_mlp, image_out_size = self._build_optional_mlp(
                    image_cnn_output,
                    self.image_mlp_cfg,
                )

            fused_input_size = state_out_size + image_out_size
            if self.has_rnn and self.is_rnn_before_mlp:
                self.rnn = self._build_rnn(
                    self.rnn_name,
                    fused_input_size,
                    self.rnn_units,
                    self.rnn_layers,
                )
                if self.rnn_ln:
                    self.layer_norm = nn.LayerNorm(self.rnn_units)
                self.fusion_mlp, fused_output_size = self._build_optional_mlp(
                    self.rnn_units,
                    self.fusion_mlp_cfg,
                )
                trunk_output_size = fused_output_size
            else:
                self.fusion_mlp, fused_output_size = self._build_optional_mlp(
                    fused_input_size,
                    self.fusion_mlp_cfg,
                )
                trunk_output_size = fused_output_size
                if self.has_rnn:
                    self.rnn = self._build_rnn(
                        self.rnn_name,
                        fused_output_size,
                        self.rnn_units,
                        self.rnn_layers,
                    )
                    if self.rnn_ln:
                        self.layer_norm = nn.LayerNorm(self.rnn_units)
                    trunk_output_size = self.rnn_units

            self.estimator_mlp = None
            self.estimator = None
            self.last_estimator_prediction = None
            estimator_input_size = 0
            if self.estimator_target_size > 0 and (
                self.estimation_coef > 0.0 or self.append_estimator_to_actor_critic
            ):
                self.estimator_mlp, estimator_out_size = self._build_optional_mlp(
                    trunk_output_size,
                    self.estimation_mlp_cfg,
                )
                self.estimator = nn.Linear(estimator_out_size, self.estimator_target_size)
                if self.append_estimator_to_actor_critic:
                    estimator_input_size = self.estimator_target_size

            downstream_input_size = trunk_output_size + estimator_input_size
            self.actor_mlp, actor_out_size = self._build_optional_mlp(
                downstream_input_size,
                self.actor_mlp_cfg,
            )
            critic_input_size = downstream_input_size
            self.critic_state_mlp = None
            if self.critic_obs_key is not None:
                critic_state_input_shape = input_shape[self.critic_obs_key]
                critic_state_input_size = critic_state_input_shape[0]
                self.critic_state_mlp, critic_state_out_size = self._build_optional_mlp(
                    critic_state_input_size,
                    self.critic_state_mlp_cfg,
                )
                critic_input_size += critic_state_out_size
            self.critic_mlp, critic_out_size = self._build_optional_mlp(
                critic_input_size,
                self.critic_mlp_cfg,
            )

            self.value = self._build_value_layer(critic_out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            self.mu = nn.Linear(actor_out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config["mu_activation"])
            mu_init = self.init_factory.create(**self.space_config["mu_init"])

            self.sigma_act = self.activations_factory.create(self.space_config["sigma_activation"])
            sigma_init = self.init_factory.create(**self.space_config["sigma_init"])
            if self.fixed_sigma:
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                    requires_grad=True,
                )
            else:
                self.sigma = nn.Linear(actor_out_size, actions_num)

            linear_init = self.init_factory.create(**self.initializer)
            cnn_init = None
            if self.image_cnn is not None:
                cnn_init = self.init_factory.create(**self.cnn["initializer"])

            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
                    if cnn_init is not None:
                        cnn_init(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)
                if isinstance(module, nn.Linear):
                    linear_init(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)

            linear_init(self.value.weight)
            mu_init(self.mu.weight)
            if self.fixed_sigma:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)

        def _build_optional_mlp(self, input_size, cfg):
            units = cfg.get("units", [])
            if len(units) == 0:
                return nn.Identity(), input_size

            mlp = self._build_mlp(
                input_size=input_size,
                units=units,
                activation=cfg.get("activation", self.activation),
                norm_func_name=self.normalization,
                dense_func=nn.Linear,
                d2rl=cfg.get("d2rl", False),
                norm_only_first_layer=cfg.get("norm_only_first_layer", False),
            )
            return mlp, units[-1]

        def _resolve_vector_key(self, input_shape):
            candidates = []
            if self.vector_obs_key is not None:
                candidates.append(self.vector_obs_key)
            if self.state_key is not None:
                candidates.append(self.state_key)
            candidates.extend(["observations", "observation", "state"])

            for candidate in candidates:
                if candidate in input_shape:
                    return candidate

            for key in input_shape.keys():
                if key != self.image_observation_key and len(input_shape[key]) == 1:
                    return key
            raise KeyError(
                f"Could not resolve vector observation key from input shape keys: {list(input_shape.keys())}"
            )

        def _resolve_image_key(self, input_shape):
            candidates = []
            if self.image_observation_key is not None:
                candidates.append(self.image_observation_key)
            candidates.extend(["img_observation", "image_observation"])

            for candidate in candidates:
                if candidate in input_shape:
                    return candidate
            return None

        def _resolve_optional_key(self, input_shape, key):
            if key is not None and key in input_shape:
                return key
            return None

        def _get_vector_obs(self, obs):
            if self.vector_obs_key in obs:
                return obs[self.vector_obs_key]
            if self.state_key in obs:
                return obs[self.state_key]
            for candidate in ["observations", "observation", "state"]:
                if candidate in obs:
                    return obs[candidate]
            for key, value in obs.items():
                if key != self.image_obs_key and len(value.shape) == 2:
                    return value
            raise KeyError(f"Could not find vector observation in keys: {list(obs.keys())}")

        def _get_image_obs(self, obs):
            if self.image_obs_key is None:
                return None
            return obs.get(self.image_obs_key, None)

        def _get_optional_obs(self, obs, key):
            if key is None or not isinstance(obs, dict):
                return None
            return obs.get(key, None)

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)
            dones = obs_dict.get("dones", None)
            bptt_len = obs_dict.get("bptt_len", 0)

            vector_obs = self._get_vector_obs(obs)
            vector_features = self.state_mlp(vector_obs)

            features = [vector_features]
            image_obs = self._get_image_obs(obs)
            if image_obs is not None:
                if self.permute_input and len(image_obs.shape) == 4:
                    image_obs = image_obs.permute((0, 3, 1, 2))
                image_features = self.image_cnn(image_obs)
                image_features = image_features.contiguous().view(image_features.size(0), -1)
                image_features = self.image_activation(image_features)
                image_features = self.image_mlp(image_features)
                features.append(image_features)

            if len(features) == 1:
                out = features[0]
            else:
                out = torch.cat(features, dim=1)

            if self.has_rnn and self.is_rnn_before_mlp:
                seq_length = obs_dict.get("seq_length", 1)
                batch_size = out.size(0)
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1).transpose(0, 1)

                if states is None:
                    states = self.get_default_rnn_state()
                    states = tuple(state.to(out.device) for state in states)
                if len(states) == 1:
                    states = states[0]

                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)

                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1).contiguous().reshape(batch_size, -1)
                if self.rnn_ln:
                    out = self.layer_norm(out)
                if not isinstance(states, tuple):
                    states = (states,)

                out = self.fusion_mlp(out)
            else:
                out = self.fusion_mlp(out)

            if self.has_rnn and not self.is_rnn_before_mlp:
                seq_length = obs_dict.get("seq_length", 1)
                batch_size = out.size(0)
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1).transpose(0, 1)

                if states is None:
                    states = self.get_default_rnn_state()
                    states = tuple(state.to(out.device) for state in states)
                if len(states) == 1:
                    states = states[0]

                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)

                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1).contiguous().reshape(batch_size, -1)
                if self.rnn_ln:
                    out = self.layer_norm(out)
                if not isinstance(states, tuple):
                    states = (states,)

            estimator_prediction = None
            if self.estimator_mlp is not None and self.estimator is not None:
                estimator_prediction = self.estimator(self.estimator_mlp(out))
            self.last_estimator_prediction = (
                estimator_prediction.detach() if estimator_prediction is not None else None
            )

            downstream_input = out
            if self.append_estimator_to_actor_critic and estimator_prediction is not None:
                downstream_input = torch.cat([downstream_input, estimator_prediction], dim=1)

            actor_out = self.actor_mlp(downstream_input)
            critic_input = downstream_input
            critic_obs = self._get_optional_obs(obs, self.critic_obs_key)
            if critic_obs is not None and self.critic_state_mlp is not None:
                critic_features = self.critic_state_mlp(critic_obs)
                critic_input = torch.cat([critic_input, critic_features], dim=1)
            critic_out = self.critic_mlp(critic_input)

            self._aux_loss = None
            gain_target = self._get_optional_obs(obs, self.estimator_target_obs_key)
            if (
                gain_target is not None
                and estimator_prediction is not None
                and self.estimation_coef > 0.0
            ):
                self._aux_loss = {
                    self.estimator_loss_name: self.estimation_coef
                    * F.mse_loss(estimator_prediction, gain_target)
                }

            value = self.value_act(self.value(critic_out))
            mu = self.mu_act(self.mu(actor_out))
            if self.fixed_sigma:
                logstd = self.sigma_act(self.sigma)
            else:
                logstd = self.sigma_act(self.sigma(actor_out))

            return mu, mu * 0 + logstd, value, states

        def load(self, params):
            self.separate = params.get("separate", False)
            self.value_activation = params.get("value_activation", "None")
            self.normalization = params.get("normalization", None)

            self.vector_obs_key = params.get("observation_key", None)
            self.state_key = params.get("state_key", "state")
            self.image_observation_key = params.get("image_observation_key", "img_observation")
            self.critic_observation_key = params.get("critic_observation_key", None)
            self.gain_target_key = params.get("gain_target_key", None)
            self.sysid_target_key = params.get("sysid_target_key", None)
            self.gain_target_dim = int(params.get("gain_target_dim", 0))
            self.sysid_target_dim = int(params.get("sysid_target_dim", 0))
            self.gain_estimation_coef = float(params.get("gain_estimation_coef", 0.0))
            self.sysid_estimation_coef = float(params.get("sysid_estimation_coef", 0.0))
            self.append_estimator_to_actor_critic = bool(
                params.get("append_estimated_sysid_to_actor_critic", False)
            )

            self.fusion_mlp_cfg = params["mlp"]
            self.activation = self.fusion_mlp_cfg["activation"]
            self.initializer = self.fusion_mlp_cfg["initializer"]

            self.state_mlp_cfg = params.get(
                "state_mlp",
                {
                    "units": [],
                    "activation": self.activation,
                    "initializer": self.initializer,
                },
            )
            self.image_mlp_cfg = params.get(
                "image_mlp",
                {
                    "units": [],
                    "activation": self.activation,
                    "initializer": self.initializer,
                },
            )
            self.actor_mlp_cfg = params.get(
                "actor_mlp",
                {
                    "units": [],
                    "activation": self.activation,
                    "initializer": self.initializer,
                },
            )
            self.critic_mlp_cfg = params.get(
                "critic_mlp",
                {
                    "units": [],
                    "activation": self.activation,
                    "initializer": self.initializer,
                },
            )
            self.critic_state_mlp_cfg = params.get(
                "critic_state_mlp",
                {
                    "units": [],
                    "activation": self.activation,
                    "initializer": self.initializer,
                },
            )
            self.gain_estimation_mlp_cfg = params.get(
                "gain_estimation_mlp",
                {
                    "units": [],
                    "activation": self.activation,
                    "initializer": self.initializer,
                },
            )
            self.estimation_mlp_cfg = params.get(
                "sysid_estimation_mlp",
                self.gain_estimation_mlp_cfg,
            )

            self.cnn = params.get("cnn", None)
            self.permute_input = False
            self.image_activation_name = self.activation
            if self.cnn is not None:
                self.permute_input = self.cnn.get("permute_input", True)
                self.image_activation_name = self.cnn.get("activation", self.activation)
                self.cnn.setdefault("initializer", {"name": "default"})

            self.space_config = params["space"]["continuous"]
            self.fixed_sigma = self.space_config["fixed_sigma"]

            self.has_rnn = "rnn" in params
            self.rnn_ln = False
            self.is_rnn_before_mlp = False
            if self.has_rnn:
                self.rnn_units = params["rnn"]["units"]
                self.rnn_layers = params["rnn"]["layers"]
                self.rnn_name = params["rnn"]["name"]
                self.rnn_ln = params["rnn"].get("layer_norm", False)
                self.is_rnn_before_mlp = params["rnn"].get("before_mlp", False)

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            if self.rnn_name == "lstm":
                return (
                    torch.zeros((self.rnn_layers, self.num_seqs, self.rnn_units)),
                    torch.zeros((self.rnn_layers, self.num_seqs, self.rnn_units)),
                )
            return (torch.zeros((self.rnn_layers, self.num_seqs, self.rnn_units)),)

        def get_aux_loss(self):
            return self._aux_loss
