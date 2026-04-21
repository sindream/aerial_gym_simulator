import distutils
import os

import gym
import isaacgym
import torch
import yaml
from gym import spaces

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.helpers import parse_arguments
from rl_games.algos_torch import a2c_continuous, model_builder, players
from rl_games.algos_torch.aerial_multimodal_models import (
    ModelA2CContinuousLogStdMultimodal,
)
from rl_games.algos_torch.aerial_multimodal_network_builder import (
    AerialMultimodalA2CBuilder,
)
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

model_builder.register_network("aerial_multimodal_actor_critic", AerialMultimodalA2CBuilder)
model_builder.register_model(
    "continuous_a2c_logstd_multimodal",
    ModelA2CContinuousLogStdMultimodal,
)


class DictObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._vector_key = None
        self._image_key = None
        self._is_dict_space = isinstance(env.observation_space, spaces.Dict)

        if self._is_dict_space:
            self._vector_key = "state" if "state" in env.observation_space.spaces else "observations"
            if "img_observation" in env.observation_space.spaces:
                self._image_key = "img_observation"

            wrapper_spaces = {self._vector_key: env.observation_space.spaces[self._vector_key]}
            if self._image_key is not None:
                wrapper_spaces[self._image_key] = env.observation_space.spaces[self._image_key]
            self.observation_space = spaces.Dict(wrapper_spaces)
        else:
            self.observation_space = env.observation_space

    def _filter_obs(self, observations):
        if not self._is_dict_space:
            return observations

        policy_observations = {
            self._vector_key: observations[self._vector_key],
        }
        if self._image_key is not None and self._image_key in observations:
            policy_observations[self._image_key] = observations[self._image_key]
        return policy_observations

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)
        return self._filter_obs(observations)

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)

        dones = torch.where(
            terminated | truncated,
            torch.ones_like(terminated),
            torch.zeros_like(terminated),
        )

        if not isinstance(infos, dict):
            infos = {}
        infos["time_outs"] = truncated
        infos["terminated"] = terminated
        infos["truncated"] = truncated

        return self._filter_obs(observations), rewards, dones, infos

    def reset_done(self):
        observations = self.env.reset_done()
        return self._filter_obs(observations)


class AERIALRLGPUEnvDict(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.env = DictObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {
            "action_space": self.env.action_space,
            "observation_space": self.env.observation_space,
        }
        if hasattr(self.env, "state_space"):
            info["state_space"] = self.env.state_space
        if hasattr(self.env, "value_size"):
            info["value_size"] = self.env.value_size
        info["agents"] = self.get_number_of_agents()
        return info

    def close(self):
        self.env.close()


class MultimodalRunner(Runner):
    def __init__(self, algo_observer=None):
        super().__init__(algo_observer=algo_observer)
        self.algo_factory.register_builder(
            "a2c_continuous_multimodal",
            lambda **kwargs: a2c_continuous.A2CAgent(**kwargs),
        )
        self.player_factory.register_builder(
            "a2c_continuous_multimodal",
            lambda **kwargs: players.PpoPlayerContinuous(**kwargs),
        )


def register_aerial_envs():
    env_configurations.register(
        "position_setpoint_task",
        {
            "env_creator": lambda **kwargs: task_registry.make_task(
                "position_setpoint_task", **kwargs
            ),
            "vecenv_type": "AERIAL-RLGPU-DICT",
        },
    )

    env_configurations.register(
        "position_setpoint_task_sim2real",
        {
            "env_creator": lambda **kwargs: task_registry.make_task(
                "position_setpoint_task_sim2real", **kwargs
            ),
            "vecenv_type": "AERIAL-RLGPU-DICT",
        },
    )

    env_configurations.register(
        "position_setpoint_task_sim2real_px4",
        {
            "env_creator": lambda **kwargs: task_registry.make_task(
                "position_setpoint_task_sim2real_px4", **kwargs
            ),
            "vecenv_type": "AERIAL-RLGPU-DICT",
        },
    )

    env_configurations.register(
        "position_setpoint_task_acceleration_sim2real",
        {
            "env_creator": lambda **kwargs: task_registry.make_task(
                "position_setpoint_task_acceleration_sim2real", **kwargs
            ),
            "vecenv_type": "AERIAL-RLGPU-DICT",
        },
    )

    env_configurations.register(
        "navigation_task",
        {
            "env_creator": lambda **kwargs: task_registry.make_task("navigation_task", **kwargs),
            "vecenv_type": "AERIAL-RLGPU-DICT",
        },
    )

    env_configurations.register(
        "drone_racing_task",
        {
            "env_creator": lambda **kwargs: task_registry.make_task("drone_racing_task", **kwargs),
            "vecenv_type": "AERIAL-RLGPU-DICT",
        },
    )

    env_configurations.register(
        "position_setpoint_task_reconfigurable",
        {
            "env_creator": lambda **kwargs: task_registry.make_task(
                "position_setpoint_task_reconfigurable", **kwargs
            ),
            "vecenv_type": "AERIAL-RLGPU-DICT",
        },
    )

    env_configurations.register(
        "position_setpoint_task_morphy",
        {
            "env_creator": lambda **kwargs: task_registry.make_task(
                "position_setpoint_task_morphy", **kwargs
            ),
            "vecenv_type": "AERIAL-RLGPU-DICT",
        },
    )

    env_configurations.register(
        "position_setpoint_task_sim2real_end_to_end",
        {
            "env_creator": lambda **kwargs: task_registry.make_task(
                "position_setpoint_task_sim2real_end_to_end", **kwargs
            ),
            "vecenv_type": "AERIAL-RLGPU-DICT",
        },
    )


vecenv.register(
    "AERIAL-RLGPU-DICT",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnvDict(config_name, num_actors, **kwargs),
)

register_aerial_envs()


def get_args():
    custom_parameters = [
        {
            "name": "--seed",
            "type": int,
            "default": 0,
            "required": False,
            "help": "Random seed, if larger than 0 will overwrite the value in yaml config.",
        },
        {
            "name": "--tf",
            "required": False,
            "help": "run tensorflow runner",
            "action": "store_true",
        },
        {
            "name": "--train",
            "required": False,
            "help": "train network",
            "action": "store_true",
        },
        {
            "name": "--play",
            "required": False,
            "help": "play(test) network",
            "action": "store_true",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "required": False,
            "help": "path to checkpoint",
        },
        {
            "name": "--file",
            "type": str,
            "default": "ppo_drone_racing_multimodal.yaml",
            "required": False,
            "help": "path to config",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": -1,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--sigma",
            "type": float,
            "required": False,
            "help": "sets new sigma value in case if 'fixed_sigma: True' in yaml config",
        },
        {
            "name": "--track",
            "action": "store_true",
            "help": "if toggled, this experiment will be tracked with Weights and Biases",
        },
        {
            "name": "--wandb-project-name",
            "type": str,
            "default": "rl_games",
            "help": "the wandb's project name",
        },
        {
            "name": "--wandb-entity",
            "type": str,
            "default": None,
            "help": "the entity (team) of wandb's project",
        },
        {
            "name": "--task",
            "type": str,
            "default": None,
            "help": "Override task from config file if provided.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": None,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--use_warp",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": None,
            "help": "Choose whether to use warp or Isaac Gym rendering pipeline.",
        },
    ]

    args = parse_arguments(description="RL Policy", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def update_config(config, args):
    if args["task"] is not None:
        config["params"]["config"]["env_name"] = args["task"]
    if args["experiment_name"] is not None:
        config["params"]["config"]["name"] = args["experiment_name"]

    if args["headless"] is not None:
        config["params"]["config"]["env_config"]["headless"] = args["headless"]
    if args["num_envs"] > 0:
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    if args["use_warp"] is not None:
        config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]

    if args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]

    config["params"]["config"]["player"] = {"use_vecenv": True}
    return config


if __name__ == "__main__":
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(get_args())
    config_name = args["file"]

    print("Loading config:", config_name)
    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)
        config = update_config(config, args)

        runner = MultimodalRunner()
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if args["track"] and rank == 0:
        import wandb

        wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )

    runner.run(args)

    if args["track"] and rank == 0:
        wandb.finish()
