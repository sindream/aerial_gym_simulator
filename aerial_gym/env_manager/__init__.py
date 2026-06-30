import isaacgym
from aerial_gym.config.env_config.env_with_obstacles import EnvWithObstaclesCfg
from aerial_gym.config.env_config.empty_env import EmptyEnvCfg
from aerial_gym.config.env_config.forest_env import ForestEnvCfg
from aerial_gym.config.env_config.env_config_2ms import EnvCfg2Ms
from aerial_gym.config.env_config.dynamic_environment import DynamicEnvironmentCfg
from aerial_gym.config.env_config.drone_racing_env import DroneRacingEnvCfg
from aerial_gym.config.env_config.drone_racing_ned_env import DroneRacingNedEnvCfg
from aerial_gym.config.env_config.quintic_tracking_env import QuinticTrackingEnvCfg
from aerial_gym.config.env_config.aigp_racing_env import AIGPRacingEnvCfg

from aerial_gym.registry.env_registry import env_config_registry

env_config_registry.register("env_with_obstacles", EnvWithObstaclesCfg)
env_config_registry.register("empty_env", EmptyEnvCfg)
env_config_registry.register("forest_env", ForestEnvCfg)
env_config_registry.register("empty_env_2ms", EnvCfg2Ms)
env_config_registry.register("dynamic_env", DynamicEnvironmentCfg)
env_config_registry.register("drone_racing_env", DroneRacingEnvCfg)
env_config_registry.register("drone_racing_ned_env", DroneRacingNedEnvCfg)
env_config_registry.register("quintic_tracking_env", QuinticTrackingEnvCfg)
env_config_registry.register("aigp_racing_env", AIGPRacingEnvCfg)
