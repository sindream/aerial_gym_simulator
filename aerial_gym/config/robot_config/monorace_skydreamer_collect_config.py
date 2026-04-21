import numpy as np

from aerial_gym.config.robot_config.monorace_skydreamer_config import MonoRaceSkyDreamerCfg


class MonoRaceSkyDreamerCollectCfg(MonoRaceSkyDreamerCfg):
    class robot_asset(MonoRaceSkyDreamerCfg.robot_asset):
        name = "monorace_skydreamer_collect_quadrotor"

    class control_allocator_config(MonoRaceSkyDreamerCfg.control_allocator_config):
        class motor_model_config(MonoRaceSkyDreamerCfg.control_allocator_config.motor_model_config):
            use_rps = True
            # Use deterministic nominal actuator parameters for clean warm-start data.
            motor_thrust_constant_min = 1.55e-6
            motor_thrust_constant_max = 1.55e-6
            motor_time_constant_increasing_min = 0.03
            motor_time_constant_increasing_max = 0.03
            motor_time_constant_decreasing_min = 0.03
            motor_time_constant_decreasing_max = 0.03
            max_thrust = 14.39
            min_thrust = 0.05
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = 0.05
            use_discrete_approximation = True
