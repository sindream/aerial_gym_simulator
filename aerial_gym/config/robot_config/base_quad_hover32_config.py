from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg


NOMINAL_MASS_KG = 0.25
NOMINAL_HOVER_THROTTLE = 0.26
NOMINAL_MAX_THRUST_PER_MOTOR_N = NOMINAL_MASS_KG * 9.81 / (
    4.0 * NOMINAL_HOVER_THROTTLE
)


class BaseQuadHover32Cfg(BaseQuadCfg):
    # The quad URDF nominal mass is 0.25 kg. Set per-motor max thrust so
    # hover uses 26% throttle: max_thrust = m * g / (4 motors * 0.26).
    nominal_mass_kg = NOMINAL_MASS_KG
    nominal_hover_throttle = NOMINAL_HOVER_THROTTLE

    class control_allocator_config(BaseQuadCfg.control_allocator_config):
        class motor_model_config(BaseQuadCfg.control_allocator_config.motor_model_config):
            max_thrust = NOMINAL_MAX_THRUST_PER_MOTOR_N
