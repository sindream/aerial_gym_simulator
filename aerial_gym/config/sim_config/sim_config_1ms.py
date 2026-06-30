from aerial_gym.config.sim_config.sim_config_2ms import SimCfg2Ms


class SimCfg1Ms(SimCfg2Ms):
    class sim(SimCfg2Ms.sim):
        dt = 0.001

