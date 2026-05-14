from aerial_gym.config.asset_config.racing_track_asset_config import (
    RANDOM_CYLINDER_OBSTACLE_COUNT,
    RaceCylinderObstacleAssetParams,
    RaceHorizontalCylinderObstacleAssetParams,
    RaceTiltedCylinderObstacleAssetParams,
    TRACK_BOUNDS_MIN,
    TRACK_BOUNDS_MAX,
    TRACK_GATE_ASSET_CLASSES,
)


TRACK_INCLUDE_ASSET_TYPE = {
    f"race_gate_{gate_index + 1:02d}": True for gate_index in range(len(TRACK_GATE_ASSET_CLASSES))
}

TRACK_ASSET_TYPE_TO_DICT_MAP = {
    f"race_gate_{gate_index + 1:02d}": gate_asset_class
    for gate_index, gate_asset_class in enumerate(TRACK_GATE_ASSET_CLASSES)
}
TRACK_INCLUDE_ASSET_TYPE["race_cylinders"] = RANDOM_CYLINDER_OBSTACLE_COUNT > 0
TRACK_ASSET_TYPE_TO_DICT_MAP["race_cylinders"] = RaceCylinderObstacleAssetParams
TRACK_INCLUDE_ASSET_TYPE["race_horizontal_cylinders"] = (
    RaceHorizontalCylinderObstacleAssetParams.num_assets > 0
)
TRACK_ASSET_TYPE_TO_DICT_MAP["race_horizontal_cylinders"] = RaceHorizontalCylinderObstacleAssetParams
TRACK_INCLUDE_ASSET_TYPE["race_tilted_cylinders"] = (
    RaceTiltedCylinderObstacleAssetParams.num_assets > 0
)
TRACK_ASSET_TYPE_TO_DICT_MAP["race_tilted_cylinders"] = RaceTiltedCylinderObstacleAssetParams


class DroneRacingEnvCfg:
    class env:
        num_envs = 64
        num_env_actions = 0
        env_spacing = 50.0

        num_physics_steps_per_env_step_mean = 2
        num_physics_steps_per_env_step_std = 0

        render_viewer_every_n_steps = 1
        reset_on_collision = True
        collision_force_threshold = 0.0001
        create_ground_plane = True
        sample_timestep_for_latency = False
        perturb_observations = False
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False

        use_warp = True
        lower_bound_min = TRACK_BOUNDS_MIN
        lower_bound_max = TRACK_BOUNDS_MIN
        upper_bound_min = TRACK_BOUNDS_MAX
        upper_bound_max = TRACK_BOUNDS_MAX

    class env_config:
        include_asset_type = TRACK_INCLUDE_ASSET_TYPE
        asset_type_to_dict_map = TRACK_ASSET_TYPE_TO_DICT_MAP
