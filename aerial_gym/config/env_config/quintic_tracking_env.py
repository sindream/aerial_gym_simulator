class QuinticTrackingEnvCfg:
    class env:
        num_envs = 1024
        num_env_actions = 0
        env_spacing = 25.0

        # base_sim_1ms runs at 1000 Hz. Holding one action for 20 physics
        # steps gives a 50 Hz policy interface.
        num_physics_steps_per_env_step_mean = 20
        num_physics_steps_per_env_step_std = 0

        render_viewer_every_n_steps = 1
        collision_force_threshold = 0.010
        manual_camera_trigger = False
        reset_on_collision = True
        create_ground_plane = False
        sample_timestep_for_latency = False
        perturb_observations = False
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False

        use_warp = False
        lower_bound_min = [-10.0, -10.0, 0.0]
        lower_bound_max = [-10.0, -10.0, 0.0]
        upper_bound_min = [10.0, 10.0, 8.0]
        upper_bound_max = [10.0, 10.0, 8.0]

    class env_config:
        include_asset_type = {}
        asset_type_to_dict_map = {}
