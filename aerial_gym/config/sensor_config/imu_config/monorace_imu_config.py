class MonoRaceImuConfig:
    num_sensors = 1
    sensor_type = "imu"

    world_frame = False
    enable_noise = True
    enable_bias = True

    bias_std = [
        9.782812831313576e-07,
        9.782812831313576e-07,
        9.782812831313576e-07,
        2.6541629581345176e-05,
        2.6541629581345176e-05,
        2.6541629581345176e-05,
    ]

    imu_noise_std = [
        0.0020,
        0.0020,
        0.0020,
        0.0012,
        0.0012,
        0.0012,
    ]

    max_measurement_value = [
        156.96,
        156.96,
        156.96,
        35.0,
        35.0,
        35.0,
    ]

    max_bias_init_value = [
        1.0e-03,
        1.0e-03,
        1.0e-03,
        1.0e-03,
        1.0e-03,
        1.0e-03,
    ]

    gravity_compensation = False

    randomize_placement = False
    min_euler_rotation_deg = [0.0, 0.0, 0.0]
    max_euler_rotation_deg = [0.0, 0.0, 0.0]

