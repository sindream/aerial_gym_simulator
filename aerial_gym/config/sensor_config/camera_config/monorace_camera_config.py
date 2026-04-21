class MonoRaceCameraConfig:
    num_sensors = 1
    sensor_type = "camera"

    height = 48
    width = 64
    horizontal_fov_deg = 150.0
    max_range = 40.0
    min_range = 0.05

    calculate_depth = True
    return_pointcloud = False
    pointcloud_in_world_frame = False
    segmentation_camera = True

    euler_frame_rot_deg = [-90.0, 0.0, -90.0]

    normalize_range = True
    normalize_range = (
        False
        if (return_pointcloud is True and pointcloud_in_world_frame is True)
        else normalize_range
    )

    far_out_of_range_value = max_range if normalize_range is True else -1.0
    near_out_of_range_value = -max_range if normalize_range is True else -1.0

    randomize_placement = False
    min_translation = [0.12, 0.0, 0.02]
    max_translation = [0.12, 0.0, 0.02]
    min_euler_rotation_deg = [0.0, 0.0, 0.0]
    max_euler_rotation_deg = [0.0, 0.0, 0.0]

    nominal_position = [0.12, 0.0, 0.02]
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]

    use_collision_geometry = False

    class sensor_noise:
        enable_sensor_noise = False
        pixel_dropout_prob = 0.0
        pixel_std_dev_multiplier = 0.0
