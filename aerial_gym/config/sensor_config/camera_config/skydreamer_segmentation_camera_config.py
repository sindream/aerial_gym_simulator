class SkyDreamerSegmentationCameraConfig:
    num_sensors = 1
    sensor_type = "camera"

    # SkyDreamer uses 64x64 segmentation masks for policy training.
    height = 64
    width = 64

    # Paper nominal intrinsics: fx = 25/64 * W -> HFOV ~= 104 degrees.
    horizontal_fov_deg = 104.0
    # The racing track spans roughly 100 m in x, so keep the far plane comfortably beyond that.
    max_range = 120.0
    min_range = 0.1

    calculate_depth = True
    return_pointcloud = False
    pointcloud_in_world_frame = False
    segmentation_camera = True

    euler_frame_rot_deg = [-90.0, 0.0, -90.0]

    normalize_range = False
    far_out_of_range_value = -1.0
    near_out_of_range_value = -1.0

    randomize_placement = True
    min_translation = [0.12, 0.0, 0.02]
    max_translation = [0.12, 0.0, 0.02]
    # SkyDreamer randomizes camera roll and yaw in [-5, 5] degrees and pitch in [45, 55]
    # degrees. The AerialGym camera convention uses a negative pitch for the same upward tilt.
    min_euler_rotation_deg = [-5.0, -55.0, -5.0]
    max_euler_rotation_deg = [5.0, -45.0, 5.0]

    nominal_position = [0.12, 0.0, 0.02]
    nominal_orientation_euler_deg = [0.0, -50.0, 0.0]

    use_collision_geometry = False

    class sensor_noise:
        enable_sensor_noise = False
        pixel_dropout_prob = 0.0
        pixel_std_dev_multiplier = 0.0
