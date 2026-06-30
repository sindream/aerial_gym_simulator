class AIGPSegmentationCameraConfig:
    num_sensors = 1
    sensor_type = "camera"

    # Technical spec camera is 640x360 with nominal fx=fy=320. Rendering a
    # 4x downsample keeps the same aspect/FOV while making vectorized training
    # tractable.
    height = 90
    width = 160
    horizontal_fov_deg = 90.0
    max_range = 220.0
    min_range = 0.05

    calculate_depth = True
    return_pointcloud = False
    pointcloud_in_world_frame = False
    segmentation_camera = True

    euler_frame_rot_deg = [-90.0, 0.0, -90.0]

    normalize_range = False
    far_out_of_range_value = -1.0
    near_out_of_range_value = -1.0

    randomize_placement = False
    min_translation = [0.12, 0.0, 0.02]
    max_translation = [0.12, 0.0, 0.02]
    min_euler_rotation_deg = [0.0, 0.0, 0.0]
    max_euler_rotation_deg = [0.0, 0.0, 0.0]

    nominal_position = [0.12, 0.0, 0.02]
    # BODY_NED to camera in the spec is a 20 degree upward tilt. In this
    # IsaacGym camera convention, negative pitch tilts the camera upward.
    nominal_orientation_euler_deg = [0.0, -20.0, 0.0]

    use_collision_geometry = False

    class sensor_noise:
        enable_sensor_noise = False
        pixel_dropout_prob = 0.0
        pixel_std_dev_multiplier = 0.0
