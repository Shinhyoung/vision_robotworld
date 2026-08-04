"""Intel RealSense D455 driver bring-up.

    ros2 launch roboworld_bringup realsense.launch.py

Depth is aligned to color -- the ICD requires the two images to share pixel
coordinates and the pose backend back-projects depth with the *color*
intrinsics, so an unaligned stream silently produces wrong poses.

WSL2 note (claude.md section 1): the D455 is not visible to WSL until it is
attached with ``usbipd-win``. Run docs/setup_wsl.md section 4 first, or use
``ros2 bag play`` instead.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument("camera_name", default_value="camera"),
        DeclareLaunchArgument("camera_namespace", default_value="camera"),
        DeclareLaunchArgument("width", default_value="640"),
        DeclareLaunchArgument("height", default_value="480"),
        DeclareLaunchArgument("fps", default_value="30"),
    ]

    width = LaunchConfiguration("width")
    height = LaunchConfiguration("height")
    fps = LaunchConfiguration("fps")

    realsense = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name=LaunchConfiguration("camera_name"),
        namespace=LaunchConfiguration("camera_namespace"),
        output="screen",
        parameters=[{
            # REQUIRED by the ICD: depth registered into the color frame.
            "align_depth.enable": True,
            "enable_color": True,
            "enable_depth": True,
            "enable_sync": True,
            "rgb_camera.color_profile": [width, "x", height, "x", fps],
            "depth_module.depth_profile": [width, "x", height, "x", fps],
            # The index station is stop-and-go; temporal filtering across a
            # moving belt would smear the edges we register against.
            "temporal_filter.enable": False,
            "spatial_filter.enable": True,
            "hole_filling_filter.enable": False,
            "pointcloud.enable": False,
        }],
    )

    return LaunchDescription(arguments + [realsense])
