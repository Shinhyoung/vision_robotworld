"""Replay a recorded station capture through the full pipeline.

    ros2 launch roboworld_bringup rosbag_replay.launch.py bag:=/path/to/bag

This is the recommended hardware-free workflow (claude.md section 1): record
once with a real D455, then every agent develops against the identical frames.
Record with::

    ros2 bag record -o station_capture \\
        /camera/camera/color/image_raw \\
        /camera/camera/aligned_depth_to_color/image_raw \\
        /camera/camera/color/camera_info \\
        /roboworld/trigger
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument("bag", description="Path to the rosbag directory"),
        DeclareLaunchArgument("rate", default_value="1.0"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument("part_id", default_value="guide_block"),
        DeclareLaunchArgument("inspection_backend", default_value="statistical"),
        DeclareLaunchArgument("pose_backend", default_value="icp"),
    ]

    play = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play", LaunchConfiguration("bag"),
            "--rate", LaunchConfiguration("rate"),
            "--loop",
        ],
        output="screen",
    )

    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("roboworld_bringup"), "launch", "pipeline.launch.py"]
            )
        ),
        launch_arguments={
            "camera_source": "rosbag",
            "use_mock_camera": "false",
            # The bag carries its own triggers; do not synthesise more.
            "use_trigger_node": "false",
            "part_id": LaunchConfiguration("part_id"),
            "inspection_backend": LaunchConfiguration("inspection_backend"),
            "pose_backend": LaunchConfiguration("pose_backend"),
        }.items(),
    )

    return LaunchDescription(arguments + [play, pipeline])
