"""Full pipeline bring-up.

    ros2 launch roboworld_bringup pipeline.launch.py

Defaults are hardware-free (mock camera source, CPU backends) so a fresh clone
runs end to end. Switch to the real station with:

    ros2 launch roboworld_bringup pipeline.launch.py \\
        camera_source:=realsense \\
        inspection_backend:=efficientad \\
        pose_backend:=foundationpose

Arguments are applied as dotted parameter overrides on top of the YAML files in
``roboworld_bringup/config``; the YAML remains the single source of truth.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument(
            "part_id", default_value="guide_block",
            description="Part type at the station (see config/parts.yaml)"),
        DeclareLaunchArgument(
            "camera_source", default_value="mock",
            description="realsense | rosbag | mock"),
        DeclareLaunchArgument(
            "inspection_backend", default_value="statistical",
            description="efficientad | statistical | stub"),
        DeclareLaunchArgument(
            "pose_backend", default_value="icp",
            description="foundationpose | icp | stub"),
        DeclareLaunchArgument(
            "use_mock_camera", default_value="false",
            description="Publish CAD-rendered frames on the realsense topics"),
        DeclareLaunchArgument(
            "use_trigger_node", default_value="true",
            description="Emit periodic triggers instead of waiting for the PLC"),
        DeclareLaunchArgument(
            "trigger_period_s", default_value="4.0",
            description="Trigger interval when use_trigger_node is true"),
        DeclareLaunchArgument(
            "mock_defect_every", default_value="3",
            description="Inject a defect on every Nth mock cycle (0 = never)"),
        DeclareLaunchArgument(
            "config_dir", default_value="",
            description="Override the config directory (defaults to this package's share)"),
        DeclareLaunchArgument(
            "log_level", default_value="info"),
    ]

    part_id = LaunchConfiguration("part_id")
    config_dir = LaunchConfiguration("config_dir")
    log_level = LaunchConfiguration("log_level")
    common = [{"config_dir": config_dir}, {"part_id": part_id}]
    log_args = ["--ros-args", "--log-level", log_level]

    inspection = Node(
        package="roboworld_inspection",
        executable="inspection_node",
        name="inspection_node",
        output="screen",
        parameters=common + [
            {"backend": LaunchConfiguration("inspection_backend")},
        ],
        arguments=log_args,
    )

    pose = Node(
        package="roboworld_pose",
        executable="pose_node",
        name="pose_node",
        output="screen",
        parameters=common + [
            {"backend": LaunchConfiguration("pose_backend")},
        ],
        arguments=log_args,
    )

    pipeline = Node(
        package="roboworld_pipeline",
        executable="pipeline_node",
        name="pipeline_node",
        output="screen",
        parameters=common + [
            {"camera.source": LaunchConfiguration("camera_source")},
            {"mock_defect_every": LaunchConfiguration("mock_defect_every")},
        ],
        arguments=log_args,
    )

    mock_camera = GroupAction(
        condition=IfCondition(LaunchConfiguration("use_mock_camera")),
        actions=[
            Node(
                package="roboworld_pipeline",
                executable="mock_camera_node",
                name="mock_camera_node",
                output="screen",
                parameters=common + [
                    {"defect_every": LaunchConfiguration("mock_defect_every")},
                ],
                arguments=log_args,
            )
        ],
    )

    trigger = GroupAction(
        condition=IfCondition(LaunchConfiguration("use_trigger_node")),
        actions=[
            Node(
                package="roboworld_pipeline",
                executable="trigger_node",
                name="trigger_node",
                output="screen",
                parameters=common + [
                    {"period_s": LaunchConfiguration("trigger_period_s")},
                ],
                arguments=log_args,
            )
        ],
    )

    return LaunchDescription(arguments + [inspection, pose, pipeline, mock_camera, trigger])
