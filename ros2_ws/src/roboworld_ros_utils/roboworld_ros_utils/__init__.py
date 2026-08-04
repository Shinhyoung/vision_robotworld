"""ROS 2 adapter helpers shared by the RoboWorld nodes.

Everything that touches ``rclpy`` or a ROS message type lives in this package
(or in the node packages), keeping ``roboworld_core`` importable without a ROS
installation.
"""

from .conversions import (
    camera_info_to_intrinsics,
    depth_to_meters,
    empty_image,
    frame_to_messages,
    image_to_numpy,
    intrinsics_to_camera_info,
    make_header,
    messages_to_frame,
    numpy_to_image,
    pose_to_msg,
    pose_to_stamped,
    seconds_to_stamp,
    stamp_to_seconds,
    stamped_to_pose,
    transform_to_matrix,
)
from .params import config_from_node, declare_override, dotted_overrides, node_kwargs
from .qos import qos_from_config, sensor_qos

__all__ = [
    "camera_info_to_intrinsics",
    "config_from_node",
    "declare_override",
    "dotted_overrides",
    "node_kwargs",
    "depth_to_meters",
    "empty_image",
    "frame_to_messages",
    "image_to_numpy",
    "intrinsics_to_camera_info",
    "make_header",
    "messages_to_frame",
    "numpy_to_image",
    "pose_to_msg",
    "pose_to_stamped",
    "qos_from_config",
    "seconds_to_stamp",
    "sensor_qos",
    "stamp_to_seconds",
    "stamped_to_pose",
    "transform_to_matrix",
]
