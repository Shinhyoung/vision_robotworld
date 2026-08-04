"""Conversions between ROS messages and the ROS-free core dataclasses.

Owned by the ROS2 agent (claude.md section 3.4). This is the only place where a
``sensor_msgs/Image`` becomes a numpy array or a :class:`Pose` becomes a
``geometry_msgs/PoseStamped``, so encoding and unit rules are enforced once.

Unit rules (ICD sections 3-4):

* depth ``16UC1`` is millimeters, ``32FC1`` is meters -> core always uses meters,
* positions are meters, orientations unit quaternions ``(x, y, z, w)``,
* ``0`` (or NaN) in a depth image means "no return" and stays 0.0 in the core.
"""

from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Pose as PoseMsg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from roboworld_core.geometry import Pose
from roboworld_core.types import CameraIntrinsics, Frame

#: Depth encodings we accept from realsense-ros / rosbag playback.
DEPTH_ENCODINGS = ("16UC1", "32FC1", "mono16")


def stamp_to_seconds(stamp: TimeMsg) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def seconds_to_stamp(seconds: float) -> TimeMsg:
    sec = int(seconds)
    return TimeMsg(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


def make_header(stamp_seconds: float, frame_id: str) -> Header:
    return Header(stamp=seconds_to_stamp(stamp_seconds), frame_id=frame_id)


# -- images --------------------------------------------------------------
def image_to_numpy(msg: Image) -> np.ndarray:
    """Decode a ``sensor_msgs/Image`` into a numpy array.

    cv_bridge is avoided on purpose: it drags in an OpenCV ABI that has to match
    the ROS distro exactly, and the handful of encodings this pipeline uses are
    trivial to decode directly.
    """
    encoding = msg.encoding.lower()
    if encoding in ("rgb8", "bgr8"):
        array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return array[..., ::-1].copy() if encoding == "bgr8" else array.copy()
    if encoding in ("mono8", "8uc1"):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width).copy()
    if encoding in ("16uc1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
    if encoding == "32fc1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    raise ValueError(
        f"unsupported image encoding '{msg.encoding}' "
        f"(supported: rgb8, bgr8, mono8, 16UC1, mono16, 32FC1)"
    )


def numpy_to_image(array: np.ndarray, encoding: str, header: Header) -> Image:
    """Encode a numpy array as a ``sensor_msgs/Image``."""
    data = np.ascontiguousarray(array)
    height, width = data.shape[0], data.shape[1]
    channels = data.shape[2] if data.ndim == 3 else 1
    return Image(
        header=header,
        height=height,
        width=width,
        encoding=encoding,
        is_bigendian=0,
        step=width * channels * data.dtype.itemsize,
        data=data.tobytes(),
    )


def empty_image(header: Header) -> Image:
    """A zero-sized image, the contract's "field not populated" marker."""
    return Image(header=header, height=0, width=0, encoding="", is_bigendian=0, step=0, data=b"")


def depth_to_meters(msg: Image, depth_units_m: float = 0.001) -> np.ndarray:
    """Convert a depth image to float32 meters, keeping 0 as "no return"."""
    array = image_to_numpy(msg)
    if msg.encoding.lower() in ("16uc1", "mono16"):
        depth = array.astype(np.float32) * float(depth_units_m)
    else:
        depth = array.astype(np.float32)
    return np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)


# -- camera --------------------------------------------------------------
def camera_info_to_intrinsics(msg: CameraInfo) -> CameraIntrinsics:
    """Extract pinhole intrinsics from ``sensor_msgs/CameraInfo``."""
    k = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
    return CameraIntrinsics(
        width=int(msg.width),
        height=int(msg.height),
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        frame_id=msg.header.frame_id,
        distortion=tuple(float(v) for v in msg.d),
    )


def intrinsics_to_camera_info(intrinsics: CameraIntrinsics, header: Header) -> CameraInfo:
    matrix = intrinsics.matrix
    info = CameraInfo(header=header, height=intrinsics.height, width=intrinsics.width)
    info.distortion_model = "plumb_bob"
    info.d = list(intrinsics.distortion) or [0.0] * 5
    info.k = [float(v) for v in matrix.reshape(9)]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [
        intrinsics.fx, 0.0, intrinsics.cx, 0.0,
        0.0, intrinsics.fy, intrinsics.cy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    return info


def messages_to_frame(
    color: Image,
    depth: Image,
    camera_info: CameraInfo,
    part_id: str,
    sequence: int,
    depth_units_m: float = 0.001,
) -> Frame:
    """Assemble a core :class:`Frame` from the three camera messages."""
    return Frame(
        color=image_to_numpy(color),
        depth=depth_to_meters(depth, depth_units_m),
        intrinsics=camera_info_to_intrinsics(camera_info),
        stamp=stamp_to_seconds(color.header.stamp),
        sequence=sequence,
        part_id=part_id,
    )


def frame_to_messages(frame: Frame) -> tuple[Image, Image, CameraInfo]:
    """Inverse of :func:`messages_to_frame`; used by the mock camera node."""
    header = make_header(frame.stamp, frame.intrinsics.frame_id)
    return (
        numpy_to_image(frame.color.astype(np.uint8), "rgb8", header),
        numpy_to_image(frame.depth.astype(np.float32), "32FC1", header),
        intrinsics_to_camera_info(frame.intrinsics, header),
    )


# -- poses ---------------------------------------------------------------
def pose_to_msg(pose: Pose) -> PoseMsg:
    msg = PoseMsg()
    msg.position.x, msg.position.y, msg.position.z = (float(v) for v in pose.position)
    quaternion = pose.orientation
    msg.orientation.x = float(quaternion[0])
    msg.orientation.y = float(quaternion[1])
    msg.orientation.z = float(quaternion[2])
    msg.orientation.w = float(quaternion[3])
    return msg


def pose_to_stamped(pose: Pose, stamp_seconds: float, frame_id: str | None = None) -> PoseStamped:
    """Wrap a core pose as ``PoseStamped``. ``frame_id`` defaults to the pose's own."""
    stamped = PoseStamped()
    stamped.header = make_header(stamp_seconds, frame_id or pose.frame_id)
    stamped.pose = pose_to_msg(pose)
    return stamped


def stamped_to_pose(msg: PoseStamped) -> Pose:
    position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    orientation = np.array(
        [msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w]
    )
    return Pose(position, orientation, msg.header.frame_id)


def transform_to_matrix(transform) -> np.ndarray:
    """``geometry_msgs/TransformStamped`` -> 4x4 homogeneous matrix."""
    from roboworld_core.geometry import make_transform, matrix_from_quaternion

    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return make_transform(
        matrix_from_quaternion(
            np.array([rotation.x, rotation.y, rotation.z, rotation.w])
        ),
        np.array([translation.x, translation.y, translation.z]),
    )
