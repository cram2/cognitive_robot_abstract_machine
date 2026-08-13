from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING, TypeVar

import rclpy
from geometry_msgs.msg import (
    PoseStamped,
    Vector3Stamped,
    PointStamped,
    TransformStamped,
    QuaternionStamped,
)
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_py import InvalidArgumentException
from tf2_ros import Buffer, TransformListener

if TYPE_CHECKING:
    from networkx import MultiDiGraph

logger = logging.getLogger(__name__)

TransformableMsg = TypeVar(
    "TransformableMsg", PoseStamped, PointStamped, QuaternionStamped, Vector3Stamped
)


@dataclass
class TFWrapper:
    """
    A wrapper for ROS2's very beautiful tf library.
    """

    node: rclpy.node.Node = None
    """The ROS2 node that this TFWrapper is associated with."""
    tf_buffer_size: Optional[Duration] = None
    """The size of the tf buffer. If None, the default buffer size is used."""
    tf_buffer: Buffer = field(init=False)
    """The tf buffer used by this TFWrapper."""
    tf_listener: TransformListener = field(init=False)
    """The tf listener used by this TFWrapper."""

    def __post_init__(self):
        self.tf_buffer = Buffer(self.tf_buffer_size)
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        try:
            self.get_tf_root()
        except Exception as e:
            logger.warning(str(e))

    def get_tf_roots(self) -> List[str]:
        """
        Returns a list of all tf roots in the tf tree.
        """
        graph = self.get_graph()
        return [node for node in graph.nodes() if not list(graph.predecessors(node))]

    def get_graph(self) -> MultiDiGraph:
        """
        Returns a networkx MultiDiGraph representation of the tf tree.
        """
        from networkx.drawing.nx_pydot import read_dot

        dot_string = self.tf_buffer._allFramesAsDot()
        cleaned_dot_string = dot_string.replace("\n", " ")

        dot_file = io.StringIO(cleaned_dot_string)
        graph = read_dot(dot_file)
        return graph

    def get_tf_root(self) -> str:
        """
        Returns the tf root.
        """
        tf_roots = self.get_tf_roots()
        assert len(tf_roots) < 2, f"There are more than one tf tree: {tf_roots}."
        assert len(tf_roots) > 0, "There is no tf tree."
        return tf_roots.pop()

    def get_full_frame_names(self, frame_name: str) -> List[str]:
        """
        Search for namespaced frames that include frame_name.
        :param frame_name: The frame name to search for.
        :return: A list of frames that include frame_name.
        """
        ret = list()
        tf_frames = self.tf_buffer._getFrameStrings()
        for tf_frame in tf_frames:
            try:
                frame = tf_frame[tf_frame.index("/") + 1 :]
                if frame == frame_name or frame_name == tf_frame:
                    ret.append(tf_frame)
            except ValueError:
                continue
        if len(ret) == 0:
            raise KeyError(
                f"Could not find frame {frame_name} in the buffer of the tf Listener."
            )
        return ret

    def get_tf_frames(self) -> List[str]:
        """
        Get all frames in the tf buffer.
        :return: A list of frames.
        """
        return self.tf_buffer._getFrameStrings()

    def wait_for_transform(
        self, target_frame: str, source_frame: str, time: Time, timeout: Duration
    ) -> bool:
        """
        Wait for a transform to become available.
        :param target_frame: The target frame.
        :param source_frame: The source frame.
        :param time: The time at which the transform is requested.
        :param timeout: The maximum time to wait for the transform.
        :return: True if the transform is available, False otherwise.
        """
        return self.tf_buffer.can_transform(target_frame, source_frame, time, timeout)

    def lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
        time: Optional[Time] = None,
        timeout: float = 5.0,
    ) -> TransformStamped:
        """
        Look up a transform between two frames.
        :param target_frame: The target frame.
        :param source_frame: The source frame.
        :param time: The time at which the transform is requested.
        :param timeout: The maximum time to wait for the transform.
        :return: The transform between the two frames.
        """
        if not target_frame:
            raise InvalidArgumentException("target frame can not be empty")
        if not source_frame:
            raise InvalidArgumentException("source frame can not be empty")
        if time is None:
            time = Time()
        return self.tf_buffer.lookup_transform(
            str(target_frame),
            str(source_frame),  # source frame
            time,
            Duration(seconds=timeout),
        )

    def transform_msg(
        self, target_frame: str, msg: TransformableMsg, timeout: float = 5
    ) -> TransformableMsg:
        """
        Transforms a message into a different target frame.
        :param target_frame: The target frame.
        :param msg: The message to transform.
        :param timeout: The maximum time to wait for the transform.
        :return: The transformed message.
        """
        if isinstance(msg, PoseStamped):
            return self.transform_pose(target_frame, msg, timeout)
        elif isinstance(msg, PointStamped):
            return self.transform_point(target_frame, msg, timeout)
        elif isinstance(msg, Vector3Stamped):
            return self.transform_vector(target_frame, msg, timeout)
        elif isinstance(msg, QuaternionStamped):
            return self.transform_quaternion(target_frame, msg, timeout)
        else:
            raise NotImplementedError(f"tf transform message of type '{type(msg)}'")

    def transform_pose(
        self, target_frame: str, pose: PoseStamped, timeout: float = 5.0
    ) -> PoseStamped:
        """
        Transforms a pose stamped into a different target frame.
        :param target_frame: The target frame.
        :param pose: The pose to transform.
        :param timeout: The maximum time to wait for the transform.
        :return: The transformed pose.
        """
        from tf2_geometry_msgs import do_transform_pose_stamped

        transform = self.lookup_transform(
            target_frame, pose.header.frame_id, pose.header.stamp, timeout
        )
        new_pose = do_transform_pose_stamped(pose, transform)
        return new_pose

    def transform_vector(
        self, target_frame: str, vector: Vector3Stamped, timeout: float = 5
    ) -> Vector3Stamped:
        """
        Transforms a vector stamped into a different target frame.
        :param target_frame: The target frame.
        :param vector: The vector to transform.
        :param timeout: The maximum time to wait for the transform.
        :return: The transformed vector.
        """
        from tf2_geometry_msgs import do_transform_vector3

        transform = self.lookup_transform(
            target_frame, vector.header.frame_id, vector.header.stamp, timeout
        )
        new_pose = do_transform_vector3(vector, transform)
        return new_pose

    def transform_quaternion(
        self, target_frame: str, quaternion: QuaternionStamped, timeout: float = 5
    ) -> QuaternionStamped:
        """
        Transforms a quaternion stamped into a different target frame.
        :param target_frame: The target frame.
        :param quaternion: The quaternion to transform.
        :param timeout: The maximum time to wait for the transform.
        :return: The transformed quaternion.
        """
        p = PoseStamped()
        p.header = quaternion.header
        p.pose.orientation = quaternion.quaternion
        new_pose = self.transform_pose(target_frame, p, timeout)
        new_quaternion = QuaternionStamped()
        new_quaternion.header = new_pose.header
        new_quaternion.quaternion = new_pose.pose.orientation
        return new_quaternion

    def transform_point(
        self, target_frame: str, point: PointStamped, timeout: float = 5
    ) -> PointStamped:
        """
        Transforms a point stamped into a different target frame.
        :param target_frame: The target frame.
        :param point: The point to transform.
        :param timeout: The maximum time to wait for the transform.
        :return: The transformed point.
        """
        from tf2_geometry_msgs import do_transform_point

        transform = self.lookup_transform(
            target_frame, point.header.frame_id, point.header.stamp, timeout
        )
        new_pose = do_transform_point(point, transform)
        return new_pose

    def lookup_pose(
        self, target_frame: str, source_frame: str, time: Optional[Time] = None
    ) -> PoseStamped:
        """
        Look up a pose between two frames.
        :param target_frame: The target frame.
        :param source_frame: The source frame.
        :param time: The time at which the transform is requested.
        :return: The pose in the target frame.
        """
        p = PoseStamped()
        p.header.frame_id = str(source_frame)
        if time is not None:
            p.header.stamp = time
        p.pose.orientation.w = 1.0
        return self.transform_pose(target_frame, p)

    def lookup_point(
        self, target_frame: str, source_frame: str, time: Optional[Time] = None
    ) -> PointStamped:
        """
        Look up a point between two frames.
        :param target_frame: The target frame.
        :param source_frame: The source frame.
        :param time: The time at which the transform is requested.
        :return: The point in the target frame.
        """
        t = self.lookup_transform(target_frame, source_frame, time)
        p = PointStamped()
        p.header.frame_id = t.header.frame_id
        p.point.x = t.transform.translation.x
        p.point.y = t.transform.translation.y
        p.point.z = t.transform.translation.z
        return p
