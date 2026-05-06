import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import (
    Vector3 as RosVector3,
    Pose as RosPose,
    Point as RosPoint,
    Quaternion as RosQuaternion,
)
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import ColorRGBA, Header
from typing_extensions import Any
from visualization_msgs.msg import MarkerArray, Marker

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)


@dataclass
class PosePublisher(ModelChangeCallback):
    pose: HomogeneousTransformationMatrix = field(kw_only=True)
    """
    The pose to publish.
    """
    node: rclpy.node.Node = field(kw_only=True)
    """
    ROS node handle, used to create the publisher.
    """
    lifetime: int = 0
    """
    Lifetime of the PosePublisher and viz marker in seconds. If the lifetime is 0 the marker will stay indefinitely.
    """
    text: str = None
    """
    Text to display at the pose position 
    """
    topic_name: str = "/semworld/viz_marker"
    """
    Topic name to publish the pose marker on.
    """

    publisher: Any = field(init=False)
    """
    Ros publisher for viz marker
    """
    end_time: float = field(init=False)
    """
    End time for this PosePublisher, used for lifetime only if given lifetime is greater than 0
    """
    qos_profile: QoSProfile = field(
        default_factory=lambda: QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
    )
    """QoS profile for the publisher."""

    def _notify(self, *args, **kwargs):
        if self.lifetime > 0 and time.time() >= self.end_time:
            self.pause()
        marker_array = self._create_marker_array()
        self.publisher.publish(marker_array)

    def __post_init__(self):
        if not self._world:
            self._world = self.pose.reference_frame._world
        super().__post_init__()
        self.fixed_frame = str(self._world.root.name)
        self.publisher = self.node.create_publisher(
            MarkerArray, self.topic_name, self.qos_profile
        )
        time.sleep(0.2)
        self.end_time = time.time() + self.lifetime

        self._notify()

    def with_tf_publisher(self):
        """
        Launches a tf publisher in conjunction with the PosePublisher.
        """
        TFPublisher(_world=self._world, node=self.node)

    def _create_marker_array(self) -> MarkerArray:
        """
        Creates a MarkerArray to visualize a Pose in RViz. The pose is visualized as an arrow for each axis to represent
        the position and orientation of the pose.
        """
        marker_array = MarkerArray()
        position = self.pose.to_position().to_np()[:3]
        orientation = self.pose.to_rotation_matrix().to_quaternion().to_np()

        p = RosPose(
            position=RosPoint(**dict(zip(["x", "y", "z"], position.tolist()))),
            orientation=RosQuaternion(
                **dict(zip(["x", "y", "z", "w"], orientation.tolist()))
            ),
        )
        for i in range(3):
            axis = [0.0, 0.0, 0.0]
            axis[i] = 0.5  # Defines the length of the arrow
            color = [0.0, 0.0, 0.0, 1.0]
            color[i] = 1.0

            c = ColorRGBA(**dict(zip(["r", "g", "b", "a"], color)))

            end_point = RosPoint(**dict(zip(["x", "y", "z"], np.array(axis).tolist())))

            marker_array.markers.append(
                self._create_marker(
                    c,
                    i,
                    p,
                    RosPoint(),
                    end_point,
                )
            )
        if self.text:
            marker_array.markers.append(
                Marker(
                    action=Marker.ADD,
                    type=Marker.TEXT_VIEW_FACING,
                    text=self.text,
                    ns=f"pose/{self.pose.reference_frame.name}/{id(self)}",
                    id=4,
                    frame_locked=True,
                    pose=p,
                    scale=RosVector3(z=0.1),
                    lifetime=Duration(
                        sec=(
                            round(self.end_time - time.time())
                            if self.lifetime > 0
                            else 0
                        )
                    ),
                    color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
                    header=Header(
                        frame_id=str(self.pose.reference_frame.name),
                    ),
                )
            )
        return marker_array

    def _create_marker(
        self,
        color: ColorRGBA,
        _id: int,
        pose: RosPose,
        start_point: RosPoint,
        end_point: RosPoint,
    ) -> Marker:
        """
        Creates a visualization marker for one axis of the pose.
        :param color: The color of the axis.
        :param _id: The id of the axis to identify the arrow.
        :param pose: The pose to publish
        :param start_point: The start point of the arrow.
        :param end_point: The end point of the arrow.
        """
        m = Marker()
        m.action = Marker.ADD
        m.type = Marker.ARROW
        m.id = _id
        m.header.frame_id = str(self.pose.reference_frame.name)
        m.pose = pose
        m.lifetime = Duration(
            sec=round(self.end_time - time.time()) if self.lifetime > 0 else 0
        )
        m.points = [start_point, end_point]

        m.scale = RosVector3(x=0.025, y=0.05, z=0.1)
        m.color = color
        m.ns = f"pose/{self.pose.reference_frame.name}/{id(self)}"
        m.frame_locked = True

        return m

    def __hash__(self):
        return hash(id(self))
