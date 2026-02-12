import time
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from builtin_interfaces.msg import Duration
import numpy as np
import rclpy
from std_msgs.msg import ColorRGBA, Header
from typing_extensions import List, Any, Dict
from visualization_msgs.msg import MarkerArray, Marker

from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from geometry_msgs.msg import (
    Vector3,
    Pose,
    Point,
    Quaternion,
)


@dataclass
class PosePublisher(StateChangeCallback):
    pose: HomogeneousTransformationMatrix
    """
    The pose to publish.
    """
    node: rclpy.node.Node
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
    _ns: UUID = field(init=False, default_factory=uuid4)
    """
    Unique id to differentiate the different pose marker
    """
    end_time: float = field(init=False)
    """
    End time for this PosePublisher, used for lifetime only if given lifetime is greater than 0
    """
    fixed_frame: str = field(init=False)
    """
    The frame in which the marker are published, is set to world root
    """

    def _notify(self):
        if self.lifetime > 0 and time.time() >= self.end_time:
            self.pause()
        world = self.pose.reference_frame._world
        global_pose = world.transform(self.pose, self.world.root)
        marker_array = self._create_marker_array(global_pose)
        self.publisher.publish(marker_array)

    def __post_init__(self):
        self.fixed_frame = str(self.world.root.name)
        self.publisher = self.node.create_publisher(MarkerArray, self.topic_name, 10)
        time.sleep(0.2)
        self.end_time = time.time() + self.lifetime
        self._notify()

    def _create_marker_array(
        self, global_pose: HomogeneousTransformationMatrix
    ) -> MarkerArray:
        """
        Creates a MarkerArray to visualize a Pose in RViz. The pose is visualized as an arrow for each axis to represent
        the position and orientation of the pose.
        :param global_pose: The pose to in global frame
        """
        marker_array = MarkerArray()
        position = global_pose.to_position().to_np()[:3]
        orientation = global_pose.to_rotation_matrix().to_quaternion().to_np()

        p = Pose(
            position=Point(**dict(zip(["x", "y", "z"], position.tolist()))),
            orientation=Quaternion(
                **dict(zip(["x", "y", "z", "w"], orientation.tolist()))
            ),
        )
        for i in range(3):
            axis = [0.0, 0.0, 0.0]
            axis[i] = 0.5  # Defines the length of the arrow
            color = [0.0, 0.0, 0.0, 1.0]
            color[i] = 1.0

            c = ColorRGBA(**dict(zip(["r", "g", "b", "a"], color)))

            end_point = Point(**dict(zip(["x", "y", "z"], np.array(axis).tolist())))

            marker_array.markers.append(
                self._create_marker(
                    c,
                    i,
                    p,
                    Point(),
                    end_point,
                )
            )
        if self.text:
            marker_array.markers.append(
                Marker(
                    action=Marker.ADD,
                    type=Marker.TEXT_VIEW_FACING,
                    text=self.text,
                    ns=str(self._ns),
                    id=4,
                    pose=p,
                    scale=Vector3(z=0.1),
                    lifetime=Duration(
                        sec=(
                            round(self.end_time - time.time())
                            if self.lifetime > 0
                            else 0
                        )
                    ),
                    color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
                    header=Header(
                        frame_id=self.fixed_frame,
                    ),
                )
            )
        return marker_array

    def _create_marker(
        self,
        color: ColorRGBA,
        _id: int,
        pose: Pose,
        start_point: Point,
        end_point: Point,
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
        m.header.frame_id = self.fixed_frame
        m.pose = pose
        m.lifetime = Duration(
            sec=round(self.end_time - time.time()) if self.lifetime > 0 else 0
        )
        m.points = [start_point, end_point]

        m.scale = Vector3(x=0.025, y=0.05, z=0.1)
        m.color = color
        m.ns = str(self._ns)

        return m
