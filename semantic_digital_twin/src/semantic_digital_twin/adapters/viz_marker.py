import atexit
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy.node

from .. import logger
from ..callbacks.callback import StateChangeCallback

try:
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import Vector3, Point, Quaternion, Pose
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Vector3, Point, PoseStamped, Quaternion, Pose
except ImportError as e:
    logger.warning(
        f"Could not import ros messages, viz marker will not be available: {e}"
    )

from scipy.spatial.transform import Rotation

from ..world_description.geometry import (
    FileMesh,
    Box,
    Sphere,
    Cylinder,
    TriangleMesh,
)
from ..world import World


@dataclass
class VizMarkerPublisher(StateChangeCallback):
    """
    Publishes an Array of visualization marker which represent the situation in the World
    """

    node: rclpy.node.Node
    """
    The ROS2 node that will be used to publish the visualization marker.
    """

    topic_name: str = "/semworld/viz_marker"
    """
    The name of the topic to which the Visualization Marker should be published.
    """

    reference_frame: str = "map"
    """
    The reference frame of the visualization marker.
    """

    throttle_state_updates: int = 1
    """
    Only published every n-th state update.
    """

    def __post_init__(self):
        """
        Initializes the publisher and registers the callback to the world.
        """
        super().__post_init__()
        self.pub = self.node.create_publisher(MarkerArray, self.topic_name, 10)
        time.sleep(0.2)
        self.notify()

    def _notify(self):
        """
        Publishes the Marker Array on world changes.
        """
        if self.world.state.version % self.throttle_state_updates != 0:
            return
        marker_array = self._make_marker_array()
        self.pub.publish(marker_array)

    def _make_marker_array(self) -> MarkerArray:
        """
        Creates the Marker Array to be published. There is one Marker for link for each object in the Array, each Object
        creates a name space in the visualization Marker. The type of Visualization Marker is decided by the collision
        tag of the URDF.

        :return: An Array of Visualization Marker
        """
        marker_array = MarkerArray()
        for body in self.world.bodies:
            for i, collision in enumerate(body.collision):
                msg = Marker()
                msg.header.frame_id = self.reference_frame
                msg.ns = body.name.name
                msg.id = i
                msg.action = Marker.ADD
                msg.pose = self.transform_to_pose(
                    (
                        self.world.compute_forward_kinematics(self.world.root, body)
                        @ collision.origin
                    ).to_np()
                )
                msg.color = ColorRGBA(
                    r=float(collision.color.R),
                    g=float(collision.color.G),
                    b=float(collision.color.B),
                    a=float(collision.color.A),
                )
                msg.lifetime = Duration(sec=0)

                if isinstance(collision, FileMesh):
                    msg.type = Marker.MESH_RESOURCE
                    msg.mesh_resource = "file://" + collision.filename
                    msg.scale = Vector3(
                        x=float(collision.scale.x),
                        y=float(collision.scale.y),
                        z=float(collision.scale.z),
                    )
                    msg.mesh_use_embedded_materials = True
                elif isinstance(collision, TriangleMesh):
                    f = collision.file
                    msg.type = Marker.MESH_RESOURCE
                    msg.mesh_resource = "file://" + f.name
                    msg.scale = Vector3(
                        x=float(collision.scale.x),
                        y=float(collision.scale.y),
                        z=float(collision.scale.z),
                    )
                    msg.mesh_use_embedded_materials = True
                elif isinstance(collision, Cylinder):
                    msg.type = Marker.CYLINDER
                    msg.scale = Vector3(
                        x=float(collision.width),
                        y=float(collision.width),
                        z=float(collision.height),
                    )
                elif isinstance(collision, Box):
                    msg.type = Marker.CUBE
                    msg.scale = Vector3(
                        x=float(collision.scale.x),
                        y=float(collision.scale.y),
                        z=float(collision.scale.z),
                    )
                elif isinstance(collision, Sphere):
                    msg.type = Marker.SPHERE
                    msg.scale = Vector3(
                        x=float(collision.radius * 2),
                        y=float(collision.radius * 2),
                        z=float(collision.radius * 2),
                    )

                marker_array.markers.append(msg)
        return marker_array

    @staticmethod
    def transform_to_pose(transform: np.ndarray) -> Pose:
        """
        Converts a 4x4 transformation matrix to a PoseStamped message.

        :param transform: The transformation matrix to convert.
        :return: A PoseStamped message.
        """
        pose = Pose()
        pose.position = Point(**dict(zip(["x", "y", "z"], transform[:3, 3])))
        pose.orientation = Quaternion(
            **dict(
                zip(
                    ["x", "y", "z", "w"],
                    Rotation.from_matrix(transform[:3, :3]).as_quat(),
                )
            )
        )
        return pose
