import os
from dataclasses import dataclass, field
from time import sleep

from rclpy.duration import Duration
from rclpy.time import Time
from visualization_msgs.msg import MarkerArray, Marker

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import FileMesh
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class Callback:
    last_msg: MarkerArray = field(init=False, default=None)

    def __call__(self, msg: MarkerArray):
        self.last_msg = msg


def test_visualization_marker(rclpy_node, cylinder_bot_world):
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(node=rclpy_node, world=cylinder_bot_world)
    viz = VizMarkerPublisher(
        world=cylinder_bot_world, node=rclpy_node, use_visuals=False
    )
    tf_wrapper.wait_for_transform(
        "map",
        "bot",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    callback = Callback()

    sub = rclpy_node.create_subscription(
        msg_type=MarkerArray,
        topic=viz.topic_name,
        callback=callback,
        qos_profile=viz.qos_profile,
    )
    for i in range(30):
        if callback.last_msg is not None:
            break
        sleep(0.1)
    else:
        assert False, "Callback timed out"
    assert len(callback.last_msg.markers) == 2
    assert callback.last_msg.markers[0].ns == "environment"
    assert callback.last_msg.markers[0].type == Marker.CYLINDER

    callback.last_msg = None

    drive = cylinder_bot_world.get_connections_by_type(OmniDrive)[0]
    new_pose = HomogeneousTransformationMatrix.from_xyz_rpy(1, 1)
    drive.origin = new_pose

    for i in range(30):
        transform = tf_wrapper.lookup_transform("map", "bot")
        sleep(0.1)
        if (
            transform.transform.translation.x == 1
            and transform.transform.translation.y == 1
        ):
            break
    else:
        assert False, "TF lookup timed out"
    assert callback.last_msg is None


def test_visualization_marker_pr2(rclpy_node, pr2_world_state_reset):
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(node=rclpy_node, world=pr2_world_state_reset)
    viz = VizMarkerPublisher(
        world=pr2_world_state_reset, node=rclpy_node, use_visuals=False
    )
    tf_wrapper.wait_for_transform(
        "odom_combined",
        "base_footprint",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    callback = Callback()

    sub = rclpy_node.create_subscription(
        msg_type=MarkerArray,
        topic=viz.topic_name,
        callback=callback,
        qos_profile=viz.qos_profile,
    )
    for i in range(30):
        if callback.last_msg is not None:
            break
        sleep(0.1)
    else:
        assert False, "Callback timed out"
    assert len(callback.last_msg.markers) == 54


def test_visualization_marker_tracy(rclpy_node, tracy_world):
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(node=rclpy_node, world=tracy_world)
    viz = VizMarkerPublisher(world=tracy_world, node=rclpy_node, use_visuals=True)

    callback = Callback()

    sub = rclpy_node.create_subscription(
        msg_type=MarkerArray,
        topic=viz.topic_name,
        callback=callback,
        qos_profile=viz.qos_profile,
    )
    for i in range(30):
        if callback.last_msg is not None:
            break
        sleep(0.1)
    else:
        assert False, "Callback timed out"

    # table has no texture, so white should be used
    for marker in callback.last_msg.markers:
        if marker.ns == str(tracy_world.get_body_by_name("table").name):
            assert marker.color.r == 1.0
            assert marker.color.g == 1.0
            assert marker.color.b == 1.0
            assert marker.color.a == 1.0
            break
    else:
        assert False, "Marker not found"

    # ur5 has texture, 0,0,0,0 must be used for correct visualization
    for marker in callback.last_msg.markers:
        if marker.ns == str(tracy_world.get_body_by_name("left_forearm_link").name):
            assert marker.color.r == 0.0
            assert marker.color.g == 0.0
            assert marker.color.b == 0.0
            assert marker.color.a == 0.0
            break
    else:
        assert False, "Marker not found"


def test_trimesh(rclpy_node):
    world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "semantic_digital_twin",
            "resources",
            "stl",
            "milk.stl",
        )
    ).parse()
    visual: FileMesh = world.root.visual.shapes[0]
    world.root.visual.shapes[0] = visual.to_triangle_mesh()
    with world.modify_world():
        body2 = Body(name=PrefixedName("body2"))
        body_C_body2 = FixedConnection(parent=world.root, child=body2)
        world.add_connection(body_C_body2)
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(node=rclpy_node, world=world)
    viz = VizMarkerPublisher(world=world, node=rclpy_node, use_visuals=True)

    assert tf_wrapper.wait_for_transform(
        str(world.root.name),
        str(body2.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )
