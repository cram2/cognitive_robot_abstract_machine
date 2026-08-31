import os
from dataclasses import dataclass, field
from time import sleep

import pytest
from rclpy.duration import Duration
from rclpy.time import Time
from typing_extensions import List, Set, Tuple
from visualization_msgs.msg import MarkerArray, Marker

from .test_tf_publisher import publisher_ignoring_existing_frames
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
    ShapeSource,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import WorldHasMultipleTfPublishersError
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Mesh, Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Region


@dataclass
class Callback:
    last_msg: MarkerArray = field(init=False, default=None)

    def __call__(self, msg: MarkerArray):
        self.last_msg = msg


def test_visualization_marker(rclpy_node, cylinder_bot_world):
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(node=rclpy_node, _world=cylinder_bot_world)
    viz = VizMarkerPublisher(
        _world=cylinder_bot_world,
        node=rclpy_node,
        shape_source=ShapeSource.COLLISION_ONLY,
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
    assert len(callback.last_msg.markers) == 3
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
    tf_publisher = TFPublisher(node=rclpy_node, _world=pr2_world_state_reset)
    viz = VizMarkerPublisher(
        _world=pr2_world_state_reset,
        node=rclpy_node,
        shape_source=ShapeSource.COLLISION_ONLY,
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
    assert len(callback.last_msg.markers) == 53


def test_visualization_marker_tracy(rclpy_node, tracy_world, ros_publishers):
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = ros_publishers.adopt(
        TFPublisher(node=rclpy_node, _world=tracy_world)
    )
    viz = ros_publishers.adopt(VizMarkerPublisher(_world=tracy_world, node=rclpy_node))

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


def test_visualization_marker_regions_are_transparent(rclpy_node, cylinder_bot_world):
    region = Region(
        name=PrefixedName("region"),
        area=ShapeCollection(
            shapes=[Box(scale=Scale(0.1, 0.1, 0.1), color=Color(A=1.0))]
        ),
    )
    connection = FixedConnection(parent=cylinder_bot_world.root, child=region)
    with cylinder_bot_world.modify_world():
        cylinder_bot_world.add_region(region)
        cylinder_bot_world.add_connection(connection)

    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(node=rclpy_node, _world=cylinder_bot_world)
    viz = VizMarkerPublisher(
        _world=cylinder_bot_world,
        node=rclpy_node,
        shape_source=ShapeSource.COLLISION_ONLY,
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

    # the shape's own color is fully opaque, but region markers must always be
    # forced transparent regardless of that
    for marker in callback.last_msg.markers:
        if marker.ns == str(region.name):
            assert marker.color.a == pytest.approx(viz.region_alpha)
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
    visual: Mesh = world.root.visual.shapes[0]
    world.root.visual.shapes[0] = visual
    with world.modify_world():
        body2 = Body(name=PrefixedName("body2"))
        body_C_body2 = FixedConnection(parent=world.root, child=body2)
        world.add_connection(body_C_body2)
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(node=rclpy_node, _world=world)
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)

    assert tf_wrapper.wait_for_transform(
        str(world.root.name),
        str(body2.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )


# %% marker frames


def world_with_visible_bodies(*names: str) -> Tuple[World, List[Body]]:
    """
    Build a world whose root carries one body with a visual shape per given name.
    """
    world = World()
    root = Body(name=PrefixedName("map"))
    bodies = [
        Body(
            name=PrefixedName(name),
            visual=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
        )
        for name in names
    ]
    with world.modify_world():
        world.add_body(root)
        for body in bodies:
            world.add_connection(FixedConnection(parent=root, child=body))
    return world, bodies


def published_child_frames(tf_publisher: TFPublisher) -> Set[str]:
    """
    The tf frames the publisher broadcasts a transform for.
    """
    return {
        transform.child_frame_id
        for transform in tf_publisher.tf_model_callback.tf_message.transforms
    }


def marker_frames(viz: VizMarkerPublisher) -> Set[str]:
    """
    The tf frames the markers are positioned in.
    """
    return {marker.header.frame_id for marker in viz.markers.markers}


def test_markers_are_stamped_with_the_frame_names_the_tf_publisher_publishes(
    rclpy_node,
):
    world, _ = world_with_visible_bodies("milk", "milk")
    tf_publisher = TFPublisher(node=rclpy_node, _world=world)

    viz = VizMarkerPublisher(_world=world, node=rclpy_node)

    assert marker_frames(viz) == published_child_frames(tf_publisher)


def test_the_tf_publisher_of_the_world_is_adopted(rclpy_node):
    world, _ = world_with_visible_bodies("milk")
    tf_publisher = TFPublisher(node=rclpy_node, _world=world)

    viz = VizMarkerPublisher(_world=world, node=rclpy_node)

    assert viz.tf_publisher is tf_publisher


def test_a_tf_publisher_is_started_while_nothing_publishes_tf(rclpy_node):
    world, _ = world_with_visible_bodies("milk")

    viz = VizMarkerPublisher(_world=world, node=rclpy_node)

    assert TFPublisher.all_callbacks_of_this_type_from_world(world) == [
        viz.tf_publisher
    ]


def test_the_given_tf_publisher_is_kept(rclpy_node):
    world, _ = world_with_visible_bodies("milk")
    tf_publisher = TFPublisher(node=rclpy_node, _world=world)

    viz = VizMarkerPublisher(_world=world, node=rclpy_node, tf_publisher=tf_publisher)

    assert viz.tf_publisher is tf_publisher


def test_markers_refuse_to_choose_between_several_tf_publishers(rclpy_node):
    world, _ = world_with_visible_bodies("milk")
    TFPublisher(node=rclpy_node, _world=world)
    TFPublisher(node=rclpy_node, _world=world)

    with pytest.raises(WorldHasMultipleTfPublishersError):
        VizMarkerPublisher(_world=world, node=rclpy_node)


def test_markers_are_stamped_by_the_tf_publisher_the_marker_publisher_starts(
    rclpy_node,
):
    """
    The connections run against the order the bodies were added, so both publishers pick
    a different body for the plain name unless they share one assignment.
    """
    world = World()
    root = Body(name=PrefixedName("map"))
    bodies = [
        Body(
            name=PrefixedName("milk"),
            visual=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
        )
        for _ in range(2)
    ]
    with world.modify_world():
        world.add_body(root)
        for body in bodies:
            world.add_body(body)
        for body in reversed(bodies):
            world.add_connection(FixedConnection(parent=root, child=body))

    viz = VizMarkerPublisher(_world=world, node=rclpy_node)

    assert marker_frames(viz) == published_child_frames(viz.tf_publisher)


def test_a_body_added_later_reaches_both_publishers_under_one_name(rclpy_node):
    world, _ = world_with_visible_bodies("milk")
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)

    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=Body(
                    name=PrefixedName("milk"),
                    visual=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
                ),
            )
        )

    assert marker_frames(viz) == published_child_frames(viz.tf_publisher)


def test_region_markers_are_stamped_like_the_frames_they_share_a_name_with(rclpy_node):
    world, _ = world_with_visible_bodies("table")
    region = Region(
        name=PrefixedName("table"),
        area=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
    )
    with world.modify_world():
        world.add_region(region)
        world.add_connection(FixedConnection(parent=world.root, child=region))

    viz = VizMarkerPublisher(_world=world, node=rclpy_node)

    assert marker_frames(viz) == published_child_frames(viz.tf_publisher)


def test_markers_of_bodies_another_publisher_broadcasts_keep_its_frame_names(
    rclpy_node,
):
    world, (odom, milk) = world_with_visible_bodies("odom_combined", "milk")
    publisher_ignoring_existing_frames(world, rclpy_node, [str(odom.name)])

    viz = VizMarkerPublisher(_world=world, node=rclpy_node)

    assert marker_frames(viz) == {str(odom.name), str(milk.name)}


# %% publisher lifetime


def test_a_stopped_marker_publisher_is_no_longer_notified(rclpy_node):
    world, _ = world_with_visible_bodies("milk")
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)

    viz.stop()

    assert VizMarkerPublisher.all_callbacks_of_this_type_from_world(world) == []


def test_stopping_the_marker_publisher_stops_the_tf_publisher_it_started(rclpy_node):
    world, _ = world_with_visible_bodies("milk")
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)

    viz.stop()

    assert TFPublisher.all_callbacks_of_this_type_from_world(world) == []


def test_stopping_the_marker_publisher_keeps_the_tf_publisher_it_was_given(rclpy_node):
    world, _ = world_with_visible_bodies("milk")
    tf_publisher = TFPublisher(node=rclpy_node, _world=world)
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)

    viz.stop()

    assert TFPublisher.all_callbacks_of_this_type_from_world(world) == [tf_publisher]
