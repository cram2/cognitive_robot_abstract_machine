from unittest.mock import MagicMock, patch
import pytest
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_py import LookupException

from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    HomogeneousTransformationMatrixToRos2Converter,
)
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    Connection6DoF,
)
from semantic_digital_twin.world_description.world_entity import Body, Region


def test_tf_publisher(rclpy_node, pr2_world_state_reset):
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(
        node=rclpy_node,
        world=pr2_world_state_reset,
    )

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        "pr2/base_footprint",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )
    transform = tf_wrapper.lookup_transform("odom_combined", "pr2/base_footprint")
    odom_combined = pr2_world_state_reset.get_kinematic_structure_entities_by_name(
        "odom_combined"
    )[0]
    base_footprint = pr2_world_state_reset.get_kinematic_structure_entities_by_name(
        "base_footprint"
    )[0]
    fk = pr2_world_state_reset.compute_forward_kinematics(odom_combined, base_footprint)
    transform2 = HomogeneousTransformationMatrixToRos2Converter.convert(fk)
    assert transform.transform == transform2.transform

    tf_wrapper.lookup_transform("odom_combined", "pr2/r_gripper_tool_frame")


def test_tf_publisher_ignore_robot(rclpy_node, pr2_world_setup):
    with pr2_world_setup.modify_world():
        box = Body(name=PrefixedName("box"))
        c = FixedConnection(parent=pr2_world_setup.root, child=box)
        pr2_world_setup.add_connection(c)
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher.create_with_ignore_robot(
        node=rclpy_node,
        robot=pr2_world_setup.get_semantic_annotations_by_type(AbstractRobot)[0],
    )

    assert not tf_wrapper.wait_for_transform(
        "pr2/base_link",
        "pr2/base_footprint",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        "box",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )
    transform = tf_wrapper.lookup_transform("odom_combined", "box")

    odom_combined = pr2_world_setup.get_kinematic_structure_entities_by_name(
        "odom_combined"
    )[0]
    base_footprint = pr2_world_setup.get_kinematic_structure_entities_by_name("box")[0]
    fk = pr2_world_setup.compute_forward_kinematics(odom_combined, base_footprint)
    transform2 = HomogeneousTransformationMatrixToRos2Converter.convert(fk)
    assert transform.transform == transform2.transform


def test_tf_publisher_kitchen(rclpy_node, pr2_apartment_world):
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(
        node=rclpy_node,
        world=pr2_apartment_world,
    )
    VizMarkerPublisher(world=pr2_apartment_world, node=rclpy_node)

    milk = pr2_apartment_world.get_kinematic_structure_entities_by_name("milk.stl")[0]

    assert tf_wrapper.wait_for_transform(
        "apartment/apartment_root",
        "pr2/base_footprint",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        str(milk.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    # delete dofs
    with pr2_apartment_world.modify_world():
        new_connection = FixedConnection(
            parent=milk.parent_kinematic_structure_entity, child=milk
        )
        pr2_apartment_world.remove_connection(milk.parent_connection)
        pr2_apartment_world.add_connection(new_connection)

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        str(milk.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    # add dofs
    with pr2_apartment_world.modify_world():
        new_connection = Connection6DoF.create_with_dofs(
            world=pr2_apartment_world,
            parent=milk.parent_kinematic_structure_entity,
            child=milk,
        )
        pr2_apartment_world.remove_connection(milk.parent_connection)
        pr2_apartment_world.add_connection(new_connection)

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        str(milk.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    # delete body
    with pr2_apartment_world.modify_world():
        pr2_apartment_world.remove_connection(milk.parent_connection)
        pr2_apartment_world.remove_kinematic_structure_entity(milk)

    # make sure wrapper forgets about it too
    tf_wrapper.tf_buffer.clear()
    assert not tf_wrapper.wait_for_transform(
        "odom_combined",
        str(milk.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    # add body back
    with pr2_apartment_world.modify_world():
        new_connection = Connection6DoF.create_with_dofs(
            world=pr2_apartment_world,
            parent=pr2_apartment_world.root,
            child=milk,
        )
        pr2_apartment_world.add_connection(new_connection)

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        str(milk.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )


def test_tf_publisher_with_Regions(rclpy_node, pr2_world_state_reset):
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(
        node=rclpy_node,
        world=pr2_world_state_reset,
    )

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        "pr2/base_footprint",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )
    region = Region(name=PrefixedName("region"))
    connection = FixedConnection(parent=pr2_world_state_reset.root, child=region)
    with pr2_world_state_reset.modify_world():
        pr2_world_state_reset.add_region(region)
        pr2_world_state_reset.add_connection(connection)


def test_empty_world(rclpy_node):
    world = World()

    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(
        node=rclpy_node,
        world=world,
    )
    assert not tf_wrapper.wait_for_transform(
        "muh1",
        "muh2",
        timeout=Duration(seconds=0.5),
        time=Time(),
    )


def test_static_world(rclpy_node):
    world = World()
    with world.modify_world():
        body1 = Body(name=PrefixedName("body1"))
        body2 = Body(name=PrefixedName("body2"))
        body3 = Body(name=PrefixedName("body3"))
        body4 = Body(name=PrefixedName("body4"))
        body5 = Body(name=PrefixedName("body5"))

        body1_C_body2 = FixedConnection(parent=body1, child=body2)
        world.add_connection(body1_C_body2)
        body2_C_body3 = FixedConnection(parent=body2, child=body3)
        world.add_connection(body2_C_body3)

        body1_C_body4 = FixedConnection(parent=body1, child=body4)
        world.add_connection(body1_C_body4)
        body4_C_body5 = FixedConnection(parent=body4, child=body5)
        world.add_connection(body4_C_body5)

    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(
        node=rclpy_node,
        world=world,
        ignored_kinematic_structure_entities={body2, body3, body5},
    )

    # body1 not ignored -> publish
    assert tf_wrapper.wait_for_transform(
        str(body1.name),
        str(body2.name),  # ignored
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    # both parent and child are ignored -> no publish
    assert not tf_wrapper.wait_for_transform(
        str(body2.name),  # ignored
        str(body3.name),  # ignored
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    # both parent and child are not ignored -> publish
    assert tf_wrapper.wait_for_transform(
        str(body1.name),
        str(body4.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    # body4 not ignored -> publish
    assert tf_wrapper.wait_for_transform(
        str(body4.name),
        str(body5.name),  # ignored
        timeout=Duration(seconds=1.0),
        time=Time(),
    )


def test_static_world2(rclpy_node):
    world = World()
    with world.modify_world():
        body1 = Body(name=PrefixedName("body1"))
        body2 = Body(name=PrefixedName("body2"))

        body1_C_body2 = FixedConnection(parent=body1, child=body2)
        world.add_connection(body1_C_body2)

    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(
        node=rclpy_node,
        world=world,
        ignored_kinematic_structure_entities={body2},
    )

    # even though body2 is ignored, there is no other transform for body1, so we need to publish anyway
    assert tf_wrapper.wait_for_transform(
        str(body1.name),
        str(body2.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )


def test_double_tf_publisher(rclpy_node, pr2_world_state_reset):
    tf_publisher = TFPublisher.create_with_ignore_existing_tf(
        node=rclpy_node,
        world=pr2_world_state_reset,
    )
    all_frames = [
        str(kse.name) for kse in pr2_world_state_reset.kinematic_structure_entities
    ]
    # we have to patch this, because the tf wrapper inside the call won't receive any tf messages,
    # because other publisher is not ticked.
    with patch.object(TFWrapper, "get_tf_frames", return_value=all_frames):
        tf_publisher2 = TFPublisher.create_with_ignore_existing_tf(
            node=rclpy_node,
            world=pr2_world_state_reset,
        )
    tf_publisher2.tf_pub.publish = MagicMock()
    with pr2_world_state_reset.modify_world():
        pr2_world_state_reset.state.positions[0] += 0.1

    assert tf_publisher2.tf_pub.publish.called
    published_msg = tf_publisher2.tf_pub.publish.call_args[0][0]
    assert len(published_msg.transforms) == 0
