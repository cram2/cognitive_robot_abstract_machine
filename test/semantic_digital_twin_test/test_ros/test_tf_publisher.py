from copy import deepcopy
from unittest.mock import MagicMock, patch
import pytest
from typing_extensions import List, Set, Tuple
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_py import LookupException

from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    HomogeneousTransformationMatrixToRos2Converter,
)
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher, TfFrameNames
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
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
        _world=pr2_world_state_reset,
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


def test_clear(rclpy_node, pr2_world_copy):
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(
        node=rclpy_node,
        _world=pr2_world_copy,
    )

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        "pr2/base_footprint",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    world_copy = deepcopy(pr2_world_copy)
    with pr2_world_copy.modify_world():
        pr2_world_copy.clear()
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(
        node=rclpy_node,
        _world=pr2_world_copy,
    )

    with pr2_world_copy.modify_world():
        pr2_world_copy.merge_world(world_copy)

    tf_wrapper.tf_buffer.clear()
    pr2_world_copy.notify_state_change()

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        "pr2/base_footprint",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )


def test_tf_publisher_ignore_robot(rclpy_node, pr2_world_copy):
    with pr2_world_copy.modify_world():
        box = Body(name=PrefixedName("box"))
        c = FixedConnection(parent=pr2_world_copy.root, child=box)
        pr2_world_copy.add_connection(c)
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher.create_with_ignore_robot(
        node=rclpy_node,
        robot=pr2_world_copy.get_semantic_annotations_by_type(AbstractRobot)[0],
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

    odom_combined = pr2_world_copy.get_kinematic_structure_entities_by_name(
        "odom_combined"
    )[0]
    base_footprint = pr2_world_copy.get_kinematic_structure_entities_by_name("box")[0]
    fk = pr2_world_copy.compute_forward_kinematics(odom_combined, base_footprint)
    transform2 = HomogeneousTransformationMatrixToRos2Converter.convert(fk)
    assert transform.transform == transform2.transform


def test_tf_publisher_kitchen(rclpy_node, pr2_apartment_world, ros_publishers):
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = ros_publishers.adopt(
        TFPublisher(
            node=rclpy_node,
            _world=pr2_apartment_world,
        )
    )
    ros_publishers.adopt(
        VizMarkerPublisher(_world=pr2_apartment_world, node=rclpy_node)
    )

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
        _world=pr2_world_state_reset,
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
        _world=world,
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
        _world=world,
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
        _world=world,
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


# %% tf frame naming


def world_with_bodies(*names: str) -> Tuple[World, List[Body]]:
    """
    Build a world whose root carries one body per given name.
    """
    world = World()
    root = Body(name=PrefixedName("map"))
    bodies = [Body(name=PrefixedName(name)) for name in names]
    with world.modify_world():
        world.add_body(root)
        for body in bodies:
            world.add_connection(FixedConnection(parent=root, child=body))
    return world, bodies


def attach_body(world: World, name: str) -> Body:
    body = Body(name=PrefixedName(name))
    with world.modify_world():
        world.add_connection(FixedConnection(parent=world.root, child=body))
    return body


def test_frame_is_named_after_the_entity():
    world, (milk,) = world_with_bodies("milk")

    assert TfFrameNames().assign(milk) == "milk"


def test_first_of_two_equally_named_bodies_keeps_the_plain_frame_name():
    world, (first, second) = world_with_bodies("milk", "milk")
    frame_names = TfFrameNames()

    assert frame_names.assign(first) == "milk"
    assert frame_names.assign(second) == f"milk_{second.id.hex}"


def test_frame_names_are_reused_on_every_republish():
    world, (first, second) = world_with_bodies("milk", "milk")
    frame_names = TfFrameNames()
    published = [frame_names.assign(first), frame_names.assign(second)]

    assert [frame_names.assign(first), frame_names.assign(second)] == published


def test_a_body_arriving_later_does_not_rename_the_bodies_already_published():
    world, (first,) = world_with_bodies("milk")
    frame_names = TfFrameNames()
    published = frame_names.assign(first)

    frame_names.assign(attach_body(world, "milk"))

    assert frame_names.assign(first) == published


def test_removing_a_body_leaves_the_frame_name_of_the_other_one_alone():
    world, (first, second) = world_with_bodies("milk", "milk")
    frame_names = TfFrameNames()
    frame_names.assign(first)
    published = frame_names.assign(second)

    with world.modify_world():
        world.remove_kinematic_structure_entity(first)

    assert frame_names.assign(second) == published


def published_child_frames(tf_publisher: TFPublisher) -> Set[str]:
    """
    The tf frames the publisher broadcasts a transform for.
    """
    return {
        transform.child_frame_id
        for transform in tf_publisher.tf_model_callback.tf_message.transforms
    }


def publisher_ignoring_existing_frames(
    world: World, node: Node, existing_frames: List[str]
) -> TFPublisher:
    """
    Build a publisher for a world whose tf tree already carries the given frames.
    """
    with patch.object(TFWrapper, "get_tf_frames", return_value=existing_frames):
        return TFPublisher.create_with_ignore_existing_tf(node=node, world=world)


def test_a_body_another_publisher_broadcasts_keeps_that_frame_name(rclpy_node):
    world, (odom, milk) = world_with_bodies("odom_combined", "milk")

    tf_publisher = publisher_ignoring_existing_frames(
        world, rclpy_node, [str(odom.name)]
    )

    assert published_child_frames(tf_publisher) == {str(odom.name), str(milk.name)}


def test_two_equally_named_bodies_another_publisher_broadcasts_get_a_frame_each(
    rclpy_node,
):
    world, (first, second) = world_with_bodies("milk", "milk")

    tf_publisher = publisher_ignoring_existing_frames(
        world, rclpy_node, [str(first.name)]
    )

    assert published_child_frames(tf_publisher) == {
        str(first.name),
        f"{second.name}_{second.id.hex}",
    }


def test_two_equally_named_bodies_reach_the_tree_as_two_frames(rclpy_node):
    world, (first, second) = world_with_bodies("milk", "milk")
    tf_wrapper = TFWrapper(node=rclpy_node)
    TFPublisher(node=rclpy_node, _world=world)

    assert tf_wrapper.wait_for_transform(
        "map", "milk", timeout=Duration(seconds=1.0), time=Time()
    )
    published_frames = set(tf_wrapper.get_tf_frames())
    told_apart_by_identifier = {
        f"milk_{first.id.hex}",
        f"milk_{second.id.hex}",
    }

    assert len(published_frames & told_apart_by_identifier) == 1
