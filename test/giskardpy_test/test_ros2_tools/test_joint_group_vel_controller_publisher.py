from dataclasses import dataclass, field
from typing import List, Optional

import pytest
from py_trees.common import Status
from std_msgs.msg import Float64MultiArray

from giskardpy.tree.behaviors.joint_group_vel_controller_publisher import (
    JointGroupVelController,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class RecordingPublisher:
    """
    Stand-in for a ROS publisher that stores the last published message.
    """

    published_message: Optional[Float64MultiArray] = None
    """
    The most recently published message, or ``None`` if nothing was published.
    """

    def publish(self, message: Float64MultiArray) -> None:
        self.published_message = message


@dataclass
class ConnectionSpec:
    """
    Describes a connection to be built for the controller test.
    """

    name: str
    """
    The connection name; ``"finger"`` substrings opt the joint out of clamping.
    """

    velocity: float
    """
    The commanded velocity written to the world state before ticking.
    """

    connection_type: type = RevoluteConnection
    """
    The concrete connection class to instantiate.
    """


def build_controller(
    specs: List[ConnectionSpec], minimum_valid_velocity: float = 0.0
) -> JointGroupVelController:
    """
    Build a controller over a chain of freshly created connections.
    """
    world = World()
    connections: List[ActiveConnection1DOF] = []
    with world.modify_world():
        parent = Body(name=PrefixedName("base"))
        for index, spec in enumerate(specs):
            child = Body(name=PrefixedName(f"body_{index}"))
            connection = spec.connection_type.create_with_dofs(
                world=world,
                parent=parent,
                child=child,
                axis=Vector3.Z(),
                name=PrefixedName(spec.name),
            )
            world.add_connection(connection)
            connections.append(connection)
            parent = child
    for spec, connection in zip(specs, connections):
        connection.velocity = spec.velocity
    return JointGroupVelController(
        cmd_topic="test_cmd",
        connections=connections,
        minimum_valid_velocity=minimum_valid_velocity,
    )


def tick(controller: JointGroupVelController) -> List[float]:
    """
    Replace the publisher with a recorder, tick once, and return the data.
    """
    recorder = RecordingPublisher()
    controller.cmd_pub = recorder
    assert controller.update() == Status.RUNNING
    return list(recorder.published_message.data)


def test_publishes_one_value_per_connection(init_rospy):
    specs = [
        ConnectionSpec(name="joint_a", velocity=0.1),
        ConnectionSpec(name="joint_b", velocity=0.2),
        ConnectionSpec(name="joint_c", velocity=0.3),
    ]
    controller = build_controller(specs)

    data = tick(controller)

    assert len(data) == len(specs)


def test_below_threshold_velocity_is_raised_to_minimum(init_rospy):
    specs = [
        ConnectionSpec(name="slow_positive", velocity=0.01),
        ConnectionSpec(name="slow_negative", velocity=-0.01),
    ]
    controller = build_controller(specs, minimum_valid_velocity=0.03)

    data = tick(controller)

    assert data[0] == pytest.approx(0.03)
    assert data[1] == pytest.approx(-0.03)


def test_default_minimum_valid_velocity_does_not_clamp(init_rospy):
    specs = [
        ConnectionSpec(name="slow_positive", velocity=0.01),
        ConnectionSpec(name="slow_negative", velocity=-0.01),
    ]
    controller = build_controller(specs)

    data = tick(controller)

    assert data[0] == pytest.approx(0.01)
    assert data[1] == pytest.approx(-0.01)


def test_velocities_outside_threshold_are_unchanged(init_rospy):
    specs = [
        ConnectionSpec(name="fast", velocity=0.1),
        ConnectionSpec(name="stopped", velocity=0.0),
    ]
    controller = build_controller(specs)

    data = tick(controller)

    assert data[0] == pytest.approx(0.1)
    assert data[1] == pytest.approx(0.0)


def test_prismatic_and_finger_joints_are_not_clamped(init_rospy):
    specs = [
        ConnectionSpec(
            name="prismatic_joint",
            velocity=0.01,
            connection_type=PrismaticConnection,
        ),
        ConnectionSpec(name="gripper_finger_joint", velocity=0.01),
    ]
    controller = build_controller(specs, minimum_valid_velocity=0.03)

    data = tick(controller)

    assert data[0] == pytest.approx(0.01)
    assert data[1] == pytest.approx(0.01)
