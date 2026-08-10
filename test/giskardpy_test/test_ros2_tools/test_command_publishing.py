from copy import deepcopy
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pytest
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from giskardpy.middleware.ros2.command_publishing import (
    DriveVelocityCommandPublisher,
    JointGroupVelocityCommandPublisher,
    JointVelocityCommandPublisher,
    JointMinimumVelocities,
    MinimumVelocity,
)
from giskardpy.middleware.ros2.exceptions import UnknownMinimumVelocityJointError
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    DifferentialDrive,
    OmniDrive,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% test doubles


@dataclass
class RecordingPublisher:
    """
    Stand-in for a ROS publisher that stores every published message.

    Messages are copied, so that publishers which reuse one message object across cycles
    cannot rewrite what was already recorded.
    """

    published_messages: List[Any] = field(default_factory=list)
    """
    Copies of all published messages, oldest first.
    """

    @property
    def published_message(self) -> Optional[Any]:
        """
        The most recently published message, or ``None`` if nothing was published.
        """
        if not self.published_messages:
            return None
        return self.published_messages[-1]

    def publish(self, message: Any) -> None:
        self.published_messages.append(deepcopy(message))


@dataclass
class ParameterServingNode:
    """
    Stand-in for a velocity controller that offers the ``joint`` parameter Giskard reads
    to learn which connection a namespace controls.
    """

    controller_name: str
    """
    Name of the node, which doubles as the namespace of its parameter service.
    """

    joint_name: str
    """
    Value served for the ``joint`` parameter.
    """

    node: Node = field(init=False)
    """
    The node offering the parameter service.
    """

    executor: SingleThreadedExecutor = field(init=False)
    """
    Executor spinning :attr:`node`.
    """

    spinner: Thread = field(init=False)
    """
    Thread running :attr:`executor`.
    """

    def __post_init__(self):
        self.node = Node(self.controller_name)
        self.node.declare_parameter("joint", self.joint_name)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spinner = Thread(
            target=self.executor.spin, daemon=True, name=f"{self.controller_name} spin"
        )
        self.spinner.start()

    def shutdown(self) -> None:
        """
        Stop spinning and destroy the node.
        """
        self.executor.shutdown()
        self.spinner.join(2.0)
        self.node.destroy_node()


@dataclass
class ConnectionSpec:
    """
    Describes a connection to be built for a publisher test.
    """

    name: str
    """
    The connection name.
    """

    velocity: float
    """
    The commanded velocity written to the world state before publishing.
    """

    multiplier: float = 1.0
    """
    Scaling between the world state value and the value of the connection.
    """

    connection_type: Type[ActiveConnection1DOF] = RevoluteConnection
    """
    The concrete connection class to instantiate.
    """


def add_joints(world: World, parent: Body, specs: List[ConnectionSpec]) -> None:
    """
    Append a chain of connections to a body and apply their commanded velocities.
    """
    connections: List[ActiveConnection1DOF] = []
    with world.modify_world():
        for index, spec in enumerate(specs):
            child = Body(name=PrefixedName(f"body_{index}"))
            connection = spec.connection_type.create_with_dofs(
                world=world,
                parent=parent,
                child=child,
                axis=Vector3.Z(),
                name=PrefixedName(spec.name),
                multiplier=spec.multiplier,
            )
            world.add_connection(connection)
            connections.append(connection)
            parent = child
    for spec, connection in zip(specs, connections):
        connection.velocity = spec.velocity


def build_world_with_joints(specs: List[ConnectionSpec]) -> World:
    """
    Build a world holding a chain of connections with their commanded velocities.
    """
    world = World()
    add_joints(world, Body(name=PrefixedName("base")), specs)
    return world


def add_removable_joint(world: World, parent: Body) -> ActiveConnection1DOF:
    """
    Add a joint on a branch of its own.

    Removing it re-lays-out the world state, which shifts the state columns of every
    degree of freedom that was added afterwards.
    """
    with world.modify_world():
        connection = RevoluteConnection.create_with_dofs(
            world=world,
            parent=parent,
            child=Body(name=PrefixedName("removable_body")),
            axis=Vector3.Z(),
            name=PrefixedName("removable_joint"),
        )
        world.add_connection(connection)
    return connection


def remove_joint(world: World, connection: ActiveConnection1DOF) -> None:
    """
    Remove a joint, its child body and its degree of freedom from the world.
    """
    with world.modify_world():
        world.remove_connection(connection)
        world.remove_kinematic_structure_entity(connection.child)
        world.remove_degree_of_freedom(connection.raw_dof)


def build_group_publisher(
    specs: List[ConnectionSpec],
    minimum_valid_velocity: float = 0.0,
    minimum_velocity_overrides: Optional[Dict[str, float]] = None,
) -> JointGroupVelocityCommandPublisher:
    """
    Build a group publisher over a chain of freshly created connections.
    """
    return group_publisher_for(
        build_world_with_joints(specs),
        specs,
        minimum_valid_velocity,
        minimum_velocity_overrides,
    )


def group_publisher_for(
    world: World,
    specs: List[ConnectionSpec],
    minimum_valid_velocity: float = 0.0,
    minimum_velocity_overrides: Optional[Dict[str, float]] = None,
) -> JointGroupVelocityCommandPublisher:
    """
    Build a group publisher over connections that already exist in a world.
    """
    return JointGroupVelocityCommandPublisher(
        world=world,
        command_topic="test_cmd",
        connections=[world.get_connection_by_name(spec.name) for spec in specs],
        minimum_velocities=JointMinimumVelocities.from_magnitudes(
            minimum_valid_velocity, minimum_velocity_overrides
        ),
    )


def publish_group(controller: JointGroupVelocityCommandPublisher) -> List[float]:
    """
    Replace the publisher with a recorder, publish once, and return the data.
    """
    recorder = RecordingPublisher()
    controller.command_publisher = recorder
    controller.publish()
    return list(recorder.published_message.data)


def publish_drive(publisher: DriveVelocityCommandPublisher) -> RecordingPublisher:
    """
    Replace the publisher with a recorder and publish one twist.
    """
    recorder = RecordingPublisher()
    publisher.velocity_publisher = recorder
    publisher.publish()
    return recorder


def add_drive(
    world: World,
    parent: Body,
    drive_type: Union[Type[OmniDrive], Type[DifferentialDrive]],
) -> None:
    """
    Append a drive to a body.
    """
    with world.modify_world():
        drive = drive_type.create_with_dofs(
            world=world,
            parent=parent,
            child=Body(name=PrefixedName("base_footprint")),
            name=PrefixedName("brumbrum"),
        )
        world.add_connection(drive)


def build_drive_world(
    drive_type: Union[Type[OmniDrive], Type[DifferentialDrive]],
) -> World:
    """
    Build a world whose only connection is a drive.
    """
    world = World()
    add_drive(world, Body(name=PrefixedName("odom")), drive_type)
    return world


# %% minimum velocity math


def test_scalar_below_minimum_keeps_its_sign():
    minimum_velocity = MinimumVelocity(0.03)

    assert minimum_velocity.enforce_on_scalar(0.01) == pytest.approx(0.03)
    assert minimum_velocity.enforce_on_scalar(-0.01) == pytest.approx(-0.03)


def test_scalar_at_zero_stays_zero():
    assert MinimumVelocity(0.03).enforce_on_scalar(0.0) == pytest.approx(0.0)


def test_scalar_above_minimum_is_unchanged():
    assert MinimumVelocity(0.03).enforce_on_scalar(0.1) == pytest.approx(0.1)


def test_zero_magnitude_disables_raising():
    assert MinimumVelocity().enforce_on_scalar(0.001) == pytest.approx(0.001)
    assert MinimumVelocity().enforce_on_vector([0.001, 0.001]) == pytest.approx(
        [0.001, 0.001]
    )


def test_vector_below_minimum_is_scaled_to_the_minimum_norm():
    raised = MinimumVelocity(0.05).enforce_on_vector([0.02, 0.02])

    assert raised[0] == pytest.approx(raised[1])
    assert np.linalg.norm(raised) == pytest.approx(0.05)


def test_vector_below_minimum_keeps_its_direction():
    raised = MinimumVelocity(0.05).enforce_on_vector([0.03, -0.01])

    assert raised[0] / raised[1] == pytest.approx(0.03 / -0.01)


def test_single_element_vector_matches_the_scalar_case():
    minimum_velocity = MinimumVelocity(0.05)

    assert minimum_velocity.enforce_on_vector([-0.01]) == pytest.approx(
        [minimum_velocity.enforce_on_scalar(-0.01)]
    )


def test_vector_above_minimum_is_unchanged():
    assert MinimumVelocity(0.05).enforce_on_vector([0.3, 0.4]) == pytest.approx(
        [0.3, 0.4]
    )


def test_zero_vector_stays_zero():
    assert MinimumVelocity(0.05).enforce_on_vector([0.0, 0.0]) == pytest.approx(
        [0.0, 0.0]
    )


# %% joint group publisher


def test_publishes_one_value_per_connection(init_rospy):
    specs = [
        ConnectionSpec(name="joint_a", velocity=0.1),
        ConnectionSpec(name="joint_b", velocity=0.2),
        ConnectionSpec(name="joint_c", velocity=0.3),
    ]
    controller = build_group_publisher(specs)

    data = publish_group(controller)

    assert len(data) == len(specs)


def test_below_threshold_velocity_is_raised_to_minimum(init_rospy):
    specs = [
        ConnectionSpec(name="slow_positive", velocity=0.01),
        ConnectionSpec(name="slow_negative", velocity=-0.01),
    ]
    controller = build_group_publisher(specs, minimum_valid_velocity=0.03)

    data = publish_group(controller)

    assert data[0] == pytest.approx(0.03)
    assert data[1] == pytest.approx(-0.03)


def test_default_minimum_valid_velocity_does_not_clamp(init_rospy):
    specs = [
        ConnectionSpec(name="slow_positive", velocity=0.01),
        ConnectionSpec(name="slow_negative", velocity=-0.01),
    ]
    controller = build_group_publisher(specs)

    data = publish_group(controller)

    assert data[0] == pytest.approx(0.01)
    assert data[1] == pytest.approx(-0.01)


def test_velocities_outside_threshold_are_unchanged(init_rospy):
    specs = [
        ConnectionSpec(name="fast", velocity=0.1),
        ConnectionSpec(name="stopped", velocity=0.0),
    ]
    controller = build_group_publisher(specs)

    data = publish_group(controller)

    assert data[0] == pytest.approx(0.1)
    assert data[1] == pytest.approx(0.0)


def test_joint_with_zero_override_is_not_raised(init_rospy):
    specs = [
        ConnectionSpec(name="exempt_joint", velocity=0.01),
        ConnectionSpec(name="clamped_joint", velocity=0.01),
    ]
    controller = build_group_publisher(
        specs,
        minimum_valid_velocity=0.03,
        minimum_velocity_overrides={"exempt_joint": 0.0},
    )

    data = publish_group(controller)

    assert data[0] == pytest.approx(0.01)
    assert data[1] == pytest.approx(0.03)


def test_joint_override_replaces_the_default_minimum(init_rospy):
    specs = [
        ConnectionSpec(name="stiff_joint", velocity=0.01),
        ConnectionSpec(name="clamped_joint", velocity=0.01),
    ]
    controller = build_group_publisher(
        specs,
        minimum_valid_velocity=0.03,
        minimum_velocity_overrides={"stiff_joint": 0.1},
    )

    data = publish_group(controller)

    assert data[0] == pytest.approx(0.1)
    assert data[1] == pytest.approx(0.03)


def test_group_publisher_follows_its_joints_through_a_model_change(init_rospy):
    specs = [
        ConnectionSpec(name="joint_a", velocity=0.1),
        ConnectionSpec(name="joint_b", velocity=0.2),
    ]
    world = World()
    base = Body(name=PrefixedName("base"))
    removable = add_removable_joint(world, base)
    add_joints(world, base, specs)
    controller = group_publisher_for(world, specs)
    commanded_dofs = [world.get_connection_by_name(spec.name).raw_dof for spec in specs]
    columns_before = world.state.column_indices(commanded_dofs)
    publish_group(controller)

    remove_joint(world, removable)

    assert world.state.column_indices(commanded_dofs) != columns_before
    assert publish_group(controller) == pytest.approx([0.1, 0.2])


def test_group_publisher_publishes_the_velocities_of_the_current_cycle(init_rospy):
    specs = [ConnectionSpec(name="joint_a", velocity=0.1)]
    world = build_world_with_joints(specs)
    controller = group_publisher_for(world, specs)
    recorder = RecordingPublisher()
    controller.command_publisher = recorder

    controller.publish()
    world.get_connection_by_name("joint_a").velocity = 0.4
    controller.publish()

    recorded = [list(message.data) for message in recorder.published_messages]
    assert recorded[0] == pytest.approx([0.1])
    assert recorded[1] == pytest.approx([0.4])


# %% minimum velocity overrides that apply to nothing


def test_an_override_of_an_uncommanded_joint_is_rejected(init_rospy):
    """
    An override that matches no commanded joint would silently do nothing, which reads
    like the joint is exempt while the hardware still receives the default minimum.
    """
    specs = [ConnectionSpec(name="joint_a", velocity=0.01)]

    with pytest.raises(UnknownMinimumVelocityJointError):
        build_group_publisher(
            specs,
            minimum_valid_velocity=0.03,
            minimum_velocity_overrides={"typo_joint": 0.0},
        )


def test_the_rejected_override_names_the_joint_it_could_not_find(init_rospy):
    specs = [ConnectionSpec(name="joint_a", velocity=0.01)]

    with pytest.raises(UnknownMinimumVelocityJointError) as rejection:
        build_group_publisher(specs, minimum_velocity_overrides={"typo_joint": 0.0})

    assert rejection.value.joint_name == "typo_joint"


def test_overrides_of_commanded_joints_are_accepted(init_rospy):
    specs = [
        ConnectionSpec(name="joint_a", velocity=0.01),
        ConnectionSpec(name="joint_b", velocity=0.01),
    ]

    controller = build_group_publisher(
        specs,
        minimum_valid_velocity=0.03,
        minimum_velocity_overrides={"joint_a": 0.0, "joint_b": 0.1},
    )

    assert publish_group(controller) == pytest.approx([0.01, 0.1])


# %% per joint publisher


@pytest.fixture()
def joint_parameter_nodes(init_rospy):
    nodes: List[ParameterServingNode] = []

    def _serve(controller_name: str, joint_name: str) -> None:
        nodes.append(
            ParameterServingNode(controller_name=controller_name, joint_name=joint_name)
        )

    try:
        yield _serve
    finally:
        for node in nodes:
            node.shutdown()


def build_joint_publisher(
    world: World,
    namespaces: List[str],
    minimum_valid_velocity: float = 0.0,
    minimum_velocity_overrides: Optional[Dict[str, float]] = None,
) -> JointVelocityCommandPublisher:
    """
    Build a publisher that commands one joint per namespace.
    """
    return JointVelocityCommandPublisher(
        world=world,
        namespaces=namespaces,
        minimum_velocities=JointMinimumVelocities.from_magnitudes(
            minimum_valid_velocity, minimum_velocity_overrides
        ),
    )


def publish_per_joint(publisher: JointVelocityCommandPublisher) -> List[float]:
    """
    Replace all publishers with recorders, publish once, and return the values.
    """
    recorders = [RecordingPublisher() for _ in publisher.publishers]
    publisher.publishers = recorders
    publisher.publish()
    return [recorder.published_message.data for recorder in recorders]


def test_publishes_one_message_per_namespace(joint_parameter_nodes):
    specs = [
        ConnectionSpec(name="joint_a", velocity=0.1),
        ConnectionSpec(name="joint_b", velocity=0.2),
    ]
    world = build_world_with_joints(specs)
    joint_parameter_nodes("joint_a_controller", "joint_a")
    joint_parameter_nodes("joint_b_controller", "joint_b")

    publisher = build_joint_publisher(
        world, ["joint_a_controller", "joint_b_controller"]
    )

    values = publish_per_joint(publisher)

    assert values == pytest.approx([0.1, 0.2])


def test_per_joint_velocity_is_raised_to_minimum(joint_parameter_nodes):
    specs = [
        ConnectionSpec(name="slow_joint", velocity=-0.01),
        ConnectionSpec(name="exempt_joint", velocity=0.01),
    ]
    world = build_world_with_joints(specs)
    joint_parameter_nodes("slow_joint_controller", "slow_joint")
    joint_parameter_nodes("exempt_joint_controller", "exempt_joint")

    publisher = build_joint_publisher(
        world,
        ["slow_joint_controller", "exempt_joint_controller"],
        minimum_valid_velocity=0.03,
        minimum_velocity_overrides={"exempt_joint": 0.0},
    )

    values = publish_per_joint(publisher)

    assert values == pytest.approx([-0.03, 0.01])


def test_per_joint_velocity_includes_the_multiplier(joint_parameter_nodes):
    specs = [ConnectionSpec(name="scaled_joint", velocity=0.0, multiplier=2.0)]
    world = build_world_with_joints(specs)
    connection = world.get_connection_by_name("scaled_joint")
    world.state[connection.raw_dof.id].velocity = 0.05
    joint_parameter_nodes("scaled_joint_controller", "scaled_joint")

    publisher = build_joint_publisher(world, ["scaled_joint_controller"])

    values = publish_per_joint(publisher)

    assert values == pytest.approx([0.1])


# %% drive publisher


def build_drive_publisher(
    world: World,
    minimum_linear_velocity: float = 0.0,
    minimum_angular_velocity: float = 0.0,
) -> DriveVelocityCommandPublisher:
    """
    Build a drive publisher for the only drive of the world.
    """
    return DriveVelocityCommandPublisher(
        world=world,
        command_topic="/cmd_vel",
        connection=world.get_connection_by_name("brumbrum"),
        minimum_linear_velocity=MinimumVelocity(minimum_linear_velocity),
        minimum_angular_velocity=MinimumVelocity(minimum_angular_velocity),
    )


def test_slow_drive_command_is_raised_without_turning(init_rospy):
    world = build_drive_world(OmniDrive)
    drive = world.get_connection_by_name("brumbrum")
    world.state[drive.x_velocity.id].velocity = 0.02
    world.state[drive.y_velocity.id].velocity = 0.02
    publisher = build_drive_publisher(world, minimum_linear_velocity=0.05)

    command = publish_drive(publisher).published_message

    assert command.linear.x == pytest.approx(command.linear.y)
    assert np.linalg.norm([command.linear.x, command.linear.y]) == pytest.approx(0.05)


def test_linear_and_angular_minimums_are_independent(init_rospy):
    world = build_drive_world(OmniDrive)
    drive = world.get_connection_by_name("brumbrum")
    world.state[drive.x_velocity.id].velocity = 0.5
    world.state[drive.yaw.id].velocity = 0.01
    publisher = build_drive_publisher(
        world, minimum_linear_velocity=0.05, minimum_angular_velocity=0.03
    )

    command = publish_drive(publisher).published_message

    assert command.linear.x == pytest.approx(0.5)
    assert command.angular.z == pytest.approx(0.03)


def test_drive_at_rest_stays_at_rest(init_rospy):
    world = build_drive_world(OmniDrive)
    publisher = build_drive_publisher(
        world, minimum_linear_velocity=0.05, minimum_angular_velocity=0.03
    )

    command = publish_drive(publisher).published_message

    assert command.linear.x == pytest.approx(0.0)
    assert command.linear.y == pytest.approx(0.0)
    assert command.angular.z == pytest.approx(0.0)


def test_drive_without_minimums_publishes_unchanged_velocities(init_rospy):
    world = build_drive_world(OmniDrive)
    drive = world.get_connection_by_name("brumbrum")
    world.state[drive.x_velocity.id].velocity = 0.001
    world.state[drive.yaw.id].velocity = 0.002
    publisher = build_drive_publisher(world)

    command = publish_drive(publisher).published_message

    assert command.linear.x == pytest.approx(0.001)
    assert command.angular.z == pytest.approx(0.002)


def test_drive_publisher_follows_its_dofs_through_a_model_change(init_rospy):
    world = World()
    odom = Body(name=PrefixedName("odom"))
    removable = add_removable_joint(world, odom)
    add_drive(world, odom, OmniDrive)
    drive = world.get_connection_by_name("brumbrum")
    world.state[drive.x_velocity.id].velocity = 0.3
    world.state[drive.yaw.id].velocity = 0.4
    publisher = build_drive_publisher(world)
    columns_before = world.state.column_indices([drive.x_velocity, drive.yaw])
    publish_drive(publisher)

    remove_joint(world, removable)

    assert world.state.column_indices([drive.x_velocity, drive.yaw]) != columns_before
    command = publish_drive(publisher).published_message
    assert command.linear.x == pytest.approx(0.3)
    assert command.angular.z == pytest.approx(0.4)


def test_drive_publisher_publishes_the_velocities_of_the_current_cycle(init_rospy):
    world = build_drive_world(OmniDrive)
    drive = world.get_connection_by_name("brumbrum")
    world.state[drive.x_velocity.id].velocity = 0.3
    publisher = build_drive_publisher(world)
    recorder = RecordingPublisher()
    publisher.velocity_publisher = recorder

    publisher.publish()
    world.state[drive.x_velocity.id].velocity = 0.6
    publisher.publish()

    assert [
        message.linear.x for message in recorder.published_messages
    ] == pytest.approx([0.3, 0.6])


def test_differential_drive_is_raised_along_x_only(init_rospy):
    world = build_drive_world(DifferentialDrive)
    drive = world.get_connection_by_name("brumbrum")
    world.state[drive.x_velocity.id].velocity = -0.01
    publisher = build_drive_publisher(world, minimum_linear_velocity=0.05)

    command = publish_drive(publisher).published_message

    assert command.linear.x == pytest.approx(-0.05)
    assert command.linear.y == pytest.approx(0.0)
