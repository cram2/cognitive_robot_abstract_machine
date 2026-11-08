"""
Tests for mounting Tracy into a world and simulating it physically in MuJoCo; skipped
where Tracy's description is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import pytest

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.adapters.multi_sim import MujocoBuilder, MujocoSim
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy, TracyJoint
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.utils import tracy_installed
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    GravityCompensation,
    PositionServo,
)

pytestmark = pytest.mark.skipif(
    not tracy_installed(), reason="iai_tracy_description is not installed"
)


@pytest.fixture
def mounted_tracy() -> Tracy:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    return RobotSpecification(Tracy).spawn(world)


def test_a_stationary_robot_is_bolted_to_the_world_through_a_fixed_drive():
    """
    Tracy has no mobile base, so its drive is a fixed connection like every robot's
    odom, and the localization pose it is spawned with is where its root ends up.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    world_T_odom = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, y=2.0)

    tracy = RobotSpecification(Tracy, world_T_odom=world_T_odom).spawn(world)

    odom_C_robot = tracy.root.parent_connection.parent.parent_connection
    root_C_odom = odom_C_robot.parent.parent_connection
    assert isinstance(odom_C_robot, FixedConnection)
    assert isinstance(root_C_odom, FixedConnection)
    assert root_C_odom.parent is world.root
    root_position = world.compute_forward_kinematics_np(world.root, tracy.root)[:3, 3]
    assert root_position[:2] == pytest.approx([1.0, 2.0])


def test_the_mounting_table_is_a_table_at_the_robots_root(mounted_tracy):
    """
    The table the arms are bolted onto is a robot part of its own, so it is set up with
    the robot, and a table, so its top is where objects can stand.
    """
    table = mounted_tracy.table

    assert isinstance(table, Table)
    assert table.root is mounted_tracy.root
    assert table in mounted_tracy._robot_parts
    widest_slab = max(
        table.root.collision.as_bounding_box_collection_in_frame(
            mounted_tracy._world.root
        ),
        key=lambda box: (box.max_x - box.min_x) * (box.max_y - box.min_y),
    )
    assert table.top_z == pytest.approx(widest_slab.max_z)


def test_every_arm_and_gripper_degree_of_freedom_is_driven_by_one_servo(
    mounted_tracy, tmp_path
):
    """
    A mounted Tracy declares a position servo on each arm and gripper degree of freedom,
    with the joint's own dynamics on the joint, and the compiled model drives each of
    those degrees of freedom with exactly one servo actuator.
    """
    world = mounted_tracy._world
    shoulder = world.get_connection_by_name(TracyJoint.LEFT_SHOULDER_PAN)
    shoulder_drive = mounted_tracy.left_arm.drive_of_each_joint[
        TracyJoint.LEFT_SHOULDER_PAN
    ]
    [shoulder_servo] = [
        actuator for actuator in world.actuators if shoulder.raw_dof in actuator.dofs
    ]
    assert isinstance(shoulder_servo, PositionServo)
    assert shoulder_servo.gains == shoulder_drive.gains
    assert shoulder.dynamics.armature == shoulder_drive.armature
    assert shoulder.dynamics.damping == shoulder_drive.joint_damping

    servoed_degrees_of_freedom = {
        connection.raw_dof
        for arm in mounted_tracy.all_arms
        for connection in arm.active_connections + arm.end_effector.active_connections
    }
    driven_degrees_of_freedom = [
        degree_of_freedom
        for actuator in world.actuators
        for degree_of_freedom in actuator.dofs
    ]
    assert set(driven_degrees_of_freedom) == servoed_degrees_of_freedom
    assert len(driven_degrees_of_freedom) == len(servoed_degrees_of_freedom)

    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))
    assert {actuator.target for actuator in builder.spec.actuators} == {
        degree_of_freedom.name.name for degree_of_freedom in servoed_degrees_of_freedom
    }


def test_the_grippers_keep_the_descriptions_own_velocity_limit(mounted_tracy):
    """
    Slowing the arms down for safety must not slow the fingers down with them.
    """
    world = mounted_tracy._world
    knuckle = world.get_connection_by_name(TracyJoint.LEFT_GRIPPER_LEFT_KNUCKLE)
    wrist = world.get_connection_by_name(TracyJoint.LEFT_WRIST_3)
    description = URDFParser.from_file(Tracy.get_ros_file_path()).create_world()
    described_knuckle_velocity = description.get_connection_by_name(
        TracyJoint.LEFT_GRIPPER_LEFT_KNUCKLE
    ).raw_dof.limits.upper.velocity

    assert knuckle.raw_dof.limits.upper.velocity == described_knuckle_velocity
    assert wrist.raw_dof.limits.upper.velocity == pytest.approx(0.2)


def test_the_servoed_parts_carry_their_weight_and_the_links_pass_through_each_other(
    mounted_tracy, tmp_path
):
    """
    Every arm and gripper link is gravity-compensated, and the compiled model excludes
    contacts between the robot's own links while the fingers keep colliding with the
    table they may press into.

    Pairs the description itself lists as adjacent, such as the table and the arm bases
    mounted on it, stay excluded.
    """
    for arm in mounted_tracy.all_arms:
        for body in arm.bodies + arm.end_effector.bodies:
            assert body.get_simulator_property_of_type(GravityCompensation) == (
                GravityCompensation(fraction=1.0)
            )
    builder = MujocoBuilder()
    builder.build_world(
        world=mounted_tracy._world, file_path=str(tmp_path / "scene.xml")
    )
    excluded_pairs = {
        frozenset((exclude.bodyname1, exclude.bodyname2))
        for exclude in builder.spec.excludes
    }
    fingertip = mounted_tracy.left_arm.end_effector.thumb.tip.name.name
    wrist = mounted_tracy._world.get_connection_by_name(
        TracyJoint.LEFT_WRIST_3
    ).child.name.name
    table = mounted_tracy.root.name.name
    assert frozenset((fingertip, wrist)) in excluded_pairs
    assert frozenset((fingertip, table)) not in excluded_pairs


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_servos_hold_the_parked_arms_up(mounted_tracy):
    """
    A joint with no servo would be written straight into MuJoCo and sag under its own
    weight between one control cycle and the next.
    """
    world = mounted_tracy._world
    for arm in mounted_tracy.all_arms:
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    world.notify_state_change()
    tool_frame = mounted_tracy.left_arm.end_effector.tool_frame
    parked = world.compute_forward_kinematics_np(world.root, tool_frame)[:3, 3]

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=1))
        simulated = simulation.simulator.get_body_position(
            body_name=tool_frame.name.name
        ).result
    finally:
        simulation.stop_simulation()

    assert list(simulated) == pytest.approx(list(parked), abs=0.01)


# %% the gripper's own geometry


def test_gripper_geometry_names_the_pads_and_the_driving_joint(mounted_tracy):
    gripper = mounted_tracy.left_arm.end_effector

    assert gripper.thumb.tip.name.name == "left_robotiq_85_left_finger_tip_link"
    assert gripper.finger.tip.name.name == "left_robotiq_85_right_finger_tip_link"
    assert gripper.knuckle_joint.name.name == TracyJoint.LEFT_GRIPPER_LEFT_KNUCKLE
    assert (
        mounted_tracy.right_arm.end_effector.knuckle_joint.name.name
        == TracyJoint.RIGHT_GRIPPER_LEFT_KNUCKLE
    )


# %% an arm swung into a wall


@dataclass
class ArmSwungIntoAWall:
    """
    Tracy with its arms parked and a wall standing on its table right next to its left
    gripper, in the way of the gripper when the left shoulder lifts.
    """

    tracy: Tracy
    """
    The robot whose left arm is swung into the wall.
    """

    wall: Body
    """
    The wall standing in the way of the left gripper.
    """

    swing_speed: float = 0.5
    """
    How fast the left shoulder lift is commanded to turn, in radians per second.
    """

    control_period: timedelta = field(
        default_factory=lambda: timedelta(milliseconds=20)
    )
    """
    How much simulated time passes between two commands, as between two ticks of a
    controller.
    """

    @classmethod
    def next_to(cls, tracy: Tracy) -> ArmSwungIntoAWall:
        """
        :param tracy: A robot mounted into a world, whose arms are parked by this.
        :return: The robot with a wall standing on its table, 5 cm from the side of its
            parked left gripper that faces the middle of the table.
        """
        world = tracy._world
        for arm in tracy.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        wall_scale = Scale(0.4, 0.04, 0.8)
        wall = Body(
            name=PrefixedName("wall"),
            collision=ShapeCollection([Box(scale=wall_scale)]),
            visual=ShapeCollection([Box(scale=wall_scale)]),
        )
        with world.modify_world():
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=wall,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.95,
                        y=0.22,
                        z=tracy.table.top_z + wall_scale.z / 2,
                        reference_frame=world.root,
                    ),
                )
            )
        world.notify_state_change()
        return cls(tracy=tracy, wall=wall)

    @property
    def shoulder_lift(self) -> ActiveConnection1DOF:
        """
        The joint that swings the left gripper into the wall as its position grows.
        """
        return self.tracy._world.get_connection_by_name(TracyJoint.LEFT_SHOULDER_LIFT)

    def command_one_control_period(self, simulation: MujocoSim) -> None:
        """
        Move the shoulder lift on by one control period at :attr:`swing_speed` from
        where the world shows it, the way a controller commands it, and let the physics
        carry out that period.

        :param simulation: The running physics of the robot's world.
        """
        world = self.tracy._world
        degree_of_freedom = self.shoulder_lift.raw_dof
        world.state[degree_of_freedom.id].position += (
            self.swing_speed * self.control_period.total_seconds()
        )
        world.notify_state_change()
        simulation.step_simulation(self.control_period)


@pytest.fixture
def arm_swung_into_a_wall(mounted_tracy) -> ArmSwungIntoAWall:
    return ArmSwungIntoAWall.next_to(mounted_tracy)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_an_arm_swung_into_a_wall_is_shown_where_the_wall_stops_it(
    arm_swung_into_a_wall,
):
    """
    Commanded on through a wall, the arm is stopped by it, and the world shows the
    gripper where the physics holds it, on its own side of the wall, rather than having
    passed through.
    """
    scene = arm_swung_into_a_wall
    world = scene.tracy._world
    tool_frame = scene.tracy.left_arm.end_effector.tool_frame

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        for _ in range(50):
            scene.command_one_control_period(simulation)
        simulated = simulation.simulator.get_body_position(
            body_name=tool_frame.name.name
        ).result
    finally:
        simulation.stop_simulation()

    shown = world.compute_forward_kinematics_np(world.root, tool_frame)[:3, 3]
    [wall_box] = scene.wall.collision.as_bounding_box_collection_in_frame(world.root)
    assert shown[1] > wall_box.max_y
    # MuJoCo reports body poses from before its last step, the world its joints after
    assert list(shown) == pytest.approx(list(simulated), abs=1e-3)
