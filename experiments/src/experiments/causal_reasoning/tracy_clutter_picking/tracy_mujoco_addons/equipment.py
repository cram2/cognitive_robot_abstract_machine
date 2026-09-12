"""
Read Tracy's description out of its own ROS package and equip it to be driven by direct
MuJoCo actuator control.

Physically simulating the joints Giskard actively commands, and letting Giskard itself
drive them, was tried first and traced to a real architectural gap: Giskard's own QP
control loop reads ``world.state`` as its belief of the robot's current position, but
for a physically simulated DOF that same state is also written by Giskard's own prior
command, not exclusively by MuJoCo's true physics readback -- so Giskard can be
satisfied by its own prior write, not by the robot actually having moved. Driving the
actuators directly via
:meth:`~experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.real_time_simulation.RealTimeSimulation.command`,
and planning Cartesian/joint goals against an isolated scratch copy of the world (see
:mod:`~experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.trajectory_planning`) rather than through Giskard's
live closed loop, sidesteps this entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import mujoco
from typing_extensions import Dict, Iterable, Tuple, Type

from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.grasp_contact import (
    ContactParameters,
    mujoco_geom_for,
)
from semantic_digital_twin.adapters.multi_sim import MujocoActuator, MujocoBody
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot, AbstractRobotPart
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Actuator, Body

# %% servo tuning


@dataclass(frozen=True)
class ServoGains:
    """
    How hard a position servo pulls its joint towards the angle it was given, and how
    much passive resistance its joint itself has.
    """

    stiffness: float
    """
    Restoring torque per radian away from the set point, in newton metres.
    """

    actuator_damping: float
    """
    Opposing torque per radian per second the servo itself applies, in newton metre
    seconds.
    """

    torque_limit: float
    """
    The largest torque the servo may exert, in newton metres.
    """

    joint_damping: float
    """
    Passive viscous damping of the joint itself, independent of the servo -- always
    resists motion, whether or not the servo is actively driving.
    """

    armature: float
    """
    Rotor inertia added to the joint, damping high-frequency numerical response without
    changing its real, low-frequency behaviour.
    """

    @classmethod
    def ur10e(cls, torque_limit: float, joint_damping: float) -> ServoGains:
        """
        The gains of one UR10e joint size class, taken as-is from MuJoCo Menagerie's own
        ``universal_robots_ur10e/ur10e.xml``: its ``<general gainprm="5000"
        biasprm="0 -5000 -500">`` and ``<joint armature="0.1">`` sit on the base
        ``ur10e`` default class, applying identically to every joint regardless of
        size; only the torque limit and the joint's own passive damping differ per size
        class. Tracy's own UR10 (not UR10e) arms are close enough to reuse this.

        :param torque_limit: The size class's torque limit, in newton metres.
        :param joint_damping: The size class's passive joint damping.
        :return: The gains.
        """
        return cls(
            stiffness=5_000.0,
            actuator_damping=500.0,
            torque_limit=torque_limit,
            joint_damping=joint_damping,
            armature=0.1,
        )

    @classmethod
    def robotiq_85_knuckle(cls) -> ServoGains:
        """
        Tuning for a Robotiq-85 knuckle joint; no MuJoCo Menagerie or otherwise
        pre-tuned reference exists for this gripper, so this is the cube-stacking
        demo's own, empirically raised value.

        :return: The gains.
        """
        return cls(
            stiffness=100.0,
            actuator_damping=10.0,
            torque_limit=10.0,
            joint_damping=0.0,
            armature=0.05,
        )


def ur10e_arm_gains() -> Dict[str, ServoGains]:
    """
    Real, per-joint-size UR10e gains and torque limits, keyed by joint name with Tracy's
    own ``left_``/``right_`` prefix stripped.

    :return: The gains of every arm joint.
    """
    # "size4" in ur10e.xml: the two shoulder joints, which carry the whole rest of the
    # arm's weight and so need the most torque and the most passive damping to settle
    # without ringing; "size3" the elbow; "size2" the three wrist joints, which carry
    # only the gripper and so need much less of either.
    return {
        "shoulder_pan_joint": ServoGains.ur10e(torque_limit=330.0, joint_damping=10.0),
        "shoulder_lift_joint": ServoGains.ur10e(torque_limit=330.0, joint_damping=10.0),
        "elbow_joint": ServoGains.ur10e(torque_limit=150.0, joint_damping=5.0),
        "wrist_1_joint": ServoGains.ur10e(torque_limit=56.0, joint_damping=2.0),
        "wrist_2_joint": ServoGains.ur10e(torque_limit=56.0, joint_damping=2.0),
        "wrist_3_joint": ServoGains.ur10e(torque_limit=56.0, joint_damping=2.0),
    }


@dataclass(frozen=True)
class TracyServoTuning:
    """
    The servos Tracy's arms and grippers are equipped with.
    """

    arm_joint_gains: Dict[str, ServoGains] = field(default_factory=ur10e_arm_gains)
    """
    Each arm joint's gains, keyed by joint name without the arm's ``left_``/``right_``
    prefix.
    """

    gripper_joint_gains: ServoGains = field(
        default_factory=ServoGains.robotiq_85_knuckle
    )
    """
    The gains of every gripper joint.
    """

    gripper_joint_velocity_limit: float = 1.0
    """
    Velocity limit, in radians per second, given to every gripper joint's own degree of
    freedom, overriding whatever ``iai_tracy_description`` itself declares.

    The parsed URDF's own knuckle joint velocity limit is roughly ``0.032`` rad/s -- at
    that speed, closing through the gripper's own ~0.8 rad range takes about 25 real
    seconds, which joint-space planning faithfully respects since it clamps its own
    reference velocity to this limit. Unlike the arm's gains (sourced from real UR10e
    hardware data), no such reference backs this gripper's own declared limit, so
    raising it trades an unproven number for a still-modest, more usable one.

    ``1.0`` (not higher) is a real, tested ceiling: ``2.0`` made the QP solver settle
    about 0.016 rad short of the actual target and stay there indefinitely, so the plan
    never finished -- ``1.0`` converges cleanly and still closes the gripper about 25x
    faster than the original limit.
    """

    def arm_gains_for(self, joint_name: str) -> ServoGains:
        """
        :param joint_name: Name of an arm joint, possibly ``left_``/``right_``-prefixed,
            e.g. ``"left_shoulder_pan_joint"``.
        :return: Its gains.
        """
        unprefixed = joint_name.removeprefix("left_").removeprefix("right_")
        return self.arm_joint_gains[unprefixed]


def joint_state_of_type(robot_part: AbstractRobotPart, state_type) -> JointState:
    """
    The one of ``robot_part``'s own joint states with the given ``state_type``.

    :param robot_part: The robot part (an arm or a gripper) to search.
    :param state_type: The state type to find, e.g. :attr:`StaticJointState.PARK`.
    :return: The joint state.
    """
    return next(
        joint_state
        for joint_state in robot_part.joint_states
        if joint_state.state_type == state_type
    )


# %% mounting


def parse_tracy(
    mount_root_name: PrefixedName = PrefixedName("tracy_mount", "tracy_mujoco_addons"),
) -> World:
    """
    Read Tracy out of its own ``iai_tracy_description`` ROS package, without any
    actuator: an actuator parsed into one world cannot be merged into another (see
    :func:`mount_stationary_robot`), and
    :func:`equip_arms_with_servos`/:func:`equip_grippers_with_servos` install their own
    once Tracy is mounted.

    :param mount_root_name: Name given to the parsed Tracy's own synthetic world root,
        so it never collides with a merge target's own root. Tracy's real kinematic
        root, the body named ``"table"``, is a descendant of this synthetic node, not
        the node itself, so renaming it does not affect
        :meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.from_world`'s
        later lookup.
    :return: A world holding only Tracy's own body tree.
    """
    tracy_world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
    with tracy_world.modify_world():
        for actuator in list(tracy_world.actuators):
            tracy_world.remove_actuator(actuator)
        tracy_world.root.name = mount_root_name
    return tracy_world


def mount_stationary_robot(
    world: World,
    robot_class: Type[AbstractRobot],
    robot_world: World,
    mount_position: Point3,
    mount_yaw: float = 0.0,
) -> AbstractRobot:
    """
    Bolt an already-parsed, fixed-base robot into ``world`` at ``mount_position``.

    Takes a parsed world rather than a robot class to parse, so a robot whose
    description is not a ROS package can be read by its caller from whichever format it
    does ship in. Its root is attached with a :class:`FixedConnection`: a robot with no
    mobile base has nothing for an active drive connection to move.

    Any actuator ``robot_world`` carries is dropped first: an actuator parsed into one
    world cannot be merged into another. Callers that need actuators add them to the
    merged world afterwards (see :func:`equip_arms_with_servos`).

    :param world: The world to mount the robot into, modified in place.
    :param robot_class: The robot to read out of the merged world.
    :param robot_world: The parsed robot, consumed by the merge.
    :param mount_position: Where the robot's root is bolted, in ``world``'s root frame.
    :param mount_yaw: Which way the robot is turned to face. A bolted arm has no base to
        move afterwards, so it has to face its whole task from this one pose.
    :return: The mounted robot.
    """
    with robot_world.modify_world():
        for actuator in list(robot_world.actuators):
            robot_world.remove_actuator(actuator)
    with world.modify_world():
        mount = FixedConnection(
            parent=world.root,
            child=robot_world.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=mount_position.x,
                y=mount_position.y,
                z=mount_position.z,
                yaw=mount_yaw,
            ),
        )
        world.merge_world(robot_world, mount)
    return robot_class.from_world(world)


def tracy_table_mount_position(
    tracy_world: World, x: float, y: float
) -> Tuple[Point3, float]:
    """
    Where to bolt a parsed-but-not-yet-mounted Tracy so its own built-in table's legs
    rest exactly on the floor (``z=0``), and the resulting height of that table's own
    top surface once mounted there.

    :param tracy_world: Tracy's own parsed world, as returned by :func:`parse_tracy`,
        not yet merged into anything.
    :param x: X-coordinate to mount Tracy's root at, in the merge target's root frame.
    :param y: Y-coordinate to mount Tracy's root at, in the merge target's root frame.
    :return: The mount position, and the world-frame height its table's own top surface
        ends up at once mounted there.
    """
    table = tracy_world.get_body_by_name("table")
    table_bounding_box = table.collision.as_bounding_box_collection_in_frame(
        tracy_world.root
    ).bounding_box()
    mount_z = -table_bounding_box.min_z
    tabletop = max(table.collision, key=lambda shape: shape.scale.x * shape.scale.y)
    root_transform_table = tracy_world.compute_forward_kinematics_np(
        tracy_world.root, table
    )
    tabletop_local_top_z = float(tabletop.origin.to_np()[2, 3] + tabletop.scale.z / 2)
    table_top_z = mount_z + float(root_transform_table[2, 3]) + tabletop_local_top_z
    return Point3(x, y, mount_z), table_top_z


def table_top_z(robot: Tracy) -> float:
    """
    Height of an already-mounted Tracy's own table's top surface above the world root,
    in metres, read via forward kinematics.

    Unlike :func:`tracy_table_mount_position`, this needs no freshly parsed, unmounted
    Tracy: it reads the table height directly off ``robot``, so it also works for a
    Tracy that is already mounted somewhere this module did not mount it itself (e.g.
    the physical robot, fetched live from its own world service).

    :param robot: The already-mounted robot whose table height is read.
    :return: The height.
    """
    table = robot.root
    tabletop = max(table.collision, key=lambda shape: shape.scale.x * shape.scale.y)
    root_transform_table = robot._world.compute_forward_kinematics_np(
        robot._world.root, table
    )
    return float(
        root_transform_table[2, 3]
        + tabletop.origin.to_np()[2, 3]
        + tabletop.scale.z / 2
    )


# %% physical simulation


class CollisionGroup(IntEnum):
    """
    MuJoCo ``contype``/``conaffinity`` bits telling apart what may touch what: a contact
    between two geoms is generated only if one's ``contype`` shares a bit with the
    other's ``conaffinity``.
    """

    ROBOT = 1
    """
    Tracy's own moving links; see :func:`exclude_self_collision`.
    """

    EXTERNAL = 2
    """
    Things Tracy is meant to actually touch -- loose objects, a table, anything that is
    not the robot's own body.
    """


def apply_gravity_compensation(world: World, robot: Tracy) -> None:
    """
    Give every arm and gripper link MuJoCo's own gravity compensation.

    Without it, each link's own position servo would have to spend part of its available
    torque fighting gravity instead of tracking its commanded target. This covers the
    gripper's own links too, not just the arm's own chain up to the wrist: without it,
    the gripper -- an entirely separate semantic annotation hanging off the arm's end,
    not part of ``arm.active_connections`` -- settles wherever gravity pulls it
    regardless of its own actuator's commanded target, since its comparatively weak
    servo never has enough authority to fight the whole uncompensated finger assembly's
    own weight.

    :param world: The world to modify in place.
    :param robot: The robot to compensate.
    """
    with world.modify_world():
        for arm in robot.get_arms():
            for body in arm.bodies + arm.end_effector.bodies:
                body.simulator_additional_properties.append(
                    MujocoBody(gravitation_compensation_factor=1.0)
                )


def exclude_self_collision(world: World, robot: Tracy) -> None:
    """
    Let the robot's own links pass through each other, without also excusing them from
    colliding with anything else.

    A description's links overlap wherever they meet; sweeping an arm through its own
    park pose swings it through several such overlaps, which a position servo cannot
    push through by itself (confirmed directly in the cube-stacking demo: one joint sat
    pinned at its starting angle the whole run, the signature of a real contact force).

    Gives every one of the robot's own collision geoms :attr:`CollisionGroup.ROBOT` as
    ``contype`` and :attr:`CollisionGroup.EXTERNAL` as ``conaffinity``: two robot geoms
    then never generate a contact, while a robot geom still collides with anything
    external.

    ``robot.bodies_with_collision`` also includes ``robot.root`` itself -- Tracy's own
    table, since both arms are rooted there rather than at a separate torso link --
    which must be skipped: it is exactly the kind of thing the robot should keep
    colliding with, not one of the robot's own moving links.

    :param world: The world to relax, modified in place.
    :param robot: The robot to exclude self-collision on.
    """
    with world.modify_world():
        for body in robot.bodies_with_collision:
            if body is robot.root:
                continue
            for shape in body.collision:
                mujoco_geom = mujoco_geom_for(shape)
                mujoco_geom.contype = CollisionGroup.ROBOT
                mujoco_geom.conaffinity = CollisionGroup.EXTERNAL


def _servo_actuator(
    gains: ServoGains, degree_of_freedom: DegreeOfFreedom
) -> MujocoActuator:
    """
    Build a MuJoCo actuator that servos ``degree_of_freedom`` to a commanded position
    with a PD law, clamped to ``gains``' own torque limit and the degree of freedom's
    own position limits.

    :param gains: Gains and torque clamp to build the servo with.
    :param degree_of_freedom: The degree of freedom the servo's control range is
        clamped to.
    :return: The actuator's MuJoCo definition.
    """
    limits = degree_of_freedom.limits
    return MujocoActuator(
        dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
        gain_type=mujoco.mjtGain.mjGAIN_FIXED,
        gain_parameters=[gains.stiffness] + [0.0] * 9,
        bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
        bias_parameters=[0.0, -gains.stiffness, -gains.actuator_damping] + [0.0] * 7,
        control_range=[limits.lower.position, limits.upper.position],
        force_range=[-gains.torque_limit, gains.torque_limit],
    )


def _equip_connections_with_servos(
    world: World,
    connections: Iterable[ActiveConnection1DOF],
    gains_for: Dict[str, ServoGains],
) -> Dict[str, Actuator]:
    """
    Give every one of ``connections`` a position-servo actuator, its own passive
    damping, and armature, driven directly via
    :meth:`~experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.real_time_simulation.RealTimeSimulation.command`
    rather than through Giskard.

    A mimic linkage (e.g. the Robotiq gripper's underactuated four-bar mechanism) shares
    one ``raw_dof`` across several connections; each such degree of freedom gets an
    actuator only once, since a second actuator on the same, already-equipped one would
    apply competing, duplicate servo force rather than driving anything new.
    ``dynamics`` (armature and damping), by contrast, lives on each connection -- its
    own physical MuJoCo joint -- not the shared degree of freedom, so it is set for
    every connection regardless; leaving a mimicked joint's own armature at zero
    starves the whole coupled mechanism of the numerical damping that keeps it from
    chattering under load, even though only one of its joints is ever driven directly.

    :param world: The world to add the actuators to, modified in place.
    :param connections: The connections to equip; non-1DOF connections are skipped.
    :param gains_for: Each connection's gains, by its joint name.
    :return: Each driven degree of freedom's own actuator, keyed by joint name.
    """
    actuators_by_joint_name: Dict[str, Actuator] = {}
    equipped: set[DegreeOfFreedom] = set()
    with world.modify_world():
        for connection in connections:
            if not isinstance(connection, ActiveConnection1DOF):
                continue
            degree_of_freedom = connection.raw_dof
            gains = gains_for[degree_of_freedom.name.name]
            connection.dynamics.armature = gains.armature
            connection.dynamics.damping = gains.joint_damping
            if degree_of_freedom in equipped:
                continue
            equipped.add(degree_of_freedom)
            actuator = Actuator()
            actuator.add_dof(dof=degree_of_freedom)
            actuator.simulator_additional_properties.append(
                _servo_actuator(gains, degree_of_freedom)
            )
            world.add_actuator(actuator=actuator)
            actuators_by_joint_name[degree_of_freedom.name.name] = actuator
    return actuators_by_joint_name


def equip_arms_with_servos(
    world: World, robot: Tracy, tuning: TracyServoTuning = TracyServoTuning()
) -> Dict[str, Actuator]:
    """
    Give every joint of both arms a position-servo actuator, its own passive damping,
    and armature, driven directly rather than through Giskard.

    :param world: The world to add the actuators to, modified in place.
    :param robot: The robot whose arms are driven.
    :param tuning: The servos to equip.
    :return: Each driven degree of freedom's own actuator, keyed by joint name.
    """
    connections = [
        connection
        for arm in robot.get_arms()
        for connection in arm.active_connections
        if isinstance(connection, ActiveConnection1DOF)
    ]
    return _equip_connections_with_servos(
        world,
        connections,
        {
            connection.raw_dof.name.name: tuning.arm_gains_for(
                connection.raw_dof.name.name
            )
            for connection in connections
        },
    )


def equip_grippers_with_servos(
    world: World, robot: Tracy, tuning: TracyServoTuning = TracyServoTuning()
) -> Dict[str, Actuator]:
    """
    Give every joint of both grippers a position-servo actuator, mirroring
    :func:`equip_arms_with_servos` for the end effectors it does not cover, and raise
    every gripper joint's own velocity limit to the tuning's (see
    :attr:`TracyServoTuning.gripper_joint_velocity_limit`).

    :param world: The world to add the actuators to, modified in place.
    :param robot: The robot whose grippers are driven.
    :param tuning: The servos to equip.
    :return: Each driven degree of freedom's own actuator, keyed by joint name.
    """
    connections = [
        connection
        for arm in robot.get_arms()
        for connection in arm.end_effector.active_connections
        if isinstance(connection, ActiveConnection1DOF)
    ]
    with world.modify_world():
        for degree_of_freedom in {connection.raw_dof for connection in connections}:
            degree_of_freedom.limits.upper.velocity = (
                tuning.gripper_joint_velocity_limit
            )
            degree_of_freedom.limits.lower.velocity = (
                -tuning.gripper_joint_velocity_limit
            )
    return _equip_connections_with_servos(
        world,
        connections,
        {
            connection.raw_dof.name.name: tuning.gripper_joint_gains
            for connection in connections
        },
    )


# %% loose objects


def add_box(
    world: World,
    name: str,
    position: Point3,
    scale: Scale,
    color: Color,
    yaw: float = 0.0,
    contact: ContactParameters = ContactParameters.cube(),
) -> Body:
    """
    Add a free-standing box with real collision geometry to the world, so it can be
    pushed, grasped, and stacked by real contact rather than teleported into place or
    kinematically attached to whatever is holding it.

    Its collision geom belongs to :attr:`CollisionGroup.EXTERNAL` and is allowed to
    touch both the robot and other external things, including another box -- see
    :func:`exclude_self_collision`.

    :param world: The world to add the box to, modified in place.
    :param name: Name of the box.
    :param position: Where the box's centre starts, in the world root frame.
    :param scale: Edge lengths of the box, in metres.
    :param color: Colour of the box.
    :param yaw: Rotation of the box about the vertical, in radians.
    :param contact: The box's contact parameters; a grasped cube's by default, without
        which a grasped box sinks into the fingers under MuJoCo's own soft contact
        defaults and slips back out as the arm lifts.
    :return: The newly added box.
    """
    box = Body(name=PrefixedName(name))
    shape = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=box),
        scale=scale,
        color=color,
    )
    mujoco_geom = mujoco_geom_for(shape)
    mujoco_geom.contype = CollisionGroup.EXTERNAL
    mujoco_geom.conaffinity = CollisionGroup.ROBOT | CollisionGroup.EXTERNAL
    geometry = ShapeCollection([shape], reference_frame=box)
    box.collision, box.visual = geometry, geometry
    contact.apply_to([box])

    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=position.x,
                    y=position.y,
                    z=position.z,
                    yaw=yaw,
                    reference_frame=world.root,
                ),
            )
        )
    return box


def add_cube(
    world: World, name: str, position: Point3, size: float, color: Color
) -> Body:
    """
    :func:`add_box` for a cube.

    :param world: The world to add the cube to, modified in place.
    :param name: Name of the cube.
    :param position: Where the cube starts, in the world root frame.
    :param size: Edge length of the cube, in metres.
    :param color: Colour of the cube.
    :return: The newly added cube.
    """
    return add_box(world, name, position, Scale(size, size, size), color)
