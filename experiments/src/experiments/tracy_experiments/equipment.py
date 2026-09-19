"""
Read Tracy's description out of its own ROS package and equip it to be driven by direct
MuJoCo actuator control.

Physically simulating the joints Giskard actively commands, and letting Giskard itself
drive them via ``ParkArmsAction``, was tried first (see
:mod:`~experiments.tracy_experiments.parkarms_demo`'s own docstring) and traced to a real
architectural gap: Giskard's own QP control loop reads ``world.state`` as its belief of
the robot's current position, but for a physically simulated DOF that same state is also
written by Giskard's own prior command, not exclusively by MuJoCo's true physics
readback -- so Giskard can be satisfied by its own prior write, not by the robot actually
having moved. Driving the actuators directly via
:meth:`~experiments.tracy_experiments.real_time_simulation.RealTimeSimulation.command`,
and planning Cartesian/joint goals against an isolated scratch copy of the world (see
:mod:`~experiments.tracy_experiments.trajectory_planning`) rather than through Giskard's
live closed loop, sidesteps this entirely -- proved out first by the cube-stacking demo.
"""

from __future__ import annotations

from typing_extensions import Callable, Dict, Iterable, Tuple

from semantic_digital_twin.adapters.multi_sim import MujocoBody, MujocoGeom
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobotPart
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import (
    JointDynamics,
    JointServo,
    ServoGains,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Box, Color, Scale, Shape
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Actuator,
    Body,
    PositionServo,
)

TRACY_MOUNT_ROOT_NAME = PrefixedName("tracy_mount", "tracy_experiments")
"""
Name given to the parsed Tracy's own synthetic world root once merged, so it never
collides with a merge target's own root.

Tracy's real kinematic root, the body named
``"table"`` (see :meth:`~semantic_digital_twin.robots.tracy.Tracy._get_root_body_name`),
is a descendant of this synthetic node, not the node itself, so renaming it does not
affect :meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.from_world`'s later
lookup.
"""

# %% servo tuning


_STIFFNESS = 5_000.0
_ACTUATOR_DAMPING = 500.0
_ARMATURE = 0.1
"""
Shared across every arm joint class below, matching MuJoCo Menagerie's own.

``universal_robots_ur10e/ur10e.xml``: its ``<general gainprm="5000" biasprm="0 -5000
-500">`` and ``<joint armature="0.1">`` sit on the base ``ur10e`` default class, applying
identically to every joint regardless of size; only torque limit and the joint's own
passive damping are given separately per size class there.
"""


def _joint_servo(
    torque_limit: float, joint_damping: float, armature: float = _ARMATURE
) -> JointServo:
    """
    A servo built from the shared arm stiffness/damping above, with the per-size torque
    limit, passive joint damping and armature a joint class of its own states.

    :param torque_limit: The largest torque the servo may exert, in newton metres.
    :param joint_damping: Passive viscous damping of the joint itself, independent of
        the servo -- always resists motion, whether or not the servo is actively
        driving.
    :param armature: Rotor inertia added to the joint, damping high-frequency numerical
        response without changing its real, low-frequency behaviour.
    """
    return JointServo(
        gains=ServoGains(
            stiffness=_STIFFNESS, damping=_ACTUATOR_DAMPING, torque_limit=torque_limit
        ),
        dynamics=JointDynamics(armature=armature, damping=joint_damping),
    )


ARM_JOINT_SERVO: Dict[str, JointServo] = {
    # "size4" in ur10e.xml: the two shoulder joints, which carry the whole rest of the
    # arm's weight and so need the most torque and the most passive damping to settle
    # without ringing.
    "shoulder_pan_joint": _joint_servo(330.0, 10.0),
    "shoulder_lift_joint": _joint_servo(330.0, 10.0),
    # "size3" in ur10e.xml: the elbow.
    "elbow_joint": _joint_servo(150.0, 5.0),
    # "size2" in ur10e.xml: the three wrist joints, which carry only the gripper and so
    # need much less of either.
    "wrist_1_joint": _joint_servo(56.0, 2.0),
    "wrist_2_joint": _joint_servo(56.0, 2.0),
    "wrist_3_joint": _joint_servo(56.0, 2.0),
}
"""
Real, per-joint-size UR10e gains and torque limits, taken as-is from MuJoCo Menagerie's
own ``universal_robots_ur10e/ur10e.xml``, keyed by joint name with Tracy's own
``left_``/``right_`` prefix stripped.

Tracy's own UR10 (not UR10e) arms are close enough to reuse this directly.
"""

GRIPPER_JOINT_SERVO = JointServo(
    gains=ServoGains(stiffness=100.0, damping=10.0, torque_limit=10.0),
    dynamics=JointDynamics(armature=0.05, damping=0.0),
)
"""
Tuning for a Robotiq-85 knuckle joint; no MuJoCo Menagerie or otherwise pre-tuned
reference exists for this gripper.

Matches the cube-stacking demo's own, empirically raised value.
"""

GRIPPER_JOINT_VELOCITY_LIMIT = 1.0
"""
Velocity limit, in radians per second, given to every gripper joint's own degree of
freedom, overriding whatever ``iai_tracy_description`` itself declares.

The parsed URDF's own knuckle joint velocity limit is roughly ``0.032`` rad/s -- at that
speed, closing through the gripper's own ~0.8 rad range takes about 25 real seconds,
which :func:`~experiments.tracy_experiments.trajectory_planning.plan_joint_trajectory`
faithfully respects since it clamps its own reference velocity to this limit, the same
way it does for the arm's own real, hardware-sourced joint limits. Unlike the arm's own
:data:`ARM_JOINT_SERVO` (sourced from real UR10e hardware data), no such reference
backs this gripper's own declared limit, so there is no real hardware speed being
faithfully preserved by keeping it -- raising it trades that unproven number for a
still-modest, more usable one.

``1.0`` (not higher) is a real, tested ceiling, not just a round number: pushing this to
``2.0`` was tried first and, confirmed directly by ticking the planner in isolation,
made the QP solver settle about 0.016 rad short of the actual target and stay there
indefinitely, so ``is_end_motion`` never triggers and the plan times out -- ``1.0``
converges cleanly (tested up to 1400 ticks with a stable, correct result) and still
closes the gripper about 25x faster than the original limit.
"""


def _raise_gripper_velocity_limits(world: World, robot: Tracy) -> None:
    """
    Raise every gripper joint's own degree of freedom velocity limit to
    :data:`GRIPPER_JOINT_VELOCITY_LIMIT`; see its own docstring for why.

    :param world: The world to modify in place.
    :param robot: The robot whose grippers' velocity limits are raised.
    """
    dofs = {
        connection.raw_dof
        for arm in robot.get_arms()
        for connection in arm.end_effector.active_connections
    }
    with world.modify_world():
        for dof in dofs:
            dof.limits.upper.velocity = GRIPPER_JOINT_VELOCITY_LIMIT
            dof.limits.lower.velocity = -GRIPPER_JOINT_VELOCITY_LIMIT


def _servo_tuning_for(joint_name: str) -> JointServo:
    """
    The tuning a joint's servo is built with, by its (possibly ``left_``/``right_``-
    prefixed) name.

    :param joint_name: Name of the joint, e.g. ``"left_shoulder_pan_joint"``.
    :return: Its own tuning from :data:`ARM_JOINT_SERVO`.
    """
    unprefixed = joint_name.removeprefix("left_").removeprefix("right_")
    return ARM_JOINT_SERVO[unprefixed]


def joint_state_of_type(robot_part: AbstractRobotPart, state_type) -> JointState:
    """
    The one of ``robot_part``'s own joint states with the given ``state_type``.

    :param robot_part: The robot part (an arm or a gripper) to search.
    :param state_type: The state type to find, e.g. :attr:`StaticJointState.PARK`.
    """
    return next(
        joint_state
        for joint_state in robot_part.joint_states
        if joint_state.state_type == state_type
    )


# %% mounting


def parse_tracy() -> World:
    """
    Read Tracy out of its own ``iai_tracy_description`` ROS package, without any
    actuator: an actuator parsed into one world cannot be merged into another (see
    :func:`~experiments.montessori.world.mount_stationary_robot`), and
    :func:`equip_arms_with_servos`/:func:`equip_grippers_with_servos` install their own
    once Tracy is mounted.

    :return: A world holding only Tracy's own body tree, its synthetic root renamed to
        :data:`TRACY_MOUNT_ROOT_NAME`.
    """
    from semantic_digital_twin.adapters.urdf import URDFParser

    tracy_world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
    with tracy_world.modify_world():
        for actuator in list(tracy_world.actuators):
            tracy_world.remove_actuator(actuator)
        tracy_world.root.name = TRACY_MOUNT_ROOT_NAME
    return tracy_world


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


def apply_gravity_compensation(world: World, robot: Tracy) -> None:
    """
    Give every arm and gripper link MuJoCo's own gravity compensation.

    Without it, each link's own position servo would have to spend part of its available
    torque fighting gravity instead of tracking its commanded target. This covers the
    gripper's own links too, not just the arm's own chain up to the wrist: without it,
    the gripper -- an entirely separate semantic annotation hanging off the arm's end,
    not part of ``arm.active_connections`` -- settles wherever gravity pulls it
    regardless of its own actuator's commanded target, since its comparatively weak
    servo (see :data:`GRIPPER_JOINT_SERVO`) never has enough authority to fight the
    whole uncompensated finger assembly's own weight.

    :param world: The world to modify in place.
    :param robot: The robot to compensate.
    """
    with world.modify_world():
        for arm in robot.get_arms():
            for body in arm.bodies + arm.end_effector.bodies:
                body.simulator_additional_properties.append(
                    MujocoBody(gravitation_compensation_factor=1.0)
                )


ROBOT_COLLISION_BIT = 1
"""
MuJoCo ``contype``/``conaffinity`` bit given to every one of Tracy's own collision geoms
by :func:`exclude_self_collision`.
"""

EXTERNAL_COLLISION_BIT = 2
"""
MuJoCo ``contype``/``conaffinity`` bit given to things Tracy is meant to actually touch
-- loose objects, a table, anything that is not the robot's own body.
"""


def _mujoco_geom_for(shape: Shape) -> MujocoGeom:
    """
    ``shape``'s own :class:`MujocoGeom` additional property, creating one if it has none
    yet.

    :class:`~semantic_digital_twin.adapters.multi_sim.MujocoGeomConverter` reads only
    the first ``MujocoGeom`` it finds on a shape, so a second, appended one would be
    silently ignored: callers must modify the returned instance in place rather than
    replacing it.

    :param shape: The shape to find or create a ``MujocoGeom`` on, modified in place if
        none exists yet.
    """
    existing = [
        additional_property
        for additional_property in shape.simulator_additional_properties
        if isinstance(additional_property, MujocoGeom)
    ]
    if existing:
        return existing[0]
    mujoco_geom = MujocoGeom()
    shape.simulator_additional_properties.append(mujoco_geom)
    return mujoco_geom


def exclude_self_collision(world: World, robot: Tracy) -> None:
    """
    Let the robot's own links pass through each other, without also excusing them from
    colliding with anything else.

    A description's links overlap wherever they meet; sweeping an arm through its own
    park pose swings it through several such overlaps, which a position servo cannot
    push through by itself (confirmed directly in the cube-stacking demo: one joint sat
    pinned at its starting angle the whole run, the signature of a real contact force).

    Gives every one of the robot's own collision geoms :data:`ROBOT_COLLISION_BIT` as
    both ``contype`` and ``conaffinity``: two robot geoms then never generate a contact,
    while a robot geom still collides normally with anything else, since that other
    geometry keeps MuJoCo's default bit (which overlaps :data:`ROBOT_COLLISION_BIT`).

    ``robot.bodies_with_collision`` also includes ``robot.root`` itself -- Tracy's own
    table, since both arms are rooted there rather than at a separate torso link -- which
    must be skipped: it is exactly the kind of thing this function's own docstring says
    the robot should keep colliding with, not one of the robot's own moving links.

    :param world: The world to relax, modified in place.
    :param robot: The robot to exclude self-collision on.
    """
    with world.modify_world():
        for body in robot.bodies_with_collision:
            if body is robot.root:
                continue
            for shape in body.collision:
                mujoco_geom = _mujoco_geom_for(shape)
                mujoco_geom.contype = ROBOT_COLLISION_BIT
                mujoco_geom.conaffinity = EXTERNAL_COLLISION_BIT


def _equip_connections_with_servos(
    world: World,
    connections: Iterable[ActiveConnection1DOF],
    servo_for: Callable[[str], JointServo],
) -> Dict[str, Actuator]:
    """
    Give every one of ``connections`` a position-servo actuator, its own passive
    damping, and armature, driven directly via
    :meth:`~experiments.tracy_experiments.real_time_simulation.RealTimeSimulation.command`
    rather than through Giskard.

    The servo's gains become a :class:`~semantic_digital_twin.world_description.world_entity.PositionServo`
    actuator on the driven degree of freedom; the MuJoCo actuator realising it is built
    automatically by :class:`~semantic_digital_twin.adapters.multi_sim.MujocoPositionServoConverter`,
    the same path every other robot part's own servo takes (see
    :meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobotPart._declare_servo`).

    A mimic linkage (e.g. the Robotiq gripper's underactuated four-bar mechanism) shares
    one ``raw_dof`` across several connections; each such ``dof`` gets an actuator only
    once, since a second actuator on the same, already-equipped ``dof`` would apply
    competing, duplicate servo force rather than driving anything new. ``dynamics``
    (armature and damping), by contrast, lives on each connection -- its own physical
    MuJoCo joint -- not the shared ``dof``, so it is set for every connection regardless
    of whether that connection's own ``dof`` was already equipped with an actuator;
    leaving a mimicked joint's own armature at zero starves the whole coupled mechanism
    of the numerical damping that keeps it from chattering under load, even though only
    one of its joints is ever actually driven directly.

    :param world: The world to add the actuators to, modified in place.
    :param connections: The connections to equip; non-1DOF connections are skipped.
    :param servo_for: Looks up a connection's own tuning by its joint name.
    :return: Each driven degree of freedom's own actuator, keyed by joint name.
    """
    actuators_by_joint_name: Dict[str, Actuator] = {}
    equipped_dofs: set[DegreeOfFreedom] = set()
    with world.modify_world():
        for connection in connections:
            if not isinstance(connection, ActiveConnection1DOF):
                continue
            dof = connection.raw_dof
            servo = servo_for(dof.name.name)
            connection.dynamics.armature = servo.dynamics.armature
            connection.dynamics.damping = servo.dynamics.damping
            if dof in equipped_dofs:
                continue
            equipped_dofs.add(dof)
            actuator = PositionServo(gains=servo.gains)
            actuator.add_dof(dof=dof)
            world.add_actuator(actuator=actuator)
            actuators_by_joint_name[dof.name.name] = actuator
    return actuators_by_joint_name


def equip_arms_with_servos(world: World, robot: Tracy) -> Dict[str, Actuator]:
    """
    Give every joint of both arms a position-servo actuator, its own passive damping,
    and armature, driven directly rather than through Giskard.

    :param world: The world to add the actuators to, modified in place.
    :param robot: The robot whose arms are driven.
    :return: Each driven degree of freedom's own actuator, keyed by joint name.
    """
    connections = [
        connection for arm in robot.get_arms() for connection in arm.active_connections
    ]
    return _equip_connections_with_servos(world, connections, _servo_tuning_for)


def equip_grippers_with_servos(world: World, robot: Tracy) -> Dict[str, Actuator]:
    """
    Give every joint of both grippers a position-servo actuator, mirroring
    :func:`equip_arms_with_servos` for the end effectors it does not cover.

    Also raises every gripper joint's own velocity limit; see
    :func:`_raise_gripper_velocity_limits`'s own docstring for why.

    :param world: The world to add the actuators to, modified in place.
    :param robot: The robot whose grippers are driven.
    :return: Each driven degree of freedom's own actuator, keyed by joint name.
    """
    _raise_gripper_velocity_limits(world, robot)
    connections = [
        connection
        for arm in robot.get_arms()
        for connection in arm.end_effector.active_connections
    ]
    return _equip_connections_with_servos(
        world, connections, lambda _: GRIPPER_JOINT_SERVO
    )


# %% cubes (stacking demo)

CUBE_FRICTION = [1.0, 0.05, 0.001]
"""
Contact friction (sliding, torsional, rolling; see
:attr:`~semantic_digital_twin.adapters.multi_sim.MujocoGeom.friction`) given to every
cube's collision geometry.

Matches ``coraplex_panda_demo/stacking_scene.xml``'s own proven-working cube
(``friction="1 0.05 0.001"``) exactly.
"""

CUBE_SOLVER_REFERENCE = [0.008, 1.0]
"""
Contact solver reference (see
:attr:`~semantic_digital_twin.adapters.multi_sim.MujocoGeom.solver_reference`) given to
every cube, matching ``coraplex_panda_demo``'s own proven-working cube
(``solref="0.008"``).

Stiffer than MuJoCo's own default (``0.02``): a soft contact lets a pinched cube sink
into the fingers and then slip back out as the arm lifts, rather than being held solidly
between them.
"""

CUBE_SOLVER_IMPEDANCE = [0.96, 0.99, 0.001, 0.5, 2.0]
"""
Contact solver impedance (see
:attr:`~semantic_digital_twin.adapters.multi_sim.MujocoGeom.solver_impedance`) given to
every cube, matching ``coraplex_panda_demo``'s own proven-working cube (``solimp="0.96
0.99"``, the remaining three values MuJoCo's own defaults).

Harder than MuJoCo's own default (``0.9 0.95``), for the same reason as
:data:`CUBE_SOLVER_REFERENCE`.
"""


def add_cube(
    world: World, name: str, position: Point3, size: float, color: Color
) -> Body:
    """
    Add a free-standing cube with real collision geometry to the world, so it can be
    pushed, grasped, and stacked by real contact rather than teleported into place or
    kinematically attached to whatever is holding it.

    Its collision geom gets :data:`EXTERNAL_COLLISION_BIT`, and is allowed to touch both
    the robot (:data:`ROBOT_COLLISION_BIT`) and other external things, including another
    cube -- see :func:`exclude_self_collision`. It also gets :data:`CUBE_FRICTION`,
    :data:`CUBE_SOLVER_REFERENCE`, and :data:`CUBE_SOLVER_IMPEDANCE`, without which a
    grasped cube sinks into the fingers under MuJoCo's own soft contact defaults and
    slips back out as the arm lifts.

    :param world: The world to add the cube to, modified in place.
    :param name: Name of the cube.
    :param position: Where the cube starts, in the world root frame.
    :param size: Edge length of the cube, in metres.
    :param color: Colour of the cube.
    :return: The newly added cube.
    """
    cube = Body(name=PrefixedName(name))
    shape = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=cube),
        scale=Scale(size, size, size),
        color=color,
    )
    mujoco_geom = _mujoco_geom_for(shape)
    mujoco_geom.friction = list(CUBE_FRICTION)
    mujoco_geom.solver_reference = list(CUBE_SOLVER_REFERENCE)
    mujoco_geom.solver_impedance = list(CUBE_SOLVER_IMPEDANCE)
    mujoco_geom.contype = EXTERNAL_COLLISION_BIT
    mujoco_geom.conaffinity = ROBOT_COLLISION_BIT | EXTERNAL_COLLISION_BIT
    geometry = ShapeCollection([shape], reference_frame=cube)
    cube.collision, cube.visual = geometry, geometry

    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=cube,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=position.x,
                    y=position.y,
                    z=position.z,
                    reference_frame=world.root,
                ),
            )
        )
    return cube
