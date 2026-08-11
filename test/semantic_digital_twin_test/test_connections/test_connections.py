import inspect
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood.utils import recursive_subclasses
from semantic_digital_twin.exceptions import (
    MissingConnectionAxisError,
    MissingReferenceFrameError,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.connection_properties import JointDynamics
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
    FixedConnection,
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
    ScrewConnection,
)
from semantic_digital_twin.world_description.world_entity import Body, Connection


def _add_drive(world_with_two_bodies, drive_type):
    """
    Creates a drive connection of ``drive_type`` and adds it to the world.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = drive_type.create_with_dofs(world, parent, child)
        world.add_connection(connection)
    return connection


def test_create_with_dofs_threads_parent_T_connection_expression(world_with_two_bodies):
    world, parent, child = world_with_two_bodies
    parent_T_connection = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3, y=0.4)
    with world.modify_world():
        connection = RevoluteConnection.create_with_dofs(
            world,
            parent,
            child,
            axis=Vector3.Z(),
            parent_T_connection_expression=parent_T_connection,
        )
        world.add_connection(connection)
    assert_allclose(connection.origin, parent_T_connection, atol=1e-9)


def test_connection6dof_origin_setter_round_trips_with_non_identity_constants(
    world_with_two_bodies,
):
    """
    Assigning ``.origin`` must make a subsequent read return exactly what was assigned,
    even when the connection carries non-identity ``parent_T_connection_expression``/
    ``connection_T_child_expression`` constants (e.g. a body spawned with a prior
    offset): the setter must un-compose those constants, not write the assigned
    transform in as the DOF state directly.
    """
    world, parent, child = world_with_two_bodies
    parent_T_connection = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.3, y=0.4, z=0.1
    )
    connection_T_child = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.05, yaw=0.2)
    with world.modify_world():
        connection = Connection6DoF.create_with_dofs(
            world,
            parent,
            child,
            parent_T_connection_expression=parent_T_connection,
            connection_T_child_expression=connection_T_child,
        )
        world.add_connection(connection)

    desired_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=-2.0, z=0.5, yaw=0.7, reference_frame=parent
    )
    connection.origin = desired_origin

    assert_allclose(connection.origin, desired_origin, atol=1e-9)


def test_connection6dof_origin_setter_requires_a_reference_frame(world_with_two_bodies):
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = Connection6DoF.create_with_dofs(world, parent, child)
        world.add_connection(connection)

    transformation_without_reference_frame = (
        HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0)
    )

    with pytest.raises(MissingReferenceFrameError):
        connection.origin = transformation_without_reference_frame


def test_connection6dof_origin_setter_converts_from_a_non_parent_reference_frame(
    world_with_two_bodies,
):
    """
    The assigned transform does not need to already be expressed in the parent frame:
    the setter must convert it via the world graph before un-composing the connection's
    own constant offsets.
    """
    world, parent, child = world_with_two_bodies
    grandparent = Body(name=PrefixedName("grandparent"))
    grandparent_T_parent = HomogeneousTransformationMatrix.from_xyz_rpy(x=2.0, y=1.0)
    with world.modify_world():
        world.add_connection(
            FixedConnection.create_with_dofs(
                world,
                grandparent,
                parent,
                parent_T_connection_expression=grandparent_T_parent,
            )
        )
        connection = Connection6DoF.create_with_dofs(world, parent, child)
        world.add_connection(connection)

    desired_origin_in_grandparent = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=2.5, y=1.5, z=0.2, reference_frame=grandparent
    )
    connection.origin = desired_origin_in_grandparent

    expected_origin_in_parent = world.transform(desired_origin_in_grandparent, parent)
    assert_allclose(connection.origin, expected_origin_in_parent, atol=1e-9)


def test_reference_origin_excludes_joint_state(world_with_two_bodies):
    """
    The reference origin stays at the zero configuration, the origin follows the joint.

    A simulator places a body's static frame once, at build time. Using the joint-
    carrying origin there bakes the current joint state in, and the simulator joint then
    applies it a second time.
    """
    world, parent, child = world_with_two_bodies
    parent_T_connection = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.3, y=0.4)
    with world.modify_world():
        connection = RevoluteConnection.create_with_dofs(
            world,
            parent,
            child,
            axis=Vector3.Z(),
            parent_T_connection_expression=parent_T_connection,
        )
        world.add_connection(connection)

    origin_at_zero = connection.origin_as_position_quaternion().evaluate()[0]
    reference_at_zero = connection.reference_origin_as_position_quaternion().evaluate()[
        0
    ]
    assert_allclose(origin_at_zero, reference_at_zero, atol=1e-9)

    joint_position = 0.7
    with world.modify_world():
        world.state[connection.active_dofs[0].id].position = joint_position

    origin_when_rotated = connection.origin_as_position_quaternion().evaluate()[0]
    reference_when_rotated = (
        connection.reference_origin_as_position_quaternion().evaluate()[0]
    )

    # The reference is unaffected by the joint, so it is safe as a static frame.
    assert_allclose(reference_when_rotated, reference_at_zero, atol=1e-9)
    # The origin carries the joint's half-angle quaternion about the z axis.
    expected_origin = [
        0.3,
        0.4,
        0.0,
        0.0,
        0.0,
        math.sin(joint_position / 2.0),
        math.cos(joint_position / 2.0),
    ]
    assert_allclose(origin_when_rotated, expected_origin, atol=1e-9)


@pytest.mark.parametrize("drive_type", [OmniDrive, DifferentialDrive])
def test_has_hardware_interface_round_trip(world_with_two_bodies, drive_type):
    connection = _add_drive(world_with_two_bodies, drive_type)
    assert not connection.has_hardware_interface
    assert connection.controlled_dofs == []

    connection.has_hardware_interface = True
    assert connection.has_hardware_interface
    assert set(connection.controlled_dofs) == set(connection.active_dofs)

    connection.has_hardware_interface = False
    assert not connection.has_hardware_interface
    assert connection.controlled_dofs == []


@pytest.mark.parametrize("drive_type", [OmniDrive, DifferentialDrive])
def test_has_hardware_interface_reflects_any_active_dof(
    world_with_two_bodies, drive_type
):
    connection = _add_drive(world_with_two_bodies, drive_type)
    connection.yaw.has_hardware_interface = True
    assert connection.has_hardware_interface


# %% screw connection


def _add_screw_connection(
    world_with_two_bodies,
    screw_pitch: float,
    multiplier: float = 1.0,
    parent_T_connection_expression: HomogeneousTransformationMatrix | None = None,
) -> ScrewConnection:
    """
    Creates a screw connection about the z-axis and adds it to the world.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = ScrewConnection.create_with_dofs(
            world,
            parent,
            child,
            axis=Vector3.Z(),
            screw_pitch=screw_pitch,
            multiplier=multiplier,
            parent_T_connection_expression=parent_T_connection_expression,
        )
        world.add_connection(connection)
    return connection


def _expected_screw_transform(angle: float, screw_pitch: float) -> np.ndarray:
    """
    The analytic parent_T_child matrix of a screw pair about the z-axis: rotation by
    ``angle`` coupled with translation ``screw_pitch * angle / (2 * pi)`` along z.
    """
    expected = np.eye(4)
    expected[0, 0] = math.cos(angle)
    expected[0, 1] = -math.sin(angle)
    expected[1, 0] = math.sin(angle)
    expected[1, 1] = math.cos(angle)
    expected[2, 3] = screw_pitch * angle / (2.0 * math.pi)
    return expected


def test_screw_connection_origin_couples_rotation_and_translation(
    world_with_two_bodies,
):
    world, parent, child = world_with_two_bodies
    screw_pitch = 0.01
    connection = _add_screw_connection(
        world_with_two_bodies,
        screw_pitch=screw_pitch,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.3, y=0.4
        ),
    )

    reference_at_zero = connection.reference_origin_as_position_quaternion().evaluate()[
        0
    ]

    joint_position = 0.7
    with world.modify_world():
        world.state[connection.active_dofs[0].id].position = joint_position

    expected_origin = [
        0.3,
        0.4,
        screw_pitch * joint_position / (2.0 * math.pi),
        0.0,
        0.0,
        math.sin(joint_position / 2.0),
        math.cos(joint_position / 2.0),
    ]
    origin_when_screwed = connection.origin_as_position_quaternion().evaluate()[0]
    assert_allclose(origin_when_screwed, expected_origin, atol=1e-9)

    # The reference origin stays at the zero configuration.
    reference_when_screwed = (
        connection.reference_origin_as_position_quaternion().evaluate()[0]
    )
    assert_allclose(reference_when_screwed, reference_at_zero, atol=1e-9)


@pytest.mark.parametrize("joint_position", [0.0, math.pi / 2.0, 2.0 * math.pi, -1.0])
def test_screw_connection_numeric_forward_kinematics(
    world_with_two_bodies, joint_position
):
    world, parent, child = world_with_two_bodies
    screw_pitch = 0.01
    connection = _add_screw_connection(world_with_two_bodies, screw_pitch=screw_pitch)

    with world.modify_world():
        world.state[connection.active_dofs[0].id].position = joint_position

    parent_T_child = world.compute_forward_kinematics_np(parent, child)
    assert_allclose(
        parent_T_child,
        _expected_screw_transform(joint_position, screw_pitch),
        atol=1e-9,
    )


def test_screw_connection_negative_screw_pitch(world_with_two_bodies):
    world, parent, child = world_with_two_bodies
    screw_pitch = -0.01
    connection = _add_screw_connection(world_with_two_bodies, screw_pitch=screw_pitch)

    joint_position = 0.5
    with world.modify_world():
        world.state[connection.active_dofs[0].id].position = joint_position

    # A left-handed thread rotates the same way but translates in the opposite direction.
    parent_T_child = world.compute_forward_kinematics_np(parent, child)
    assert_allclose(
        parent_T_child,
        _expected_screw_transform(joint_position, screw_pitch),
        atol=1e-9,
    )


def test_screw_connection_multiplier_mirrors_motion(world_with_two_bodies):
    world, parent, child = world_with_two_bodies
    screw_pitch = 0.01
    connection = _add_screw_connection(
        world_with_two_bodies, screw_pitch=screw_pitch, multiplier=-1.0
    )

    raw_position = 0.5
    with world.modify_world():
        world.state[connection.active_dofs[0].id].position = raw_position

    # The connection moves by multiplier * raw position.
    parent_T_child = world.compute_forward_kinematics_np(parent, child)
    assert_allclose(
        parent_T_child,
        _expected_screw_transform(-raw_position, screw_pitch),
        atol=1e-9,
    )


def test_screw_connection_rotation_angle_for_travel_distance(world_with_two_bodies):
    screw_pitch = 0.005
    connection = _add_screw_connection(world_with_two_bodies, screw_pitch=screw_pitch)

    # One full turn advances the child by one screw pitch.
    assert_allclose(
        connection.rotation_angle_for_travel_distance(screw_pitch),
        2.0 * math.pi,
    )
    # Travelling against the axis requires rotating in the opposite direction.
    assert_allclose(
        connection.rotation_angle_for_travel_distance(-screw_pitch),
        -2.0 * math.pi,
    )


def test_screw_connection_rotation_angle_for_travel_distance_left_handed_thread(
    world_with_two_bodies,
):
    screw_pitch = -0.005
    connection = _add_screw_connection(world_with_two_bodies, screw_pitch=screw_pitch)

    # A left-handed thread rotates the other way for the same travel.
    assert_allclose(
        connection.rotation_angle_for_travel_distance(0.005),
        -2.0 * math.pi,
    )


def test_screw_connection_copy_with_new_parent_preserves_screw_pitch(
    world_with_two_bodies,
):
    world, parent, child = world_with_two_bodies
    screw_pitch = 0.01
    connection = _add_screw_connection(world_with_two_bodies, screw_pitch=screw_pitch)

    new_parent = Body(name=PrefixedName("new_parent"))
    copied_connection = connection.copy_with_new_parent(
        new_parent, HomogeneousTransformationMatrix.from_xyz_rpy(x=0.1)
    )

    assert copied_connection.screw_pitch == screw_pitch
    assert copied_connection.raw_dof is connection.raw_dof
    assert copied_connection.axis == connection.axis
    assert copied_connection.parent is new_parent


# %% create_with_dofs parameter forwarding


def _connection_types_implementing_create_with_dofs() -> list[type[Connection]]:
    """
    The connection types that carry their own ``create_with_dofs`` implementation.
    """
    return [
        connection_type
        for connection_type in recursive_subclasses(Connection)
        if "create_with_dofs" in connection_type.__dict__
    ]


@pytest.mark.parametrize(
    "connection_type", _connection_types_implementing_create_with_dofs()
)
def test_create_with_dofs_accepts_the_shared_parameters(connection_type):
    """
    Every implementation must accept the parameters a polymorphic caller supplies.

    ``ConnectionSpecification.connect`` and ``RobotSpecification.spawn`` dispatch on a
    connection type they do not know statically, so these parameters are the interface
    every connection family has to honour.
    """
    parameters = inspect.signature(connection_type.create_with_dofs).parameters
    assert {
        "world",
        "parent",
        "child",
        "name",
        "parent_T_connection_expression",
        "connection_T_child_expression",
    } <= set(parameters)


@pytest.mark.parametrize(
    "connection_type, additional_arguments",
    [
        (FixedConnection, {}),
        (Connection6DoF, {}),
        (OmniDrive, {}),
        (DifferentialDrive, {}),
        (RevoluteConnection, {"axis": Vector3.Z()}),
    ],
)
def test_create_with_dofs_rejects_unknown_keyword_argument(
    world_with_two_bodies, connection_type, additional_arguments
):
    """
    A misspelled parameter must fail loudly instead of falling back to the default.

    ``translation_velocity_limit`` is the singular-form typo of a real drive parameter:
    swallowing it would hand back a drive running at the default velocity limit.
    """
    world, parent, child = world_with_two_bodies
    with pytest.raises(TypeError):
        connection_type.create_with_dofs(
            world,
            parent,
            child,
            translation_velocity_limit=0.2,
            **additional_arguments,
        )


@pytest.mark.parametrize("drive_type", [OmniDrive, DifferentialDrive])
def test_create_with_dofs_applies_drive_velocity_limits(
    world_with_two_bodies, drive_type
):
    """
    The limits are asserted with non-default values, so ignoring them is visible.
    """
    world, parent, child = world_with_two_bodies
    translation_velocity_limit = 0.25
    rotation_velocity_limit = 0.15
    with world.modify_world():
        connection = drive_type.create_with_dofs(
            world,
            parent,
            child,
            translation_velocity_limits=translation_velocity_limit,
            rotation_velocity_limits=rotation_velocity_limit,
        )
        world.add_connection(connection)

    assert_allclose(
        connection.x_velocity.limits.upper.velocity, translation_velocity_limit
    )
    assert_allclose(
        connection.x_velocity.limits.lower.velocity, -translation_velocity_limit
    )
    assert_allclose(connection.yaw.limits.upper.velocity, rotation_velocity_limit)
    assert_allclose(connection.yaw.limits.lower.velocity, -rotation_velocity_limit)


def test_create_with_dofs_applies_lateral_velocity_limit_of_omni_drive(
    world_with_two_bodies,
):
    """
    The omni drive's lateral degree of freedom shares the translation limit.
    """
    world, parent, child = world_with_two_bodies
    translation_velocity_limit = 0.25
    with world.modify_world():
        connection = OmniDrive.create_with_dofs(
            world,
            parent,
            child,
            translation_velocity_limits=translation_velocity_limit,
        )
        world.add_connection(connection)

    assert_allclose(
        connection.y_velocity.limits.upper.velocity, translation_velocity_limit
    )
    assert_allclose(
        connection.y_velocity.limits.lower.velocity, -translation_velocity_limit
    )


def test_create_with_dofs_applies_multiplier_and_offset(world_with_two_bodies):
    """
    The scaling of the single degree of freedom is set from the arguments.
    """
    world, parent, child = world_with_two_bodies
    multiplier = 2.0
    offset = 0.5
    with world.modify_world():
        connection = RevoluteConnection.create_with_dofs(
            world,
            parent,
            child,
            axis=Vector3.Z(),
            multiplier=multiplier,
            offset=offset,
        )
        world.add_connection(connection)

    assert_allclose(connection.multiplier, multiplier)
    assert_allclose(connection.offset, offset)

    raw_position = 1.0
    world.state[connection.raw_dof.id].position = raw_position
    assert_allclose(connection.position, raw_position * multiplier + offset)


@pytest.mark.parametrize("connection_type", [PrismaticConnection, RevoluteConnection])
def test_create_with_dofs_without_axis_raises(world_with_two_bodies, connection_type):
    """
    A single-DoF connection has no meaningful default axis, so it must be supplied.
    """
    world, parent, child = world_with_two_bodies
    with pytest.raises(MissingConnectionAxisError):
        connection_type.create_with_dofs(world, parent, child)


def test_create_with_dofs_registers_the_six_dof_state(world_with_two_bodies):
    """
    The seven degrees of freedom are added and the quaternion starts at identity.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = Connection6DoF.create_with_dofs(world, parent, child)
        world.add_connection(connection)

    assert len(connection.passive_dofs) == 7
    assert set(connection.passive_dofs) <= set(world.degrees_of_freedom)
    assert_allclose(world.state[connection.qw.id].position, 1.0)


def test_create_with_dofs_of_fixed_connection_adds_no_degrees_of_freedom(
    world_with_two_bodies,
):
    """
    A fixed connection has nothing to move, so it must not touch the world's DoFs.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = FixedConnection.create_with_dofs(world, parent, child)
        world.add_connection(connection)

    assert connection.dofs == []
    assert world.degrees_of_freedom == []


# %% joint dynamics


def test_joint_dynamics_custom_values():
    armature = 1.5
    dry_friction = 0.2
    damping = 0.05
    joint_dynamics = JointDynamics(
        armature=armature, dry_friction=dry_friction, damping=damping
    )
    assert_allclose(joint_dynamics.armature, armature)
    assert_allclose(joint_dynamics.dry_friction, dry_friction)
    assert_allclose(joint_dynamics.damping, damping)

    joint_prop_dict = joint_dynamics.__dict__
    assert_allclose(joint_prop_dict["armature"], armature)
    assert_allclose(joint_prop_dict["dry_friction"], dry_friction)
    assert_allclose(joint_prop_dict["damping"], damping)
