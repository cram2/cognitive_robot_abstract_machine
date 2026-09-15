"""
Unit tests for :class:`NumericPose` and the pose reads that go through it.

A pose read that builds a CasADi expression cannot be done from a second thread: CasADi
releases the GIL and counts its expression-node references without atomics, so two
threads reading poses free nodes each other still hold. These tests pin down that the
numeric read path returns the same numbers as the symbolic one and that it reaches them
without building any symbolic expression.
"""

from __future__ import annotations

import numpy as np
import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.numeric import NumericPose
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any

ROLL_PITCH_YAW_CASES = [
    (0.0, 0.0, 0.0),
    (np.pi, 0.0, 0.0),
    (0.0, np.pi, 0.0),
    (0.0, 0.0, np.pi),
    (0.3, -1.1, 2.4),
]
"""
Orientations covering each branch the conversion picks between: a positive trace, and a
largest diagonal entry at each of the three axes.
"""


def _body_at(roll: float, pitch: float, yaw: float) -> Body:
    """
    A body held at a known pose by a fixed connection to its world's root.

    :param roll: The body's rotation around the x-axis.
    :param pitch: The body's rotation around the y-axis.
    :param yaw: The body's rotation around the z-axis.
    """
    world = World()
    root = Body(name=PrefixedName("world"))
    body = Body(name=PrefixedName("object"))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1.5, -2.0, 0.25, roll, pitch, yaw
                ),
            )
        )
    return body


def _refuse(*args: Any, **kwargs: Any) -> Any:
    """
    Stand in for a symbolic call that the numeric read path must never make.
    """
    raise AssertionError("the read path built a symbolic expression")


def _position_and_quaternion(pose: Pose) -> tuple[tuple, tuple]:
    """
    A symbolic pose's own position and quaternion, read out through the symbolic path.

    :param pose: The pose to read.
    """
    return (
        tuple(pose.to_position().to_np()[:3]),
        tuple(pose.to_quaternion().to_np()[:4]),
    )


# %% the numeric conversion agrees with the symbolic one
@pytest.mark.parametrize("roll, pitch, yaw", ROLL_PITCH_YAW_CASES)
def test_a_numeric_pose_holds_the_symbolic_poses_own_numbers(
    roll: float, pitch: float, yaw: float
) -> None:
    """
    The numeric path replaces the symbolic one, so it has to reach the same quaternion
    for every branch of the conversion, including the near-degenerate half-turns.
    """
    pose = Pose.from_xyz_rpy(1.5, -2.0, 0.25, roll, pitch, yaw)
    expected_position, expected_quaternion = _position_and_quaternion(pose)

    read_out = NumericPose.of_pose(pose)

    assert read_out.position == pytest.approx(expected_position)
    assert read_out.quaternion == pytest.approx(expected_quaternion)


def test_a_numeric_pose_reads_a_transformation_matrix_as_it_reads_a_pose() -> None:
    """
    A transform and the pose it describes are the same rigid placement, so reading
    either out has to give the same numbers.
    """
    root_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.5, -2.0, 0.25, 0.3, -1.1, 2.4
    )

    read_out = NumericPose.from_transformation_matrix(root_T_body.to_np())

    assert read_out == NumericPose.of_pose(root_T_body.to_pose())


def test_a_numeric_pose_holds_only_plain_floats() -> None:
    """
    A value still holding a CasADi expression would be evaluated again by whichever
    thread renders it, which is the hazard being avoided.
    """
    read_out = NumericPose.of_pose(Pose.from_xyz_rpy(1.0, 2.0, 3.0))

    assert all(type(value) is float for value in read_out.position)
    assert all(type(value) is float for value in read_out.quaternion)


@pytest.mark.parametrize("roll, pitch, yaw", ROLL_PITCH_YAW_CASES)
def test_a_pose_built_from_a_numeric_pose_places_what_it_was_read_from(
    roll: float, pitch: float, yaw: float
) -> None:
    """
    An event keeps poses as numbers and hands them on as symbolic poses, so turning one
    back must land where the original stood.
    """
    body = _body_at(roll, pitch, yaw)
    expected_position, expected_quaternion = _position_and_quaternion(body.global_pose)

    rebuilt = Pose.from_numeric_pose(body.numeric_global_pose, body._world.root)

    rebuilt_position, rebuilt_quaternion = _position_and_quaternion(rebuilt)
    assert rebuilt_position == pytest.approx(expected_position)
    assert np.abs(np.dot(rebuilt_quaternion, expected_quaternion)) == pytest.approx(1.0)
    assert rebuilt.reference_frame is body._world.root


def test_a_numeric_pose_lists_its_position_before_its_quaternion() -> None:
    """
    The list is the wire format a pose is published as, so its order is part of the
    contract.
    """
    read_out = NumericPose(position=(1.0, 2.0, 3.0), quaternion=(0.0, 0.0, 0.0, 1.0))

    assert read_out.to_position_quaternion_list() == [
        *read_out.position,
        *read_out.quaternion,
    ]


# %% reads that build no symbolic expression
@pytest.mark.parametrize("roll, pitch, yaw", ROLL_PITCH_YAW_CASES)
def test_a_bodys_numeric_global_pose_matches_its_symbolic_one(
    roll: float, pitch: float, yaw: float
) -> None:
    """
    The numeric read is only a substitute for the symbolic one if it places the body
    where the symbolic one does.
    """
    body = _body_at(roll, pitch, yaw)
    expected_position, expected_quaternion = _position_and_quaternion(body.global_pose)

    numeric = body.numeric_global_pose

    assert numeric.position == pytest.approx(expected_position)
    assert numeric.quaternion == pytest.approx(expected_quaternion)


def test_a_bodys_numeric_global_pose_builds_no_transformation_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Wrapping forward kinematics in a transformation matrix is what makes an ordinary
    pose read a CasADi call, and is what a read from another thread must avoid.
    """
    body = _body_at(0.3, -1.1, 2.4)
    expected = body.numeric_global_pose
    monkeypatch.setattr(World, "compute_forward_kinematics", _refuse)

    assert body.numeric_global_pose == expected


# %% comparing two numeric poses
@pytest.mark.parametrize("roll, pitch, yaw", ROLL_PITCH_YAW_CASES)
def test_the_angle_between_two_numeric_poses_matches_the_symbolic_one(
    roll: float, pitch: float, yaw: float
) -> None:
    """
    Motion detection compares consecutive poses, and the numeric comparison replaces the
    symbolic rotational error, so it has to name the same rotation.

    ..note:: :meth:`RotationMatrix.rotational_distance` reports some rotations the long way
       round, as ``2*pi`` minus the angle; the numeric measure always takes the shorter
       way, so the two are compared modulo that wrap.
    """
    first = Pose.from_xyz_rpy(0.0, 0.0, 0.0, 0.2, -0.4, 1.1)
    second = Pose.from_xyz_rpy(1.0, 2.0, 3.0, roll, pitch, yaw)
    symbolic = float(
        first.to_rotation_matrix().rotational_distance(second.to_rotation_matrix())
    )
    expected = min(symbolic, 2.0 * np.pi - symbolic)

    measured = NumericPose.of_pose(first).rotational_distance(
        NumericPose.of_pose(second)
    )

    assert measured == pytest.approx(expected, abs=1e-6)


def test_the_distance_between_two_numeric_poses_matches_the_symbolic_one() -> None:
    """
    The same comparison measures how far a tracked object moved.
    """
    first = Pose.from_xyz_rpy(1.0, 2.0, 3.0)
    second = Pose.from_xyz_rpy(1.5, -2.0, 0.25)
    expected = float(first.to_position().euclidean_distance(second.to_position()))

    measured = NumericPose.of_pose(first).euclidean_distance(
        NumericPose.of_pose(second)
    )

    assert measured == pytest.approx(expected)
