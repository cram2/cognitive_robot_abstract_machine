from dataclasses import dataclass

import numpy as np
import pytest
from typing_extensions import Callable

from krrood_spatial_math.spatial_types import AxisAngle, Point3, RotationMatrix, Vector3

from giskardpy.motion_statechart.constraint_builders import GeometricConstraintBuilder
from giskardpy.motion_statechart.graph_node import MotionNodeArtifacts
from giskardpy.motion_statechart.weights import VelocityUnit
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.enforcement_strategy import IntegralStrategy
from giskardpy.qp.qp_controller_config import QPControllerConfig

ROTATION_REFERENCE_VELOCITY = 0.1
"""
Reference velocity of the rotation goals built in these tests, in rad/s.
"""


def controller_config() -> QPControllerConfig:
    """
    The controller configuration the bounds in these tests are evaluated with.
    """
    return QPControllerConfig(target_frequency=20, prediction_horizon=10)


def capped_equality_bounds(
    collection: ConstraintCollection,
) -> tuple[np.ndarray, float]:
    """
    Evaluates the bounds an integral strategy enforces for the collection's equality
    constraints.

    :return: The capped bounds, and the reachable change of the first constraint.
    """
    strategy = IntegralStrategy(
        degrees_of_freedom=[],
        constraints=collection.equality_constraints,
        qp_controller_config=controller_config(),
    )
    reachable_change = (
        strategy.reachable_change_of_constraint(strategy.constraints[0]).to_np().item()
    )
    return strategy.create_equality_bounds().to_np(), reachable_change


def test_point_goal_writes_three_equality_constraints():
    collection = ConstraintCollection()
    builder = GeometricConstraintBuilder(collection)
    builder.add_point_goal_constraints(
        frame_P_current=Point3(0.0, 0.0, 0.0),
        frame_P_goal=Point3(1.0, 2.0, 3.0),
        reference_velocity=0.1,
        quadratic_weight=1.0,
        name="goal",
    )
    assert len(collection.equality_constraints) == 3
    assert len(collection.inequality_constraints) == 0


def test_point_goal_caps_the_length_of_its_error_and_keeps_its_direction():
    error = np.array([1.0, 0.3, 0.0])
    collection = ConstraintCollection()
    GeometricConstraintBuilder(collection).add_point_goal_constraints(
        frame_P_current=Point3(0.0, 0.0, 0.0),
        frame_P_goal=Point3(*error),
        reference_velocity=0.1,
        quadratic_weight=1.0,
    )

    bounds, reachable_change = capped_equality_bounds(collection)

    assert np.linalg.norm(bounds) == pytest.approx(reachable_change)
    assert np.allclose(bounds / np.linalg.norm(bounds), error / np.linalg.norm(error))


def test_rotation_goal_writes_three_equality_constraints():
    collection = ConstraintCollection()
    builder = GeometricConstraintBuilder(collection)
    builder.add_rotation_goal_constraints(
        frame_R_current=RotationMatrix(),
        frame_R_goal=RotationMatrix.from_axis_angle(AxisAngle(Vector3.Z(), 0.5)),
        reference_velocity=0.1,
        quadratic_weight=1.0,
        name="goal",
    )
    assert len(collection.equality_constraints) == 3
    assert len(collection.inequality_constraints) == 0


def test_yaw_goal_writes_one_equality_constraint():
    collection = ConstraintCollection()
    builder = GeometricConstraintBuilder(collection)
    builder.add_yaw_goal_constraint(
        frame_R_current=RotationMatrix(),
        frame_R_goal=RotationMatrix.from_rpy(roll=0.2, pitch=-0.1, yaw=0.5),
        reference_velocity=0.1,
        quadratic_weight=1.0,
        name="goal",
    )
    assert len(collection.equality_constraints) == 1
    assert len(collection.inequality_constraints) == 0


@pytest.fixture(
    params=[
        GeometricConstraintBuilder.add_rotation_goal_constraints,
        GeometricConstraintBuilder.add_yaw_goal_constraint,
    ],
    ids=["rotation", "yaw"],
)
def add_rotation_goal(request: pytest.FixtureRequest) -> Callable[..., None]:
    """
    :return: A method of :class:`GeometricConstraintBuilder` adding the constraints of a
        rotation goal.
    """
    return request.param


def rotation_goal_bounds(
    add_rotation_goal: Callable[..., None], angle: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Builds a rotation goal about the z-axis and evaluates its bounds.

    :param add_rotation_goal: The method of :class:`GeometricConstraintBuilder` adding
        the goal's constraints.
    :param angle: How far the goal is rotated away from the current orientation.
    :return: The uncapped bounds, the capped bounds, and the reachable change.
    """
    collection = ConstraintCollection()
    add_rotation_goal(
        GeometricConstraintBuilder(collection),
        frame_R_current=RotationMatrix(),
        frame_R_goal=RotationMatrix.from_axis_angle(AxisAngle(Vector3.Z(), angle)),
        reference_velocity=ROTATION_REFERENCE_VELOCITY,
        quadratic_weight=1.0,
    )
    uncapped = np.array(
        [c.bound.to_np().item() for c in collection.equality_constraints]
    )
    capped, reachable_change = capped_equality_bounds(collection)
    return uncapped, capped, reachable_change


def test_rotation_goal_caps_the_angle_it_demands(
    add_rotation_goal: Callable[..., None],
):
    """
    The bounds are in units of the half angle, so the cap is applied to the angle:

    a far away goal demands a rotation of exactly twice the reachable change.
    """
    uncapped, capped, reachable_change = rotation_goal_bounds(
        add_rotation_goal, np.pi / 2
    )
    half_angle = np.arcsin(np.linalg.norm(uncapped))

    remaining_half_angle = np.arcsin(np.linalg.norm(uncapped) - np.linalg.norm(capped))

    assert remaining_half_angle == pytest.approx(half_angle - reachable_change)


def test_rotation_goal_demands_as_much_rotation_as_its_reference_velocity_sustains(
    add_rotation_goal: Callable[..., None],
):
    """
    Within the control horizon, the half angle may change by the reference velocity
    sustained over the control horizon, as much as a single row of the quaternion error
    was allowed to change before the angle was capped.
    """
    _, _, reachable_change = rotation_goal_bounds(add_rotation_goal, np.pi / 2)
    config = controller_config()

    assert reachable_change == pytest.approx(
        ROTATION_REFERENCE_VELOCITY
        * config.control_time_step.total_seconds()
        * config.control_horizon
    )


def test_rotation_goal_within_reach_is_not_capped(
    add_rotation_goal: Callable[..., None],
):
    uncapped, capped, reachable_change = rotation_goal_bounds(add_rotation_goal, 0.01)

    assert 0.01 < 2 * reachable_change
    assert np.allclose(capped, uncapped)


def test_node_artifacts_geometry_writes_into_its_constraints():
    artifacts = MotionNodeArtifacts()
    artifacts.geometry.add_point_goal_constraints(
        frame_P_current=Point3(0.0, 0.0, 0.0),
        frame_P_goal=Point3(1.0, 2.0, 3.0),
        reference_velocity=0.1,
        quadratic_weight=1.0,
        name="goal",
    )
    assert len(artifacts.constraints.equality_constraints) == 3


# %% weight normalization


@pytest.fixture(
    params=[
        GeometricConstraintBuilder.add_point_goal_constraints,
        GeometricConstraintBuilder.add_vector_goal_constraints,
        GeometricConstraintBuilder.add_rotation_goal_constraints,
        GeometricConstraintBuilder.add_yaw_goal_constraint,
    ],
    ids=["point", "vector", "rotation", "yaw"],
)
def add_goal(request: pytest.FixtureRequest) -> Callable[..., None]:
    """
    :return: A method of :class:`GeometricConstraintBuilder` adding the constraints of a
        goal.
    """
    return request.param


@dataclass
class EnforcedGoal:
    """
    What an integral strategy makes of the constraints of a goal.
    """

    slack_weights: np.ndarray
    """
    The quadratic slack weight of every constraint.
    """

    capped_bounds: np.ndarray
    """
    The bound of every constraint, capped to what may be demanded within the control
    horizon.
    """

    @property
    def pull(self) -> np.ndarray:
        """
        How hard every constraint pulls on the joints while they are at rest, up to a
        factor all constraints share.
        """
        return self.slack_weights * self.capped_bounds


def enforced_goal(
    add_goal: Callable[..., None], reference_velocity: float, distance: float
) -> EnforcedGoal:
    """
    Builds a goal and evaluates it with an integral strategy.

    :param add_goal: The method of :class:`GeometricConstraintBuilder` adding the goal's
        constraints.
    :param reference_velocity: The reference velocity of the goal.
    :param distance: How far away the goal is, in meters or radians.
    :return: The goal as the integral strategy enforces it.
    """
    current_and_goal = {
        GeometricConstraintBuilder.add_point_goal_constraints: (
            Point3(0.0, 0.0, 0.0),
            Point3(distance, 0.0, 0.0),
        ),
        GeometricConstraintBuilder.add_vector_goal_constraints: (
            Vector3.X(),
            Vector3(np.cos(distance), np.sin(distance), 0.0),
        ),
        GeometricConstraintBuilder.add_rotation_goal_constraints: (
            RotationMatrix(),
            RotationMatrix.from_axis_angle(AxisAngle(Vector3.Z(), distance)),
        ),
        GeometricConstraintBuilder.add_yaw_goal_constraint: (
            RotationMatrix(),
            RotationMatrix.from_axis_angle(AxisAngle(Vector3.Z(), distance)),
        ),
    }
    current, goal = current_and_goal[add_goal]
    collection = ConstraintCollection()
    add_goal(
        GeometricConstraintBuilder(collection),
        current,
        goal,
        reference_velocity=reference_velocity,
        quadratic_weight=1.0,
    )
    strategy = IntegralStrategy(
        degrees_of_freedom=[],
        constraints=collection.equality_constraints,
        qp_controller_config=controller_config(),
    )
    return EnforcedGoal(
        slack_weights=strategy.create_slack_variables().quadratic_weights.to_np(),
        capped_bounds=strategy.create_equality_bounds().to_np(),
    )


def test_slowing_down_a_goal_within_reach_does_not_change_its_weight(add_goal):
    slow = enforced_goal(add_goal, reference_velocity=0.05, distance=0.005)
    fast = enforced_goal(add_goal, reference_velocity=0.5, distance=0.005)

    assert np.allclose(slow.slack_weights, fast.slack_weights)


def test_slowing_down_a_goal_out_of_reach_does_not_change_how_hard_it_pulls(add_goal):
    slow = enforced_goal(add_goal, reference_velocity=0.05, distance=0.5)
    fast = enforced_goal(add_goal, reference_velocity=0.5, distance=0.5)

    assert np.allclose(slow.pull, fast.pull)


def test_a_quaternion_error_is_normalized_by_how_fast_it_changes_with_the_angle(
    add_rotation_goal,
):
    collection = ConstraintCollection()
    add_rotation_goal(
        GeometricConstraintBuilder(collection),
        frame_R_current=RotationMatrix(),
        frame_R_goal=RotationMatrix.from_axis_angle(AxisAngle(Vector3.Z(), 0.5)),
        reference_velocity=ROTATION_REFERENCE_VELOCITY,
        quadratic_weight=1.0,
    )

    expected = (
        VelocityUnit.ROTATION * GeometricConstraintBuilder.quaternion_error_per_radian
    )
    assert [c.velocity_unit for c in collection.equality_constraints] == [
        pytest.approx(expected)
    ] * len(collection.equality_constraints)
