import numpy as np
import pytest

from semantic_digital_twin.spatial_types import Point3, RotationMatrix, Vector3

from giskardpy.motion_statechart.constraint_builders import GeometricConstraintBuilder
from giskardpy.motion_statechart.graph_node import NodeArtifacts
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
    reach = strategy.reachable_change(strategy.constraints[0]).to_np().item()
    return strategy.create_equality_bounds().to_np(), reach


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

    bounds, reach = capped_equality_bounds(collection)

    assert np.linalg.norm(bounds) == pytest.approx(reach)
    assert np.allclose(bounds / np.linalg.norm(bounds), error / np.linalg.norm(error))


def test_rotation_goal_writes_three_equality_constraints():
    collection = ConstraintCollection()
    builder = GeometricConstraintBuilder(collection)
    builder.add_rotation_goal_constraints(
        frame_R_current=RotationMatrix(),
        frame_R_goal=RotationMatrix.from_axis_angle(Vector3.Z(), 0.5),
        reference_velocity=0.1,
        quadratic_weight=1.0,
        name="goal",
    )
    assert len(collection.equality_constraints) == 3
    assert len(collection.inequality_constraints) == 0


def rotation_goal_bounds(angle: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Builds a rotation goal about the z-axis and evaluates its bounds.

    :param angle: How far the goal is rotated away from the current orientation.
    :return: The uncapped bounds, the capped bounds, and the reachable change.
    """
    collection = ConstraintCollection()
    GeometricConstraintBuilder(collection).add_rotation_goal_constraints(
        frame_R_current=RotationMatrix(),
        frame_R_goal=RotationMatrix.from_axis_angle(Vector3.Z(), angle),
        reference_velocity=ROTATION_REFERENCE_VELOCITY,
        quadratic_weight=1.0,
    )
    uncapped = np.array(
        [c.bound.to_np().item() for c in collection.equality_constraints]
    )
    capped, reach = capped_equality_bounds(collection)
    return uncapped, capped, reach


def test_rotation_goal_caps_the_angle_it_demands():
    """
    The bounds are in units of the half angle, so the cap is applied to the angle:

    a far away goal demands a rotation of exactly twice the reachable change.
    """
    uncapped, capped, reach = rotation_goal_bounds(np.pi / 2)
    half_angle = np.arcsin(np.linalg.norm(uncapped))

    remaining_half_angle = np.arcsin(np.linalg.norm(uncapped) - np.linalg.norm(capped))

    assert remaining_half_angle == pytest.approx(half_angle - reach)


def test_rotation_goal_demands_as_much_rotation_as_its_reference_velocity_sustains():
    """
    One plan may change the half angle by the reference velocity sustained over the
    control horizon, as much as a single row of the quaternion error was allowed to
    change before the angle was capped.
    """
    _, _, reach = rotation_goal_bounds(np.pi / 2)
    config = controller_config()

    assert reach == pytest.approx(
        ROTATION_REFERENCE_VELOCITY
        * config.control_time_step.total_seconds()
        * config.control_horizon
    )


def test_rotation_goal_within_reach_is_not_capped():
    uncapped, capped, reach = rotation_goal_bounds(0.01)

    assert 0.01 < 2 * reach
    assert np.allclose(capped, uncapped)


def test_node_artifacts_geometry_writes_into_its_constraints():
    artifacts = NodeArtifacts()
    artifacts.geometry.add_point_goal_constraints(
        frame_P_current=Point3(0.0, 0.0, 0.0),
        frame_P_goal=Point3(1.0, 2.0, 3.0),
        reference_velocity=0.1,
        quadratic_weight=1.0,
        name="goal",
    )
    assert len(artifacts.constraints.equality_constraints) == 3
