from semantic_digital_twin.spatial_types import Point3, RotationMatrix, Vector3

from giskardpy.motion_statechart.constraint_builders import GeometricConstraintBuilder
from giskardpy.motion_statechart.graph_node import NodeArtifacts
from giskardpy.qp.constraint_collection import ConstraintCollection


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
