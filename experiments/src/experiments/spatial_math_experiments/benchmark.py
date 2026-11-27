"""
Measure how long the common operations of the spatial types take to build their
symbolic expressions, on the bodies of a world, without ROS::

    python -m experiments.spatial_math_experiments.benchmark

The compiled expressions are what a controller evaluates; this measures building them.
"""

from __future__ import annotations

import argparse
import timeit
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Any

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
    Unit,
)
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix3Dimensional,
    Point3Dimensional,
    Pose3Dimensional,
    RotationMatrix,
    Vector3Dimensional,
)
from krrood_spatial_math.spatial_types_2_dimensional import (
    HomogeneousTransformationMatrix2Dimensional,
    Pose2Dimensional,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% the operations measured


class SpatialOperation(StrEnum):
    """
    An operation on spatial types whose building time is measured, named as it is
    written.
    """

    TRANSFORM_POINT = "T @ point"
    TRANSFORM_VECTOR = "T @ vector"
    TRANSFORM_POSE = "T @ pose"
    COMPOSE_TRANSFORMATIONS = "T @ T"
    INVERT_TRANSFORMATION = "T.inverse()"
    ROTATE_POINT = "R @ point"
    SUBTRACT_POINTS = "point - point"
    MOVE_POINT = "point + vector"
    MULTIPLY_VECTORS = "vector.dot(vector)"
    CROSS_VECTORS = "vector.cross(other=vector)"
    MEASURE_VECTOR = "vector.norm()"
    MEASURE_DISTANCE = "point.euclidean_distance(point)"
    CHANGE_BETWEEN_PLANAR_FRAMES = "planar change between two frames"
    EXPRESS_PLANAR_POSE = "world.transform(planar pose)"


# %% the values the operations act on


@dataclass(kw_only=True)
class SpatialOperands:
    """
    Values given in the bodies of a world whose base is driven on the map and carries an
    arm fixed on it, which every :class:`SpatialOperation` acts on.
    """

    world: World
    """
    The world the bodies belong to.
    """

    map_body: Body
    """
    The body the base is driven on.
    """

    base: Body
    """
    The driven body.
    """

    base_T_arm: HomogeneousTransformationMatrix3Dimensional
    """
    A rigid change from the arm into the base.
    """

    arm_T_arm: HomogeneousTransformationMatrix3Dimensional
    """
    A rigid change composed after :attr:`base_T_arm`.
    """

    point: Point3Dimensional
    """
    A point given in the arm.
    """

    other_point: Point3Dimensional
    """
    Another point given in the arm.
    """

    vector: Vector3Dimensional
    """
    A vector given in the arm.
    """

    other_vector: Vector3Dimensional
    """
    Another vector given in the arm.
    """

    pose: Pose3Dimensional
    """
    A pose given in the arm.
    """

    planar_pose: Pose2Dimensional
    """
    A pose on the x-y plane of the base.
    """

    @classmethod
    def from_driven_base(cls) -> SpatialOperands:
        """
        :return: The operands on a new world of a map, a base driven on it and an arm
            fixed on the base.
        """
        world = World()
        map_body, base, arm = (
            Body(name=PrefixedName(name)) for name in ("map", "base", "arm")
        )
        with world.modify_world():
            world.add_kinematic_structure_entity(map_body)
            world.add_connection(
                OmniDrive.create_with_dofs(world=world, parent=map_body, child=base)
            )
            world.add_connection(FixedConnection(parent=base, child=arm))
        return cls(
            world=world,
            map_body=map_body,
            base=base,
            base_T_arm=HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
                x=1,
                y=2,
                z=3,
                roll=0.1,
                pitch=0.2,
                yaw=0.3,
                reference_frame=base,
                child_frame=arm,
            ),
            arm_T_arm=HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
                x=1, reference_frame=arm, child_frame=arm
            ),
            point=Point3Dimensional(1, 2, 3, reference_frame=arm),
            other_point=Point3Dimensional(3, 2, 1, reference_frame=arm),
            vector=Vector3Dimensional(0, 1, 5, reference_frame=arm),
            other_vector=Vector3Dimensional(1, 2, 3, reference_frame=arm),
            pose=Pose3Dimensional.from_xyz_rpy(x=1, yaw=0.5, reference_frame=arm),
            planar_pose=Pose2Dimensional.from_xy_yaw(
                x=1, y=2, yaw=0.3, reference_frame=base
            ),
        )

    def run(self, *, operation: SpatialOperation) -> Any:
        """
        :param operation: The operation to carry out once.
        :return: What the operation built.
        """
        match operation:
            case SpatialOperation.TRANSFORM_POINT:
                return self.base_T_arm @ self.point
            case SpatialOperation.TRANSFORM_VECTOR:
                return self.base_T_arm @ self.vector
            case SpatialOperation.TRANSFORM_POSE:
                return self.base_T_arm @ self.pose
            case SpatialOperation.COMPOSE_TRANSFORMATIONS:
                return self.base_T_arm @ self.arm_T_arm
            case SpatialOperation.INVERT_TRANSFORMATION:
                return self.base_T_arm.inverse()
            case SpatialOperation.ROTATE_POINT:
                return self.base_T_arm.rotation @ self.point
            case SpatialOperation.SUBTRACT_POINTS:
                return self.point - self.other_point
            case SpatialOperation.MOVE_POINT:
                return self.point + self.vector
            case SpatialOperation.MULTIPLY_VECTORS:
                return self.vector.dot(self.other_vector)
            case SpatialOperation.CROSS_VECTORS:
                return self.vector.cross(other=self.other_vector)
            case SpatialOperation.MEASURE_VECTOR:
                return self.vector.norm()
            case SpatialOperation.MEASURE_DISTANCE:
                return self.point.euclidean_distance(self.other_point)
            case SpatialOperation.CHANGE_BETWEEN_PLANAR_FRAMES:
                return HomogeneousTransformationMatrix2Dimensional(
                    reference_frame=self.map_body, child_frame=self.base
                )
            case SpatialOperation.EXPRESS_PLANAR_POSE:
                return self.world.transform(self.planar_pose, self.map_body)


# %% one measured operation


@dataclass(kw_only=True)
class SpatialOperationResult(ExperimentResult):
    """
    How long one operation took to build its expression.
    """

    operation: SpatialOperation
    """
    The operation that was measured.
    """

    duration: MeanAndStandardDeviation
    """
    How long one call took, over the repetitions.
    """


# %% measuring every operation


@dataclass(kw_only=True)
class SpatialMathBenchmark:
    """
    Times every :class:`SpatialOperation` on one set of operands.
    """

    repetitions: int = 5
    """
    How often every operation is timed.
    """

    calls_per_repetition: int = 2000
    """
    How often an operation is called within one timing, which its duration is the mean
    over.
    """

    def measure(self) -> ExperimentsTable:
        """
        :return: One row per operation, with how long one call took.
        """
        operands = SpatialOperands.from_driven_base()
        return ExperimentsTable(
            [
                SpatialOperationResult(
                    operation=operation,
                    duration=MeanAndStandardDeviation.from_measurements(
                        [
                            total
                            / self.calls_per_repetition
                            / Unit.MICROSECONDS.seconds
                            for total in timeit.repeat(
                                lambda: operands.run(operation=operation),
                                number=self.calls_per_repetition,
                                repeat=self.repetitions,
                            )
                        ],
                        unit=Unit.MICROSECONDS,
                    ),
                )
                for operation in SpatialOperation
            ]
        )


def main() -> None:
    """
    Measure every operation and print the table.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=SpatialMathBenchmark.repetitions,
        help="how often every operation is timed",
    )
    parser.add_argument(
        "--calls-per-repetition",
        type=int,
        default=SpatialMathBenchmark.calls_per_repetition,
        help="how often an operation is called within one timing",
    )
    arguments = parser.parse_args()
    table = SpatialMathBenchmark(
        repetitions=arguments.repetitions,
        calls_per_repetition=arguments.calls_per_repetition,
    ).measure()
    print(
        TypstRenderer(table).render_figure(
            "Time to build the expression of each spatial operation, as mean and "
            f"standard deviation over {arguments.repetitions} repetitions."
        )
    )


if __name__ == "__main__":
    main()
