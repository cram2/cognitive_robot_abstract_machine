from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from krrood.ormatic.utils import classproperty
from giskardpy.motion_statechart.weights import DefaultWeights, VelocityUnit
from giskardpy.qp.bound_cap import RotationBoundCap, VectorBoundCap
from giskardpy.qp.constraint import LargeNumber
from giskardpy.qp.constraint_collection import ConstraintCollection
from krrood_spatial_math.spatial_types import (
    AxisAngle,
    Point3,
    Quaternion,
    Vector3,
    RotationMatrix,
)


@dataclass
class GeometricConstraintBuilder:
    """
    Builds high-level geometric constraints (point, vector, and rotation goals, and
    Cartesian velocity limits) by translating them into the primitive constraints of a
    :class:`ConstraintCollection`.
    """

    collection: ConstraintCollection
    """
    The collection the produced constraints are written into.
    """

    @classproperty
    def quaternion_error_per_radian(cls) -> float:
        """
        How much the vector part of a quaternion error changes per radian of rotation
        near the goal: half, since it is the sine of half the angle.
        """
        return 0.5

    @staticmethod
    def _indexed_constraint_name(name: str | None, index: int) -> str | None:
        """
        Builds a per-axis constraint name, or None so the collection assigns an index-
        based name.
        """
        if name is None:
            return None
        return f"{name}/{index}"

    def add_point_goal_constraints(
        self,
        frame_P_current: Point3,
        frame_P_goal: Point3,
        cruise_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        name: str | None = None,
    ) -> None:
        """
        Adds three constraints to move frame_P_current to frame_P_goal.

        The error is capped by its length rather than per axis, so a far away goal is
        approached along the straight line to it.

        Make sure that both points are expressed relative to the same frame!
        :param frame_P_current: a vector describing a 3D point. It should depend on
            active dofs.
        :param frame_P_goal: a vector describing a 3D point
        :param cruise_velocity: m/s
        """
        frame_V_error = frame_P_goal - frame_P_current
        error_cap = VectorBoundCap(
            vector_bounds=sm.Vector([frame_V_error[i] for i in range(3)])
        )
        for i in range(3):
            self.collection.add_equality_constraint(
                task_expression=frame_P_current[i],
                equality_bound=frame_V_error[i],
                quadratic_weight=quadratic_weight,
                cruise_velocity=cruise_velocity,
                velocity_unit=VelocityUnit.TRANSLATION,
                name=self._indexed_constraint_name(name, i),
                bound_cap=error_cap,
            )

    def add_position_constraint(
        self,
        expression_current: sm.SymbolicScalar,
        expression_goal: sm.ScalarData,
        cruise_velocity: sm.ScalarData,
        velocity_unit: sm.ScalarData,
        quadratic_weight: sm.ScalarData = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        name: str | None = None,
    ) -> None:
        """
        A wrapper around add_constraint.

        Will add a constraint that tries to move expr_current to expr_goal.
        :param expression_current: a symbolic expression describing a 3D point. It
            should depend on active dofs.
        :param expression_goal: a symbolic expression describing a 3D point
        :param cruise_velocity: the speed the error is closed at
        :param velocity_unit: the speed at which expression_current typically
            changes
        :param quadratic_weight: name relative to other constraints
        """
        error = expression_goal - expression_current
        self.collection.add_equality_constraint(
            cruise_velocity=cruise_velocity,
            velocity_unit=velocity_unit,
            equality_bound=error,
            quadratic_weight=quadratic_weight,
            task_expression=expression_current,
            name=name,
        )

    def add_vector_goal_constraints(
        self,
        frame_V_current: Vector3,
        frame_V_goal: Vector3,
        cruise_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        name: str | None = None,
    ) -> None:
        """
        Adds constraints to align frame_V_current with frame_V_goal.

        Make sure that both vectors are expressed relative to the same frame and are
        normalized to a length of 1.
        :param frame_V_current: a vector describing a 3D vector. It should depend on
            active dofs.
        :param frame_V_goal: a vector describing a 3D vector
        :param cruise_velocity: rad/s
        """
        angle = sm.safe_acos(frame_V_current.dot(frame_V_goal))
        # avoid singularity by staying away from pi
        angle_limited = sm.min(
            sm.max(angle, -VelocityUnit.ROTATION), VelocityUnit.ROTATION
        )
        angle_limited = angle_limited.safe_division(angle)
        root_V_goal_normal_intermediate = frame_V_current.slerp(
            frame_V_goal, angle_limited
        )

        error = root_V_goal_normal_intermediate - frame_V_current
        for i in range(3):
            self.collection.add_equality_constraint(
                task_expression=frame_V_current[i],
                equality_bound=error[i],
                cruise_velocity=cruise_velocity,
                velocity_unit=VelocityUnit.ROTATION,
                quadratic_weight=quadratic_weight,
                name=self._indexed_constraint_name(name, i),
            )

    def add_rotation_goal_constraints(
        self,
        frame_R_current: RotationMatrix,
        frame_R_goal: RotationMatrix,
        cruise_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        name: str | None = None,
    ) -> None:
        """
        Adds constraints to move frame_R_current to frame_R_goal.

        Make sure that both are expressed relative to the same frame.
        :param frame_R_current: current rotation as rotation matrix. It should depend on
            active dofs.
        :param frame_R_goal: goal rotation as rotation matrix
        :param cruise_velocity: rad/s
        """
        quaternion_error = self._quaternion_error(frame_R_current, frame_R_goal)

        # w is redundant
        error_cap = RotationBoundCap(
            quaternion_error_bounds=sm.Vector([-quaternion_error[i] for i in range(3)])
        )
        for i in range(3):
            self.collection.add_equality_constraint(
                task_expression=quaternion_error[i],
                equality_bound=-quaternion_error[i],
                quadratic_weight=quadratic_weight,
                cruise_velocity=sm.Scalar(cruise_velocity)
                * self.quaternion_error_per_radian,
                name=self._indexed_constraint_name(name, i),
                bound_cap=error_cap,
                velocity_unit=VelocityUnit.ROTATION * self.quaternion_error_per_radian,
            )

    def add_yaw_goal_constraint(
        self,
        frame_R_current: RotationMatrix,
        frame_R_goal: RotationMatrix,
        cruise_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        name: str | None = None,
    ) -> None:
        """
        Adds one constraint turning frame_R_current about the z-axis of the frame until
        it faces the way frame_R_goal does, for a rotation that can only turn about that
        axis.

        Make sure that both are expressed relative to the same frame.
        :param frame_R_current: current rotation, turned only about the z-axis of the
            frame. It should depend on active dofs.
        :param frame_R_goal: goal rotation, of which only the yaw is reached
        :param cruise_velocity: rad/s
        """
        frame_R_goal_yaw = RotationMatrix.from_rpy(
            yaw=frame_R_goal.rpy[2], reference_frame=frame_R_goal.reference_frame
        )
        quaternion_error = self._quaternion_error(frame_R_current, frame_R_goal_yaw)
        self.collection.add_equality_constraint(
            task_expression=quaternion_error.z,
            equality_bound=-quaternion_error.z,
            quadratic_weight=quadratic_weight,
            cruise_velocity=sm.Scalar(cruise_velocity)
            * self.quaternion_error_per_radian,
            name=name,
            bound_cap=RotationBoundCap(
                quaternion_error_bounds=sm.Vector([-quaternion_error.z])
            ),
            velocity_unit=VelocityUnit.ROTATION * self.quaternion_error_per_radian,
        )

    @staticmethod
    def _quaternion_error(
        frame_R_current: RotationMatrix, frame_R_goal: RotationMatrix
    ) -> Quaternion:
        """
        :return: The rotation from frame_R_current to frame_R_goal, the shorter way
            round.
        """
        # avoid singularity
        # the sign determines in which direction the robot moves when in singularity.
        # -0.0001 preserves the old behavior from before this goal was refactored
        hack = RotationMatrix.from_axis_angle(AxisAngle(Vector3.Z(), -0.0001))
        frame_R_current = frame_R_current.dot(hack)
        current_quaternion = frame_R_current.quaternion
        goal_quaternion = frame_R_goal.quaternion
        goal_quaternion = sm.if_less(
            goal_quaternion.dot(current_quaternion),
            0,
            -goal_quaternion,
            goal_quaternion,
        )
        return current_quaternion.diff(goal_quaternion)

    def add_translational_velocity_limit(
        self,
        frame_P_current: Point3,
        max_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        max_violation: sm.ScalarData = np.inf,
        name: str | None = None,
    ) -> None:
        """
        Adds constraints to limit the translational velocity of frame_P_current.

        Be aware that the velocity is relative to frame.
        :param frame_P_current: a vector describing a 3D point. It should depend on
            active dofs.
        :param max_violation: m/s
        """
        translation_error = frame_P_current.norm()
        translation_error = sm.if_eq_zero(
            translation_error, sm.Scalar(0.01), translation_error
        )
        self.collection.add_velocity_constraint(
            upper_velocity_limit=max_velocity,
            lower_velocity_limit=-max_velocity,
            quadratic_weight=quadratic_weight,
            task_expression=translation_error,
            velocity_unit=VelocityUnit.TRANSLATION,
            lower_slack_limit=-max_violation,
            upper_slack_limit=max_violation,
            name=name,
        )

    def add_rotational_velocity_limit(
        self,
        frame_R_current: RotationMatrix,
        max_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        max_violation: sm.ScalarData = LargeNumber,
        name: str | None = None,
    ) -> None:
        """
        Add velocity constraints to limit the velocity of frame_R_current.

        Be aware that the velocity is relative to frame.
        :param frame_R_current: Rotation matrix describing the current rotation. It
            should depend on active dofs.
        :param max_velocity: rad/s
        """
        root_Q_tipCurrent = frame_R_current.quaternion
        angle_error = root_Q_tipCurrent.axis_angle.angle
        self.collection.add_velocity_constraint(
            upper_velocity_limit=max_velocity,
            lower_velocity_limit=-max_velocity,
            quadratic_weight=quadratic_weight,
            task_expression=angle_error,
            velocity_unit=VelocityUnit.ROTATION,
            lower_slack_limit=-max_violation,
            upper_slack_limit=max_violation,
            name=name,
        )
