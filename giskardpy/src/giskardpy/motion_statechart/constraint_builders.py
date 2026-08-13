from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import Optional

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.qp.constraint import LargeNumber
from giskardpy.qp.constraint_collection import ConstraintCollection
from semantic_digital_twin.spatial_types import Point3, Vector3, RotationMatrix


@dataclass
class GeometricConstraintBuilder:
    """
    Builds high-level geometric constraints (point, vector, and rotation goals, and Cartesian
    velocity limits) by translating them into the primitive constraints of a
    :class:`ConstraintCollection`.
    """

    collection: ConstraintCollection
    """
    The collection the produced constraints are written into.
    """

    @staticmethod
    def _indexed_constraint_name(name: Optional[str], index: int) -> Optional[str]:
        """
        Builds a per-axis constraint name, or None so the collection assigns an index-based name.
        """
        if name is None:
            return None
        return f"{name}/{index}"

    def add_point_goal_constraints(
        self,
        frame_P_current: Point3,
        frame_P_goal: Point3,
        reference_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        name: Optional[str] = None,
    ) -> None:
        """
        Adds three constraints to move frame_P_current to frame_P_goal.
        Make sure that both points are expressed relative to the same frame!
        :param frame_P_current: a vector describing a 3D point. It should depend on active dofs.
        :param frame_P_goal: a vector describing a 3D point
        :param reference_velocity: m/s
        """

        frame_V_error = frame_P_goal - frame_P_current
        for i in range(3):
            self.collection.add_equality_constraint(
                task_expression=frame_P_current[i],
                equality_bound=frame_V_error[i],
                quadratic_weight=quadratic_weight,
                reference_velocity=reference_velocity,
                name=self._indexed_constraint_name(name, i),
            )

    def add_position_constraint(
        self,
        expression_current: sm.SymbolicScalar,
        expression_goal: sm.ScalarData,
        reference_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData = DefaultWeights.WEIGHT_BELOW_CA,
        name: Optional[str] = None,
    ) -> None:
        """
        A wrapper around add_constraint. Will add a constraint that tries to move expr_current to expr_goal.
        :param expression_current: a symbolic expression describing a 3D point. It should depend on active dofs.
        :param expression_goal: a symbolic expression describing a 3D point
        :param reference_velocity: value used for normalization m/s
        :param quadratic_weight: name relative to other constraints
        """

        error = expression_goal - expression_current
        self.collection.add_equality_constraint(
            reference_velocity=reference_velocity,
            equality_bound=error,
            quadratic_weight=quadratic_weight,
            task_expression=expression_current,
            name=name,
        )

    def add_vector_goal_constraints(
        self,
        frame_V_current: Vector3,
        frame_V_goal: Vector3,
        reference_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData = DefaultWeights.WEIGHT_BELOW_CA,
        name: Optional[str] = None,
    ) -> None:
        """
        Adds constraints to align frame_V_current with frame_V_goal. Make sure that both vectors are expressed
        relative to the same frame and are normalized to a length of 1.
        :param frame_V_current: a vector describing a 3D vector. It should depend on active dofs.
        :param frame_V_goal: a vector describing a 3D vector
        :param reference_velocity: value used for normalization rad/s
        """

        angle = sm.safe_acos(frame_V_current.dot(frame_V_goal))
        # avoid singularity by staying away from pi
        angle_limited = sm.min(sm.max(angle, -reference_velocity), reference_velocity)
        angle_limited = angle_limited.safe_division(angle)
        root_V_goal_normal_intermediate = frame_V_current.slerp(
            frame_V_goal, angle_limited
        )

        error = root_V_goal_normal_intermediate - frame_V_current
        for i in range(3):
            self.collection.add_equality_constraint(
                task_expression=frame_V_current[i],
                equality_bound=error[i],
                reference_velocity=reference_velocity,
                quadratic_weight=quadratic_weight,
                name=self._indexed_constraint_name(name, i),
            )

    def add_rotation_goal_constraints(
        self,
        frame_R_current: RotationMatrix,
        frame_R_goal: RotationMatrix,
        reference_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        name: Optional[str] = None,
    ) -> None:
        """
        Adds constraints to move frame_R_current to frame_R_goal. Make sure that both are expressed relative to the same
        frame.
        :param frame_R_current: current rotation as rotation matrix. It should depend on active dofs.
        :param frame_R_goal: goal rotation as rotation matrix
        :param reference_velocity: value used for normalization rad/s
        """

        # avoid singularity
        # the sign determines in which direction the robot moves when in singularity.
        # -0.0001 preserves the old behavior from before this goal was refactored
        hack = RotationMatrix.from_axis_angle(Vector3.Z(), -0.0001)
        frame_R_current = frame_R_current.dot(hack)
        q_actual = frame_R_current.to_quaternion()
        q_goal = frame_R_goal.to_quaternion()
        q_goal = sm.if_less(q_goal.dot(q_actual), 0, -q_goal, q_goal)
        q_error = q_actual.diff(q_goal)

        # w is redundant
        for i in range(3):
            self.collection.add_equality_constraint(
                task_expression=q_error[i],
                equality_bound=-q_error[i],
                quadratic_weight=quadratic_weight,
                reference_velocity=reference_velocity,
                name=self._indexed_constraint_name(name, i),
            )

    def add_translational_velocity_limit(
        self,
        frame_P_current: Point3,
        max_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        max_violation: sm.ScalarData = np.inf,
        name: Optional[str] = None,
    ) -> None:
        """
        Adds constraints to limit the translational velocity of frame_P_current. Be aware that the velocity is relative
        to frame.
        :param frame_P_current: a vector describing a 3D point. It should depend on active dofs.
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
            lower_slack_limit=-max_violation,
            upper_slack_limit=max_violation,
            velocity_limit=max_velocity,
            name=name,
        )

    def add_rotational_velocity_limit(
        self,
        frame_R_current: RotationMatrix,
        max_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        max_violation: sm.ScalarData = LargeNumber,
        name: Optional[str] = None,
    ) -> None:
        """
        Add velocity constraints to limit the velocity of frame_R_current. Be aware that the velocity is relative to
        frame.
        :param frame_R_current: Rotation matrix describing the current rotation. It should depend on active dofs.
        :param max_velocity: rad/s
        """

        root_Q_tipCurrent = frame_R_current.to_quaternion()
        angle_error = root_Q_tipCurrent.to_axis_angle()[1]
        self.collection.add_velocity_constraint(
            upper_velocity_limit=max_velocity,
            lower_velocity_limit=-max_velocity,
            quadratic_weight=quadratic_weight,
            task_expression=angle_error,
            lower_slack_limit=-max_violation,
            upper_slack_limit=max_violation,
            name=name,
            velocity_limit=max_velocity,
        )
