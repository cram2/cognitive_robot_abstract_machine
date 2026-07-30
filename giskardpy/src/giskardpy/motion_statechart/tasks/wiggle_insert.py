from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Optional, Tuple

import krrood.symbolic_math.symbolic_math as symbolic_math
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import (
    DebugExpression,
    NodeArtifacts,
    Task,
)
from semantic_digital_twin.spatial_types import Point3, Vector3, RotationMatrix
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class WiggleInsert(Task):
    """
    Presses the tip link down towards a hole while wiggling it with random translational
    and angular noise.

    The wiggle is driven per control cycle through auxiliary variables that are updated
    in :meth:`on_tick`.
    """

    root_link: Body = field(kw_only=True)
    """
    Root link of the kinematic chain.
    """

    tip_link: Body = field(kw_only=True)
    """
    Tip link that is pressed into the hole.
    """

    hole_point: Point3 = field(kw_only=True)
    """
    Center point of the hole.
    """

    noise_translation: float = field(kw_only=True, default=0.5)
    """
    Strength of the translational wiggle.

    Default is an untuned preset, not a calibrated value.
    """

    noise_angle: float = field(kw_only=True, default=10)
    """
    Strength of the angular wiggle.

    Default is an untuned preset, not a calibrated value.
    """

    down_velocity: float = field(kw_only=True, default=0.2)
    """
    Reference velocity for pressing down in m/s.

    Default is an untuned preset, not a calibrated value.
    """

    hole_normal: Optional[Vector3] = field(default=None, kw_only=True)
    """
    Vector perpendicular to the hole.

    Defaults to the root z-axis.
    """

    threshold: float = field(default=0.01, kw_only=True)
    """
    Distance to the hole point below which the task is achieved.
    """

    random_walk: bool = field(default=True, kw_only=True)
    """
    If ``True`` a random walk is used, otherwise an uncorrelated random sample.
    """

    vector_momentum_factor: float = field(default=0.9, kw_only=True)
    """
    Influence of the previous translational momentum during a random walk.
    """

    angular_momentum_factor: float = field(default=0.9, kw_only=True)
    """
    Influence of the previous angular momentum during a random walk.
    """

    center_pull_strength_angle: float = field(default=0.1, kw_only=True)
    """
    Pull back towards the starting angle during a random walk.
    """

    center_pull_strength_vector: float = field(default=0.25, kw_only=True)
    """
    Pull back towards the hole point during a random walk.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    Priority weight relative to other tasks.
    """

    _control_frequency: float = field(default=0.0, init=False, repr=False)
    """
    Control frequency, used to scale the per-cycle noise.
    """

    _perpendicular_basis_first: np.ndarray = field(default=None, init=False, repr=False)
    """
    First basis vector of the plane perpendicular to the hole normal.
    """

    _perpendicular_basis_second: np.ndarray = field(
        default=None, init=False, repr=False
    )
    """
    Second basis vector of the plane perpendicular to the hole normal.
    """

    _current_angle: float = field(default=0.0, init=False, repr=False)
    """
    Current accumulated wiggle angle.
    """

    _angular_momentum: float = field(default=0.0, init=False, repr=False)
    """
    Current angular momentum of the random walk.
    """

    _current_vector: np.ndarray = field(default=None, init=False, repr=False)
    """
    Current accumulated wiggle translation.
    """

    _vector_momentum: np.ndarray = field(default=None, init=False, repr=False)
    """
    Current translational momentum of the random walk.
    """

    _random_translation: Vector3 = field(default=None, init=False, repr=False)
    """
    Auxiliary variable holding the current translational noise in the root frame.
    """

    _random_angle: symbolic_math.FloatVariable = field(
        default=None, init=False, repr=False
    )
    """
    Auxiliary variable holding the current angular noise.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        # The previous default was a zero vector, which has no well-defined perpendicular plane;
        # the root z-axis is used instead so the default is usable.
        hole_normal = context.world.transform(
            target_frame=self.root_link,
            spatial_object=(
                self.hole_normal
                if self.hole_normal is not None
                else Vector3.Z(reference_frame=self.root_link)
            ),
        )

        control_dt = context.qp_controller_config.control_dt
        self._control_frequency = 1 / control_dt

        self._current_angle = 0.0
        self._angular_momentum = 0.0
        self._current_vector = np.zeros(3)
        self._vector_momentum = np.zeros(3)
        self._perpendicular_basis_first, self._perpendicular_basis_second = (
            self._calculate_perpendicular_basis(hole_normal.to_np()[:3])
        )

        root_P_current = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        root_P_hole = context.world.transform(
            target_frame=self.root_link, spatial_object=self.hole_point
        )

        self._random_translation = Vector3.create_with_variables(
            f"{self.name}_rand_translation"
        )
        self._random_translation.reference_frame = self.root_link
        context.float_variable_data.register_expression(self._random_translation)

        root_P_hole_wiggled = root_P_hole + self._random_translation
        artifacts.geometry.add_point_goal_constraints(
            frame_P_current=root_P_current,
            frame_P_goal=root_P_hole_wiggled,
            reference_velocity=self.down_velocity,
            quadratic_weight=self.weight,
            name=f"{self.name}_point_goal",
        )

        self._random_angle = symbolic_math.FloatVariable(f"{self.name}_rand_angle")
        context.float_variable_data.register_expression(self._random_angle)

        tip_V_hole_normal = context.world.transform(
            target_frame=self.tip_link, spatial_object=hole_normal
        )
        tip_R_hole_normal = RotationMatrix.from_axis_angle(
            angle=self._random_angle, axis=tip_V_hole_normal
        )
        root_R_current = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()
        root_R_goal = root_R_current.dot(tip_R_hole_normal)

        artifacts.geometry.add_rotation_goal_constraints(
            frame_R_current=root_R_current,
            frame_R_goal=root_R_goal,
            reference_velocity=self.down_velocity,
            quadratic_weight=self.weight + 1,
        )

        artifacts.debug_expressions.append(
            DebugExpression(f"{self.name}/root_P_hole", root_P_hole)
        )
        artifacts.debug_expressions.append(
            DebugExpression(f"{self.name}/root_P_hole_wiggled", root_P_hole_wiggled)
        )

        distance = root_P_current.euclidean_distance(root_P_hole)
        artifacts.observation = distance <= self.threshold
        return artifacts

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        if self.random_walk:
            translation = self._random_walk_translation()
            angle = self._random_walk_angle()
        else:
            translation = self._random_sample_translation()
            angle = self._random_sample_angle()
        context.float_variable_data.set_value(self._random_translation, translation)
        context.float_variable_data.set_value(self._random_angle, angle)
        return None

    def _random_sample_angle(self) -> float:
        """
        Draw an uncorrelated random angular offset for the current control cycle.

        :return: The sampled angle.
        """
        self._current_angle = (
            (random.random() - 0.5) * self.noise_angle
        ) / self._control_frequency
        return self._current_angle

    def _random_walk_angle(self) -> float:
        """
        Advance the angular random walk by one control cycle, blending in new random
        change, applying momentum, and pulling back towards the starting angle.

        :return: The updated accumulated angle.
        """
        random_angular_change = (
            (random.random() - 0.5) * self.noise_angle
        ) / self._control_frequency
        self._angular_momentum = (
            self.angular_momentum_factor * self._angular_momentum
            + (1 - self.angular_momentum_factor) * random_angular_change
        )
        self._current_angle += self._angular_momentum
        # Normalize the difference to the starting angle to [-pi, pi] before pulling back.
        angle_difference = ((0 - self._current_angle + np.pi) % (2 * np.pi)) - np.pi
        self._current_angle += angle_difference * self.center_pull_strength_angle
        self._current_angle = self._current_angle % (2 * np.pi)
        return self._current_angle

    def _random_sample_translation(self) -> np.ndarray:
        """
        Draw an uncorrelated random translational offset within the plane perpendicular
        to the hole normal.

        :return: The sampled translation vector.
        """
        self._current_vector = (
            (random.random() - 0.5)
            * self.noise_translation
            * self._perpendicular_basis_first
            + (random.random() - 0.5)
            * self.noise_translation
            * self._perpendicular_basis_second
        ) / self._control_frequency
        return self._current_vector

    def _random_walk_translation(self) -> np.ndarray:
        """
        Advance the translational random walk by one control cycle, blending in new
        random change, applying momentum, and pulling back towards the hole center.

        :return: The updated accumulated translation vector.
        """
        random_vector_change = (
            (random.random() - 0.5)
            * self.noise_translation
            * self._perpendicular_basis_first
            + (random.random() - 0.5)
            * self.noise_translation
            * self._perpendicular_basis_second
        ) / self._control_frequency
        self._vector_momentum = (
            self.vector_momentum_factor * self._vector_momentum
            + (1 - self.vector_momentum_factor) * random_vector_change
        )
        self._current_vector += self._vector_momentum
        # Pull back towards the hole center (the zero vector).
        self._current_vector += (
            0 - self._current_vector
        ) * self.center_pull_strength_vector
        return self._current_vector

    @staticmethod
    def _calculate_perpendicular_basis(
        normal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return two orthonormal vectors spanning the plane perpendicular to ``normal``.

        :param normal: Unit vector to compute the perpendicular plane for.
        :return: Two orthonormal vectors spanning the plane perpendicular to ``normal``.
        """
        if abs(normal[0]) >= abs(normal[1]):
            first = np.array([-normal[2], 0, normal[0]])
        else:
            first = np.array([0, -normal[2], normal[1]])

        first = first / np.linalg.norm(first)
        second = np.cross(normal, first)
        second = second / np.linalg.norm(second)
        return first, second
