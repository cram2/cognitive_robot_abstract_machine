from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.model_registries import (
    ModelRegistry,
    UniformPriorRegistry,
)
from semantic_digital_twin.exceptions import SurfaceGraspNotOnSurfaceError
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.spatial_types import (
    AxisAngle,
    HomogeneousTransformationMatrix,
    Point3,
    RotationMatrix,
    Twist,
    Vector3,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates

# %% the result of trying a grasp


@dataclass
class GraspResult:
    """
    What happened to an object when a robot tried to pick it up by a grasp.
    """

    object_raised: bool
    """
    Whether the object ended up clear of where it rested.
    """

    motion_completed: bool
    """
    Whether the robot went through every step of the pick-up within its time limit.
    """

    object_displacement: Twist
    """
    How the object moved from where it rested to where it was held at the end, along
    the world's axes.
    """

    slip: Twist
    """
    How the object moved relative to the grasp while it was lifted and held, along the
    grasp's axes.
    """

    @property
    def lifted(self) -> bool:
        """
        :return: Whether the object ended up clear of where it rested and the robot went
            through every step of the pick-up.
        """
        return self.object_raised and self.motion_completed


# %% grasps on an object's surface


@dataclass
class SurfaceGrasp:
    """
    A grasp of an object, placed on the object's surface independently of the robot
    taking it, in the frame of the object's root body.

    With ``root_V_outward`` the horizontal direction at :attr:`yaw`, ``root_P_surface``
    is the first point of the grasp surface that a horizontal ray hits when it comes
    from ``root_V_outward`` at :attr:`height`. The gripper closes at
    ``root_P_grasp = root_P_surface - depth * root_V_outward``. The x-axis of
    ``root_T_grasp`` is the direction of approach, straight down tilted by :attr:`pitch`
    towards ``-root_V_outward``; its y-axis is the closing direction, ``root_V_outward``
    made perpendicular to the approach and turned by :attr:`roll` about it.
    """

    yaw: float
    """
    The yaw of ``root_V_outward`` about the root body's z-axis, in radians.
    """

    height: float
    """
    The height of ``root_P_surface``, as a fraction of the grasp surface's bounding box
    from its bottom to its top.
    """

    depth: float
    """
    How far behind ``root_P_surface`` the gripper closes, in meters.
    """

    pitch: float
    """
    How far the direction of approach is tilted from straight down, in radians.
    """

    roll: float
    """
    How far the closing direction is turned about the direction of approach, in radians.
    """

    result: GraspResult | None = field(default=None)
    """
    What happened when the grasp was tried; ``None`` while it has not been.
    """

    def grasp_candidate(self, graspable: HasGraspCandidates) -> GraspCandidate:
        """
        :param graspable: The object to take hold of.
        :return: This grasp as ``root_T_grasp`` on the object's root body; its axes are
            those a :class:`GraspResult` measures slip along.
        :raises SurfaceGraspNotOnSurfaceError: If the object's grasp surface does not
            reach ``root_P_surface``.
        """
        root_V_outward = self._outward_direction()
        root_P_surface = self._surface_point(graspable, root_V_outward)
        if root_P_surface is None:
            raise SurfaceGraspNotOnSurfaceError(surface_grasp=self, graspable=graspable)
        root_V_approach = Vector3.NEGATIVE_Z() * math.cos(
            self.pitch
        ) - root_V_outward * math.sin(self.pitch)
        root_V_across = root_V_outward - root_V_approach * root_V_outward.dot(
            root_V_approach
        )
        root_V_across.scale(1)
        root_V_closing = root_V_across * math.cos(self.roll) + root_V_approach.cross(
            root_V_across
        ) * math.sin(self.roll)
        root_T_grasp = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=root_P_surface - root_V_outward * self.depth,
            rotation_matrix=RotationMatrix.from_vectors(
                x=root_V_approach, y=root_V_closing
            ),
            reference_frame=graspable.root,
        )
        return GraspCandidate(graspable, root_T_grasp.pose)

    def reaches_surface_of(self, graspable: HasGraspCandidates) -> bool:
        """
        :param graspable: The object to take hold of.
        :return: Whether the object's grasp surface reaches ``root_P_surface``.
        """
        return self._surface_point(graspable, self._outward_direction()) is not None

    def _outward_direction(self) -> Vector3:
        """
        :return: ``root_V_outward``.
        """
        return Vector3(math.cos(self.yaw), math.sin(self.yaw), 0.0)

    def _surface_point(
        self, graspable: HasGraspCandidates, root_V_outward: Vector3
    ) -> Point3 | None:
        """
        :param graspable: The object whose grasp surface the ray is cast against.
        :param root_V_outward: The horizontal direction the ray comes from.
        :return: ``root_P_surface``; ``None`` if the ray hits nothing.
        """
        mesh = graspable.grasp_surface()
        outward = root_V_outward.to_np()[:3]
        lowest, highest = mesh.bounds
        origin = (lowest + highest) / 2 + outward * float(
            np.linalg.norm(highest - lowest)
        )
        origin[2] = lowest[2] + self.height * (highest[2] - lowest[2])
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=origin[None, :], ray_directions=-outward[None, :]
        )
        if len(locations) == 0:
            return None
        return Point3(
            *locations[int(np.argmin(np.linalg.norm(locations - origin, axis=1)))]
        )


# %% stating where an object may be grasped


def any_surface_grasp(result: Match[GraspResult] | None = None) -> Match[SurfaceGrasp]:
    """
    :param result: The result the grasp is asked to have had; any result when ``None``.
    :return: A pattern of a surface grasp with every parameter left free, for a
        statement to narrow down with ``where``.
    """
    return a(SurfaceGrasp)(
        yaw=..., height=..., depth=..., pitch=..., roll=..., result=result
    )


def lifting_surface_grasp() -> Match[SurfaceGrasp]:
    """
    :return: A pattern of a surface grasp that lifted the object, with every other
        parameter left free. Only a model learned over tried grasps and their results
        can answer it.
    """
    return any_surface_grasp(
        result=a(GraspResult)(
            object_raised=True,
            motion_completed=True,
            object_displacement=_any_twist(),
            slip=_any_twist(),
        )
    )


def _any_twist() -> Match[Twist]:
    """
    :return: A pattern of a twist with every component left free.
    """
    return a(Twist)(
        translation=a(Vector3)(x=..., y=..., z=...),
        rotation=a(AxisAngle)(axis=a(Vector3)(x=..., y=..., z=...), angle=...),
    )


def draw_surface_grasps(
    statement: Match[SurfaceGrasp],
    number_of_grasps: int,
    model_registry: ModelRegistry | None = None,
) -> list[SurfaceGrasp]:
    """
    :param statement: A statement of where an object may be grasped.
    :param number_of_grasps: How many grasps to draw, as the statement's limit.
    :param model_registry: Answers the statement; ``None`` draws uniformly from what it
        allows.
    :return: Grasps answering the statement.
    """
    if number_of_grasps == 0:
        return []
    return list(
        statement.limit(number_of_grasps).evaluate(
            backend=ProbabilisticBackend(model_registry or UniformPriorRegistry())
        )
    )
