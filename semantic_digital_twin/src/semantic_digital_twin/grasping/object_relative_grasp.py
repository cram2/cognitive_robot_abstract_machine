"""
Grasps described relative to the object they take hold of, independently of the robot
taking it, and what happened when a robot tried them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields

from typing_extensions import TYPE_CHECKING, Self

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.model_registries import (
    ModelRegistry,
    UniformPriorRegistry,
)
from krrood_spatial_math.spatial_types import Twist, Vector3
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import IsGraspable

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

    @classmethod
    def lifting_pattern(cls) -> Match[Self]:
        """
        :return: A pattern of a result that lifted the object, with how the object moved
            left free. Only a model learned over tried grasps and their results can
            answer a statement asking for it.
        """
        return a(cls)(
            object_raised=True,
            motion_completed=True,
            object_displacement=_any_twist(),
            slip=_any_twist(),
        )


def _any_twist() -> Match[Twist]:
    """
    :return: A pattern of a twist with every component left free.
    """
    return a(Twist)(
        translation=a(Vector3)(x=..., y=..., z=...),
        rotation_vector=a(Vector3)(x=..., y=..., z=...),
    )


# %% grasps relative to an object


@dataclass
class ObjectRelativeGrasp(ABC):
    """
    A grasp of an object, described by parameters relative to the object independently
    of the robot taking it, in the frame of the object's root body.
    """

    result: GraspResult | None = field(default=None, kw_only=True)
    """
    What happened when the grasp was tried; ``None`` while it has not been.
    """

    @classmethod
    def any_pattern(cls, result: Match[GraspResult] | None = None) -> Match[Self]:
        """
        :param result: The result the grasp is asked to have had; any result when
            ``None``.
        :return: A pattern of this kind of grasp with every parameter left free, for a
            statement to narrow down with ``where``.
        """
        result_field_names = {
            result_field.name for result_field in fields(ObjectRelativeGrasp)
        }
        return a(cls)(
            **{
                parameter.name: ...
                for parameter in fields(cls)
                if parameter.name not in result_field_names
            },
            result=result,
        )

    @abstractmethod
    def reaches_surface_of(self, graspable: IsGraspable) -> bool:
        """
        :param graspable: The object to take hold of.
        :return: Whether the place this grasp names reaches the surface of the object.
        """

    @abstractmethod
    def grasp_candidate(self, graspable: IsGraspable) -> GraspCandidate:
        """
        :param graspable: The object to take hold of.
        :return: This grasp as ``root_T_grasp`` on the object's root body, approaching
            along its x-axis and closing along its y-axis; its axes are those a
            :class:`GraspResult` measures slip along.
        :raises GraspNotOnObjectError: If the grasp does not
            :meth:`reach the surface of <reaches_surface_of>` the object.
        """


def draw_grasps(
    statement: Match[ObjectRelativeGrasp],
    number_of_grasps: int,
    model_registry: ModelRegistry | None = None,
) -> list[ObjectRelativeGrasp]:
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
        statement.limit(n=number_of_grasps).evaluate(
            backend=ProbabilisticBackend(
                model_registry=model_registry or UniformPriorRegistry()
            )
        )
    )
