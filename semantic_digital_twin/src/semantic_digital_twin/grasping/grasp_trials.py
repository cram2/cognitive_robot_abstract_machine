"""
Trying grasps on an object to collect the data a model of grasps is learned from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Callable, Iterator

from krrood.entity_query_language.query.match import Match
from krrood.parametrization.model_registries import (
    ModelRegistry,
    UniformPriorRegistry,
)
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.grasping.surface_grasp import (
    GraspResult,
    SurfaceGrasp,
    any_surface_grasp,
    draw_surface_grasps,
)
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates

# %% performing a grasp


class GraspPerformer(ABC):
    """
    Performs grasps and tells what happened, for example a robot in a physics simulation
    or in the real world.
    """

    @abstractmethod
    def perform(self, grasp: GraspCandidate) -> GraspResult:
        """
        :param grasp: The grasp to perform.
        :return: What happened to the object.
        """


# %% the trials


@dataclass
class GraspTrialRecord:
    """
    One tried grasp as it is recorded.
    """

    grasped_part: HasGraspCandidates
    """
    The part the grasp was placed on.
    """

    grasp: SurfaceGrasp
    """
    The grasp, with its result.
    """


@dataclass
class GraspTrials:
    """
    Grasps drawn from the statement of where an object's grasped part may be grasped,
    each tried and given its result.
    """

    graspable: HasGraspCandidates
    """
    The object to grasp.
    """

    performer: GraspPerformer
    """
    Tries each grasp.
    """

    number_of_trials: int = 20
    """
    How many grasps are drawn.
    """

    model_registry: ModelRegistry = field(default_factory=UniformPriorRegistry)
    """
    Answers the statement.
    """

    grasp_pattern: Callable[[], Match[SurfaceGrasp]] = any_surface_grasp
    """
    Builds the pattern of the grasps asked for, such as
    :func:`~semantic_digital_twin.grasping.surface_grasp.lifting_surface_grasp`.
    """

    @property
    def grasped_part(self) -> HasGraspCandidates:
        """
        :return: The part of the object the grasps are placed on.
        """
        return self.graspable.grasped_part()

    def drawn_grasps(self) -> list[SurfaceGrasp]:
        """
        :return: :attr:`number_of_trials` grasps answering the statement of where
            :attr:`grasped_part` may be grasped.
        """
        return draw_surface_grasps(
            self.grasped_part.surface_grasp_statement(self.grasp_pattern),
            self.number_of_trials,
            self.model_registry,
        )

    def run(self) -> Iterator[GraspTrialRecord]:
        """
        Try every drawn grasp.

        :return: A record of each grasp tried, with its result, as they are tried. A
            grasp naming a point the grasped part's surface does not reach is skipped.
        """
        part = self.grasped_part
        for surface_grasp in self.drawn_grasps():
            if not surface_grasp.reaches_surface_of(part):
                continue
            surface_grasp.result = self.performer.perform(
                surface_grasp.grasp_candidate(part)
            )
            yield GraspTrialRecord(grasped_part=part, grasp=surface_grasp)
