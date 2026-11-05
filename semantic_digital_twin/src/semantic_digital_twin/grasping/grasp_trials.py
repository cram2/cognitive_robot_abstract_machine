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
from semantic_digital_twin.grasping.grasp_strategies import GraspStrategy
from semantic_digital_twin.grasping.object_relative_grasp import (
    GraspResult,
    ObjectRelativeGrasp,
    draw_grasps,
)
from semantic_digital_twin.semantic_annotations.mixins import IsGraspable

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

    grasped_part: IsGraspable
    """
    The part the grasp was placed on.
    """

    grasp: ObjectRelativeGrasp
    """
    The grasp, with its result.
    """


@dataclass
class GraspTrials:
    """
    Grasps drawn from the statement of where an object's grasped part may be grasped,
    each tried and given its result.
    """

    graspable: IsGraspable
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

    result_pattern: Callable[[], Match[GraspResult]] | None = None
    """
    Builds the pattern of the result the grasps asked for are to have had, such as
    :meth:`GraspResult.lifting_pattern`; any result when ``None``.
    """

    grasp_strategy: GraspStrategy | None = None
    """
    Where the grasps are drawn from; the :attr:`grasped_part`'s own strategy when
    ``None``.
    """

    @property
    def grasped_part(self) -> IsGraspable:
        """
        :return: The part of the object the grasps are placed on.
        """
        return self.graspable.grasped_part()

    def drawn_grasps(self) -> list[ObjectRelativeGrasp]:
        """
        :return: :attr:`number_of_trials` grasps answering the statement of where
            :attr:`grasp_strategy` allows :attr:`grasped_part` to be grasped.
        """
        part = self.grasped_part
        strategy = self.grasp_strategy or part.grasp_strategy
        return draw_grasps(
            strategy.statement(part, self.result_pattern),
            self.number_of_trials,
            self.model_registry,
        )

    def run(self) -> Iterator[GraspTrialRecord]:
        """
        Try every drawn grasp.

        :return: A record of each grasp tried, with its result, as they are tried. A
            grasp that does not reach the surface of the grasped part is skipped.
        """
        part = self.grasped_part
        for grasp in self.drawn_grasps():
            if not grasp.reaches_surface_of(part):
                continue
            grasp.result = self.performer.perform(grasp.grasp_candidate(part))
            yield GraspTrialRecord(grasped_part=part, grasp=grasp)
