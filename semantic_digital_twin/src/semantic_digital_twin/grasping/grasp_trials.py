"""
Trying grasps on an object to collect the data a model of grasps is learned from.

The grasps are drawn from the statement of where the object's annotation may be grasped,
through a model registry: uniformly from what it allows at first, from a learned model
later. Each tried grasp is recorded together with what happened, so the records hold the
grasp's parameters and its result as plain numbers and truth values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Iterator, List

from krrood.parametrization.model_registries import (
    ModelRegistry,
    UniformPriorRegistry,
)
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.grasping.surface_grasp import (
    GraspResult,
    SurfaceGrasp,
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
    Grasps drawn from the statement of where an object may be grasped, each tried and
    given its result.

    The grasps are placed on the object's grasped part: its handle when it has one.
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

    require_lifting: bool = False
    """
    Whether to ask the model only for grasps it expects to lift the object, which needs
    a model learned over tried grasps and their results.
    """

    @property
    def grasped_part(self) -> HasGraspCandidates:
        """
        :return: The part of the object the grasps are placed on.
        """
        return self.graspable.grasped_part()

    def drawn_grasps(self) -> List[SurfaceGrasp]:
        """
        :return: :attr:`number_of_trials` grasps answering the statement of where
            :attr:`grasped_part` may be grasped.
        """
        return draw_surface_grasps(
            self.grasped_part.surface_grasp_statement(self.require_lifting),
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
