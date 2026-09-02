from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import ClassVar, List, Optional, Set, Tuple, Type

from krrood.patterns.specificity_ranking import mro_depth, sole_maximum

from coraplex.exceptions import AmbiguousFailureDetector, FailureRefinementCycle
from coraplex.plans.failures import PlanFailure

# %% detectors


@dataclass
class FailureDetector(ABC):
    """
    A detector that narrows a failure raised during plan execution down to a more
    specific one.

    A detector declares the failure type it consumes, the failure type it produces, and
    the parameter mixins the failing action has to carry for the detection to be
    meaningful.
    """

    input_failure_type: ClassVar[Type[PlanFailure]] = PlanFailure
    """
    The failure type this detector refines.
    """

    output_failure_type: ClassVar[Type[PlanFailure]] = PlanFailure
    """
    The failure type this detector produces.
    """

    required_parameter_mixins: ClassVar[Tuple[Type, ...]] = ()
    """
    The parameter mixins the failing action has to inherit from for this detector to
    apply.
    """

    def applies(self, failure: PlanFailure) -> bool:
        """
        :param failure: The failure to check.
        :return: Whether this detector can refine the failure.
        """
        if not isinstance(failure, self.input_failure_type):
            return False
        action_node = failure.action_node
        if action_node is None:
            return False
        return all(
            isinstance(action_node.action, mixin)
            for mixin in self.required_parameter_mixins
        )

    @abstractmethod
    def detect(self, failure: PlanFailure) -> PlanFailure:
        """
        Narrow the failure down to a more specific one.

        ..warning:: Callers have to check :meth:`applies` first; detectors dereference
            the failing action without guarding.

        :param failure: The failure to refine.
        :return: The refined failure, or the given failure itself if there is nothing to
            refine.
        """


# %% refinement


@dataclass
class FailureRefiner:
    """
    Refines a failure that happens during plan execution by repeatedly applying the most
    specific of its failure detectors until no detector applies anymore.
    """

    failure_detectors: List[FailureDetector] = field(default_factory=list)
    """
    The failure detectors that narrow down the failure that happened.
    """

    def most_specific_detector(
        self,
        failure: PlanFailure,
        candidates: Optional[List[FailureDetector]] = None,
    ) -> Optional[FailureDetector]:
        """
        :param failure: The failure to find a detector for.
        :param candidates: The detectors to choose from, all of them by default.
        :return: The single most specific applicable detector, or None if no detector
            applies.
        :raises AmbiguousFailureDetector: If several detectors are equally specific.
        """
        if candidates is None:
            candidates = self.failure_detectors
        applicable = [detector for detector in candidates if detector.applies(failure)]
        return sole_maximum(
            applicable,
            key=lambda detector: (
                mro_depth(detector.input_failure_type),
                len(detector.required_parameter_mixins),
            ),
            collision_error=lambda detectors: AmbiguousFailureDetector(
                failure=failure, detectors=detectors
            ),
        )

    def confirmed_refinement(self, failure: PlanFailure) -> Optional[PlanFailure]:
        """
        Let the applicable detectors examine the failure, most specific first, until one
        confirms that its own failure type describes what happened.

        A detector declines by returning the failure it was given, which hands the same
        failure to the next most specific detector. An action can carry the parameters
        of several detectors, so the most specific one is not necessarily the one that
        recognises what went wrong.

        :param failure: The failure to refine by one step.
        :return: The refined failure, or None if no detector confirmed this failure.
        """
        candidates = list(self.failure_detectors)
        while True:
            detector = self.most_specific_detector(failure, candidates)
            if detector is None:
                return None

            refined = detector.detect(failure)
            if refined is not failure:
                return refined

            candidates = [
                candidate for candidate in candidates if candidate is not detector
            ]

    def refine(self, failure: PlanFailure) -> PlanFailure:
        """
        Refine the failure until no detector confirms the result anymore.

        Every refinement step records where it came from, both as
        :attr:`~coraplex.plans.failures.PlanFailure.refined_from` and as the cause of the
        refined exception.

        :param failure: The failure that was raised during plan execution.
        :return: The most specific failure the detectors could confirm.
        :raises FailureRefinementCycle: If the detectors produce a failure type that was
            already produced before.
        """
        current = failure
        seen_failure_types: Set[Type[PlanFailure]] = {type(failure)}

        while True:
            refined = self.confirmed_refinement(current)
            if refined is None:
                return current

            refined.refined_from = current
            refined.__cause__ = current

            # A detector that stays within the failure type it was given makes no progress,
            # so refining its output again would not terminate.
            if type(refined) is type(current):
                return refined

            if type(refined) in seen_failure_types:
                raise FailureRefinementCycle(
                    failure=failure, repeated_failure_type=type(refined)
                )

            seen_failure_types.add(type(refined))
            current = refined
