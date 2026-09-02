from dataclasses import dataclass

import pytest
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.exceptions import AmbiguousFailureDetector, FailureRefinementCycle
from coraplex.failure_handling.failure_refiner import FailureDetector, FailureRefiner
from coraplex.plans.factories import code, execute_single
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.parameter_mixins import (
    JointStatesKept,
    TargetLocationMovedTo,
    UsedArm,
)

# %% stub failures


@dataclass
class ExecutionFailure(PlanFailure):
    """
    The unrefined failure the stub detectors start from.
    """


@dataclass
class SpecificExecutionFailure(ExecutionFailure):
    """
    A more specific starting failure, so that detectors declared for the base and for
    the subclass compete.
    """


@dataclass
class RefinedFailure(PlanFailure):
    """
    The output of the first refinement hop.
    """


@dataclass
class FurtherRefinedFailure(PlanFailure):
    """
    The output of the second refinement hop.
    """


@dataclass
class AlternativeRefinedFailure(PlanFailure):
    """
    The output of detectors that compete with the ones producing
    :class:`RefinedFailure`.
    """


# %% stub detectors


@dataclass
class SingleHopDetector(FailureDetector):
    """
    Refines an execution failure into a :class:`RefinedFailure`.
    """

    input_failure_type = ExecutionFailure
    output_failure_type = RefinedFailure

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return RefinedFailure(node=failure.node)


@dataclass
class AlternativeSingleHopDetector(FailureDetector):
    """
    Refines an execution failure just as specifically as :class:`SingleHopDetector`,
    which makes the two of them ambiguous.
    """

    input_failure_type = ExecutionFailure
    output_failure_type = AlternativeRefinedFailure

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return AlternativeRefinedFailure(node=failure.node)


@dataclass
class SecondHopDetector(FailureDetector):
    """
    Refines a :class:`RefinedFailure` further, forming the second link of a two-hop
    chain.
    """

    input_failure_type = RefinedFailure
    output_failure_type = FurtherRefinedFailure

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return FurtherRefinedFailure(node=failure.node)


@dataclass
class SubclassDetector(FailureDetector):
    """
    Declared for a subclass of :class:`SingleHopDetector`'s input type and therefore
    more specific than it.
    """

    input_failure_type = SpecificExecutionFailure
    output_failure_type = FurtherRefinedFailure

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return FurtherRefinedFailure(node=failure.node)


@dataclass
class NoOpDetector(FailureDetector):
    """
    Hands the very same failure back, signalling that there is nothing left to refine.
    """

    input_failure_type = ExecutionFailure

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return failure


@dataclass
class SameTypeDetector(FailureDetector):
    """
    Returns a fresh failure of the type it was given, which makes no progress in the
    failure hierarchy.
    """

    input_failure_type = ExecutionFailure
    output_failure_type = ExecutionFailure

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return ExecutionFailure(node=failure.node)


@dataclass
class ArmRequiringDetector(FailureDetector):
    """
    Only applies to actions that use an arm.
    """

    input_failure_type = ExecutionFailure
    output_failure_type = AlternativeRefinedFailure
    required_parameter_mixins = (UsedArm,)

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return AlternativeRefinedFailure(node=failure.node)


@dataclass
class TargetLocationRequiringDetector(FailureDetector):
    """
    Applies to actions that move to a target location.
    """

    input_failure_type = ExecutionFailure
    output_failure_type = RefinedFailure
    required_parameter_mixins = (TargetLocationMovedTo,)

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return RefinedFailure(node=failure.node)


@dataclass
class TargetLocationAndJointStatesRequiringDetector(FailureDetector):
    """
    Requires one mixin more than :class:`TargetLocationRequiringDetector` and therefore
    wins the tiebreak against it.
    """

    input_failure_type = ExecutionFailure
    output_failure_type = AlternativeRefinedFailure
    required_parameter_mixins = (TargetLocationMovedTo, JointStatesKept)

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return AlternativeRefinedFailure(node=failure.node)


@dataclass
class DecliningTargetLocationDetector(FailureDetector):
    """
    The most specific candidate for a navigation failure, yet it always declines by
    handing the failure back.
    """

    input_failure_type = ExecutionFailure
    output_failure_type = RefinedFailure
    required_parameter_mixins = (TargetLocationMovedTo,)

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return failure


@dataclass
class CycleOpeningDetector(FailureDetector):
    """
    Leaves the refined failure class behind, the first half of a refinement cycle.
    """

    input_failure_type = RefinedFailure
    output_failure_type = AlternativeRefinedFailure

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return AlternativeRefinedFailure(node=failure.node)


@dataclass
class CycleClosingDetector(FailureDetector):
    """
    Refines back into the failure class the chain already visited, closing the cycle.
    """

    input_failure_type = AlternativeRefinedFailure
    output_failure_type = RefinedFailure

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return RefinedFailure(node=failure.node)


# %% fixtures


@pytest.fixture
def navigation_action_node() -> PlanNode:
    return execute_single(NavigateAction(target_location=Pose()))


# %% detector applicability


def test_detector_applies_to_a_failure_of_its_input_type(navigation_action_node):
    failure = ExecutionFailure(node=navigation_action_node)

    assert SingleHopDetector().applies(failure)


def test_detector_does_not_apply_to_a_failure_of_another_type(navigation_action_node):
    failure = ExecutionFailure(node=navigation_action_node)

    assert not SecondHopDetector().applies(failure)


def test_detector_does_not_apply_without_an_enclosing_action_node():
    failure = ExecutionFailure(node=code(lambda: None))

    assert not SingleHopDetector().applies(failure)


def test_detector_does_not_apply_when_a_required_mixin_is_missing(
    navigation_action_node,
):
    failure = ExecutionFailure(node=navigation_action_node)

    assert not ArmRequiringDetector().applies(failure)


def test_detector_applies_when_all_required_mixins_are_present(navigation_action_node):
    failure = ExecutionFailure(node=navigation_action_node)

    assert TargetLocationRequiringDetector().applies(failure)


# %% refinement fixpoint


def test_refinement_without_detectors_returns_the_failure_unchanged(
    navigation_action_node,
):
    failure = ExecutionFailure(node=navigation_action_node)

    assert FailureRefiner().refine(failure) is failure


def test_refinement_returns_the_failure_unchanged_when_no_detector_applies(
    navigation_action_node,
):
    failure = ExecutionFailure(node=navigation_action_node)
    refiner = FailureRefiner(failure_detectors=[ArmRequiringDetector()])

    assert refiner.refine(failure) is failure


def test_refinement_stops_when_a_detector_returns_the_same_failure(
    navigation_action_node,
):
    failure = ExecutionFailure(node=navigation_action_node)
    refiner = FailureRefiner(failure_detectors=[NoOpDetector()])

    assert refiner.refine(failure) is failure


def test_refinement_stops_when_a_detector_returns_the_same_failure_type(
    navigation_action_node,
):
    failure = ExecutionFailure(node=navigation_action_node)
    refiner = FailureRefiner(failure_detectors=[SameTypeDetector()])

    refined = refiner.refine(failure)

    assert type(refined) is ExecutionFailure
    assert refined is not failure


def test_refinement_applies_a_single_detector(navigation_action_node):
    failure = ExecutionFailure(node=navigation_action_node)
    refiner = FailureRefiner(failure_detectors=[SingleHopDetector()])

    refined = refiner.refine(failure)

    assert isinstance(refined, RefinedFailure)
    assert refined.node is navigation_action_node


def test_refinement_chains_detectors_until_none_applies(navigation_action_node):
    failure = ExecutionFailure(node=navigation_action_node)
    refiner = FailureRefiner(
        failure_detectors=[SingleHopDetector(), SecondHopDetector()]
    )

    assert isinstance(refiner.refine(failure), FurtherRefinedFailure)


# %% declining detectors


def test_a_declining_detector_hands_over_to_the_next_most_specific(
    navigation_action_node,
):
    refiner = FailureRefiner(
        failure_detectors=[DecliningTargetLocationDetector(), SingleHopDetector()]
    )
    failure = ExecutionFailure(node=navigation_action_node)

    refined = refiner.refine(failure)

    assert isinstance(refined, RefinedFailure)
    assert refined.refined_from is failure


def test_refinement_returns_the_failure_when_every_detector_declines(
    navigation_action_node,
):
    refiner = FailureRefiner(
        failure_detectors=[DecliningTargetLocationDetector(), NoOpDetector()]
    )
    failure = ExecutionFailure(node=navigation_action_node)

    assert refiner.refine(failure) is failure


def test_a_decline_that_leaves_equally_specific_detectors_is_ambiguous(
    navigation_action_node,
):
    refiner = FailureRefiner(
        failure_detectors=[
            DecliningTargetLocationDetector(),
            SingleHopDetector(),
            AlternativeSingleHopDetector(),
        ]
    )
    failure = ExecutionFailure(node=navigation_action_node)

    with pytest.raises(AmbiguousFailureDetector):
        refiner.refine(failure)


# %% provenance


def test_refinement_records_the_failure_it_came_from(navigation_action_node):
    failure = ExecutionFailure(node=navigation_action_node)
    refiner = FailureRefiner(failure_detectors=[SingleHopDetector()])

    refined = refiner.refine(failure)

    assert refined.refined_from is failure
    assert refined.__cause__ is failure


def test_refinement_records_the_provenance_of_every_hop(navigation_action_node):
    failure = ExecutionFailure(node=navigation_action_node)
    refiner = FailureRefiner(
        failure_detectors=[SingleHopDetector(), SecondHopDetector()]
    )

    refined = refiner.refine(failure)

    assert isinstance(refined.refined_from, RefinedFailure)
    assert refined.refined_from.refined_from is failure


# %% specificity selection


def test_the_detector_declared_for_the_failure_subclass_wins(navigation_action_node):
    failure = SpecificExecutionFailure(node=navigation_action_node)
    refiner = FailureRefiner(
        failure_detectors=[SingleHopDetector(), SubclassDetector()]
    )

    assert isinstance(refiner.refine(failure), FurtherRefinedFailure)


def test_the_detector_requiring_more_mixins_wins(navigation_action_node):
    failure = ExecutionFailure(node=navigation_action_node)
    refiner = FailureRefiner(
        failure_detectors=[
            TargetLocationRequiringDetector(),
            TargetLocationAndJointStatesRequiringDetector(),
        ]
    )

    assert isinstance(refiner.refine(failure), AlternativeRefinedFailure)


def test_equally_specific_detectors_are_ambiguous(navigation_action_node):
    failure = ExecutionFailure(node=navigation_action_node)
    refiner = FailureRefiner(
        failure_detectors=[SingleHopDetector(), AlternativeSingleHopDetector()]
    )

    with pytest.raises(AmbiguousFailureDetector):
        refiner.refine(failure)


# %% cycle detection


def test_a_detector_chain_that_revisits_a_failure_type_is_rejected(
    navigation_action_node,
):
    failure = RefinedFailure(node=navigation_action_node)
    refiner = FailureRefiner(
        failure_detectors=[CycleOpeningDetector(), CycleClosingDetector()]
    )

    with pytest.raises(FailureRefinementCycle):
        refiner.refine(failure)
