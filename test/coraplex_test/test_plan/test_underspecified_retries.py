"""
Tests for which outcomes make an underspecified step try its next candidate.

A candidate that cannot be carried out must not take the whole plan down while other
candidates are still on offer.
"""

from dataclasses import dataclass, field

import pytest
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from giskardpy.qp.exceptions import InfeasibleException
from typing_extensions import List, Optional, Type

from coraplex.plans.executables import UnderspecifiedExecutable
from coraplex.plans.failures import EmptyUnderspecified, PlanFailure

# %% a step whose candidates fail on demand


@dataclass
class FailingOnDemand:
    """
    Stands in for the executable a grounded candidate parses into.
    """

    raises: Optional[Exception] = None
    """
    What executing this candidate raises, or ``None`` when it succeeds.
    """

    def execute(self) -> None:
        if self.raises is not None:
            raise self.raises


@dataclass
class CandidateStub:
    """
    Stands in for one grounded candidate of an underspecified step.
    """

    executable: FailingOnDemand

    def parse(self) -> FailingOnDemand:
        return self.executable


@dataclass
class StepWithCandidates:
    """
    Stands in for the underspecified node, handing out its candidates in order.
    """

    outcomes: List[Optional[Exception]]
    """
    What each candidate raises when it is executed, in the order they are offered.
    """

    current_candidate: Optional[CandidateStub] = field(default=None, init=False)
    tried: int = field(default=0, init=False)
    stopped: bool = field(default=False, init=False)

    def advance(self) -> bool:
        if self.tried >= len(self.outcomes):
            return False
        self.current_candidate = CandidateStub(
            FailingOnDemand(self.outcomes[self.tried])
        )
        self.tried += 1
        return True

    def stop_grounding(self) -> None:
        self.stopped = True


# %% which outcomes are retried


@pytest.mark.parametrize(
    "outcome",
    [
        PlanFailure(),
        CollisionViolatedError(violated_collisions=[], thresholds=[]),
        InfeasibleException(),
        TimeoutError(),
    ],
    ids=["plan_failure", "collision", "infeasible", "timeout"],
)
def test_a_candidate_that_cannot_be_carried_out_is_followed_by_the_next(outcome):
    """
    Every way a motion fails to be carried out leaves the remaining candidates a chance.
    """
    step = StepWithCandidates([outcome, None])

    UnderspecifiedExecutable(node=step, context=None).execute()

    assert step.tried == 2
    assert step.stopped


def test_a_step_whose_candidates_all_fail_reports_it():
    """
    Once nothing is left to try, the step fails rather than swallowing the exhaustion.
    """
    step = StepWithCandidates([PlanFailure(), PlanFailure()])

    with pytest.raises(EmptyUnderspecified):
        UnderspecifiedExecutable(node=step, context=None).execute()

    assert step.tried == 2
    assert not step.stopped


def test_an_unrelated_error_is_not_retried():
    """
    A candidate that fails for a reason the plan cannot route around takes the plan
    down, rather than being retried until the candidates run out.
    """
    step = StepWithCandidates([ValueError("not a motion outcome"), None])

    with pytest.raises(ValueError):
        UnderspecifiedExecutable(node=step, context=None).execute()

    assert step.tried == 1
