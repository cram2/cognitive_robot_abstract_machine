"""
ORM round-trip tests for :mod:`experiments.montessori.sorting_results`: confirms segmind
events persist through ORMatic with a resolvable reference to the specific insertion
attempt -- and thus the specific :class:`~coraplex.plans.plan.Plan` -- they were
detected during, not just loosely grouped under the shape's overall result.
"""

from __future__ import annotations

from sqlalchemy import select

from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import PlanNode
from experiments.montessori.sorting_results import (
    InsertionOutcome,
    ShapeInsertionAttempt,
    ShapeInsertionResult,
    SortingIterationResult,
)
from experiments.orm.ormatic_interface import ShapeInsertionResultDAO
from krrood.ormatic.data_access_objects.helper import to_dao
from segmind.datastructures.events import InsertionEvent, PickUpEvent
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body

# %% events persisted under their own attempt


def _minimal_plan() -> Plan:
    """
    The smallest ``Plan`` :class:`~coraplex.orm.model.PlanMapping` can persist: a bare
    ``Plan()`` has no nodes, and ``Plan.root`` (which persistence needs) requires
    exactly one node with no parent, so a single bare node is added.
    """
    plan = Plan()
    plan.add_node(PlanNode())
    return plan


def test_events_are_persisted_under_their_own_attempt(montessori_results_session):
    """
    Two attempts at inserting the same shape detect different segmind events (a pick-up
    on the first, an insertion on the second, each with its own realized Plan); after a
    round trip through the database, each event must still be reachable only through its
    own attempt, with that attempt's own distinct plan.
    """
    session = montessori_results_session
    tracked_shape = Body(name=PrefixedName("circular_hole_1_shape"))
    pick_up_event = PickUpEvent(tracked_object=tracked_shape)
    insertion_event = InsertionEvent(tracked_object=tracked_shape)

    result = SortingIterationResult(
        iteration=1,
        shape_results=[
            ShapeInsertionResult(
                shape_key="circular_hole_1",
                outcome=InsertionOutcome.FELL_THROUGH,
                attempts=[
                    ShapeInsertionAttempt(plan=_minimal_plan(), events=[pick_up_event]),
                    ShapeInsertionAttempt(
                        plan=_minimal_plan(), events=[insertion_event]
                    ),
                ],
            )
        ],
    )

    session.add(to_dao(result))
    session.commit()

    [shape_result] = session.scalars(select(ShapeInsertionResultDAO)).all()
    attempts = [
        attempt_association.target for attempt_association in shape_result.attempts
    ]
    assert len(attempts) == 2

    def event_type_names(attempt) -> set[str]:
        return {
            type(event_association.target).__name__
            for event_association in attempt.events
        }

    pick_up_attempt = next(
        a for a in attempts if event_type_names(a) == {"PickUpEventDAO"}
    )
    insertion_attempt = next(
        a for a in attempts if event_type_names(a) == {"InsertionEventDAO"}
    )

    assert pick_up_attempt.database_id != insertion_attempt.database_id
    assert pick_up_attempt._plan_id != insertion_attempt._plan_id
    assert pick_up_attempt._plan_id is not None
    assert insertion_attempt._plan_id is not None
