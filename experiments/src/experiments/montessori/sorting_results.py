"""
What one shape-sorting run recorded: every shape's insertion outcome, attempt by attempt.

Kept in a module of its own rather than beside the script that produces it: a class
defined in a script run via ``python -m ...`` is loaded twice, once as ``__main__`` and
once under its real dotted path -- which is the path the generated
``experiments.orm.ormatic_interface`` maps -- leaving two unrelated class objects of the
same name and no DAO for whichever one the script instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from coraplex.plans.plan import Plan
from segmind.datastructures.events import DetectionEvent


class InsertionOutcome(StrEnum):
    """
    How one shape's insertion attempts ended.
    """

    FELL_THROUGH = "fell_through"
    DID_NOT_FALL_THROUGH = "did_not_fall_through"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"


@dataclass
class ShapeInsertionAttempt:
    """
    One insertion attempt's realized plan and the segmind events detected while it ran.
    """

    plan: Plan
    """
    The whole realized plan tree of this attempt, expanded down to its motions rather
    than only its top-level insertion node.
    """

    events: list[DetectionEvent] = field(default_factory=list)
    """
    Segmind events whose timestamp fell within this attempt's time window.
    """


@dataclass
class ShapeInsertionResult:
    """
    One shape's insertion outcome from a single sorting run.
    """

    shape_key: str
    """
    This shape's name, with its trailing ``"_shape"`` suffix removed.
    """

    outcome: InsertionOutcome
    """
    How this shape's insertion attempts ended.
    """

    attempts: list[ShapeInsertionAttempt] = field(default_factory=list)
    """
    Every attempt made for this shape, in order; the last one is what settled
    :attr:`outcome`.
    """


@dataclass
class SortingIterationResult:
    """
    Every attempted shape's :class:`ShapeInsertionResult` from one sorting run.
    """

    iteration: int
    """
    This run's 1-based index among the iterations that were repeated.
    """

    shape_results: list[ShapeInsertionResult] = field(default_factory=list)
    """
    Every attempted shape's own outcome from this iteration.
    """
