"""
Example domain classes for the EQL verbalization documentation.

These small dataclasses are the *domain* that the verbalization user guide
(:doc:`/eql/user/verbalization`) queries over — *"Find a Robot whose battery is
greater than 50"*, *"the sum of amounts among BankTransactions …"*, and so on.

They live here, under ``src``, for one reason: Sphinx AutoAPI documents this
module like any other, so the source-link resolver
(:class:`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.AutoAPIResolver`)
resolves the class/attribute hyperlinks in the rendered examples to real AutoAPI
pages — both on the published GitHub Pages site and in a locally built docs tree —
with no hand-maintained mock API.

They are illustrative only: not used by the verbalization engine itself, and safe
to import without side effects.  Grouped by the doc section that uses them.

Canonical class list (the single source of truth;
``test_example_domain.py::_EXAMPLE_CLASSES`` enforces that this stays in sync):

* Robots & missions — ``Robot``, ``Mission``
* Boolean & list attributes — ``Task``, ``Worker``
* Money (aggregation / nested / dates) — ``AmountDetails``, ``BankTransaction``
* Employees (disambiguation / grouping / having) — ``Employee``
* Custom predicates — ``Location``, ``IsReachable``, ``Department``, ``StaffMember``, ``WorksIn``
* Birds (rule-tree / inference) — ``Bird``, ``LoveBirds``, ``BirdView``, ``StrongLoveBird``,
  ``WeakLoveBird``
* Containers (deep-nesting rule trees) — ``Handle``, ``Container``, ``FixedConnection``,
  ``PrismaticConnection``, ``Drawer``, ``Cabinet``
* Causal queries (``cause`` / ``causes_effect()``) — ``Pick``
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing_extensions import Mapping


from krrood.entity_query_language.factories import Symbol
from krrood.entity_query_language.predicate import Predicate
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Prepositions
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Adjective,
    clause,
    Copula,
    Noun,
    Verb,
)

# %% Robots & missions (the quick-start + cross-variable examples)


@dataclass(kw_only=True)
class Robot:
    """
    An example autonomous robot.
    """

    name: str
    """The robot's identifier string."""

    battery: int
    """
    Charge level from 0 to 100.
    """

    operational: bool
    """Whether the robot is operational (a boolean terminal for predicative chains)."""


@dataclass(kw_only=True)
class Mission:
    """
    A task assigned to a :class:`Robot`, with a numeric priority.
    """

    assigned_to: Robot
    """The robot the mission is assigned to."""

    priority: int
    """
    Priority rank (higher is more urgent).
    """


# %% Boolean & list attributes (predicative / indexed-attribute examples)


@dataclass(kw_only=True)
class Task:
    """
    A unit of work with a boolean completion flag.
    """

    name: str
    """The task's name."""

    completed: bool
    """
    Whether the task is finished — drives the predicative *"is completed"* form.
    """


@dataclass(kw_only=True)
class Worker:
    """
    A worker holding an ordered list of :class:`Task` objects.
    """

    name: str
    """The worker's name."""

    tasks: list[Task]
    """
    The worker's tasks (indexable, e.g. ``tasks[0]``).
    """


# %% Money (aggregation, nested-attribute, and date-range examples)


@dataclass(kw_only=True)
class AmountDetails:
    """
    A monetary amount, nested inside a :class:`BankTransaction`.
    """

    amount: float
    """The amount of money."""


@dataclass(kw_only=True)
class BankTransaction:
    """
    A bank transaction with a nested amount and a booking date.
    """

    amount_details: AmountDetails
    """The transaction's amount (a nested attribute chain)."""

    booking_date: datetime.datetime
    """
    When the transaction was booked (used for *"between … and …"* date folding).
    """


# %% Employees (same-type disambiguation, grouping, having examples)


@dataclass(kw_only=True)
class Employee:
    """
    An employee record with numeric salary fields used in grouping examples.
    """

    name: str
    """The employee's name."""

    department: str
    """
    The employee's department name (a grouping key).
    """

    salary: int
    """Current salary."""

    starting_salary: int
    """
    Salary at hiring time.
    """


# %% Custom predicates (fragment-built verbalization examples)


@dataclass(kw_only=True, eq=False)
class Location:
    """
    A named place, used as the location of the :class:`IsReachable` predicate.
    """

    name: str
    """
    The location's name.
    """


@dataclass(kw_only=True, eq=False)
class IsReachable(Predicate):
    """
    Two-field custom predicate: *"<location> is reachable for <body>"*.
    """

    location: object
    """
    The place whose reachability is asserted.
    """

    body: object
    """
    The entity the location must be reachable for.
    """

    def __call__(self):
        return True

    @classmethod
    def _verbalization_fragment_(
        cls, *, fields: Mapping[str, VerbalizationFragment]
    ) -> VerbalizationFragment:
        return clause(
            Noun(content=fields["location"]),
            Copula(),
            Adjective(word="reachable"),
            Prepositions.FOR,
            Noun(content=fields["body"]),
        )


@dataclass(kw_only=True, eq=False)
class Department:
    """
    A department, used as the second field of :class:`WorksIn`.
    """

    name: str
    """
    The department's name.
    """


@dataclass(kw_only=True, eq=False)
class StaffMember:
    """
    A staff member belonging to a :class:`Department`.
    """

    name: str
    """
    The staff member's name.
    """

    department: Department
    """
    The department the staff member works in.
    """


@dataclass(kw_only=True, eq=False)
class WorksIn(Predicate):
    """
    Multi-field custom predicate: *"<employee> works in <department>"*.
    """

    employee: object
    """
    The employee (first positional field of the predicate).
    """

    department: object
    """
    The department (second positional field of the predicate).
    """

    def __call__(self):
        return True

    @classmethod
    def _verbalization_fragment_(
        cls, *, fields: Mapping[str, VerbalizationFragment]
    ) -> VerbalizationFragment:
        return clause(
            Noun(content=fields["employee"]),
            Verb(lemma="work"),
            Prepositions.IN,
            Noun(content=fields["department"]),
        )


# %% Birds (rule-tree / inference verbalization examples)


@dataclass(kw_only=True)
class Bird:
    """
    An example bird.
    """

    name: str
    """
    The bird's name.
    """


@dataclass(kw_only=True)
class LoveBirds:
    """
    A pair of :class:`Bird` objects with a strength-of-bond flag.
    """

    bird_1: Bird
    """
    The first bird of the pair.
    """

    bird_2: Bird
    """
    The second bird of the pair.
    """

    strong_love: bool
    """
    Whether the two birds share a strong bond.
    """


@dataclass(kw_only=True)
class BirdView(Symbol):
    """
    A perceived view over a :class:`Bird` (the base of the inferred bird symbols).
    """

    bird: Bird
    """
    The bird this view is about.
    """


@dataclass(kw_only=True)
class StrongLoveBird(BirdView):
    """
    A bird inferred to be in a strong-love pairing.
    """


@dataclass(kw_only=True)
class WeakLoveBird(BirdView):
    """
    A bird inferred to be in a weak-love pairing.
    """


# %% Furniture (deeply nested chains + aggregated-antecedent rule examples)


@dataclass(kw_only=True)
class Handle:
    """
    A handle attached to a container.
    """

    name: str
    """
    The handle's name.
    """


@dataclass(kw_only=True)
class Container:
    """
    A container that may hold handles and nest inside other containers.
    """

    name: str
    """
    The container's name.
    """


@dataclass(kw_only=True)
class FixedConnection:
    """
    A rigid connection between a container and a handle.
    """

    parent: Container
    """
    The container side of the connection.
    """

    child: Handle
    """
    The handle side of the connection.
    """


@dataclass(kw_only=True)
class PrismaticConnection:
    """
    A sliding connection between two containers (a drawer-in-cabinet joint).
    """

    parent: Container
    """
    The outer container.
    """

    child: Container
    """
    The inner (sliding) container.
    """


@dataclass(kw_only=True)
class Drawer:
    """
    A drawer inferred from a container plus a handle.
    """

    container: Container
    """
    The drawer's container.
    """

    handle: Handle
    """
    The drawer's handle.
    """


@dataclass(kw_only=True)
class Cabinet:
    """
    A cabinet inferred from a container plus a set of drawers.
    """

    container: Container
    """
    The cabinet's container.
    """

    drawers: list
    """
    The cabinet's drawers (aggregated antecedent → plural *"there are … drawers"*).
    """


# %% Causal queries (do() / causes_effect() examples)


@dataclass(kw_only=True)
class Pick:
    """
    A robot's attempt to grasp an object, whose success may hinge on where its arm was
    positioned.
    """

    arm: float
    """
    The arm's position at the time of the attempt.
    """

    grasped: bool
    """
    Whether the attempt succeeded.
    """
