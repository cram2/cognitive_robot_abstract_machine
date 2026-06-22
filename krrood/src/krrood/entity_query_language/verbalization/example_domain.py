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
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List

from krrood.entity_query_language.factories import Symbol
from krrood.entity_query_language.predicate import Predicate

# ── Robots & missions (the quick-start + cross-variable examples) ────────────


@dataclass
class Robot:
    """An example autonomous robot."""

    name: str
    """The robot's identifier string."""

    battery: int
    """Charge level from 0 to 100."""

    operational: bool
    """Whether the robot is operational (a boolean terminal for predicative chains)."""


@dataclass
class Mission:
    """A task assigned to a :class:`Robot`, with a numeric priority."""

    assigned_to: Robot
    """The robot the mission is assigned to."""

    priority: int
    """Priority rank (higher is more urgent)."""


# ── Boolean & list attributes (predicative / indexed-attribute examples) ─────


@dataclass
class Task:
    """A unit of work with a boolean completion flag."""

    name: str
    """The task's name."""

    completed: bool
    """Whether the task is finished — drives the predicative *"is completed"* form."""


@dataclass
class Worker:
    """A worker holding an ordered list of :class:`Task` objects."""

    name: str
    """The worker's name."""

    tasks: List[Task]
    """The worker's tasks (indexable, e.g. ``tasks[0]``)."""


# ── Money (aggregation, nested-attribute, and date-range examples) ───────────


@dataclass
class AmountDetails:
    """A monetary amount, nested inside a :class:`BankTransaction`."""

    amount: float
    """The amount of money."""


@dataclass
class BankTransaction:
    """A bank transaction with a nested amount and a booking date."""

    amount_details: AmountDetails
    """The transaction's amount (a nested attribute chain)."""

    booking_date: datetime.datetime
    """When the transaction was booked (used for *"between … and …"* date folding)."""


# ── Employees (same-type disambiguation, grouping, having examples) ──────────


@dataclass
class Employee:
    """An employee record with numeric salary fields used in grouping examples."""

    name: str
    """The employee's name."""

    department: str
    """The employee's department name (a grouping key)."""

    salary: int
    """Current salary."""

    starting_salary: int
    """Salary at hiring time."""


# ── Custom predicates (template-driven verbalization examples) ───────────────


@dataclass(eq=False)
class Location:
    """A named place, used as the body of the :class:`IsReachable` predicate."""

    name: str
    """The location's name."""


@dataclass(eq=False)
class IsReachable(Predicate):
    """Single-field custom predicate: *"<body> is reachable"*."""

    body: object
    """The thing whose reachability is asserted."""

    def __call__(self):
        return True

    @classmethod
    def _verbalization_template_(cls):
        return "{body} is reachable"


@dataclass(eq=False)
class Department:
    """A department, used as the second field of :class:`WorksIn`."""

    name: str
    """The department's name."""


@dataclass(eq=False)
class StaffMember:
    """A staff member belonging to a :class:`Department`."""

    name: str
    """The staff member's name."""

    department: Department
    """The department the staff member works in."""


@dataclass(eq=False)
class WorksIn(Predicate):
    """Multi-field custom predicate: *"<employee> works in <department>"*."""

    employee: object
    """The employee (first positional argument of the template)."""

    department: object
    """The department (second positional argument of the template)."""

    def __call__(self):
        return True

    @classmethod
    def _verbalization_template_(cls):
        return "{employee} works in {department}"


# ── Birds (rule-tree / inference verbalization examples) ─────────────────────


@dataclass
class Bird:
    """An example bird."""

    name: str
    """The bird's name."""


@dataclass
class LoveBirds:
    """A pair of :class:`Bird` objects with a strength-of-bond flag."""

    bird_1: Bird
    """The first bird of the pair."""

    bird_2: Bird
    """The second bird of the pair."""

    strong_love: bool
    """Whether the two birds share a strong bond."""


@dataclass
class BirdView(Symbol):
    """A perceived view over a :class:`Bird` (the base of the inferred bird symbols)."""

    bird: Bird
    """The bird this view is about."""


@dataclass
class StrongLoveBird(BirdView):
    """A bird inferred to be in a strong-love pairing."""


@dataclass
class WeakLoveBird(BirdView):
    """A bird inferred to be in a weak-love pairing."""


# ── Furniture (deeply nested chains + aggregated-antecedent rule examples) ───


@dataclass
class Handle:
    """A handle attached to a container."""

    name: str
    """The handle's name."""


@dataclass
class Container:
    """A container that may hold handles and nest inside other containers."""

    name: str
    """The container's name."""


@dataclass
class FixedConnection:
    """A rigid connection between a container and a handle."""

    parent: Container
    """The container side of the connection."""

    child: Handle
    """The handle side of the connection."""


@dataclass
class PrismaticConnection:
    """A sliding connection between two containers (a drawer-in-cabinet joint)."""

    parent: Container
    """The outer container."""

    child: Container
    """The inner (sliding) container."""


@dataclass
class Drawer:
    """A drawer inferred from a container plus a handle."""

    container: Container
    """The drawer's container."""

    handle: Handle
    """The drawer's handle."""


@dataclass
class Cabinet:
    """A cabinet inferred from a container plus a set of drawers."""

    container: Container
    """The cabinet's container."""

    drawers: list
    """The cabinet's drawers (aggregated antecedent → plural *"there are … drawers"*)."""
