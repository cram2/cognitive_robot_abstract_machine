"""
This module demonstrates, with a concrete constructed input, that
:meth:`~random_events.product_algebra.Event.complement` can require time and output size
exponential in the number of input simple sets -- and pins down precisely which property
of the input causes that, as opposed to merely "many simple sets" or "many variables".

Background
----------
An :class:`~random_events.product_algebra.Event` with ``n`` simple sets is a DNF
formula ``OR_i (AND_j x_j in I_ij)``. Complementing it via De Morgan gives a CNF
``AND_i (OR_j x_j not in I_ij)``, which then has to be re-expressed as a DNF (a union
of boxes) because that is the only representation composite sets have. Re-expanding a
CNF into a DNF means picking one disjunct from every one of the ``n`` clauses, which
can require up to (branching factor) ** n terms in the worst case.

That worst case is only reachable when the ``n`` clauses are *mutually independent*
-- no two clauses share a variable, so no two candidate terms can ever merge or
subsume one another. :func:`disjoint_clause_event` builds exactly that: ``n`` simple
sets, each pinning its own private pair of fresh binary variables, giving a complement
that provably needs ``2 ** n`` simple sets -- not an artifact of a naive algorithm, the
true minimal answer is that large.

:func:`shared_dimension_event` is the control: the same number of clauses, but all of
them constrain the *same* two variables. Because clauses can now interact along shared
axes, later clauses split and absorb earlier clauses' leftover pieces instead of
multiplying them, and the complement stays linear in ``n``. For any fixed number of
variables ``d``, the arrangement complexity of ``n`` axis-aligned boxes in ``R^d`` is
bounded by ``O(n ** d)`` (a standard computational-geometry result), so exponential
blow-up in ``n`` is only reachable if the number of *independent* variables is also
allowed to grow with ``n`` -- which is exactly what :func:`disjoint_clause_event` does
and :func:`shared_dimension_event` deliberately does not.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import tqdm

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    TypstRenderer,
)
from experiments.random_events_experiments.scalability_experiment import (
    time_composite_set_operation,
)
from random_events.product_algebra import Event, SimpleEvent
from random_events.set import Set
from random_events.variable import Symbolic


class ComplementWorstCaseConstruction(enum.StrEnum):
    """
    The two event constructions compared by this experiment.
    """

    DISJOINT_CLAUSES = (
        "Disjoint clauses (private variable pair per clause) -- exponential"
    )
    """
    Each clause owns a fresh, private pair of variables: no two clauses can interact, so
    the complement is forced to its exponential worst case.
    """

    SHARED_DIMENSIONS = "Shared dimensions (control) -- polynomial"
    """
    Every clause constrains the same two variables: clauses can split and absorb each
    other's pieces, capping the complement at a linear number of simple sets.
    """

    def __str__(self) -> str:
        return self.value


def disjoint_clause_event(number_of_clauses: int) -> Event:
    """
    Build the adversarial event that forces
    :meth:`~random_events.product_algebra.Event.complement` into its exponential worst
    case.

    Creates ``number_of_clauses`` simple events over ``2 * number_of_clauses`` fresh
    binary symbolic variables (domain ``{0, 1}``). Clause ``i`` pins its own private
    pair of variables ``(x_2i, x_2i+1)`` to ``1`` and leaves every other variable
    unconstrained. Because every clause's variable support is disjoint from every
    other clause's, De Morgan's law turns the complement into a conjunction of
    ``number_of_clauses`` independent binary choices with no possibility of merging
    terms across clauses: the minimal disjoint-box representation of the complement
    requires exactly ``2 ** number_of_clauses`` simple sets.

    :param number_of_clauses: Number of mutually independent clauses to construct.
    :return: The event whose complement is exponential in ``number_of_clauses``.
    """
    variables = [
        Symbolic(name=f"x{index}", domain=Set.from_iterable([0, 1]))
        for index in range(2 * number_of_clauses)
    ]
    simple_events = [
        SimpleEvent.from_data({variables[2 * i]: 1, variables[2 * i + 1]: 1})
        for i in range(number_of_clauses)
    ]
    return Event.from_simple_sets(*simple_events)


def shared_dimension_event(number_of_clauses: int) -> Event:
    """
    Build the control event: the same number of clauses as
    :func:`disjoint_clause_event`, but all clauses constrain the *same* two variables
    instead of private ones.

    Creates two symbolic variables ``x``, ``y`` with a domain sized to
    ``number_of_clauses``, and one simple event per clause pinning ``(x, y)`` to the
    diagonal cell ``(i, i)``. Because every clause shares the same pair of dimensions,
    later clauses can split and absorb earlier clauses' leftover pieces instead of
    multiplying them, so the complement stays proportional to ``number_of_clauses``.

    :param number_of_clauses: Number of clauses to construct, all sharing dimensions.
    :return: The event whose complement stays linear in ``number_of_clauses``.
    """
    x = Symbolic(name="x", domain=Set.from_iterable(range(number_of_clauses)))
    y = Symbolic(name="y", domain=Set.from_iterable(range(number_of_clauses)))
    simple_events = [
        SimpleEvent.from_data({x: index, y: index}) for index in range(number_of_clauses)
    ]
    return Event.from_simple_sets(*simple_events)


@dataclass
class ComplementWorstCaseExperimentResult(ExperimentResult):
    """
    One measurement of :meth:`~random_events.product_algebra.Event.complement` on a
    specific construction.
    """

    construction: ComplementWorstCaseConstruction
    """
    Which of the two constructions this measurement was taken on.
    """

    number_of_clauses: int
    """
    Number of simple sets the measured event was built from.
    """

    number_of_dimensions: int
    """
    Number of variables the measured event is defined over.
    """

    complement_simple_sets: int
    """
    Number of simple sets in the computed complement.
    """

    duration: float
    """
    Time spent computing the complement, in seconds.
    """


def measure_complement(
    construction: ComplementWorstCaseConstruction, event: Event
) -> ComplementWorstCaseExperimentResult:
    """
    Measure :meth:`~random_events.product_algebra.Event.complement` on a single event.

    :param construction: Which construction ``event`` was built with.
    :param event: The event to complement.
    :return: The measurement, including the resulting complement's simple set count.
    """
    measurement = time_composite_set_operation(event.complement)
    return ComplementWorstCaseExperimentResult(
        construction=construction,
        number_of_clauses=len(event.simple_sets),
        number_of_dimensions=len(event.variables),
        complement_simple_sets=measurement.resulting_simple_sets,
        duration=measurement.duration,
    )


def main():
    disjoint_clause_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    shared_dimension_clause_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    disjoint_table = ExperimentsTable(
        [
            measure_complement(
                ComplementWorstCaseConstruction.DISJOINT_CLAUSES,
                disjoint_clause_event(number_of_clauses),
            )
            for number_of_clauses in tqdm.tqdm(
                disjoint_clause_counts, desc="Disjoint Clauses"
            )
        ]
    )
    shared_table = ExperimentsTable(
        [
            measure_complement(
                ComplementWorstCaseConstruction.SHARED_DIMENSIONS,
                shared_dimension_event(number_of_clauses),
            )
            for number_of_clauses in tqdm.tqdm(
                shared_dimension_clause_counts, desc="Shared Dimensions"
            )
        ]
    )

    print(
        TypstRenderer(disjoint_table).render_figure(
            "Complement of n mutually independent clauses, each pinning its own "
            "private pair of fresh binary variables to 1 and leaving every other "
            "variable unconstrained. Because no two clauses share a variable, no two "
            "candidate complement terms can ever merge, so the minimal complement "
            "provably requires 2^n simple sets -- exponential in the number of "
            "clauses, not the number of variables per se."
        )
    )
    print()
    print(
        TypstRenderer(shared_table).render_figure(
            "Control: the same number of clauses as the table above, but every "
            "clause constrains the SAME two shared variables instead of private "
            "ones. Because clauses can now interact along shared axes, later clauses "
            "split and absorb earlier clauses' leftover pieces instead of "
            "multiplying them, so the complement stays linear in the number of "
            "clauses however far the sweep is pushed. This isolates the precise "
            "precondition for exponential complement blow-up: mutually independent "
            "clauses (pairwise disjoint variable support), not merely a large "
            "number of simple sets. For any fixed number of variables d, the "
            "complement is bounded by O(n^d), which is polynomial -- exponential "
            "behaviour requires the number of independent variables to grow with n."
        )
    )


if __name__ == "__main__":
    main()
