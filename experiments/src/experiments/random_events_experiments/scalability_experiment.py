"""
This module provides functions and dataclass definitions for conducting scalability
experiments with the product algebra of the random_events package.

It measures how the algebraic operations on
:class:`~random_events.product_algebra.Event` (union, intersection, difference,
complement, simplify and make_disjoint) scale along three independent factors: the
number of variables involved, the domain size of the symbolic variables involved, and
the number of simple sets an event is composed of.
"""

from __future__ import annotations

import enum
import functools
import inspect
import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import plotly.graph_objects as go
import tqdm

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
)
from random_events.interval import Interval, closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.set import Set
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Continuous, Symbolic, Variable

CONTINUOUS_VARIABLE_BOUNDS = (0.0, 100.0)
"""
Lower and upper bound of the universe continuous variables are sampled from.
"""


class ProductAlgebraOperation(enum.Enum):
    """
    The product algebra operations measured by this experiment.

    Every member's value is the actual :class:`~random_events.product_algebra.Event`
    method that performs the operation. Binary operations (union, intersection,
    difference) take a second event as argument; unary operations (complement, simplify,
    make_disjoint) take none, see :attr:`is_binary`.
    """

    UNION = enum.member(Event.union_with)
    INTERSECTION = enum.member(Event.intersection_with)
    DIFFERENCE = enum.member(Event.difference_with)
    COMPLEMENT = enum.member(Event.complement)
    SIMPLIFY = enum.member(Event.simplify)
    MAKE_DISJOINT = enum.member(Event.make_disjoint)

    @property
    def field_prefix(self) -> str:
        """:return: Lowercase name of this operation, used as its dataclass field
        prefix."""
        return self.name.lower()

    @property
    def duration_field(self) -> str:
        """:return: Name of the dataclass field holding this operation's duration."""
        return f"{self.field_prefix}_duration"

    @property
    def resulting_simple_sets_field(self) -> str:
        """:return: Name of the dataclass field holding this operation's resulting
        simple set count."""
        return f"{self.field_prefix}_resulting_simple_sets"

    @property
    def is_binary(self) -> bool:
        """:return: Whether this operation takes a second event as argument."""
        return len(inspect.signature(self.value).parameters) == 2


@dataclass
class RandomEventFactory:
    """
    Generates random events over a mix of symbolic and continuous variables for the
    scalability experiment.

    Variables alternate between symbolic and continuous, so every generated event
    involves both variable kinds regardless of :attr:`number_of_variables`.
    """

    number_of_variables: int
    """
    How many variables an event is defined over, split between symbolic and continuous.
    """

    domain_size: int
    """
    Number of distinct elements in the domain of every symbolic variable.
    """

    number_of_simple_sets: int
    """
    Number of simple events that are unioned to form an event.
    """

    def make_variables(self) -> list[Variable]:
        """
        :return: Fresh variables, one per :attr:`number_of_variables`, alternating
            between symbolic variables with a domain of :attr:`domain_size` distinct
            elements and continuous variables over the real line.
        """
        variables: list[Variable] = []
        for index in range(self.number_of_variables):
            if index % 2 == 0:
                variables.append(
                    Symbolic(
                        name=f"symbolic_variable_{index}",
                        domain=Set.from_iterable(range(self.domain_size)),
                    )
                )
            else:
                variables.append(Continuous(name=f"continuous_variable_{index}"))
        return variables

    def make_event(self, variables: list[Variable]) -> Event:
        """
        Create a random event over the given variables.

        :param variables: The variables the event is defined over.
        :return: An event composed of :attr:`number_of_simple_sets` random simple
            events.
        """
        return Event.from_simple_sets(
            *[self._make_simple_event(variables) for _ in range(self.number_of_simple_sets)]
        )

    def _make_simple_event(self, variables: list[Variable]) -> SimpleEvent:
        """
        :param variables: The variables the simple event is defined over.
        :return: A simple event that assigns every symbolic variable a random,
            non-empty subset of its domain and every continuous variable a random
            bounded interval.
        """
        assignment = {
            variable: (
                self._make_symbolic_value(variable)
                if isinstance(variable, Symbolic)
                else self._make_continuous_value()
            )
            for variable in variables
        }
        return SimpleEvent.from_data(assignment)

    def _make_symbolic_value(self, variable: Symbolic) -> list[int]:
        """
        :param variable: The symbolic variable to sample a value for.
        :return: A random, non-empty subset of the variable's domain elements.
        """
        return random.sample(
            list(variable.domain.all_elements), random.randint(1, self.domain_size)
        )

    @staticmethod
    def _make_continuous_value() -> Interval:
        """
        :return: A random bounded interval within :data:`CONTINUOUS_VARIABLE_BOUNDS`.
        """
        lower_bound, upper_bound = CONTINUOUS_VARIABLE_BOUNDS
        lower = random.uniform(lower_bound, upper_bound)
        upper = random.uniform(lower, upper_bound)
        return closed(lower, upper)


@dataclass
class ProductAlgebraScalabilityExperimentResult(ExperimentResult):
    """
    Raw measurements from a single pass over all product algebra operations.

    All durations are in seconds. Resulting simple set counts reflect the actual events
    produced, which is not necessarily equal to the number of simple sets that were
    requested, since equal simple events collapse into one.
    """

    number_of_variables: int
    """
    Number of variables the measured events are defined over.
    """

    domain_size: int
    """
    Number of distinct elements in the domain of every symbolic variable.
    """

    number_of_simple_sets: int
    """
    Number of simple sets requested when constructing the measured events.
    """

    union_duration: float
    """
    Time spent computing the union of two random events.
    """

    union_resulting_simple_sets: int
    """
    Number of simple sets in the result of the union.
    """

    intersection_duration: float
    """
    Time spent computing the intersection of two random events.
    """

    intersection_resulting_simple_sets: int
    """
    Number of simple sets in the result of the intersection.
    """

    difference_duration: float
    """
    Time spent computing the difference of two random events.
    """

    difference_resulting_simple_sets: int
    """
    Number of simple sets in the result of the difference.
    """

    complement_duration: float
    """
    Time spent computing the complement of a random event.
    """

    complement_resulting_simple_sets: int
    """
    Number of simple sets in the result of the complement.
    """

    simplify_duration: float
    """
    Time spent simplifying a random event.
    """

    simplify_resulting_simple_sets: int
    """
    Number of simple sets in the result of the simplification.
    """

    make_disjoint_duration: float
    """
    Time spent making a random event disjoint.
    """

    make_disjoint_resulting_simple_sets: int
    """
    Number of simple sets in the result of making the event disjoint.
    """

    @property
    def total_duration(self) -> float:
        """:return: Sum of every operation's duration for this run."""
        return sum(
            getattr(self, operation.duration_field) for operation in ProductAlgebraOperation
        )

    @property
    def total_resulting_simple_sets(self) -> int:
        """:return: Sum of every operation's resulting simple set count for this
        run."""
        return sum(
            getattr(self, operation.resulting_simple_sets_field)
            for operation in ProductAlgebraOperation
        )


@dataclass
class ProductAlgebraScalabilityAggregateResult(ExperimentResult):
    """
    Aggregated statistics over multiple product algebra measurement runs at a fixed
    number of variables, domain size and number of simple sets.

    Every duration and simple-set-count field is a :class:`MeanAndStandardDeviation`
    computed across all iterations of :func:`run_scalability_experiment`.
    """

    number_of_variables: int
    """
    Number of variables the measured events are defined over.
    """

    domain_size: int
    """
    Number of distinct elements in the domain of every symbolic variable.
    """

    number_of_simple_sets: int
    """
    Number of simple sets requested when constructing the measured events.
    """

    total_duration: float
    """
    Mean, across iterations, of the summed duration (s) of every
    :class:`ProductAlgebraOperation`.

    Kept at full precision, unlike the per-operation :class:`MeanAndStandardDeviation`
    fields below, so it stays visible on a log-scale scalability plot even when
    individual operations run in microseconds.
    """

    total_resulting_simple_sets: float
    """
    Mean, across iterations, of the summed resulting simple set count of every
    :class:`ProductAlgebraOperation`.
    """

    union_duration: MeanAndStandardDeviation
    """
    Statistics over union duration (s) across iterations.
    """

    union_resulting_simple_sets: MeanAndStandardDeviation
    """
    Statistics over the number of simple sets in the union result across iterations.
    """

    intersection_duration: MeanAndStandardDeviation
    """
    Statistics over intersection duration (s) across iterations.
    """

    intersection_resulting_simple_sets: MeanAndStandardDeviation
    """
    Statistics over the number of simple sets in the intersection result across
    iterations.
    """

    difference_duration: MeanAndStandardDeviation
    """
    Statistics over difference duration (s) across iterations.
    """

    difference_resulting_simple_sets: MeanAndStandardDeviation
    """
    Statistics over the number of simple sets in the difference result across
    iterations.
    """

    complement_duration: MeanAndStandardDeviation
    """
    Statistics over complement duration (s) across iterations.
    """

    complement_resulting_simple_sets: MeanAndStandardDeviation
    """
    Statistics over the number of simple sets in the complement result across
    iterations.
    """

    simplify_duration: MeanAndStandardDeviation
    """
    Statistics over simplification duration (s) across iterations.
    """

    simplify_resulting_simple_sets: MeanAndStandardDeviation
    """
    Statistics over the number of simple sets in the simplification result across
    iterations.
    """

    make_disjoint_duration: MeanAndStandardDeviation
    """
    Statistics over make-disjoint duration (s) across iterations.
    """

    make_disjoint_resulting_simple_sets: MeanAndStandardDeviation
    """
    Statistics over the number of simple sets in the make-disjoint result across
    iterations.
    """


def time_composite_set_operation(
    operation: Callable[[], AbstractCompositeSet]
) -> tuple[float, int]:
    """
    Measure the wall-clock duration and resulting simple set count of an operation.

    :param operation: A zero-argument callable that performs one algebraic operation.
    :return: Tuple of (duration in seconds, number of simple sets in the result).
    """
    begin = time.perf_counter()
    result = operation()
    duration = time.perf_counter() - begin
    return round(duration, 6), len(result.simple_sets)


def product_algebra_scalability_experiment(
    number_of_variables: int, domain_size: int, number_of_simple_sets: int
) -> ProductAlgebraScalabilityExperimentResult:
    """
    Run a single pass over all product algebra operations on freshly generated random
    events and return timing and structural measurements.

    :param number_of_variables: Number of variables the events are defined over.
    :param domain_size: Number of distinct elements in the domain of every symbolic
        variable.
    :param number_of_simple_sets: Number of simple sets requested for each event.
    :return: Timing and structural measurements for this single run.
    """
    factory = RandomEventFactory(number_of_variables, domain_size, number_of_simple_sets)
    variables = factory.make_variables()
    event_a = factory.make_event(variables)
    event_b = factory.make_event(variables)

    measurements = {}
    for operation in ProductAlgebraOperation:
        arguments = (event_a, event_b) if operation.is_binary else (event_a,)
        duration, resulting_simple_sets = time_composite_set_operation(
            functools.partial(operation.value, *arguments)
        )
        measurements[operation.duration_field] = duration
        measurements[operation.resulting_simple_sets_field] = resulting_simple_sets

    return ProductAlgebraScalabilityExperimentResult(
        number_of_variables=number_of_variables,
        domain_size=domain_size,
        number_of_simple_sets=number_of_simple_sets,
        **measurements,
    )


def run_scalability_experiment(
    number_of_variables: int,
    domain_size: int,
    number_of_simple_sets: int,
    iterations: int = 10,
) -> ProductAlgebraScalabilityAggregateResult:
    """
    Repeatedly run :func:`product_algebra_scalability_experiment` with freshly sampled
    random events and aggregate timing and structural measurements across all runs.

    :param number_of_variables: Number of variables the events are defined over.
    :param domain_size: Number of distinct elements in the domain of every symbolic
        variable.
    :param number_of_simple_sets: Number of simple sets requested for each event.
    :param iterations: Number of independent runs to aggregate.
    :return: Aggregated mean and standard deviation for all measurements.
    """
    results = [
        product_algebra_scalability_experiment(
            number_of_variables, domain_size, number_of_simple_sets
        )
        for _ in range(iterations)
    ]

    def stats(attribute: str) -> MeanAndStandardDeviation:
        return MeanAndStandardDeviation.from_measurements(
            [getattr(result, attribute) for result in results]
        )

    return ProductAlgebraScalabilityAggregateResult(
        number_of_variables=number_of_variables,
        domain_size=domain_size,
        number_of_simple_sets=number_of_simple_sets,
        total_duration=statistics.mean(result.total_duration for result in results),
        total_resulting_simple_sets=statistics.mean(
            result.total_resulting_simple_sets for result in results
        ),
        **{
            operation.duration_field: stats(operation.duration_field)
            for operation in ProductAlgebraOperation
        },
        **{
            operation.resulting_simple_sets_field: stats(
                operation.resulting_simple_sets_field
            )
            for operation in ProductAlgebraOperation
        },
    )


@dataclass
class ScalabilitySweep:
    """
    Aggregate results produced by varying a single factor, ready to be summarized
    alongside sweeps of the other factors.
    """

    label: str
    """
    Human-readable name of the varied factor, used as its legend entry.
    """

    table: ExperimentsTable
    """
    Aggregate results, one row per value the factor was swept over.
    """

    x_attribute: str
    """
    Name of the configuration field that was varied across the table's rows.
    """

    @property
    def rows(self) -> list[ProductAlgebraScalabilityAggregateResult]:
        """:return: The table's rows, typed as aggregate scalability results."""
        return self.table.experiments

    @property
    def swept_values(self) -> list[int]:
        """:return: The actual configuration values the factor was swept over."""
        return [getattr(row, self.x_attribute) for row in self.rows]

    @property
    def total_durations(self) -> list[float]:
        """:return: Per-row mean of the summed duration across all operations."""
        return [row.total_duration for row in self.rows]

    @property
    def total_resulting_simple_sets(self) -> list[float]:
        """:return: Per-row mean of the summed resulting simple set count across all
        operations."""
        return [row.total_resulting_simple_sets for row in self.rows]


SCALABILITY_SWEEP_COLORS = {
    "Number of Variables": "#2a78d6",
    "Domain Size": "#eb6834",
    "Number of Simple Sets": "#1baf7a",
}
"""
Fixed categorical color per sweep label, in the palette's slot order.
"""

MARKER_SIZE_RANGE = (8, 40)
"""
Minimum and maximum marker diameter (px) used to encode resulting simple set counts.
"""


def plot_scalability_summary(sweeps: list[ScalabilitySweep]) -> go.Figure:
    """
    Summarize every scalability sweep in a single figure.

    Each sweep's swept values are normalised to a shared ``[0, 1]`` "fraction of range"
    x-axis, so factors with very different natural ranges (a handful of variables vs. up
    to a hundred simple sets) can be compared directly. The y-axis is the total duration
    summed over every :class:`ProductAlgebraOperation`, on a log scale since durations
    span microseconds to seconds. Marker size encodes the total resulting simple set
    count, so structural growth stays visible without a second axis.

    :param sweeps: The scalability sweeps to summarize, one per varied factor.
    :return: A single Plotly figure summarizing all sweeps.
    """
    max_resulting_simple_sets = max(
        size for sweep in sweeps for size in sweep.total_resulting_simple_sets
    )
    min_marker_size, max_marker_size = MARKER_SIZE_RANGE

    fig = go.Figure()
    for sweep in sweeps:
        values = sweep.swept_values
        x = (
            [index / (len(values) - 1) for index in range(len(values))]
            if len(values) > 1
            else [0.0]
        )
        durations = [max(duration, 1e-7) for duration in sweep.total_durations]
        sizes = sweep.total_resulting_simple_sets
        marker_sizes = [
            min_marker_size
            + (max_marker_size - min_marker_size)
            * math.sqrt(size / max_resulting_simple_sets)
            for size in sizes
        ]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=durations,
                mode="lines+markers",
                name=sweep.label,
                line=dict(color=SCALABILITY_SWEEP_COLORS[sweep.label], width=2),
                marker=dict(size=marker_sizes, color=SCALABILITY_SWEEP_COLORS[sweep.label]),
                customdata=list(zip(values, sizes)),
                hovertemplate=(
                    f"{sweep.label}: %{{customdata[0]}}<br>"
                    "Total duration: %{y:.6f} s<br>"
                    "Total resulting simple sets: %{customdata[1]:.1f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Product Algebra Scalability Summary",
        xaxis_title="Fraction of Swept Range",
        yaxis_title="Total Duration Across All Operations (s)",
        yaxis_type="log",
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font=dict(color="#0b0b0b"),
    )
    fig.update_xaxes(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7")
    fig.update_yaxes(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7")
    return fig


def main():
    baseline_number_of_variables = 3
    baseline_domain_size = 4
    baseline_number_of_simple_sets = 5

    variable_counts = [1, 2, 3, 4, 5, 6, 7, 8]
    domain_sizes = [2, 4, 8, 16, 32]
    simple_set_counts = [1, 2, 5, 10, 20, 30, 40, 50, 65, 80, 100]

    variable_table = ExperimentsTable(
        [
            run_scalability_experiment(
                number_of_variables, baseline_domain_size, baseline_number_of_simple_sets
            )
            for number_of_variables in tqdm.tqdm(
                variable_counts, desc="Number of Variables"
            )
        ]
    )
    domain_size_table = ExperimentsTable(
        [
            run_scalability_experiment(
                baseline_number_of_variables, domain_size, baseline_number_of_simple_sets
            )
            for domain_size in tqdm.tqdm(domain_sizes, desc="Domain Size")
        ]
    )
    simple_set_table = ExperimentsTable(
        [
            run_scalability_experiment(
                baseline_number_of_variables, baseline_domain_size, number_of_simple_sets
            )
            for number_of_simple_sets in tqdm.tqdm(
                simple_set_counts, desc="Number of Simple Sets"
            )
        ]
    )

    print(
        TypstRenderer(variable_table).render_figure(
            "Product algebra operation timings and resulting simple set counts as "
            "the number of variables varies, with domain size and simple sets per "
            "event held fixed at their baseline values. Each row aggregates several "
            "independently sampled event pairs to report mean and standard "
            "deviation."
        )
    )
    print()
    print(
        TypstRenderer(domain_size_table).render_figure(
            "Product algebra operation timings and resulting simple set counts as "
            "the symbolic domain size varies, with the number of variables and "
            "simple sets per event held fixed at their baseline values. Each row "
            "aggregates several independently sampled event pairs to report mean "
            "and standard deviation."
        )
    )
    print()
    print(
        TypstRenderer(simple_set_table).render_figure(
            "Product algebra operation timings and resulting simple set counts as "
            "the number of simple sets per event varies, with the number of "
            "variables and domain size held fixed at their baseline values. Each "
            "row aggregates several independently sampled event pairs to report "
            "mean and standard deviation."
        )
    )

    sweeps = [
        ScalabilitySweep("Number of Variables", variable_table, "number_of_variables"),
        ScalabilitySweep("Domain Size", domain_size_table, "domain_size"),
        ScalabilitySweep(
            "Number of Simple Sets", simple_set_table, "number_of_simple_sets"
        ),
    ]
    plot_scalability_summary(sweeps).show()


if __name__ == "__main__":
    main()
