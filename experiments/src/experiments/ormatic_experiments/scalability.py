"""
This module provides functions and dataclass definitions for conducting
scalability experiments with the ORMatic library. It includes tools for
collecting mappable classes, alternative mappings, and type mappings, as well
as computing timing and structural statistics across multiple ORMatic
generation runs.
"""

import logging
import random
import time
import tempfile
from contextlib import contextmanager
from dataclasses import is_dataclass, dataclass
from typing import Type, List, Set, Tuple

import plotly.graph_objects as go
import tqdm

import coraplex.orm.ormatic_interface  # type: ignore
import coraplex.plans.plan_node
import semantic_digital_twin  # type: ignore
from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
)
from krrood.class_diagrams import ClassDiagram
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.helper import get_classes_of_ormatic_interface
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.type_dict import TypeDict
from krrood.utils import recursive_subclasses
from coraplex.robot_plans.actions.base import ActionDescription


def build_cram_class_sets() -> Tuple[Set[Type], List[Type], dict]:
    """
    Collect all mappable classes, alternative mappings, and type mappings from
    the coraplex ORM interface.

    Filters out non-dataclasses and AlternativeMapping subclasses from the raw
    interface, then augments with the original classes of every registered
    AlternativeMapping so the full set is consistent.

    :return: Tuple of (classes, alternative_mappings, type_mappings) ready to
             pass to :func:`run_scalability_experiment`.
    """
    classes, alternative_mappings, type_mappings = get_classes_of_ormatic_interface(
        coraplex.orm.ormatic_interface
    )
    classes = set(classes)

    alternative_mappings += [
        alternative_mapping
        for alternative_mapping in recursive_subclasses(AlternativeMapping)
    ]
    alternative_mappings = list(set(alternative_mappings))
    classes = {
        c for c in classes if is_dataclass(c) and not issubclass(c, AlternativeMapping)
    }
    classes |= {am.original_class() for am in recursive_subclasses(AlternativeMapping)}
    alternative_mappings = [
        am
        for am in recursive_subclasses(AlternativeMapping)
        if am.original_class() in classes
    ]
    return classes, alternative_mappings, type_mappings


@dataclass
class ORMaticScalabilityExperimentResult(ExperimentResult):
    """
    Raw measurements from a single ORMatic generation run.

    All durations are in seconds, rounded to two decimal places.
    Structural counts reflect the class diagram that was actually built for
    the given filtered class set.
    """

    total_duration: float
    """Wall-clock time from ClassDiagram creation to file write completion."""
    class_diagram_creation_duration: float
    """Time spent constructing the ClassDiagram."""
    ormatic_reasoning_duration: float
    """Time spent in ORMatic.make_all_tables()."""
    writing_to_file_duration: float
    """Time spent serialising the generated SQLAlchemy code to a temp file."""
    number_of_classes: int
    """Number of classes in the filtered input set."""
    number_of_associations: int
    """Number of association edges in the resulting class diagram."""
    number_of_inheritances: int
    """Number of inheritance edges in the resulting class diagram."""


@dataclass
class ORMaticScalabilityAggregateResult(ExperimentResult):
    """
    Aggregated statistics over multiple ORMatic generation runs at a fixed drop probability.

    Every numeric field is a :class:`MeanAndStandardDeviation` computed across
    all iterations of :func:`run_scalability_experiment`.  Structural counts
    (classes, associations, inheritances) vary between iterations because the
    class subset is resampled each time.
    """

    class_drop_probability: float
    """Fraction of classes randomly excluded from each iteration's input set."""
    number_of_classes: MeanAndStandardDeviation
    """Statistics over the size of the filtered class set across iterations."""
    number_of_associations: MeanAndStandardDeviation
    """Statistics over association edge count across iterations."""
    number_of_inheritances: MeanAndStandardDeviation
    """Statistics over inheritance edge count across iterations."""
    total_duration: MeanAndStandardDeviation
    """Statistics over total generation time (s) across iterations."""
    class_diagram_creation_duration: MeanAndStandardDeviation
    """Statistics over ClassDiagram construction time (s) across iterations."""
    ormatic_reasoning_duration: MeanAndStandardDeviation
    """Statistics over ORMatic reasoning time (s) across iterations."""
    writing_to_file_duration: MeanAndStandardDeviation
    """Statistics over file serialisation time (s) across iterations."""


@contextmanager
def _silence_ormatic_logger():
    logger = logging.getLogger("krrood.ormatic")
    original_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(original_level)


def ormatic_scalability_experiment(
    filtered_classes: Set[Type],
    alternative_mappings: List[Type],
    type_mappings: dict,
) -> ORMaticScalabilityExperimentResult:
    """
    Run a single ORMatic generation pass over a pre-determined class set and
    return timing and structural measurements.

    ORMatic log output is suppressed for the duration of the run to keep
    benchmark output readable.

    :param filtered_classes: The exact set of classes to map in this run.
    :param alternative_mappings: AlternativeMapping subclasses to register with ORMatic.
    :param type_mappings: Custom type-to-column mappings forwarded to :class:`TypeDict`.
    :return: Timing breakdown and class-diagram statistics for this single run.
    """
    with _silence_ormatic_logger():
        return _ormatic_scalability_experiment(
            filtered_classes, alternative_mappings, type_mappings
        )


def _ormatic_scalability_experiment(
    filtered_classes: Set[Type],
    alternative_mappings: List[Type],
    type_mappings: dict,
) -> ORMaticScalabilityExperimentResult:
    """
    Inner implementation of a single ORMatic generation pass without logger suppression.

    Measures three sequential phases: ClassDiagram construction, ORMatic table reasoning
    (``make_all_tables``), and writing the generated SQLAlchemy file to a temp file.

    :param filtered_classes: The exact set of classes to map in this run.
    :param alternative_mappings: AlternativeMapping subclasses to register with ORMatic.
    :param type_mappings: Custom type-to-column mappings forwarded to :class:`TypeDict`.
    :return: Timing breakdown and class-diagram statistics for this single run.
    """
    begin = time.perf_counter()

    class_diagram = ClassDiagram(
        list(sorted(filtered_classes, key=lambda c: c.__name__, reverse=True))
    )

    class_diagram_creation_time = time.perf_counter()

    ormatic = ORMatic(
        class_diagram,
        type_mappings=TypeDict(type_mappings),
        alternative_mappings=alternative_mappings,
    )
    ormatic.make_all_tables()

    ormatic_reasoning_time = time.perf_counter()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        ormatic.to_sqlalchemy_file(f)

    writing_to_file_time = time.perf_counter()

    return ORMaticScalabilityExperimentResult(
        total_duration=round(writing_to_file_time - begin, 2),
        class_diagram_creation_duration=round(class_diagram_creation_time - begin, 2),
        ormatic_reasoning_duration=round(
            ormatic_reasoning_time - class_diagram_creation_time, 2
        ),
        writing_to_file_duration=round(
            writing_to_file_time - ormatic_reasoning_time, 2
        ),
        number_of_classes=len(filtered_classes),
        number_of_associations=len(class_diagram.associations),
        number_of_inheritances=len(class_diagram.inheritance_relations),
    )


def run_scalability_experiment(
    classes: List[Type],
    alternative_mappings: List[Type],
    type_mappings: dict,
    class_drop_probability: float = 0.3,
    iterations: int = 10,
    required_classes: List[Type] | None = None,
) -> ORMaticScalabilityAggregateResult:
    """
    Repeatedly sample a random subset of classes and run the ORMatic generation
    pipeline, then aggregate timing and structural measurements across all runs.

    Each iteration independently resamples the class subset so that the reported
    standard deviations reflect variability from both class-set composition and
    runtime noise.  AlternativeMapping original classes and any ``required_classes``
    are always present in every iteration's input, regardless of the drop probability.

    :param classes: Full pool of candidate classes to sample from.
    :param alternative_mappings: AlternativeMapping subclasses to register with ORMatic.
    :param type_mappings: Custom type-to-column mappings forwarded to :class:`TypeDict`.
    :param class_drop_probability: Per-class probability of exclusion in each iteration.
    :param iterations: Number of independent generation runs to aggregate.
    :param required_classes: Classes pinned into every iteration's input set.
    :return: Aggregated mean and standard deviation for all measurements.
    """
    pinned = set(required_classes) if required_classes else set()
    results = []
    for _ in range(iterations):
        filtered_classes = {
            c for c in classes if random.uniform(0, 1) > class_drop_probability
        }
        filtered_classes |= {
            alternative_mapping.original_class()
            for alternative_mapping in alternative_mappings
        }
        filtered_classes |= pinned
        results.append(
            ormatic_scalability_experiment(
                filtered_classes, alternative_mappings, type_mappings
            )
        )

    return ORMaticScalabilityAggregateResult(
        class_drop_probability=class_drop_probability,
        number_of_classes=MeanAndStandardDeviation.from_measurements(
            [r.number_of_classes for r in results]
        ),
        number_of_associations=MeanAndStandardDeviation.from_measurements(
            [r.number_of_associations for r in results]
        ),
        number_of_inheritances=MeanAndStandardDeviation.from_measurements(
            [r.number_of_inheritances for r in results]
        ),
        total_duration=MeanAndStandardDeviation.from_measurements(
            [r.total_duration for r in results]
        ),
        class_diagram_creation_duration=MeanAndStandardDeviation.from_measurements(
            [r.class_diagram_creation_duration for r in results]
        ),
        ormatic_reasoning_duration=MeanAndStandardDeviation.from_measurements(
            [r.ormatic_reasoning_duration for r in results]
        ),
        writing_to_file_duration=MeanAndStandardDeviation.from_measurements(
            [r.writing_to_file_duration for r in results]
        ),
    )


def plot_scalability(table: ExperimentsTable) -> go.Figure:
    """
    Produce a band plot of number of classes (x) vs mean total runtime (y).

    The shaded band covers mean ± 1 standard deviation of the total duration.
    The x-axis uses the mean number of classes from each
    :class:`ORMaticScalabilityAggregateResult` row in ``table``.

    :param table: An :class:`ExperimentsTable` whose rows are
                  :class:`ORMaticScalabilityAggregateResult` instances, typically
                  produced by calling :func:`run_scalability_experiment` at
                  several different drop probabilities.
    :return: A Plotly figure ready for display or export.
    """
    rows: List[ORMaticScalabilityAggregateResult] = table.experiments

    x = [r.number_of_classes.mean for r in rows]
    y = [r.total_duration.mean for r in rows]
    y_upper = [
        r.total_duration.mean + r.total_duration.standard_deviation for r in rows
    ]
    y_lower = [
        r.total_duration.mean - r.total_duration.standard_deviation for r in rows
    ]

    fig = go.Figure(
        [
            go.Scatter(
                x=x + x[::-1],
                y=y_upper + y_lower[::-1],
                fill="toself",
                fillcolor="rgba(0, 100, 250, 0.2)",
                line=dict(color="rgba(0, 0, 0, 0)"),
                showlegend=True,
                name="±1 std",
            ),
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=dict(color="rgb(0, 100, 250)"),
                name="mean",
            ),
        ]
    )
    fig.update_layout(
        xaxis_title="Number of Classes",
        yaxis_title="Total Duration (s)",
        title="ORMatic Scalability: Classes vs Runtime",
    )
    return fig


def main():
    classes, alternative_mappings, type_mappings = build_cram_class_sets()
    required_classes = [coraplex.plans.plan_node.UnderspecifiedNode, ActionDescription]
    results = []
    for class_drop_probability in tqdm.tqdm(
        [
            0.9,
            0.8,
            0.7,
            0.6,
            0.5,
            0.4,
            0.3,
            0.2,
            0.1,
            0.0,
        ]
    ):
        results.append(
            run_scalability_experiment(
                classes,
                alternative_mappings,
                type_mappings,
                class_drop_probability,
                iterations=10,
                required_classes=required_classes,
            )
        )

    table = ExperimentsTable(results)
    print(TypstRenderer(table).render_table())
    plot_scalability(table).show()


if __name__ == "__main__":
    main()
