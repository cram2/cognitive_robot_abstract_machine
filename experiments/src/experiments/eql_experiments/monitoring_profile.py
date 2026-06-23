"""
Profile the runtime overhead of EQL's monitoring machinery.

EQL adds two independent monitoring layers around query construction and evaluation:

- **Layer 1 - call-stack capture** (the ``@monitored`` decorator in
  :mod:`krrood.entity_query_language._monitoring`). On every monitored object's
  ``__post_init__`` it runs :func:`inspect.stack`, builds a
  :class:`~krrood.entity_query_language._stack.StackFrame` per frame and filters out
  site-packages. It fires for ``Variable``, ``InstantiatedVariable`` and ``Query`` both at
  construction time and during evaluation when inferred instances are created. It is globally
  toggleable through :meth:`~krrood.entity_query_language._monitoring.MonitoredRegistry.disabled`.
- **Layer 2 - evaluation observers** (:mod:`krrood.entity_query_language.evaluation`). The
  default evaluation context attaches ``EvaluationTracker``, ``SatisfiedConditionTracker`` and
  ``InferenceRecorder``, which are invoked for every ``on_evaluate_enter`` / ``on_result_yielded``
  / ``on_conclusions_processed`` event dispatched from the hot wrapper
  ``SymbolicExpression._evaluate_``.

This module measures both layers two ways:

1. An A/B wall-clock stress test across three monitoring levels
   (:class:`MonitoringLevel`) that reports mean +- standard deviation per iteration and the
   overhead of each level relative to the fully-disabled baseline.
2. A :class:`line_profiler.LineProfiler` pass over every hot monitoring method, giving per-line
   timings so the most expensive lines are visible.

Both a console summary and a saved markdown + CSV report are produced. No ``krrood`` source is
modified - the monitoring levels are realised purely through the existing public
:meth:`MonitoredRegistry.disabled` and
:func:`~krrood.entity_query_language.evaluation_context.set_evaluation_context` APIs.

Run with (the ``experiments`` package must be importable, e.g. installed editable or with
``experiments/src`` on ``PYTHONPATH``)::

    python -m experiments.eql_experiments.monitoring_profile

This module deliberately does not use ``from __future__ import annotations``: the
:class:`~experiments.experiment_definitions.ExperimentResult` introspector requires the dataclass
field annotations to be real classes rather than strings.
"""

import csv
import io
import platform
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, List

from line_profiler import LineProfiler

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language._monitoring import MonitoredRegistry, monitored
from krrood.entity_query_language._stack import CallStack, StackFrame
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.evaluation import (
    EvaluationTracker,
    InferenceRecorder,
    SatisfiedConditionTracker,
    is_condition_participant,
)
from krrood.entity_query_language.evaluation_context import (
    EvaluationContext,
    _evaluation_context_var,
    set_evaluation_context,
)
from krrood.entity_query_language.factories import (
    an,
    and_,
    entity,
    inference,
    set_of,
    variable,
)
from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
)


# %% Self-contained domain types


@dataclass(unsafe_hash=True)
class Gadget(Symbol):
    """A small symbol with a few attributes used to exercise filters and aggregations."""

    name: str
    """
    The name of the gadget.
    """

    weight: int
    """
    The weight of the gadget.
    """

    category: str
    """
    The category of the gadget.
    """


@dataclass(unsafe_hash=True)
class Owner(Symbol):
    """A symbol used as a second source to exercise multi-source (cross-product) queries."""

    name: str
    """
    The name of the owner.
    """


_CATEGORIES = ("tool", "toy", "device")


def _build_gadgets(count: int) -> List[Gadget]:
    """
    Construct a deterministic list of :class:`Gadget` instances.

    :param count: Number of gadgets to create.
    :return: The freshly created gadgets.
    """
    return [
        Gadget(name=f"Gadget{i}", weight=(i % 7) + 1, category=_CATEGORIES[i % 3])
        for i in range(count)
    ]


def _build_owners(count: int) -> List[Owner]:
    """
    Construct a deterministic list of :class:`Owner` instances.

    :param count: Number of owners to create.
    :return: The freshly created owners.
    """
    return [Owner(name=f"Owner{i}") for i in range(count)]


# %% Workloads


@dataclass
class Workload:
    """
    A single, self-contained unit of EQL work.

    Each workload rebuilds its query and domain on every call so that both
    construction-time and evaluation-time monitoring overhead are measured, and clears the
    :class:`~krrood.symbol_graph.symbol_graph.SymbolGraph` first so repeated runs stay bounded.
    """

    name: str
    """Human-readable identifier shown in reports."""
    run: Callable[[], int]
    """Builds and fully consumes the query, returning the number of results produced."""


def _workload_filter() -> int:
    """Single-source query with a conjunctive numeric/string filter."""
    SymbolGraph().clear()
    gadgets = _build_gadgets(40)
    g = variable(Gadget, domain=gadgets)
    query = an(entity(g).where(and_(g.weight > 2, g.category == "tool")))
    return len(list(query.evaluate()))


def _workload_multi_source() -> int:
    """Two-source (cross-product) query with conditions on both sources."""
    SymbolGraph().clear()
    gadgets = _build_gadgets(20)
    owners = _build_owners(4)
    g = variable(Gadget, domain=gadgets)
    o = variable(Owner, domain=owners)
    query = an(set_of(g, o).where(and_(g.weight > 1, g.weight < 6, o.name != "Owner0")))
    return len(list(query.evaluate()))


def _workload_aggregation() -> int:
    """Aggregation query (count + max) over a filtered domain."""
    SymbolGraph().clear()
    gadgets = _build_gadgets(40)
    g = variable(Gadget, domain=gadgets)
    query = an(set_of(eql.count(g), eql.max(g.weight, default=0)).where(g.weight > 2))
    return len(list(query.evaluate()))


def _workload_inference() -> int:
    """Generative inference query that creates a new instance (exercises ``InferenceRecorder``)."""
    SymbolGraph().clear()
    make_gadget = inference(Gadget)
    query = entity(make_gadget(name="Inferred", weight=5, category="tool"))
    return len(list(query.evaluate()))


def build_workloads() -> List[Workload]:
    """
    :return: The standard set of workloads exercising filtering, multi-source joins,
        aggregation and generative inference.
    """
    return [
        Workload("filter", _workload_filter),
        Workload("multi_source", _workload_multi_source),
        Workload("aggregation", _workload_aggregation),
        Workload("inference", _workload_inference),
    ]


# %% Monitoring levels


class MonitoringLevel(Enum):
    """The three monitoring configurations compared by the A/B stress test."""

    FULL = "full"
    """Both layers active: stack capture on, evaluation observers on (normal behaviour)."""
    MONITORING_OFF = "monitoring_off"
    """Layer 1 disabled via :meth:`MonitoredRegistry.disabled`; evaluation observers still on."""
    BOTH_OFF = "both_off"
    """Layer 1 disabled and an observer-less evaluation context installed, so neither layer runs."""


@contextmanager
def apply_monitoring_level(level: MonitoringLevel) -> Iterator[None]:
    """
    Activate the given monitoring *level* for the duration of the ``with`` block.

    For :attr:`MonitoringLevel.BOTH_OFF` an empty :class:`EvaluationContext` is installed; because
    ``SymbolicExpression._evaluate_`` only builds the default observer context when none is
    present, this suppresses all observer work for nested evaluations.

    :param level: The monitoring configuration to activate.
    """
    if level is MonitoringLevel.FULL:
        yield
        return
    with monitored.disabled():
        if level is MonitoringLevel.MONITORING_OFF:
            yield
            return
        token = set_evaluation_context(EvaluationContext(observers=[]))
        try:
            yield
        finally:
            _evaluation_context_var.reset(token)


# %% A/B wall-clock measurement


@dataclass
class MonitoringLevelResult(ExperimentResult):
    """One row of the A/B comparison table: timing for a single monitoring level."""

    level: str
    """The :class:`MonitoringLevel` value this row reports."""
    milliseconds_per_iteration: MeanAndStandardDeviation
    """Mean +- standard deviation of the wall-clock time for one pass over all workloads."""
    overhead_vs_baseline_ms: float
    """Mean milliseconds added relative to the ``both_off`` baseline (0.0 for the baseline)."""
    overhead_vs_baseline_percent: float
    """Overhead relative to the ``both_off`` baseline as a percentage (0.0 for the baseline)."""


def measure_level(
    level: MonitoringLevel,
    workloads: List[Workload],
    iterations: int,
    warmups: int,
) -> List[float]:
    """
    Time ``iterations`` passes over all *workloads* at the given monitoring *level*.

    :param level: The monitoring configuration to measure.
    :param workloads: The workloads to run, in order, once per iteration.
    :param iterations: Number of measured iterations.
    :param warmups: Number of leading iterations to discard before measuring.
    :return: Per-iteration durations in milliseconds (length ``iterations``).
    """
    durations_ms: List[float] = []
    with apply_monitoring_level(level):
        for index in range(iterations + warmups):
            start = time.perf_counter()
            for workload in workloads:
                workload.run()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if index >= warmups:
                durations_ms.append(elapsed_ms)
    return durations_ms


def run_ab_comparison(
    workloads: List[Workload],
    iterations: int,
    warmups: int,
) -> ExperimentsTable:
    """
    Run the A/B stress test across all three monitoring levels and aggregate the results.

    :param workloads: The workloads to run each iteration.
    :param iterations: Number of measured iterations per level.
    :param warmups: Number of leading iterations to discard per level.
    :return: A table with one :class:`MonitoringLevelResult` row per monitoring level, ordered
        ``both_off``, ``monitoring_off``, ``full``.
    """
    ordered_levels = [
        MonitoringLevel.BOTH_OFF,
        MonitoringLevel.MONITORING_OFF,
        MonitoringLevel.FULL,
    ]
    stats = {
        level: MeanAndStandardDeviation.from_measurements(
            measure_level(level, workloads, iterations, warmups)
        )
        for level in ordered_levels
    }
    baseline_ms = stats[MonitoringLevel.BOTH_OFF].mean

    rows: List[MonitoringLevelResult] = []
    for level in ordered_levels:
        mean_ms = stats[level].mean
        overhead_ms = round(mean_ms - baseline_ms, 2)
        overhead_percent = (
            round(overhead_ms / baseline_ms * 100, 1) if baseline_ms else 0.0
        )
        rows.append(
            MonitoringLevelResult(
                level=level.value,
                milliseconds_per_iteration=stats[level],
                overhead_vs_baseline_ms=overhead_ms,
                overhead_vs_baseline_percent=overhead_percent,
            )
        )
    return ExperimentsTable(rows)


# %% Per-line profiling


def monitoring_target_functions() -> List[Callable]:
    """
    Collect every hot monitoring function to feed to :class:`line_profiler.LineProfiler`.

    Layer 1 contributes the wrapped ``__post_init__`` of each monitored class (where
    :func:`inspect.stack` is called) plus the stack-building helpers. Layer 2 contributes the
    observer callbacks and the dispatch/wrapper that invokes them.

    :return: The functions to profile line by line.
    """
    targets: List[Callable] = [
        StackFrame.from_frame_info,
        StackFrame.from_raw_frame,
        StackFrame._from_frame,
        CallStack.capture,
        CallStack.filter,
        MonitoredRegistry.is_monitored,
        EvaluationTracker.on_evaluate_enter,
        EvaluationTracker.on_result_yielded,
        EvaluationTracker._snapshot_evaluated,
        SatisfiedConditionTracker.on_evaluate_enter,
        SatisfiedConditionTracker.on_result_yielded,
        SatisfiedConditionTracker.on_conclusions_processed,
        InferenceRecorder.on_result_yielded,
        is_condition_participant,
        SymbolicExpression._evaluate_,
    ]
    for monitored_class in monitored.monitored_classes:
        # The wrapped ``new_post_init`` closure is a distinct function object per monitored
        # class; profiling it directly is what attributes the ``inspect.stack()`` cost.
        post_init = monitored_class.__dict__.get("__post_init__")
        if post_init is not None:
            targets.append(post_init)
    return targets


def profile_lines(workloads: List[Workload], repetitions: int) -> str:
    """
    Run a line-by-line profile of the monitoring functions under :attr:`MonitoringLevel.FULL`.

    :param workloads: The workloads to drive the profiler.
    :param repetitions: Number of full passes over the workloads to profile.
    :return: The formatted ``line_profiler`` statistics as text.
    """
    profiler = LineProfiler()
    with warnings.catch_warnings():
        # The monitored ``__post_init__`` carries ``functools.wraps`` metadata; we intentionally
        # profile the wrapper (not ``__wrapped__``) because the stack capture lives there.
        warnings.simplefilter("ignore", UserWarning)
        for function in monitoring_target_functions():
            profiler.add_function(function)

    def driver() -> None:
        for _ in range(repetitions):
            for workload in workloads:
                workload.run()

    with apply_monitoring_level(MonitoringLevel.FULL):
        profiler.runcall(driver)

    buffer = io.StringIO()
    profiler.print_stats(stream=buffer)
    return buffer.getvalue()


# %% Reporting


def _markdown_ab_table(table: ExperimentsTable) -> str:
    """
    Render the A/B comparison as a GitHub-flavoured markdown table.

    :param table: The table produced by :func:`run_ab_comparison`.
    :return: The markdown table as a string.
    """
    headers = MonitoringLevelResult.get_column_names()
    lines = [
        "| " + " | ".join(h.replace("_", " ") for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table.experiments:
        lines.append(
            "| " + " | ".join(str(value) for value in row.get_column_values()) + " |"
        )
    return "\n".join(lines)


def write_report(
    table: ExperimentsTable,
    line_stats: str,
    iterations: int,
    warmups: int,
    line_repetitions: int,
    output_dir: Path,
) -> Path:
    """
    Write the markdown report and CSV of A/B timings.

    :param table: The A/B comparison table.
    :param line_stats: The per-line profiler output.
    :param iterations: Measured iterations per level (recorded in the report header).
    :param warmups: Discarded warm-up iterations per level.
    :param line_repetitions: Number of passes used for the line-profiled run.
    :param output_dir: Directory to write ``monitoring_profile_report.md`` and ``.csv`` into.
    :return: Path to the markdown report.
    """
    report_path = output_dir / "monitoring_profile_report.md"
    csv_path = output_dir / "monitoring_profile_timings.csv"

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "level",
                "mean_ms",
                "std_ms",
                "overhead_vs_baseline_ms",
                "overhead_vs_baseline_percent",
            ]
        )
        for row in table.experiments:
            writer.writerow(
                [
                    row.level,
                    row.milliseconds_per_iteration.mean,
                    row.milliseconds_per_iteration.standard_deviation,
                    row.overhead_vs_baseline_ms,
                    row.overhead_vs_baseline_percent,
                ]
            )

    report = (
        "# EQL Monitoring Overhead Profile\n\n"
        f"- Platform: `{platform.platform()}`\n"
        f"- Python: `{platform.python_version()}`\n"
        f"- Measured iterations per level: {iterations} (after {warmups} warm-ups)\n"
        f"- Line-profiled passes: {line_repetitions}\n\n"
        "## A/B wall-clock comparison (authoritative overhead)\n\n"
        "Time for one pass over all workloads. `both_off` is the fully-disabled baseline; "
        "overhead columns are relative to it.\n\n"
        f"{_markdown_ab_table(table)}\n\n"
        "## Per-line profile (relative attribution only)\n\n"
        "Measured under `full`. Absolute times here are inflated by `line_profiler`'s own "
        "overhead; use them to compare lines, not to size total cost.\n\n"
        "```\n"
        f"{line_stats}"
        "```\n"
    )
    report_path.write_text(report)
    return report_path


# %% Entry point


def main() -> None:
    """Run the A/B stress test and per-line profile, print a summary and write the report."""
    iterations = 30
    warmups = 5
    line_repetitions = 20

    workloads = build_workloads()

    table = run_ab_comparison(workloads, iterations=iterations, warmups=warmups)
    line_stats = profile_lines(workloads, repetitions=line_repetitions)

    output_dir = Path(__file__).parent
    report_path = write_report(
        table,
        line_stats,
        iterations=iterations,
        warmups=warmups,
        line_repetitions=line_repetitions,
        output_dir=output_dir,
    )

    print("EQL monitoring overhead - milliseconds per pass over all workloads:\n")
    print(_markdown_ab_table(table))
    print("\nTypst table:\n")
    print(TypstRenderer(table).render_table())
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
