import logging
from pathlib import Path

import experiments
import experiments.control_loop_experiments.benchmark
import experiments.control_loop_experiments.scenarios
import coraplex.orm.ormatic_interface

from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module
import experiments.control_loop_experiments.control_loop_profiler
from experiments.causal_reasoning.mutagenesis import (
    evaluation as mutagenesis_evaluation,
    exceptions as mutagenesis_exceptions,
    flat_table as mutagenesis_flat_table,
    pipelines as mutagenesis_pipelines,
    queries as mutagenesis_queries,
    report as mutagenesis_report,
    run_pipeline as mutagenesis_run_pipeline,
)
from experiments.causal_reasoning.tracy_clutter_picking import (
    evaluation as tracy_evaluation,
    exceptions as tracy_exceptions,
    flat_table as tracy_flat_table,
    pipelines as tracy_pipelines,
    queries as tracy_queries,
    report as tracy_report,
    run_pipeline as tracy_run_pipeline,
)

# benchmarking measures a running system instead of describing it
ignored_classes = set(classes_of_module(experiments.control_loop_experiments.scenarios))
ignored_classes |= set(
    classes_of_module(experiments.control_loop_experiments.benchmark)
)
ignored_classes |= set(
    classes_of_module(experiments.control_loop_experiments.control_loop_profiler)
)

# the causal-query comparisons fit and measure models over their domains instead of
# describing the domains; both experiments name their pipelines, outcomes and
# reports alike, which one interface cannot map twice
for comparison_module in (
    mutagenesis_evaluation,
    mutagenesis_exceptions,
    mutagenesis_flat_table,
    mutagenesis_pipelines,
    mutagenesis_queries,
    mutagenesis_report,
    mutagenesis_run_pipeline,
    tracy_evaluation,
    tracy_exceptions,
    tracy_flat_table,
    tracy_pipelines,
    tracy_queries,
    tracy_report,
    tracy_run_pipeline,
):
    ignored_classes |= set(classes_of_module(comparison_module))

# Create an ORMatic object with the classes to be mapped
ormatic = ORMatic.from_package(
    [experiments], [coraplex.orm.ormatic_interface], ignored_classes, type_mappings={}
)
logging.getLogger("krrood").setLevel(logging.DEBUG)

# Generate the ORM classes
ormatic.make_all_tables()

ormatic_interface_path = (
    Path(__file__).parent.parent
    / "src"
    / "experiments"
    / "orm"
    / "ormatic_interface.py"
)
with open(ormatic_interface_path, "w") as f:
    ormatic.to_sqlalchemy_file(f)
