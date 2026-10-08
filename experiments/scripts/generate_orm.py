import logging
from pathlib import Path

import experiments
import experiments.control_loop_experiments.benchmark
import experiments.control_loop_experiments.scenarios
import coraplex.orm.ormatic_interface

from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module
import experiments.control_loop_experiments.control_loop_profiler
import experiments.causal_reasoning.do_query
import experiments.causal_reasoning.mutagenesis.do_query
from experiments.causal_reasoning.tracy_clutter_picking import (
    do_query as tracy_do_query,
    exceptions as tracy_exceptions,
    synthetic as tracy_synthetic,
)

# benchmarking measures a running system instead of describing it
ignored_classes = set(classes_of_module(experiments.control_loop_experiments.scenarios))
ignored_classes |= set(
    classes_of_module(experiments.control_loop_experiments.benchmark)
)
ignored_classes |= set(
    classes_of_module(experiments.control_loop_experiments.control_loop_profiler)
)

# a question put to a circuit, the answer read back and the closed-form stand-in for the
# simulator are not stored, so mapping them would add a table per question and one for
# every answer, with nothing ever writing to them
for measuring_module in (
    experiments.causal_reasoning.do_query,
    experiments.causal_reasoning.mutagenesis.do_query,
    tracy_do_query,
    tracy_exceptions,
    tracy_synthetic,
):
    ignored_classes |= set(classes_of_module(measuring_module))

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
