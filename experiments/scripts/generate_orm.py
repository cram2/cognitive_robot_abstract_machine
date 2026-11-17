"""
Generate the ORM interface of the experiments, on top of that of coraplex.
"""

import logging
from pathlib import Path

import experiments
import experiments.circuit_learning.datasets
import experiments.circuit_learning.measurements
import experiments.circuit_learning.methods
import experiments.circuit_learning.settings
import experiments.control_loop_experiments.benchmark
import experiments.control_loop_experiments.scenarios
import coraplex.orm.ormatic_interface

from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module
import experiments.control_loop_experiments.control_loop_profiler

# benchmarking measures a running system instead of describing it
ignored_classes = set(classes_of_module(experiments.control_loop_experiments.scenarios))
ignored_classes |= set(
    classes_of_module(experiments.control_loop_experiments.benchmark)
)
ignored_classes |= set(
    classes_of_module(experiments.control_loop_experiments.control_loop_profiler)
)
# of the circuit learning benchmark, only its records and measured rows are stored; its
# database and runner import this interface, so no generation can import or map them
for circuit_learning_module in (
    experiments.circuit_learning.datasets,
    experiments.circuit_learning.measurements,
    experiments.circuit_learning.methods,
):
    ignored_classes |= set(classes_of_module(circuit_learning_module))
ignored_classes.add(experiments.circuit_learning.settings.BudgetedMemory)

# Create an ORMatic object with the classes to be mapped
ormatic = ORMatic.from_package(
    packages=[experiments],
    ormatic_interface_dependencies=[coraplex.orm.ormatic_interface],
    ignored_classes=ignored_classes,
    type_mappings={},
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
    ormatic.to_sqlalchemy_file(file=f)
