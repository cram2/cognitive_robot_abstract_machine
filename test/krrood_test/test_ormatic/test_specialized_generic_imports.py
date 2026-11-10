"""
A generated interface imports every module its specialized generic DAOs name, including
the modules of their type arguments.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

from random_events.interval import SimpleInterval

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.ormatic import ORMatic
from ..dataset.generic_over_argument_in_another_package import (
    HolderOfSpecializedGeneric,
    LabelledIntervals,
)

# %% generating and importing the interface


def generate_interface(tmp_path) -> ModuleType:
    """
    Generate and import the interface of the classes over an interval from another
    package.

    :param tmp_path: The directory to write the interface into.
    :return: The imported interface.
    """
    instance = ORMatic(ClassDiagram([LabelledIntervals, HolderOfSpecializedGeneric]))
    instance.make_all_tables()

    interface_file = tmp_path / "specialized_generic_interface.py"
    with open(interface_file, "w") as file:
        instance.to_sqlalchemy_file(file)

    spec = importlib.util.spec_from_file_location(
        "specialized_generic_interface", interface_file
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# %% type arguments from another package


def test_specialized_generic_with_argument_from_another_package_imports(tmp_path):
    module = generate_interface(tmp_path)

    assert (
        module.LabelledIntervals_SimpleIntervalDAO.original_class()
        == LabelledIntervals[SimpleInterval]
    )
