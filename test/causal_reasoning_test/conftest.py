"""
Fixtures for :mod:`test.causal_reasoning_test`.

Builds this test package's own, self-contained ``ormatic_interface.py`` -- mapping
only the dataclasses of :mod:`experiments.causal_reasoning.mutagenesis.domain` and
:mod:`experiments.causal_reasoning.tracy_clutter_picking.domain` -- instead of hooking into the
workspace-wide ORM build ``test/experiments_test/conftest.py`` uses. That build chains through ``semantic_digital_twin``, ``giskardpy``, ``segmind``
and ``coraplex`` before reaching ``experiments``, none of which this package's
dataclasses need.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import is_dataclass

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.helper import OrmaticInterfaceInformation
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module

from experiments.causal_reasoning.mutagenesis import domain as mutagenesis_domain
from experiments.causal_reasoning.tracy_clutter_picking import (
    domain as tracy_clutter_picking_domain,
)


def generate_sqlalchemy_interface() -> None:
    """
    Generate this package's ``ormatic_interface.py`` before tests run.

    Written atomically -- to a temporary file in the same directory, then moved into
    place -- the same way :mod:`test.krrood_test.conftest` does, so a run under
    pytest-xdist never has one worker read a half-written file another is still
    writing.
    """
    all_classes = {
        clazz
        for module in (mutagenesis_domain, tracy_clutter_picking_domain)
        for clazz in classes_of_module(module)
        if is_dataclass(clazz)
    }
    class_diagram = ClassDiagram(
        list(sorted(all_classes, key=lambda clazz: clazz.__name__, reverse=True))
    )
    instance = ORMatic(
        class_dependency_graph=class_diagram,
        interface_information=OrmaticInterfaceInformation(),
    )
    instance.make_all_tables()

    file_path = os.path.join(os.path.dirname(__file__), "ormatic_interface.py")
    directory = os.path.dirname(file_path)
    with tempfile.NamedTemporaryFile(
        "w", dir=directory, prefix="ormatic_interface.", suffix=".py.tmp", delete=False
    ) as temporary_file:
        instance.to_sqlalchemy_file(temporary_file)
        temporary_path = temporary_file.name
    os.replace(temporary_path, file_path)


# Generate ormatic_interface.py at module level, before the star import below. This
# must happen here (not in a pytest hook) because hooks run after the conftest module
# is fully imported, which means the import on the next line would fail if the
# generated file is stale or missing.
generate_sqlalchemy_interface()

try:
    from .ormatic_interface import *  # type: ignore  # noqa: F401,F403
except ImportError:
    pass
