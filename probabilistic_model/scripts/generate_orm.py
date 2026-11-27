"""
Generate the ORM interface of the probabilistic_model package.
"""

from __future__ import annotations

import logging
from pathlib import Path

import random_events.variable

import probabilistic_model.probabilistic_circuit.tensorized
from krrood.ormatic.ormatic import ORMatic
from probabilistic_model.orm.custom_types import BackendArrayType
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)

ormatic = ORMatic.from_package(
    packages=[
        probabilistic_model.probabilistic_circuit.tensorized,
        random_events.variable,
    ],
    ormatic_interface_dependencies=[],
    ignored_classes=set(),
    type_mappings={BackendArray: BackendArrayType},
)
logging.getLogger("krrood").setLevel(logging.DEBUG)

ormatic.make_all_tables()

ormatic_interface_path = (
    Path(__file__).parents[1]
    / "src"
    / "probabilistic_model"
    / "orm"
    / "ormatic_interface.py"
)
with open(ormatic_interface_path, "w") as f:
    ormatic.to_sqlalchemy_file(file=f)
