"""
Generate the ORM interface of probabilistic_model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import probabilistic_model
from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.ormatic.ormatic import ORMatic
from probabilistic_model.learning.parameters.gradient_descent.jax_gradient_descent import (
    TrainingState,
)
from probabilistic_model.orm.custom_types import BackendArrayType
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)

# the state of a running optimizer, not something a model stores
ignored_classes = {SubclassJSONSerializer, TrainingState}

dependencies = []

type_mappings = {BackendArray: BackendArrayType}


ormatic = ORMatic.from_package(
    packages=[probabilistic_model],
    ormatic_interface_dependencies=dependencies,
    ignored_classes=ignored_classes,
    type_mappings=type_mappings,
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
