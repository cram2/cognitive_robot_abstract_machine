"""
Generate the ORM interface of the cramph package.

.. warning:: Do not import giskardpy or coraplex here. Alternative mappings are collected
    through a global subclass scan, so importing a downstream package leaks its mappings
    into this interface.
"""

import logging
from pathlib import Path

import numpy as np

import cramph
import semantic_digital_twin.orm.ormatic_interface
from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.ormatic.custom_types import NumpyType
from krrood.ormatic.ormatic import ORMatic

ignored_classes = {SubclassJSONSerializer}

dependencies = [semantic_digital_twin.orm.ormatic_interface]

type_mappings = {np.ndarray: NumpyType}


ormatic = ORMatic.from_package(
    [cramph],
    dependencies,
    ignored_classes,
    type_mappings,
)
logging.getLogger("krrood").setLevel(logging.DEBUG)


ormatic.make_all_tables()

ormatic_interface_path = (
    Path(__file__).parents[1] / "src" / "cramph" / "orm" / "ormatic_interface.py"
)
with open(ormatic_interface_path, "w") as f:
    ormatic.to_sqlalchemy_file(f)
