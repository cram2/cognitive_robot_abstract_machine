import logging
from pathlib import Path

import coraplex.locations.costmaps
import coraplex.orm.model
import giskardpy.orm.ormatic_interface
from krrood.adapters.json_serializer import SubclassJSONSerializer

from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module

# ----------------------------------------------------------------------------------------------------------------------
# This script generates the ORM classes for the coraplex package
# Classes that are self_mapped and explicitly_mapped are already mapped in the model.py file. Look there for more
# information on how to map them.
# ----------------------------------------------------------------------------------------------------------------------


ignored_classes = set(classes_of_module(coraplex.locations.costmaps))
# profiling and benchmarking measure a running system instead of describing it
ignored_classes |= {SubclassJSONSerializer}

dependencies = [giskardpy.orm.ormatic_interface]

# numpy and trimesh arrive through the dependency's type mappings
type_mappings = {}


# Create an ORMatic object with the classes to be mapped
ormatic = ORMatic.from_package([coraplex], dependencies, ignored_classes, type_mappings)
logging.getLogger("krrood").setLevel(logging.DEBUG)


# Generate the ORM classes
ormatic.make_all_tables()

ormatic_interface_path = (
    Path(__file__).parents[1] / "src" / "coraplex" / "orm" / "ormatic_interface.py"
)
with open(ormatic_interface_path, "w") as f:
    ormatic.to_sqlalchemy_file(f)
