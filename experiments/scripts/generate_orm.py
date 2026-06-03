import logging
import os
from dataclasses import is_dataclass

import numpy as np

import experiments
import pycram.orm.ormatic_interface
from krrood.class_diagrams import ClassDiagram
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.helper import get_classes_of_ormatic_interface
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.type_dict import TypeDict
from krrood.ormatic.utils import classes_of_package
from krrood.utils import recursive_subclasses
from pycram.orm.model import NumpyType

# ----------------------------------------------------------------------------------------------------------------------
# This script generates the ORM classes for the pycram package
# Classes that are self_mapped and explicitly_mapped are already mapped in the model.py file. Look there for more
# information on how to map them.
# ----------------------------------------------------------------------------------------------------------------------


# import classes from the existing interface
classes, alternative_mappings, type_mappings = get_classes_of_ormatic_interface(
    pycram.orm.ormatic_interface
)
classes = set(classes)

classes |= set(classes_of_package(experiments))


alternative_mappings += [am for am in recursive_subclasses(AlternativeMapping)]
alternative_mappings = list(set(alternative_mappings))
# keep only dataclasses that are NOT AlternativeMapping subclasses
classes = {
    c for c in classes if is_dataclass(c) and not issubclass(c, AlternativeMapping)
}
classes |= {am.original_class() for am in recursive_subclasses(AlternativeMapping)}

alternative_mappings = [
    am
    for am in recursive_subclasses(AlternativeMapping)
    if am.original_class() in classes
]

# create the new ormatic interface
class_diagram = ClassDiagram(
    list(sorted(classes, key=lambda c: c.__name__, reverse=True))
)

type_mappings.update({np.ndarray: NumpyType})


# Create an ORMatic object with the classes to be mapped
ormatic = ORMatic(
    class_diagram,
    type_mappings=TypeDict(type_mappings),
    alternative_mappings=alternative_mappings,
)
logging.getLogger("krrood").setLevel(logging.DEBUG)

# Generate the ORM classes
ormatic.make_all_tables()

path = os.path.abspath(os.path.join(os.getcwd(), "../src/experiments/"))
with open(os.path.join(path, "ormatic_interface.py"), "w") as f:
    ormatic.to_sqlalchemy_file(f)
