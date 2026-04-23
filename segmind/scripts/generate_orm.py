# ----------------------------------------------------------------------------------------------------------------------
# This script generates the ORM classes for the semantic_digital_twin package.
# Dataclasses can be mapped automatically to the ORM model
# using the ORMatic library, they just have to be registered in the classes list.
# Classes that are self_mapped and explicitly_mapped are already mapped in the model.py file. Look there for more
# information on how to map them.
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
import logging
import os
from dataclasses import is_dataclass

import segmind.orm.ormatic_interface
from krrood.class_diagrams import ClassDiagram
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module, classes_of_package
from krrood.utils import recursive_subclasses

all_classes = set(classes_of_package(segmind))
all_classes -= set(classes_of_module(segmind.orm.ormatic_interface))


# keep only dataclasses that are NOT AlternativeMapping subclasses
all_classes = {
    c for c in all_classes if is_dataclass(c) and not issubclass(c, AlternativeMapping)
}

alternative_mappings = [
    am
    for am in recursive_subclasses(AlternativeMapping)
    if am.original_class() in all_classes and not am.__module__.startswith("test.")
]


def generate_orm():
    """
    Generate the ORM classes for the segmind package.
    """

    logging.basicConfig(level=logging.INFO)  # Or your preferred config
    logging.getLogger("krrood").setLevel(logging.DEBUG)

    class_diagram = ClassDiagram(
        list(sorted(all_classes, key=lambda c: c.__name__, reverse=True))
    )

    instance = ORMatic(
        class_dependency_graph=class_diagram,
        alternative_mappings=alternative_mappings,
    )
    instance.make_all_tables()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(
        os.path.join(script_dir, "..", "src", "segmind", "orm")
    )
    with open(os.path.join(path, "ormatic_interface.py"), "w") as f:
        instance.to_sqlalchemy_file(f)


if __name__ == "__main__":
    generate_orm()
