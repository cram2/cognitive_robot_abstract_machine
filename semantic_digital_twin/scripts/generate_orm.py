# ----------------------------------------------------------------------------------------------------------------------
# This script generates the ORM classes for the semantic_digital_twin package.
# Dataclasses can be mapped automatically to the ORM model
# using the ORMatic library, they just have to be registered in the classes list.
# Classes that are self_mapped and explicitly_mapped are already mapped in the model.py file. Look there for more
# information on how to map them.
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import trimesh

import semantic_digital_twin
import semantic_digital_twin.orm.model

import semantic_digital_twin.adapters.procthor.procthor_resolver
from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.ormatic.ormatic import ORMatic
from semantic_digital_twin.reasoning.predicates import ContainsType
from semantic_digital_twin.semantic_annotations.position_descriptions import (
    SemanticDirection,
)
from semantic_digital_twin.spatial_computations.forward_kinematics import (
    ForwardKinematicsManager,
)
from semantic_digital_twin.world import (
    ResetStateContextManager,
    WorldModelUpdateContextManager,
)

# remove classes that should not be mapped
ignore_classes = {
    ResetStateContextManager,
    WorldModelUpdateContextManager,
    ForwardKinematicsManager,
    semantic_digital_twin.adapters.procthor.procthor_resolver.ProcthorResolver,
    ContainsType,
    SemanticDirection,
    SubclassJSONSerializer,
}


def generate_orm():
    """
    Generate the ORM classes for the coraplex package.
    """
    logging.basicConfig(level=logging.INFO)  # Or your preferred config
    logging.getLogger("krrood").setLevel(logging.DEBUG)

    ormatic = ORMatic.from_package(
        [semantic_digital_twin],
        ormatic_interface_dependencies=[],
        ignored_classes=ignore_classes,
        type_mappings={
            trimesh.Trimesh: semantic_digital_twin.orm.model.TrimeshType,
        },
    )
    ormatic.make_all_tables()
    ormatic_interface_path = (
        Path(__file__).parent.parent
        / "src"
        / "semantic_digital_twin"
        / "orm"
        / "ormatic_interface.py"
    )

    with open(ormatic_interface_path, "w") as f:
        ormatic.to_sqlalchemy_file(f)


if __name__ == "__main__":
    generate_orm()
