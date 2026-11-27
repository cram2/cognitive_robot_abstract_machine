"""
Tests that ORMatic maps fields declared as a ``frozenset`` and loads them back as one.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import pytest
from sqlalchemy import select
from sqlalchemy.orm import configure_mappers, sessionmaker

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.helper import get_classes_of_ormatic_interface
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import create_engine
from ..dataset.frozenset_collection_classes import (
    FrozensetCollectionMember,
    FrozensetCollectionOwner,
)

# %% generated interface


@pytest.fixture(scope="module")
def module(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """
    The ORMatic interface of the frozenset classes, generated and imported once, so the
    tests do not declare the same classes into the shared registry twice.
    """
    class_diagram = ClassDiagram(
        classes=[FrozensetCollectionOwner, FrozensetCollectionMember]
    )
    instance = ORMatic(class_dependency_graph=class_diagram)
    instance.make_all_tables()

    interface_file = (
        tmp_path_factory.mktemp("frozenset_collection")
        / "frozenset_collection_interface.py"
    )
    with open(interface_file, "w") as file:
        instance.to_sqlalchemy_file(file=file)

    spec = importlib.util.spec_from_file_location(
        "frozenset_collection_interface", interface_file
    )
    generated_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generated_module
    spec.loader.exec_module(generated_module)
    get_classes_of_ormatic_interface(generated_module)
    configure_mappers()
    return generated_module


# %% round trip


def test_frozenset_fields_are_loaded_as_frozensets_of_the_stored_elements(
    module: ModuleType,
):
    """
    A frozenset of mapped members and a frozenset of strings come back as frozensets of
    equal elements.
    """
    engine = create_engine("sqlite:///:memory:")
    module.Base.metadata.create_all(engine)
    session = sessionmaker(engine)()
    original = FrozensetCollectionOwner(
        members=frozenset(
            FrozensetCollectionMember(label=label) for label in ["a", "b"]
        ),
        tags=frozenset({"red", "round"}),
    )

    dao = to_dao(original)
    session.add(dao)
    session.commit()
    reconstructed = session.scalars(select(type(dao))).one().from_dao()

    assert type(reconstructed.members) is frozenset
    assert reconstructed.members == original.members
    assert type(reconstructed.tags) is frozenset
    assert reconstructed.tags == original.tags
