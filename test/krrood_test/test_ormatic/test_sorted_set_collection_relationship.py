"""
Tests that ORMatic maps a many-to-many relationship declared as a
:class:`~sortedcontainers.SortedSet` and loads it back as one.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import pytest
from sortedcontainers import SortedSet
from sqlalchemy import select
from sqlalchemy.orm import configure_mappers, sessionmaker

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.helper import get_classes_of_ormatic_interface
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import create_engine
from ..dataset.sorted_set_collection_classes import (
    SortedSetCollectionMember,
    SortedSetCollectionOwner,
)

# %% generated interface


@pytest.fixture(scope="module")
def module(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """
    The ORMatic interface of the sorted set classes, generated and imported once, so the
    tests do not declare the same classes into the shared registry twice.
    """
    class_diagram = ClassDiagram(
        classes=[SortedSetCollectionOwner, SortedSetCollectionMember]
    )
    instance = ORMatic(class_dependency_graph=class_diagram)
    instance.make_all_tables()

    interface_file = (
        tmp_path_factory.mktemp("sorted_set_collection")
        / "sorted_set_collection_interface.py"
    )
    with open(interface_file, "w") as file:
        instance.to_sqlalchemy_file(file=file)

    spec = importlib.util.spec_from_file_location(
        "sorted_set_collection_interface", interface_file
    )
    generated_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generated_module
    spec.loader.exec_module(generated_module)
    get_classes_of_ormatic_interface(generated_module)
    configure_mappers()
    return generated_module


# %% round trip


def test_a_sorted_set_relationship_is_loaded_as_a_sorted_set_in_order(
    module: ModuleType,
):
    """
    A sorted set of members that can only be compared once they are initialised comes
    back as a sorted set of equal members in the same order.
    """
    engine = create_engine("sqlite:///:memory:")
    module.Base.metadata.create_all(engine)
    session = sessionmaker(engine)()
    original = SortedSetCollectionOwner(
        members=SortedSet(
            SortedSetCollectionMember(label=label) for label in ["c", "a", "b"]
        )
    )

    dao = to_dao(original)
    session.add(dao)
    session.commit()
    reconstructed = session.scalars(select(type(dao))).one().from_dao()

    assert isinstance(reconstructed.members, SortedSet)
    assert list(reconstructed.members) == list(original.members)
