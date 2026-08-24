"""
ORMatic-specific check that a many-to-many relationship declared as ``tuple[X, ...]``
(rather than ``list[X]`` or ``Set[X]``) can be mapped and round-tripped.

SQLAlchemy requires a collection class it can instrument in place (an appender method),
which a ``tuple`` cannot provide, so the generated relationship must use an
instrumentable collection at the ORM layer while still handing the domain object back
its declared ``tuple``.
"""

from __future__ import annotations

import importlib.util
import sys

from sqlalchemy import select
from sqlalchemy.orm import configure_mappers, sessionmaker

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.helper import get_classes_of_ormatic_interface
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import create_engine
from ..dataset.tuple_collection_classes import (
    TupleCollectionMember,
    TupleCollectionOwner,
)


def _generate_interface(tmp_path):
    """
    Generate and import an ORMatic SQLAlchemy interface for the tuple-collection
    classes.
    """
    class_diagram = ClassDiagram([TupleCollectionOwner, TupleCollectionMember])
    instance = ORMatic(class_diagram)
    instance.make_all_tables()

    interface_file = tmp_path / "tuple_collection_interface.py"
    with open(interface_file, "w") as f:
        instance.to_sqlalchemy_file(f)

    spec = importlib.util.spec_from_file_location(
        "tuple_collection_interface", interface_file
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ormatic_configures_a_tuple_valued_many_to_many_relationship(tmp_path):
    """
    Configuring the mappers of a ``tuple[X, ...]``-valued relationship must not raise,
    since SQLAlchemy cannot instrument ``tuple`` itself as a collection class.
    """
    module = _generate_interface(tmp_path)
    get_classes_of_ormatic_interface(module)

    configure_mappers()


def test_ormatic_round_trips_a_tuple_valued_many_to_many_relationship(tmp_path):
    module = _generate_interface(tmp_path)
    get_classes_of_ormatic_interface(module)
    configure_mappers()

    engine = create_engine("sqlite:///:memory:")
    module.Base.metadata.create_all(engine)
    session = sessionmaker(engine)()

    original = TupleCollectionOwner(
        members=(
            TupleCollectionMember(label="a"),
            TupleCollectionMember(label="b"),
        )
    )

    dao = to_dao(original)
    session.add(dao)
    session.commit()

    reconstructed = session.scalars(select(type(dao))).one().from_dao()

    assert isinstance(reconstructed.members, tuple)
    assert reconstructed == original
