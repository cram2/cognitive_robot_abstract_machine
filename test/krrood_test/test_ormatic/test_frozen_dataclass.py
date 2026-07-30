"""
ORMatic-specific check that ``frozen=True`` dataclasses are supported.

ORMatic reconstructs a domain object with ``cls.__new__(cls)`` followed by
``object.__setattr__`` (see ``from_dao``/``dao``), which bypasses a frozen dataclass's
blocked ``__setattr__``.  This test builds an ORMatic interface for a small frozen
object graph (flat, nested relationship, list field, and a set collection) and round-
trips an instance through the database to prove the whole to_dao → persist → from_dao
path works on frozen classes.

If this passes, the verbalization layer's frozen value objects (plans, PathStep,
SourceReference, the lexicon words, …) are safe to keep frozen.
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
from ..dataset.frozen_classes import FrozenInner, FrozenOuter


def _generate_interface(tmp_path):
    """
    Generate and import an ORMatic SQLAlchemy interface for the frozen classes.
    """
    class_diagram = ClassDiagram([FrozenOuter, FrozenInner])
    instance = ORMatic(class_diagram)
    instance.make_all_tables()

    interface_file = tmp_path / "frozen_interface.py"
    with open(interface_file, "w") as f:
        instance.to_sqlalchemy_file(f)

    spec = importlib.util.spec_from_file_location("frozen_interface", interface_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ormatic_round_trips_a_frozen_dataclass(tmp_path):
    module = _generate_interface(tmp_path)
    # Register the generated DAO classes (so to_dao/from_dao can find them).
    get_classes_of_ormatic_interface(module)
    configure_mappers()

    engine = create_engine("sqlite:///:memory:")
    module.Base.metadata.create_all(engine)
    session = sessionmaker(engine)()

    original = FrozenOuter(
        name="outer",
        inner=FrozenInner(label="a", weight=1),
        values=[1, 2, 3],
        members={FrozenInner(label="b", weight=2)},
    )

    dao = to_dao(original)
    session.add(dao)
    session.commit()

    reconstructed = session.scalars(select(type(dao))).one().from_dao()

    # Full round-trip equality — proves frozen construction (object.__setattr__) worked,
    # including the nested frozen relationship, the list field, and the set collection.
    assert reconstructed == original
    assert isinstance(reconstructed, FrozenOuter)
    assert reconstructed.inner == FrozenInner(label="a", weight=1)
    assert reconstructed.values == [1, 2, 3]
    assert reconstructed.members == {FrozenInner(label="b", weight=2)}
