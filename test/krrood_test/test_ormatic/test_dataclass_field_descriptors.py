"""
Persistence of dataclass fields whose value a
:class:`~krrood.patterns.dataclass_field_descriptor.DataclassFieldDescriptor` keeps.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from krrood.ormatic.data_access_objects.helper import to_dao

from ..dataset.dataclass_field_descriptors import (
    HolderOfValueKeptInStorage,
    IntegerHolderOfValueKeptInStorage,
    SubclassOfHolder,
)


# %% round trip


def reload(holder: object, session) -> object:
    """
    :return: The holder after a round trip through the database.
    """
    data_access_object = to_dao(holder)
    session.add(data_access_object)
    session.commit()
    session.expunge_all()
    query = select(type(data_access_object))
    return session.scalars(query).one().from_dao()


@pytest.mark.parametrize(
    "holder",
    [
        HolderOfValueKeptInStorage(value=3),
        SubclassOfHolder(value=3, other_value=4),
        IntegerHolderOfValueKeptInStorage(value=3),
    ],
    ids=lambda holder: type(holder).__name__,
)
def test_value_survives_round_trip(holder: object, session, database):
    reloaded = reload(holder, session)
    assert reloaded.value == holder.value


@pytest.mark.parametrize(
    "holder",
    [
        HolderOfValueKeptInStorage(value=3),
        SubclassOfHolder(value=3, other_value=4),
        IntegerHolderOfValueKeptInStorage(value=3),
    ],
    ids=lambda holder: type(holder).__name__,
)
def test_reloaded_value_is_kept_in_storage(holder: object, session, database):
    reloaded = reload(holder, session)
    assert reloaded.storage.value == holder.value
    assert "value" not in vars(reloaded)


def test_value_of_narrowed_field_is_a_column():
    data_access_object_type = type(to_dao(IntegerHolderOfValueKeptInStorage(value=3)))
    assert "value" in data_access_object_type.__table__.columns
