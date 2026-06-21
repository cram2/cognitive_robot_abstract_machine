import pytest
from sqlalchemy import select, inspect

from krrood.ormatic.data_access_objects.helper import to_dao
from ..dataset.example_classes import TypeVarFieldHolder, KRROODPosition
from ..dataset.ormatic_interface import TypeVarFieldHolderDAO, KRROODPositionDAO


def test_typevar_field_dao_generation():
    """TypeVar-typed fields should generate FK columns and relationships."""
    mapper = inspect(TypeVarFieldHolderDAO)
    assert hasattr(TypeVarFieldHolderDAO, "typed_field_id"), (
        "TypeVarFieldHolderDAO should have a typed_field_id column"
    )
    assert "typed_field" in mapper.relationships, (
        "TypeVarFieldHolderDAO should have a typed_field relationship"
    )


def test_typevar_field_to_dao(session, database):
    """to_dao should persist TypeVar-typed fields correctly."""
    pos = KRROODPosition(1.0, 2.0, 3.0)
    holder = TypeVarFieldHolder(typed_field=pos, name="test")

    holder_dao: TypeVarFieldHolderDAO = to_dao(holder)
    assert holder_dao.typed_field.x == 1.0
    assert holder_dao.typed_field.y == 2.0
    assert holder_dao.typed_field.z == 3.0
    assert holder_dao.name == "test"

    session.add(holder_dao)
    session.commit()

    queried = session.scalars(select(TypeVarFieldHolderDAO)).one()
    assert queried.typed_field.x == 1.0
    assert queried.name == "test"


def test_typevar_field_from_dao(session, database):
    """from_dao should reconstruct TypeVar-typed fields correctly."""
    pos = KRROODPosition(4.0, 5.0, 6.0)
    holder = TypeVarFieldHolder(typed_field=pos, name="roundtrip")

    holder_dao = to_dao(holder)
    session.add(holder_dao)
    session.commit()

    queried = session.scalars(select(TypeVarFieldHolderDAO)).one()
    reconstructed = queried.from_dao()
    assert reconstructed.typed_field == pos
    assert reconstructed.name == "roundtrip"
