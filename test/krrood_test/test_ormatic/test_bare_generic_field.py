from sqlalchemy import select, inspect

from krrood.ormatic.data_access_objects.helper import to_dao
from ..dataset.classes_with_generic import (
    BareGenericFieldHolder,
    SubClassGenericThatUpdatesGenericTypeToBuiltInType,
)
from ..dataset.ormatic_interface import (
    BareGenericFieldHolderDAO,
    SubClassGenericThatUpdatesGenericTypeToBuiltInTypeDAO, FirstGenericDAO,
)


def test_bare_generic_field_dao_generation():
    """
    A field typed to a bare, unparametrized generic class must still generate a
    relationship, since the generic class is itself mapped through its concrete
    parametrizations.
    """
    mapper = inspect(BareGenericFieldHolderDAO)
    assert "item" in mapper.relationships
    assert issubclass(SubClassGenericThatUpdatesGenericTypeToBuiltInTypeDAO, FirstGenericDAO)


def test_bare_generic_field_round_trips_to_concrete_subclass(session, database):
    """
    A concrete parametrization stored through the bare generic field must round trip
    back to its own concrete type, not the abstract base.
    """
    concrete = SubClassGenericThatUpdatesGenericTypeToBuiltInType(
        attribute_using_generic=5
    )
    holder = BareGenericFieldHolder(item=concrete)

    dao: BareGenericFieldHolderDAO = to_dao(holder)
    session.add(dao)
    session.commit()
    session.expunge_all()

    queried = session.scalars(select(BareGenericFieldHolderDAO)).one()
    assert isinstance(queried.item, SubClassGenericThatUpdatesGenericTypeToBuiltInTypeDAO)


    reconstructed = queried.from_dao()
    assert type(reconstructed.item) is SubClassGenericThatUpdatesGenericTypeToBuiltInType
    assert reconstructed.item.attribute_using_generic == 5
