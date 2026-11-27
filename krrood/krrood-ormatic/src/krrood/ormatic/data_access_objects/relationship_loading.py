"""
Loading the objects a DAO's relationships refer to, from the identity map where it holds
them and with one statement per table inside :func:`selectin_loading`.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import StrEnum

import sqlalchemy.inspection
import sqlalchemy.orm
from sqlalchemy import event
from sqlalchemy.orm import selectinload
from typing_extensions import TYPE_CHECKING, Any, Iterator

from krrood.ormatic.data_access_objects.relationships import SingleRelationship

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import DataAccessObject


# %% session information


class SessionInformationKey(StrEnum):
    """
    Keys of the entries the conversion between DAOs and domain objects writes into
    :attr:`sqlalchemy.orm.Session.info`.
    """

    BULK_LOADING_FLAG = "krrood_bulk_loading"
    BULK_LOADED_MAPPERS = "krrood_bulk_loaded_mappers"
    PLAIN_LOAD = "krrood_plain_load"
    OWNERS_OF_UNLOADED_PARTS = "krrood_owners_of_unloaded_parts"
    """
    The loaded DAOs, by the classes of the parts they have but whose part of that
    class was not loaded yet.
    """


# %% reading relationships


def _bulk_load_target_table_if_enabled(
    session: sqlalchemy.orm.Session, target_dao_class: type
) -> None:
    """
    Inside :func:`selectin_loading`, load all rows of the target's root mapper into the
    identity map, once per session and root mapper.

    :param session: The session to load into.
    :param target_dao_class: The DAO class about to be fetched.
    """
    if not session.info.get(SessionInformationKey.BULK_LOADING_FLAG, False):
        return
    root_mapper = sqlalchemy.inspection.inspect(target_dao_class).base_mapper
    loaded = session.info.setdefault(SessionInformationKey.BULK_LOADED_MAPPERS, {})
    if root_mapper in loaded:
        return
    # keep strong references to the rows: the identity map is weak, so the
    # instances would be garbage collected (and the map emptied) otherwise
    loaded[root_mapper] = session.scalars(
        sqlalchemy.select(root_mapper.class_).execution_options(
            **{SessionInformationKey.PLAIN_LOAD: True}
        )
    ).all()


def read_single_relationship(
    instance: Any, relationship: SingleRelationship
) -> DataAccessObject | None:
    """
    Read the value of a single-valued relationship on a DAO or association object, an
    unloaded many-to-one relationship from the identity map by its local foreign key.

    :param instance: The object to read the relationship from.
    :param relationship: The relationship to read.
    :return: The related DAO instance or None.
    """
    key = relationship.key
    if relationship.foreign_key_attribute is None or key in instance.__dict__:
        return getattr(instance, key)
    session = sqlalchemy.orm.object_session(instance)
    if session is None:
        return getattr(instance, key)
    foreign_key_value = getattr(instance, relationship.foreign_key_attribute)
    if foreign_key_value is not None:
        _bulk_load_target_table_if_enabled(session, relationship.target_dao_class)
    value = (
        session.get(
            relationship.target_dao_class,
            foreign_key_value,
            execution_options={SessionInformationKey.PLAIN_LOAD: True},
        )
        if foreign_key_value is not None
        else None
    )
    sqlalchemy.orm.attributes.set_committed_value(instance, key, value)
    return value


# %% bulk loading


@contextmanager
def selectin_loading(session: sqlalchemy.orm.Session) -> Iterator[None]:
    """
    Context manager that optimizes bulk reads of whole object graphs.

    Top-level ORM queries executed within its scope use selectin loading for
    their relationships. Additionally, identity-map misses during
    :meth:`DataAccessObject.from_dao` bulk-load the entire table of the missed
    class instead of fetching one row at a time, so reconstructing a graph that
    spans (most of) the database takes O(tables) instead of O(rows) queries.

    Use this context manager when you want call `from_dao()` on the result, as it is ~3 times faster.

    Usage::

        with selectin_loading(session):
            dao = session.scalars(select(SomeDAO)).one()
            domain_obj = dao.from_dao()

    :param session: The SQLAlchemy session whose queries should use selectin loading.
    """

    def _add_selectin(orm_execute_state: sqlalchemy.orm.ORMExecuteState) -> None:
        """
        Load the relationships of a top level query's results with selectin loading.

        :param orm_execute_state: The statement about to be executed.
        """
        if orm_execute_state.execution_options.get(
            SessionInformationKey.PLAIN_LOAD, False
        ):
            return None
        if orm_execute_state.is_select and not orm_execute_state.is_relationship_load:
            return orm_execute_state.invoke_statement(
                statement=orm_execute_state.statement.options(selectinload("*"))
            )

    event.listen(session, "do_orm_execute", _add_selectin)
    session.info[SessionInformationKey.BULK_LOADING_FLAG] = True
    try:
        yield
    finally:
        session.info[SessionInformationKey.BULK_LOADING_FLAG] = False
        session.info.pop(SessionInformationKey.BULK_LOADED_MAPPERS, None)
        event.remove(session, "do_orm_execute", _add_selectin)
