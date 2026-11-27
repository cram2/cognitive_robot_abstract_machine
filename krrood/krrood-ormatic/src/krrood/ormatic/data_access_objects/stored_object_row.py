"""
The row every stored object has in the table of every stored object, and the rows a
DAO's object has outside the DAO's own tables.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable
from enum import StrEnum
from functools import lru_cache
from itertools import batched

import sqlalchemy.inspection
import sqlalchemy.orm
from sqlalchemy import event
from sqlalchemy.orm import make_transient_to_detached
from sqlalchemy.sql.base import ExecutableOption
from typing_extensions import TYPE_CHECKING, Any, Callable, Sequence

from krrood.ormatic.base import Base
from krrood.ormatic.data_access_objects.relationship_loading import (
    SessionInformationKey,
)
from krrood.ormatic.data_access_objects.schema import get_dao_schema
from krrood.ormatic.exceptions import (
    PartOfDetachedObjectNotLoadedError,
    StoredObjectTableNotUniqueError,
)
from krrood.patterns.called_positionally import called_positionally
from krrood.patterns.classproperty import classproperty

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import (
        DataAccessObject,
        PartDataAccessObject,
    )
    from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState


# %% the table of every stored object


class StoredObjectColumn(StrEnum):
    """
    The columns of the table of every stored object that a selection of the ids and
    types of stored objects gives, under the names of the attributes of its rows.
    """

    DATABASE_ID = "database_id"
    """
    The id of the object, the same in the table of every class it has rows in.
    """

    POLYMORPHIC_TYPE = "polymorphic_type"
    """
    The name of the table of the class the object is loaded as.
    """


class StoredObjectRow:
    """
    The row every stored object has, whatever its class.

    The root table of every generated hierarchy takes its ids from the table of the
    mapped subclass, so an object has the same id in the tables of all its mapped base
    classes, and the row's type column names the DAO class the object is loaded as.
    """

    object_dao_in_memory: DataAccessObject | None = None
    """
    The DAO of the object, while it was created in this process and not loaded.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Insert the rows of new objects into the subclass's table before every flush.
        """
        super().__init_subclass__(**kwargs)
        event.listen(
            sqlalchemy.orm.Session, "before_flush", cls.insert_rows_of_new_objects
        )

    @staticmethod
    def mapped_class() -> type[StoredObjectRow]:
        """
        :return: The class mapping the table of every stored object.
        :raises StoredObjectTableNotUniqueError: If not exactly one class maps it.
        """
        table_classes = StoredObjectRow.__subclasses__()
        if len(table_classes) != 1:
            raise StoredObjectTableNotUniqueError(table_classes=table_classes)
        return table_classes[0]

    @property
    def object_dao(self) -> DataAccessObject:
        """
        :return: The DAO of the object this row belongs to, loaded as its own class.
        """
        if self.object_dao_in_memory is not None:
            return self.object_dao_in_memory
        return sqlalchemy.orm.object_session(self).get(
            self.dao_class_of_table(table_name=self.polymorphic_type), self.database_id
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def dao_class_of_table(*, table_name: str) -> type[DataAccessObject]:
        """
        :param table_name: The name of the table of a generated DAO class.
        :return: The DAO class objects of that table are loaded as.
        """
        return next(
            mapper.class_
            for mapper in Base.registry.mappers
            if mapper.local_table.name == table_name
            and issubclass(mapper.class_, StoredObjectRowHolder)
            and not mapper.class_.is_part
        )

    @classmethod
    def load_objects_by_class(
        cls,
        *,
        stored_objects: Sequence[StoredObjectRow | sqlalchemy.Row],
        load_objects_of_class: Callable[
            [type[DataAccessObject], list[int]], Iterable[Any]
        ],
    ) -> list[Any]:
        """
        :param stored_objects: Rows of this table, or rows with the
            :attr:`~krrood.ormatic.stored_object.StoredObjectDAO.database_id` and
            :attr:`~krrood.ormatic.stored_object.StoredObjectDAO.polymorphic_type` of stored objects.
        :param load_objects_of_class: Loads the objects of one DAO class from their ids,
            each of them with its
            :attr:`~krrood.ormatic.stored_object.StoredObjectDAO.database_id`.
        :return: What was loaded for every stored object, in the same order.
        """
        ids_by_class: dict[type[DataAccessObject], list[int]] = {}
        for stored_object in stored_objects:
            dao_class = cls.dao_class_of_table(
                table_name=stored_object.polymorphic_type
            )
            ids_by_class.setdefault(dao_class, []).append(stored_object.database_id)
        loaded_by_id = {
            loaded.database_id: loaded
            for dao_class, ids in ids_by_class.items()
            for loaded in load_objects_of_class(dao_class, ids)
        }
        return [
            loaded_by_id[stored_object.database_id] for stored_object in stored_objects
        ]

    @classmethod
    def object_daos(
        cls,
        *,
        session: sqlalchemy.orm.Session,
        stored_objects: Sequence[StoredObjectRow | sqlalchemy.Row],
        loader_options: tuple[ExecutableOption, ...] = (),
        ids_per_statement: int,
    ) -> list[DataAccessObject]:
        """
        :param session: The session the rows were loaded with.
        :param stored_objects: Rows of this table, or rows with the
            :attr:`~krrood.ormatic.stored_object.StoredObjectDAO.database_id` and
            :attr:`~krrood.ormatic.stored_object.StoredObjectDAO.polymorphic_type` of stored objects.
        :param loader_options: The options of the statements that load the DAOs.
        :param ids_per_statement: The most ids one statement binds.
        :return: The DAOs of the rows' objects, in the same order, loaded with one query
            per DAO class and batch of ids.
        """
        return cls.load_objects_by_class(
            stored_objects=stored_objects,
            load_objects_of_class=lambda dao_class, ids: (
                dao
                for batch in batched(ids, ids_per_statement)
                for dao in session.scalars(
                    sqlalchemy.select(dao_class)
                    .where(dao_class.database_id.in_(batch))
                    .options(*loader_options)
                )
            ),
        )

    def from_dao(self, *, state: FromDataAccessObjectState | None = None) -> Any:
        """
        :param state: The conversion state.
        :return: The object this row belongs to, converted from its DAO as
            :meth:`DataAccessObject.from_dao` converts it.
        """
        return self.object_dao.from_dao(state=state)

    @classmethod
    @called_positionally
    def insert_rows_of_new_objects(
        cls, session: sqlalchemy.orm.Session, flush_context: Any, instances: Any
    ) -> None:
        """
        Before a session flushes, add the DAOs and parts that its new objects hold in
        memory, insert the rows of all new objects, give their DAOs and parts the rows'
        ids, and delete the other rows of the objects it deletes.

        :param session: The session about to flush.
        :param flush_context: The flush's unit of work.
        :param instances: The objects the flush was asked for.
        """
        cls._add_objects_held_in_memory(session=session)
        cls._delete_rows_of_deleted_objects(session=session)
        new_daos = [
            dao
            for dao in session.new
            if isinstance(dao, StoredObjectRowHolder)
            and get_dao_schema(type(dao)).shares_id_with_stored_object
            and dao.database_id is None
        ]
        if new_daos:
            cls._insert_rows(session=session, daos=new_daos)

    @classmethod
    def _insert_rows(
        cls, *, session: sqlalchemy.orm.Session, daos: list[DataAccessObject]
    ) -> None:
        """
        Insert the rows of new DAOs' objects in one statement and give the rows, the
        DAOs and their parts the ids the database chose.

        :param session: The session about to flush.
        :param daos: New DAOs whose objects have no row yet.
        """
        rows = [dao.stored_object for dao in daos]
        ids = session.execute(
            sqlalchemy.insert(cls.__table__).returning(
                cls.__table__.c.database_id, sort_by_parameter_order=True
            ),
            [{"polymorphic_type": row.polymorphic_type} for row in rows],
        ).scalars()
        for row, dao, database_id in zip(rows, daos, ids, strict=True):
            dao.database_id = database_id
            for part in (dao.parts_in_memory or {}).values():
                part.database_id = database_id
            cls._make_row_persistent(session=session, row=row, database_id=database_id)

    @classmethod
    def _make_row_persistent(
        cls, *, session: sqlalchemy.orm.Session, row: StoredObjectRow, database_id: int
    ) -> None:
        """
        Turn a row inserted outside the flush into a stored object of the session if a
        field already refers to it, or into a detached one otherwise.

        :param session: The session about to flush.
        :param row: The inserted row.
        :param database_id: The id the database chose for the row.
        """
        referenced = row in session
        if referenced:
            session.expunge(row)
        row.database_id = database_id
        make_transient_to_detached(row)
        if not referenced:
            return
        # a row deleted without the session may still hold the id the database reused
        stale_row = session.identity_map.get(
            cls.__mapper__.identity_key_from_primary_key((database_id,))
        )
        if stale_row is not None:
            session.expunge(stale_row)
        session.add(row)

    @classmethod
    def _add_objects_held_in_memory(cls, *, session: sqlalchemy.orm.Session) -> None:
        """
        Add the DAOs of the objects whose new rows of this table a session holds, since
        a field referring to an object holds only its row, and the parts of the new
        DAOs, together with what they refer to in turn.

        :param session: The session about to flush.
        """
        while True:
            held_in_memory = [
                held
                for new in session.new
                for held in cls._objects_held_in_memory(new_object=new)
                if held not in session
            ]
            if not held_in_memory:
                return
            session.add_all(held_in_memory)

    @classmethod
    def _objects_held_in_memory(cls, *, new_object: Any) -> list[DataAccessObject]:
        """
        :param new_object: An object a session is about to insert.
        :return: The DAO of the object a new row of this table belongs to, or the parts
            of a new DAO.
        """
        if isinstance(new_object, cls):
            owner = new_object.object_dao_in_memory
            return [] if owner is None else [owner]
        if isinstance(new_object, StoredObjectRowHolder) and new_object.parts_in_memory:
            return list(new_object.parts_in_memory.values())
        return []

    @staticmethod
    def _delete_rows_of_deleted_objects(*, session: sqlalchemy.orm.Session) -> None:
        """
        Delete the rows that the objects of the DAOs a session is about to delete have
        in the table of every stored object and in the tables of their secondary bases.

        :param session: The session about to flush.
        """
        for dao in list(session.deleted):
            if not isinstance(dao, StoredObjectRowHolder):
                continue
            schema = get_dao_schema(type(dao))
            if not schema.shares_id_with_stored_object:
                continue
            rows = [dao.stored_object] + [
                dao.part(part_class=part_class) for part_class in schema.part_classes
            ]
            for row in rows:
                if row is not None and sqlalchemy.inspect(row).persistent:
                    session.delete(row)


# %% the rows a DAO's object has outside the DAO's own tables


class StoredObjectRowHolder:
    """
    The rows a DAO's object has outside the DAO's own tables: its row in the table of
    every stored object, and its parts in the tables of its secondary bases.
    """

    stored_object_in_memory: StoredObjectRow | None = None
    """
    The row of the DAO's object in the table of every stored object, once it was asked
    for.
    """

    parts_in_memory: (
        dict[type[PartDataAccessObject], PartDataAccessObject | None] | None
    ) = None
    """
    The parts of the DAO's object by their classes, once created or loaded; None for a
    part that the object does not have.
    """

    @classproperty
    def is_part(cls) -> bool:
        """
        :return: Whether objects of this class are parts held by the DAO of another
            object, rather than the DAO an object is converted to or loaded as.
        """
        return False

    @classmethod
    @lru_cache(maxsize=None)
    def _shares_id_with_stored_object(cls) -> bool:
        """
        :return: Whether the objects of this DAO class take their ids from their rows in
            the table of every stored object.
        """
        stored_object_table = StoredObjectRow.mapped_class().__table__
        return not cls.is_part and any(
            foreign_key.column.table is stored_object_table
            for column in sqlalchemy.inspection.inspect(
                cls
            ).base_mapper.local_table.primary_key
            for foreign_key in column.foreign_keys
        )

    @property
    def stored_object(self) -> StoredObjectRow:
        """
        :return: The row of this DAO's object in the table of every stored object,
            created for an object that was never stored.
        """
        if self.stored_object_in_memory is not None:
            return self.stored_object_in_memory
        stored_object_class = StoredObjectRow.mapped_class()
        if self.database_id is None:
            self.stored_object_in_memory = stored_object_class(
                polymorphic_type=self.__tablename__
            )
            self.stored_object_in_memory.object_dao_in_memory = self
        else:
            self.stored_object_in_memory = sqlalchemy.orm.object_session(self).get(
                stored_object_class, self.database_id
            )
        return self.stored_object_in_memory

    def part(
        self, *, part_class: type[PartDataAccessObject]
    ) -> PartDataAccessObject | None:
        """
        :param part_class: The class of one of the parts of this DAO's class.
        :return: The part of this DAO's object of that class, if the object has one.
        :raises PartOfDetachedObjectNotLoadedError: If this DAO was loaded, left its
            session and never loaded that part.
        """
        if self.parts_in_memory is not None and part_class in self.parts_in_memory:
            return self.parts_in_memory[part_class]
        instance_state = sqlalchemy.inspect(self)
        if instance_state.detached:
            raise PartOfDetachedObjectNotLoadedError(
                data_access_object=self, part_class=part_class
            )
        session = instance_state.session
        if session is None:
            return None
        owners = session.info.get(
            SessionInformationKey.OWNERS_OF_UNLOADED_PARTS, {}
        ).pop(part_class, weakref.WeakSet())
        owners.add(self)
        owners_by_id = {owner.database_id: owner for owner in owners}
        parts_by_id = {
            part.database_id: part
            for part in session.scalars(
                sqlalchemy.select(part_class)
                .where(part_class.database_id.in_(owners_by_id))
                .execution_options(**{SessionInformationKey.PLAIN_LOAD: True})
            )
        }
        for database_id, owner in owners_by_id.items():
            if owner.parts_in_memory is None:
                owner.parts_in_memory = {}
            owner.parts_in_memory[part_class] = parts_by_id.get(database_id)
        return self.parts_in_memory[part_class]

    @staticmethod
    @called_positionally
    def register_owner_of_unloaded_parts(
        target: Any, context: sqlalchemy.orm.QueryContext
    ) -> None:
        """
        When a DAO that has parts is loaded, remember it, so its parts are loaded
        together with those of the other DAOs loaded meanwhile.

        :param target: The loaded object.
        :param context: The query that loaded it.
        """
        if not isinstance(target, StoredObjectRowHolder):
            return
        part_classes = get_dao_schema(type(target)).part_classes
        if not part_classes:
            return
        owners = context.session.info.setdefault(
            SessionInformationKey.OWNERS_OF_UNLOADED_PARTS, {}
        )
        for part_class in part_classes:
            owners.setdefault(part_class, weakref.WeakSet()).add(target)


event.listen(
    Base, "load", StoredObjectRowHolder.register_owner_of_unloaded_parts, propagate=True
)
