"""
Bring a database written before the table of every stored object to it: every row gets
an id no other row holds and its row in
:class:`~krrood.ormatic.stored_object.StoredObjectDAO`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from sqlalchemy import (
    Column,
    Connection,
    MetaData,
    Table,
    and_,
    bindparam,
    exists,
    func,
    insert,
    select,
    union_all,
    update,
)

from krrood.ormatic.stored_object import StoredObjectDAO

# %% one class hierarchy


@dataclass(kw_only=True)
class StoredObjectHierarchy:
    """
    The tables of one class hierarchy whose root's ids refer to the table of every
    stored object, and the columns elsewhere that refer to its rows.
    """

    root: Table
    """
    The table of the hierarchy's root class, holding a row for every object in it.
    """

    extending_tables: tuple[Table, ...]
    """
    The tables whose ``database_id`` refers, directly or through another of them, to the
    root's.
    """

    referring_columns: tuple[Column, ...]
    """
    The columns of any table, other than a ``database_id``, that refer to a row of the
    hierarchy.
    """

    def ids_without_stored_object_row(self, *, connection: Connection) -> list[int]:
        """
        :param connection: A connection to the database.
        :return: The ids of the root's rows that have no row of their own in the table of
            every stored object, in ascending order.
        """
        stored = StoredObjectDAO.__table__
        has_own_row = exists().where(
            and_(
                stored.c.database_id == self.root.c.database_id,
                stored.c.polymorphic_type == self.root.c.polymorphic_type,
            )
        )
        return list(
            connection.scalars(
                select(self.root.c.database_id)
                .where(~has_own_row)
                .order_by(self.root.c.database_id)
            )
        )

    def renumber(self, *, connection: Connection, new_ids: dict[int, int]) -> None:
        """
        Give the rows of the hierarchy their new ids, and every column referring to them
        the same.

        :param connection: A connection to the database, its foreign keys not enforced.
        :param new_ids: The new id of each old one; none of them is an id in use.
        """
        if not new_ids:
            return
        moves = [{"old_id": old, "new_id": new} for old, new in new_ids.items()]
        for table in (self.root, *self.extending_tables):
            connection.execute(
                update(table)
                .where(table.c.database_id == bindparam("old_id"))
                .values(database_id=bindparam("new_id")),
                moves,
            )
        for column in self.referring_columns:
            connection.execute(
                update(column.table)
                .where(column == bindparam("old_id"))
                .values({column.name: bindparam("new_id")}),
                moves,
            )

    def stored_object_rows(self, *, new_ids: dict[int, int]) -> select:
        """
        :param new_ids: The ids the hierarchy's rows were given.
        :return: The rows of the table of every stored object for them.
        """
        return select(self.root.c.database_id, self.root.c.polymorphic_type).where(
            self.root.c.database_id.in_(list(new_ids.values()))
        )


# %% the database


@dataclass(kw_only=True)
class StoredObjectBackfill:
    """
    Give every row of a database whose tables numbered their rows each on its own an id
    that no other row holds, and its row in the table of every stored object.

    .. warning::
        The database's tables must already have the columns of :attr:`metadata`, and its
        foreign keys must not be enforced while :meth:`backfill` runs, as SQLite does not
        by default. Changing the constraints themselves is left to a migration tool.
    """

    metadata: MetaData
    """
    The tables of the generated interface the database is read with.
    """

    hierarchies: tuple[StoredObjectHierarchy, ...] = field(init=False)
    """
    Every class hierarchy whose root refers to the table of every stored object, in the
    order of their roots' names.
    """

    def __post_init__(self) -> None:
        """
        Find the hierarchies of :attr:`metadata`.
        """
        roots = sorted(
            (
                table
                for table in self.metadata.tables.values()
                if self.referred_table(table=table) is StoredObjectDAO.__table__
            ),
            key=lambda table: table.name,
        )
        self.hierarchies = tuple(self.hierarchy(root=root) for root in roots)

    def referred_table(self, *, table: Table) -> Table | None:
        """
        :param table: A table of :attr:`metadata`.
        :return: The table its ``database_id`` refers to, or ``None`` where it refers to
            none.
        """
        if "database_id" not in table.c:
            return None
        foreign_keys = table.c.database_id.foreign_keys
        if not foreign_keys:
            return None
        return next(iter(foreign_keys)).column.table

    def hierarchy(self, *, root: Table) -> StoredObjectHierarchy:
        """
        :param root: The table of a hierarchy's root class.
        :return: The hierarchy, with its extending tables and the columns referring to it.
        """
        members = [root]
        for member in members:
            members.extend(
                table
                for table in self.metadata.tables.values()
                if self.referred_table(table=table) is member and table not in members
            )
        referring_columns = tuple(
            column
            for table in self.metadata.tables.values()
            for column in table.c
            if column.name != "database_id"
            and any(
                foreign_key.column.table in members
                for foreign_key in column.foreign_keys
            )
        )
        return StoredObjectHierarchy(
            root=root,
            extending_tables=tuple(members[1:]),
            referring_columns=referring_columns,
        )

    def backfill(self, *, connection: Connection) -> None:
        """
        Renumber the rows that have no row in the table of every stored object after
        every id in use, and add their rows to it, creating it where it is missing.

        :param connection: A connection to the database, its foreign keys not enforced.
        """
        StoredObjectDAO.__table__.create(connection, checkfirst=True)
        next_ids = itertools.count(self.largest_id(connection=connection) + 1)
        new_ids_of_hierarchies = [
            dict(
                zip(
                    hierarchy.ids_without_stored_object_row(connection=connection),
                    next_ids,
                )
            )
            for hierarchy in self.hierarchies
        ]
        for hierarchy, new_ids in zip(self.hierarchies, new_ids_of_hierarchies):
            hierarchy.renumber(connection=connection, new_ids=new_ids)
        rows = [
            hierarchy.stored_object_rows(new_ids=new_ids)
            for hierarchy, new_ids in zip(self.hierarchies, new_ids_of_hierarchies)
            if new_ids
        ]
        if not rows:
            return
        connection.execute(
            insert(StoredObjectDAO.__table__).from_select(
                ["database_id", "polymorphic_type"], union_all(*rows)
            )
        )

    def largest_id(self, *, connection: Connection) -> int:
        """
        :param connection: A connection to the database.
        :return: The largest id any root or the table of every stored object holds, or 0
            in an empty database.
        """
        tables = (
            StoredObjectDAO.__table__,
            *(hierarchy.root for hierarchy in self.hierarchies),
        )
        return max(
            connection.scalar(select(func.coalesce(func.max(table.c.database_id), 0)))
            for table in tables
        )
