"""
How much of the objects an EQL query selects is loaded from the database, and with
which statements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import batched

import sqlalchemy
from sqlalchemy.orm import Session, lazyload, with_polymorphic
from sqlalchemy.sql.base import ExecutableOption
from typing_extensions import Any, Sequence

from krrood.ormatic.data_access_objects.dao import DataAccessObject, StoredObjectRow

# %% choices


class RelationshipLoading(StrEnum):
    """
    When the relationships of the loaded data access objects are loaded.
    """

    AS_MAPPED = "as_mapped"
    """
    Each relationship loads as its mapping declares, eagerly where it says so.
    """

    ON_ACCESS = "on_access"
    """
    Every relationship loads when it is first read.
    """


class SubclassLoading(StrEnum):
    """
    How the columns that the classes of the selected objects add to the selected class
    are loaded.
    """

    AS_MAPPED = "as_mapped"
    """
    As the mappers declare: one further statement per subclass and batch of ids.
    """

    ONE_QUERY_PER_CLASS = "one_query_per_class"
    """
    One statement per class of the selected objects, loading every column of it.
    """

    ONE_QUERY_PER_HIERARCHY = "one_query_per_hierarchy"
    """
    One statement per hierarchy, joining the tables of the selected objects' classes in
    it.
    """


class ResultContent(StrEnum):
    """
    What a query's evaluation returns for every selected object.
    """

    DATA_ACCESS_OBJECTS = "data_access_objects"
    """
    The data access object, loaded as the class of the object.
    """

    DATABASE_IDS = "database_ids"
    """
    The id of the object, without loading any of its rows.
    """


# %% configuration


@dataclass(kw_only=True, frozen=True)
class ResultLoading:
    """
    How the EQL translator loads the objects that a query selects.
    """

    relationship_loading: RelationshipLoading = RelationshipLoading.AS_MAPPED
    """
    When the relationships of the loaded data access objects are loaded.
    """

    subclass_loading: SubclassLoading = SubclassLoading.AS_MAPPED
    """
    How the columns of the selected objects' own classes are loaded.
    """

    result_content: ResultContent = ResultContent.DATA_ACCESS_OBJECTS
    """
    What the evaluation returns for every selected object.
    """

    ids_per_statement: int = 10_000
    """
    The most ids one statement binds, below the parameter limits of SQLite and
    PostgreSQL.
    """

    @property
    def loads_selected_rows(self) -> bool:
        """
        :return: Whether the evaluation loads the data access objects from the rows the
            translated query selects, as the mappers declare.
        """
        return (
            self.subclass_loading == SubclassLoading.AS_MAPPED
            and self.result_content == ResultContent.DATA_ACCESS_OBJECTS
        )

    @property
    def loader_options(self) -> tuple[ExecutableOption, ...]:
        """
        :return: The options every statement that loads data access objects takes.
        """
        if self.relationship_loading == RelationshipLoading.ON_ACCESS:
            return (lazyload("*"),)
        return ()

    def load_data_access_objects(
        self, *, session: Session, selected_objects: Sequence[sqlalchemy.Row]
    ) -> list[DataAccessObject]:
        """
        :param session: The session to load with.
        :param selected_objects: The ``database_id`` and ``polymorphic_type`` of every
            selected object, in the order of the query.
        :return: The data access objects of the selected objects, in the same order.
        """
        ids_by_class: dict[type[DataAccessObject], list[int]] = {}
        for selected_object in selected_objects:
            dao_class = StoredObjectRow.dao_class_of_table(
                table_name=selected_object.polymorphic_type
            )
            ids_by_class.setdefault(dao_class, []).append(selected_object.database_id)
        daos_by_id = {}
        for entity, ids in self._loaded_entities(ids_by_class=ids_by_class):
            for batch in batched(ids, self.ids_per_statement):
                statement = (
                    sqlalchemy.select(entity)
                    .where(entity.database_id.in_(batch))
                    .options(*self.loader_options)
                )
                for dao in session.scalars(statement):
                    daos_by_id[dao.database_id] = dao
        return [
            daos_by_id[selected_object.database_id]
            for selected_object in selected_objects
        ]

    def _loaded_entities(
        self, *, ids_by_class: dict[type[DataAccessObject], list[int]]
    ) -> list[tuple[Any, list[int]]]:
        """
        :param ids_by_class: The ids of the selected objects, by the class they are
            loaded as.
        :return: Every entity one statement loads, with the ids it loads.
        """
        if self.subclass_loading != SubclassLoading.ONE_QUERY_PER_HIERARCHY:
            return list(ids_by_class.items())
        classes_by_root: dict[type, list[type[DataAccessObject]]] = {}
        for dao_class in ids_by_class:
            root = sqlalchemy.inspect(dao_class).base_mapper.class_
            classes_by_root.setdefault(root, []).append(dao_class)
        return [
            (
                with_polymorphic(root, classes, flat=True),
                [
                    database_id
                    for dao_class in classes
                    for database_id in ids_by_class[dao_class]
                ],
            )
            for root, classes in classes_by_root.items()
        ]
