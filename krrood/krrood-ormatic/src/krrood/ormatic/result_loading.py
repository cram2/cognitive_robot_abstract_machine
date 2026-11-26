"""
How much of the objects an EQL query selects is loaded from the database, and with
which statements.
"""

from __future__ import annotations

import gc
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from itertools import batched

import sqlalchemy
from sqlalchemy import Select
from sqlalchemy.orm import Session, lazyload
from sqlalchemy.sql.base import ExecutableOption
from typing_extensions import Any, Iterable, Iterator, Sequence, get_origin

from krrood.entity_query_language.core.base_expressions import HasExpression
from krrood.entity_query_language.query.query import Entity
from krrood.ormatic.data_access_objects.dao import (
    DataAccessObject,
    StoredObjectColumn,
    StoredObjectRow,
    selectin_loading,
)
from krrood.ormatic.data_access_objects.helper import (
    get_mapping_data_access_object_class,
    get_part_class,
)

# %% choices


class RelationshipLoading(StrEnum):
    """
    When the relationships of the loaded data access objects are loaded.
    """

    AS_MAPPED = "as_mapped"
    """
    Each relationship loads as its mapping declares, eagerly where it says so. Use it
    when the relationships are read after the session closed.
    """

    ON_ACCESS = "on_access"
    """
    Every relationship loads when it is first read. Use it when few of the loaded
    objects' relationships are read, since every eager relationship costs statements.
    """

    FOR_CONVERSION = "for_conversion"
    """
    Every relationship of the selected objects loads with them, as inside
    :func:`~krrood.ormatic.data_access_objects.dao.selectin_loading`. Use it when the
    results are converted to domain objects.
    """


class SubclassLoading(StrEnum):
    """
    How the columns that the classes of the selected objects add to the selected class
    are loaded.
    """

    AS_MAPPED = "as_mapped"
    """
    As the mappers declare: one further statement per subclass and batch of ids. Use
    it when the selected class has no mapped subclasses, so the selection holds all.
    """

    ONE_QUERY_PER_CLASS = "one_query_per_class"
    """
    One statement per class of the selected objects, loading every column of it. Use
    it when the selected objects may be of many classes, such as for a secondary base.
    """


class SelectedObjectRestriction(StrEnum):
    """
    How a statement that loads the objects of one class keeps to the selected ones.
    """

    BOUND_IDS = "bound_ids"
    """
    It binds the ids of the selected objects, in batches. Use it when the selection is
    costly to evaluate again.
    """

    SELECTION_JOIN = "selection_join"
    """
    It joins the query's selection, which the database evaluates again for it. Use it
    when the selection is cheap and selects very many objects. A query with a limit
    binds the ids instead, since evaluating it again may select other objects.
    """


class ResultContent(StrEnum):
    """
    What a query's evaluation returns for every selected object.
    """

    DATA_ACCESS_OBJECTS = "data_access_objects"
    """
    The data access object, loaded as the class of the object. Use it when the objects
    are converted to domain objects or their relationships are read.
    """

    COLUMNS = "columns"
    """
    A row with the value of every column of the object's own class, without building
    its data access object. Use it when only the column values of many objects are
    read, such as for a table or a statistic.
    """

    DATABASE_IDS = "database_ids"
    """
    The id of the object, without loading any of its rows. Use it to count the
    objects, or to load them later in parts.
    """


# %% pausing the garbage collector


@dataclass(kw_only=True)
class GarbageCollectionPause(AbstractContextManager):
    """
    Keeps Python's cyclic garbage collector from running inside a ``with`` block.
    """

    collector_was_running: bool = False
    """
    Whether the collector ran when the block was entered, so it runs again after it.
    """

    def __enter__(self) -> GarbageCollectionPause:
        """
        Pauses the collector, unlike :meth:`contextlib.AbstractContextManager.__enter__`.
        """
        self.collector_was_running = gc.isenabled()
        gc.disable()
        return self

    def __exit__(self, *exception_information: Any) -> None:
        if self.collector_was_running:
            gc.enable()


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

    selected_object_restriction: SelectedObjectRestriction = (
        SelectedObjectRestriction.BOUND_IDS
    )
    """
    How a statement loading the objects of one class keeps to the selected ones.
    """

    result_content: ResultContent = ResultContent.DATA_ACCESS_OBJECTS
    """
    What the evaluation returns for every selected object.
    """

    pauses_garbage_collection: bool = False
    """
    Whether Python's cyclic garbage collector is kept from running during the
    evaluation. Use it for large results while no other thread allocates much, since
    the collector finds next to nothing among the objects a load creates.
    """

    ids_per_statement: int = 10_000
    """
    The most ids one statement binds.
    """

    @classmethod
    def from_query(
        cls,
        *,
        query: HasExpression,
        relationship_loading: RelationshipLoading = RelationshipLoading.ON_ACCESS,
    ) -> ResultLoading:
        """
        :param query: The EQL query to evaluate, or the object standing for it.
        :param relationship_loading: When the relationships of the results load.
        :return: A result loading that loads the query's objects with few statements,
            one statement per class where the selected class's objects may be of
            several classes.
        """
        expression = query._get_expression_()
        if not isinstance(expression, Entity) or not cls._selects_several_classes(
            domain_type=expression._type_
        ):
            return cls(relationship_loading=relationship_loading)
        return cls(
            relationship_loading=relationship_loading,
            subclass_loading=SubclassLoading.ONE_QUERY_PER_CLASS,
        )

    @staticmethod
    def _selects_several_classes(*, domain_type: type | None) -> bool:
        """
        :param domain_type: The class a query selects the objects of.
        :return: Whether those objects may be loaded as other classes than its own.
        """
        domain_type = get_origin(domain_type) or domain_type
        if not isinstance(domain_type, type) or get_part_class(domain_type) is not None:
            return True
        dao_class = get_mapping_data_access_object_class(domain_type)
        if dao_class is None:
            return False
        return len(sqlalchemy.inspect(dao_class).self_and_descendants) > 1

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

    def build_loader_options(
        self, *, loaded_classes: Iterable[Any]
    ) -> tuple[ExecutableOption, ...]:
        """
        :param loaded_classes: The data access object classes, or aliases of them, that a
            statement loads.
        :return: The options the statement takes, reaching also the relationships that
            subclasses of the loaded classes declare.
        """
        if self.relationship_loading != RelationshipLoading.ON_ACCESS:
            return ()
        subclass_relationships = (
            relationship.class_attribute
            for loaded_class in loaded_classes
            for mapper in sqlalchemy.inspect(loaded_class).mapper.self_and_descendants
            for relationship in mapper.relationships
            if relationship.parent is mapper
        )
        return (lazyload("*"), *map(lazyload, subclass_relationships))

    @contextmanager
    def create_evaluation_context(self, *, session: Session) -> Iterator[None]:
        """
        :param session: The session the evaluation loads with.
        :return: The context an evaluation with this result loading runs in.
        """
        with ExitStack() as stack:
            if self.pauses_garbage_collection:
                stack.enter_context(GarbageCollectionPause())
            if self.relationship_loading == RelationshipLoading.FOR_CONVERSION:
                stack.enter_context(selectin_loading(session))
            yield

    def load_selected_objects(
        self,
        *,
        session: Session,
        selection: Select,
        selected_objects: Sequence[sqlalchemy.Row],
    ) -> list[Any]:
        """
        :param session: The session to load with.
        :param selection: The statement that selected the objects.
        :param selected_objects: The :attr:`~krrood.ormatic.stored_object.StoredObjectDAO.database_id` and
            :attr:`~krrood.ormatic.stored_object.StoredObjectDAO.polymorphic_type` of every
            selected object, as *selection* gave them.
        :return: The data access object or the row of every selected object, in the
            order of *selected_objects*.
        """
        return StoredObjectRow.load_objects_by_class(
            stored_objects=selected_objects,
            load_objects_of_class=lambda dao_class, ids: (
                loaded
                for statement in self._build_loading_statements(
                    dao_class=dao_class, ids=ids, selection=selection
                )
                for loaded in self._execute(session=session, statement=statement)
            ),
        )

    def _build_loading_statements(
        self, *, dao_class: type[DataAccessObject], ids: list[int], selection: Select
    ) -> Iterator[Select]:
        """
        :param dao_class: The class the objects load as.
        :param ids: The ids of the selected objects of that class.
        :param selection: The statement that selected the objects.
        :return: The statements that load those objects.
        """
        if self.result_content == ResultContent.COLUMNS:
            tables = sqlalchemy.inspect(dao_class).selectable
            statement = sqlalchemy.select(tables).select_from(tables)
        else:
            statement = sqlalchemy.select(dao_class).options(
                *self.build_loader_options(loaded_classes=[dao_class])
            )
        if self.selected_object_restriction == SelectedObjectRestriction.BOUND_IDS:
            for batch in batched(ids, self.ids_per_statement):
                yield statement.where(dao_class.database_id.in_(batch))
            return
        selected = selection.subquery()
        yield statement.join(
            selected,
            selected.c[StoredObjectColumn.DATABASE_ID] == dao_class.database_id,
        ).where(
            selected.c[StoredObjectColumn.POLYMORPHIC_TYPE] == dao_class.__tablename__
        )

    def _execute(self, *, session: Session, statement: Select) -> Iterator[Any]:
        """
        :param session: The session to execute with.
        :param statement: A statement :meth:`_build_loading_statements` made.
        :return: What the statement loads: data access objects or rows.
        """
        if self.result_content == ResultContent.COLUMNS:
            return iter(session.execute(statement))
        return iter(session.scalars(statement))
