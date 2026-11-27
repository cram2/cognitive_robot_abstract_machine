"""
The conversion of domain objects to the DAOs that map them.
"""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING, Any

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.query.match import AbstractMatchExpression
from krrood.ormatic.data_access_objects.association_rows import AssociationRowWriter
from krrood.ormatic.data_access_objects.base import DomainObject
from krrood.ormatic.data_access_objects.helper import get_data_access_object_class
from krrood.ormatic.data_access_objects.relationships import (
    CollectionRelationship,
    SingleRelationship,
)
from krrood.ormatic.data_access_objects.schema import (
    get_alternative_partition_plan,
    get_dao_schema,
)
from krrood.ormatic.data_access_objects.stored_object_row import StoredObjectRow
from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState
from krrood.ormatic.exceptions import (
    NoDAOFoundDuringParsingError,
    QueryCannotBePersisted,
)

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import (
        DataAccessObject,
        PartDataAccessObject,
    )


# %% conversion to DAOs


class ToDataAccessObjectConversion:
    """
    The conversion of a domain object to its DAO, which traverses the object graph
    breadth first with a queue of work items, converting every reachable object.
    """

    @classmethod
    def to_dao(
        cls,
        *,
        source_object: DomainObject,
        state: ToDataAccessObjectState | None = None,
        register: bool = True,
    ) -> DataAccessObject:
        """
        Convert an object to its Data Access Object.

        :param source_object: The object to convert.
        :param state: The conversion state.
        :param register: Whether to register the result in the memo.
        :return: The converted DAO instance.
        :raises QueryCannotBePersisted: If the object is an entity query language query.
        """
        cls._refuse_queries(source_object=source_object)
        state = state or ToDataAccessObjectState()

        # Phase 1: Resolution - Check memo and apply alternative mappings
        existing = state.get(source=source_object)
        if existing is not None:
            return existing

        resolved_source = state.apply_alternative_mapping_if_needed(
            dao_clazz=cls, source_object=source_object
        )

        # Phase 2: Allocation & Registration
        result = cls()

        if register:
            state.register(source=source_object, target=result)
            if id(source_object) != id(resolved_source):
                state.register(source=resolved_source, target=result)

        # Phase 3: Queueing & Processing
        is_entry_call = len(state.work_items) == 0
        alternative_base = get_dao_schema(cls).alternative_base
        state.push_work_item(
            source_object=resolved_source,
            dao_instance=result,
            alternative_base=alternative_base,
        )

        if is_entry_call:
            cls._process_to_dao_queue(state=state)

        return result

    @staticmethod
    def _refuse_queries(*, source_object: Any) -> None:
        """
        :param source_object: An object that is to be stored.
        :raises QueryCannotBePersisted: If the object is an entity query language query.
        """
        if isinstance(source_object, (AbstractMatchExpression, SymbolicExpression)):
            raise QueryCannotBePersisted(query=source_object)

    @classmethod
    def _process_to_dao_queue(cls, *, state: ToDataAccessObjectState) -> None:
        """
        Process the work items for converting objects to DAOs.

        This uses a Breadth-First Search (BFS) approach by processing the deque as a
        FIFO queue (popleft). New work items for nested relationships are added to the
        queue during processing.

        :param state: The conversion state containing the work_items.
        """
        while state.work_items:
            work_item = state.work_items.popleft()
            if work_item.alternative_base is not None:
                work_item.dao_instance.fill_dao_if_subclass_of_alternative_mapping(
                    source_object=work_item.source_object,
                    alternative_base=work_item.alternative_base,
                    state=state,
                )
            else:
                work_item.dao_instance.fill_dao_default(
                    source_object=work_item.source_object, state=state
                )

    def fill_dao_default(
        self, *, source_object: DomainObject, state: ToDataAccessObjectState
    ) -> None:
        """
        Populate the DAO instance from a source object.

        :param source_object: The source object.
        :param state: The conversion state.
        """
        plan = get_dao_schema(type(self))

        for name in plan.data_column_names:
            setattr(self, name, getattr(source_object, name))
        self._fill_relationships_from_plan(
            source_object=source_object,
            single_relationships=plan.single_relationships,
            collection_relationships=plan.collection_relationships,
            state=state,
        )
        self._fill_parts(
            source_object=source_object, part_classes=plan.part_classes, state=state
        )

    def _fill_parts(
        self,
        *,
        source_object: DomainObject,
        part_classes: tuple[type[PartDataAccessObject], ...],
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Give this DAO the parts holding its object's rows in the tables of its secondary
        bases, filled from the source object.

        :param source_object: The source object.
        :param part_classes: The classes of the parts to fill.
        :param state: The conversion state.
        """
        self.parts_in_memory = {}
        for part_class in part_classes:
            part_dao = part_class()
            type_column_key = part_class.type_column_key()
            if type_column_key is not None:
                setattr(part_dao, type_column_key, type(self).__tablename__)
            part_dao.fill_dao_default(source_object=source_object, state=state)
            self.parts_in_memory[part_class] = part_dao

    def fill_dao_if_subclass_of_alternative_mapping(
        self,
        *,
        source_object: DomainObject,
        alternative_base: type[DataAccessObject],
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Populate the DAO instance for an alternatively mapped subclass.

        :param source_object: The source object.
        :param alternative_base: The base class using alternative mapping.
        :param state: The conversion state.
        """
        # Temporarily remove the object from the memo to allow the parent DAO to be created separately
        temp_dao = state.pop(source=source_object)

        # create dao of alternatively mapped superclass
        parent_dao = alternative_base.original_class().to_dao(
            source_object=source_object, state=state
        )

        # Restore the object in the memo dictionary
        if temp_dao is not None:
            state.register(source=source_object, target=temp_dao)

        partition = get_alternative_partition_plan(type(self), alternative_base)

        # Copy values from parent DAO and original object
        for name in partition.parent_data_column_names:
            setattr(self, name, getattr(parent_dao, name))
        for name in partition.own_data_column_names:
            setattr(self, name, getattr(source_object, name))

        # Ensure columns on intermediate ancestors are also covered
        for key in partition.intermediate_attribute_keys:
            setattr(self, key, getattr(source_object, key))

        # Fill the partitioned relationships
        self._fill_relationships_from_plan(
            source_object=parent_dao,
            single_relationships=partition.parent_single_relationships,
            collection_relationships=partition.parent_collection_relationships,
            state=state,
        )
        self._fill_relationships_from_plan(
            source_object=source_object,
            single_relationships=partition.own_single_relationships,
            collection_relationships=partition.own_collection_relationships,
            state=state,
        )
        self._fill_parts(
            source_object=source_object,
            part_classes=get_dao_schema(type(self)).part_classes,
            state=state,
        )

    def _fill_relationships_from_plan(
        self,
        *,
        source_object: Any,
        single_relationships: tuple[SingleRelationship, ...],
        collection_relationships: tuple[CollectionRelationship, ...],
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Populate relationships from a source object using a conversion plan.

        :param source_object: The source of relationship values.
        :param single_relationships: The single-valued relationship entries.
        :param collection_relationships: The collection relationship entries.
        :param state: The conversion state.
        """
        for single_relationship in single_relationships:
            self._fill_single_relationship(
                source_object=source_object,
                relationship=single_relationship,
                state=state,
            )

        for collection_relationship in collection_relationships:
            self._fill_collection_relationship(
                source_object=source_object,
                relationship=collection_relationship,
                state=state,
            )

    def _fill_single_relationship(
        self,
        *,
        source_object: Any,
        relationship: SingleRelationship,
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Populate a single-valued relationship attribute from a source object.

        :param source_object: The source of the relationship value.
        :param relationship: The single-valued relationship entry.
        :param state: The conversion state.
        """
        value = getattr(source_object, relationship.key)
        if value is None:
            setattr(self, relationship.key, None)
        else:
            setattr(
                self,
                relationship.key,
                self.get_or_queue_referenced_dao(
                    source_object=value,
                    state=state,
                    expected_type=relationship.domain_type,
                    target_dao_class=relationship.target_dao_class,
                ),
            )

    def _fill_collection_relationship(
        self,
        *,
        source_object: DomainObject,
        relationship: CollectionRelationship,
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Populate a collection-valued relationship attribute from a source object.

        :param source_object: The source of the relationship values.
        :param relationship: The collection-valued relationship entry.
        :param state: The conversion state.
        """
        source_collection = getattr(source_object, relationship.key)

        if relationship.association_class is not None:
            writer = AssociationRowWriter(
                relationship=relationship, holder=self, state=state
            )
            setattr(
                self, relationship.key, writer.write_rows(container=source_collection)
            )
            return

        data_access_objects = [
            self.get_or_queue_referenced_dao(
                source_object=item,
                state=state,
                expected_type=relationship.domain_type,
                target_dao_class=relationship.target_dao_class,
            )
            for item in source_collection
        ]

        assignable_container_type = (
            list if type(source_collection) is tuple else type(source_collection)
        )
        setattr(self, relationship.key, assignable_container_type(data_access_objects))

    def get_or_queue_referenced_dao(
        self,
        *,
        source_object: Any,
        state: ToDataAccessObjectState,
        expected_type: type,
        target_dao_class: type | None,
    ) -> DataAccessObject:
        """
        Resolve an object a relationship refers to into the DAO the relationship holds,
        queuing the object's own DAO if necessary.

        :param source_object: The object referred to.
        :param state: The conversion state.
        :param expected_type: The domain type of the field being filled.
        :param target_dao_class: The DAO class the relationship points to; for the table
            of every stored object, the relationship holds the object's row in it.
        :return: The DAO the relationship holds.
        """
        dao = self._get_or_queue_dao(
            source_object=source_object, state=state, expected_type=expected_type
        )
        if target_dao_class is not None and issubclass(
            target_dao_class, StoredObjectRow
        ):
            return dao.stored_object
        return dao

    def _get_or_queue_dao(
        self,
        *,
        source_object: Any,
        state: ToDataAccessObjectState,
        expected_type: type | None = None,
    ) -> DataAccessObject:
        """
        Resolve a source object to a DAO, queuing it if necessary.

        The parametrized alias an object was constructed through takes precedence over
        *expected_type*, so a shared object resolves to the same DAO class whichever
        field reaches it.

        :param source_object: The object to resolve.
        :param state: The conversion state.
        :param expected_type: The expected domain type of the field being filled.
        :return: The corresponding DAO instance.
        :raises QueryCannotBePersisted: If the object is an entity query language query.
        """
        self._refuse_queries(source_object=source_object)
        expected_type = getattr(source_object, "__orig_class__", None) or expected_type

        # Check if already built
        existing = state.get(source=source_object)
        if existing is not None:
            return existing

        dao_clazz = get_data_access_object_class(type(source_object), expected_type)
        if dao_clazz is None:
            raise NoDAOFoundDuringParsingError(
                obj=source_object, dao=type(self), relationship=None
            )

        # Check for alternative mapping
        mapped_object = state.apply_alternative_mapping_if_needed(
            dao_clazz=dao_clazz, source_object=source_object
        )
        if isinstance(mapped_object, dao_clazz):
            state.register(source=source_object, target=mapped_object)
            return mapped_object

        # Create new DAO instance
        result = dao_clazz()
        state.register(source=source_object, target=result)
        if id(source_object) != id(mapped_object):
            state.register(source=mapped_object, target=result)

        # Queue for filling
        alternative_base = get_dao_schema(dao_clazz).alternative_base
        state.push_work_item(
            source_object=mapped_object,
            dao_instance=result,
            alternative_base=alternative_base,
        )

        return result
