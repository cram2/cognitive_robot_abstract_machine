"""
The conversion of DAOs back to the domain objects they map.
"""

from __future__ import annotations

from collections.abc import Iterable

from typing_extensions import TYPE_CHECKING, Any

from krrood.entity_query_language.core.mapped_variable import Attribute, IndexByValue
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.association_rows import AssociationRowReader
from krrood.ormatic.data_access_objects.base import DomainObject
from krrood.ormatic.data_access_objects.domain_object_finishing import (
    DomainObjectFinisher,
)
from krrood.ormatic.data_access_objects.from_dao import (
    FromDataAccessObjectState,
    FromDataAccessObjectWorkItem,
    ObjectInheritingFromAlternativeMapping,
)
from krrood.ormatic.data_access_objects.relationship_loading import (
    read_single_relationship,
)
from krrood.ormatic.data_access_objects.relationships import (
    AssociationDataAccessObject,
    CollectionRelationship,
    SingleRelationship,
)
from krrood.ormatic.data_access_objects.schema import (
    DataAccessObjectSchema,
    get_alternative_partition_plan,
    get_dao_schema,
)

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import (
        DataAccessObject,
        PartDataAccessObject,
    )


# %% conversion from DAOs


class FromDataAccessObjectConversion:
    """
    The conversion of a DAO back to its domain object, in four phases: discovering every
    reachable DAO and allocating its domain object, populating the domain objects,
    finishing their containers, and calling their ``__post_init__``.
    """

    def from_dao(
        self,
        *,
        state: FromDataAccessObjectState | None = None,
    ) -> DomainObject:
        """
        Convert the DAO back into a domain object instance.

        :param state: The conversion state.
        :return: The converted domain object.
        """
        state = state or FromDataAccessObjectState()

        if state.has(source=self) and state.is_initialized(dao_instance=self):
            return state.resolve_result(result=state.get(source=self))

        if not state.is_processing:
            result = self._perform_from_dao_conversion(state=state)

            # if the instance that started this whole process is alternatively mapped, finally convert it
            return state.resolve_result(result=result)

        return self._register_for_conversion(state=state)

    def _perform_from_dao_conversion(
        self, *, state: FromDataAccessObjectState
    ) -> DomainObject:
        """
        Perform the four-phase conversion process.

        :param state: The conversion state.
        :return: The converted domain object.
        """
        state.is_processing = True
        state.reset_conversion_tracking()
        discovery_order = []

        if not state.has(source=self):
            state.allocate_and_memoize(
                dao_instance=self, original_clazz=self.constructable_original_class()
            )
        state.push_work_item(dao_instance=self, domain_object=state.get(source=self))

        self._discover_dependencies(state=state, discovery_order=discovery_order)
        self._fill_domain_objects(state=state, discovery_order=discovery_order)
        state.convert_alternative_mappings_to_domain_objects()
        state.copy_attributes_inherited_from_alternative_mappings()
        finisher = DomainObjectFinisher(state=state, discovery_order=discovery_order)
        finisher.finish_containers()
        finisher.call_post_inits()
        finisher.finish_sorting_containers()

        for work_item in discovery_order:
            state.mark_initialized(dao_instance=work_item.dao_instance)

        state.is_processing = False
        return state.get(source=self)

    def _discover_dependencies(
        self,
        *,
        state: FromDataAccessObjectState,
        discovery_order: list[FromDataAccessObjectWorkItem],
    ) -> None:
        """
        Phase 1: Discovery (DFS) to identify all reachable DAOs.

        :param state: The conversion state.
        :param discovery_order: List to record the discovery order.
        """
        state.discovery_mode = True
        collected_types = set()

        while state.work_items:
            # Use pop() to treat the deque as a stack (LIFO) for DFS
            work_item = state.work_items.pop()
            discovery_order.append(work_item)
            if isinstance(work_item.domain_object, AlternativeMapping):
                collected_types.add(type(work_item.domain_object))
            work_item.dao_instance._fill_from_dao(
                domain_object=work_item.domain_object, state=state
            )

        state._build_class_dependencies(alternative_mapping_types=list(collected_types))
        state.discovery_mode = False

    def _fill_domain_objects(
        self,
        *,
        state: FromDataAccessObjectState,
        discovery_order: list[FromDataAccessObjectWorkItem],
    ) -> None:
        """
        Phase 2: Filling (Bottom-Up) to initialize domain objects.

        Populate all relationships and scalars for all discovered instances. This
        ensures that all objects point to each other (even if not yet fully resolved).

        :param state: The conversion state.
        :param discovery_order: The order in which to process the instances.
        """
        for work_item in discovery_order:
            if not state.is_initialized(dao_instance=work_item.dao_instance):
                work_item.dao_instance._populate_relationships_and_scalars_from_dao(
                    domain_object=work_item.domain_object, state=state
                )

    def _register_for_conversion(
        self, *, state: FromDataAccessObjectState
    ) -> DomainObject:
        """
        Register this DAO for conversion if not already present.

        :param state: The conversion state.
        :return: The uninitialized domain object.
        """
        if not state.has(source=self):
            domain_object = state.allocate_and_memoize(
                dao_instance=self, original_clazz=self.constructable_original_class()
            )
            state.push_work_item(dao_instance=self, domain_object=domain_object)
        return state.get(source=self)

    def _populate_relationships_and_scalars_from_dao(
        self, *, domain_object: DomainObject, state: FromDataAccessObjectState
    ) -> None:
        """
        Populate the relationships and scalar columns of the domain object.

        :param domain_object: The domain object.
        :param state: The conversion state.
        """
        plan = get_dao_schema(type(self))
        for part_dao, part_plan in self._parts_and_their_plans(plan=plan):
            part_dao._populate_from_columns_and_relationships(
                domain_object=domain_object,
                data_column_names=part_plan.data_column_names,
                single_relationships=part_plan.single_relationships,
                collection_relationships=part_plan.collection_relationships,
                state=state,
            )
        alternative_parent = self._alternative_parent(state=state)
        if alternative_parent is None:
            self._populate_from_columns_and_relationships(
                domain_object=domain_object,
                data_column_names=plan.data_column_names,
                single_relationships=plan.single_relationships,
                collection_relationships=plan.collection_relationships,
                state=state,
            )
            return

        partition = get_alternative_partition_plan(type(self), plan.alternative_base)
        self._populate_from_columns_and_relationships(
            domain_object=domain_object,
            data_column_names=partition.intermediate_attribute_keys,
            single_relationships=partition.own_single_relationships,
            collection_relationships=partition.own_collection_relationships,
            state=state,
        )
        own_attribute_keys = (
            *partition.intermediate_attribute_keys,
            *(entry.key for entry in partition.own_single_relationships),
            *(entry.key for entry in partition.own_collection_relationships),
        )
        state.objects_inheriting_from_alternative_mappings.append(
            ObjectInheritingFromAlternativeMapping(
                domain_object=domain_object,
                parent=alternative_parent,
                own_attribute_keys=own_attribute_keys,
            )
        )

    def _populate_from_columns_and_relationships(
        self,
        *,
        domain_object: DomainObject,
        data_column_names: Iterable[str],
        single_relationships: Iterable[SingleRelationship],
        collection_relationships: Iterable[CollectionRelationship],
        state: FromDataAccessObjectState,
    ) -> None:
        """
        Set the given columns and relationships of this DAO on the domain object.

        :param domain_object: The domain object.
        :param data_column_names: The names of the scalar columns to set.
        :param single_relationships: The single valued relationships to set.
        :param collection_relationships: The collection valued relationships to set.
        :param state: The conversion state.
        """
        for name in data_column_names:
            object.__setattr__(domain_object, name, getattr(self, name))

        for relationship in single_relationships:
            self._populate_single_relationship(
                domain_object=domain_object,
                key=relationship.key,
                value=read_single_relationship(self, relationship),
                state=state,
            )
        for relationship in collection_relationships:
            self._populate_collection_relationship(
                domain_object=domain_object, relationship=relationship, state=state
            )

    def _fill_from_dao(
        self, *, domain_object: DomainObject, state: FromDataAccessObjectState
    ) -> DomainObject:
        """
        Populate the domain object with data from the DAO.

        :param domain_object: The domain object to populate.
        :param state: The conversion state.
        :return: The populated domain object.
        """
        if state.discovery_mode:
            return self._trigger_discovery(domain_object=domain_object, state=state)

        # Fallback for when _fill_from_dao is called directly (not during Phase 1)
        self._populate_relationships_and_scalars_from_dao(
            domain_object=domain_object, state=state
        )
        return domain_object

    def _trigger_discovery(
        self,
        *,
        domain_object: DomainObject,
        state: FromDataAccessObjectState,
    ) -> DomainObject:
        """
        Trigger discovery of dependencies without fully populating the object.

        :param domain_object: The domain object.
        :param state: The conversion state.
        :return: The domain object.
        """
        plan = get_dao_schema(type(self))
        self._discover_related_objects(plan=plan, state=state)
        for part_dao, part_plan in self._parts_and_their_plans(plan=plan):
            part_dao._discover_related_objects(plan=part_plan, state=state)

        self._alternative_parent(state=state)
        return domain_object

    def _parts_and_their_plans(
        self, *, plan: DataAccessObjectSchema
    ) -> list[tuple[PartDataAccessObject, DataAccessObjectSchema]]:
        """
        :param plan: The conversion plan of this DAO's class.
        :return: The parts this DAO holds, each with the conversion plan of its class.
        """
        return [
            (part_dao, get_dao_schema(part_class))
            for part_class in plan.part_classes
            if (part_dao := self.part(part_class=part_class)) is not None
        ]

    def _discover_related_objects(
        self, *, plan: DataAccessObjectSchema, state: FromDataAccessObjectState
    ) -> None:
        """
        Register every object this DAO's relationships refer to for conversion.

        :param plan: The conversion plan of this DAO's class.
        :param state: The conversion state.
        """
        for relationship in plan.single_relationships:
            value = read_single_relationship(self, relationship)
            if value is not None:
                value.from_dao(state=state)

        for relationship in plan.collection_relationships:
            value = getattr(self, relationship.key)
            if not value:
                continue
            if relationship.association_class is not None:
                for row in value:
                    for referenced in self._objects_of_association_row(
                        row=row, relationship=relationship
                    ):
                        referenced.from_dao(state=state)
            else:
                for item in value:
                    item.from_dao(state=state)

    def _populate_single_relationship(
        self,
        *,
        domain_object: Any,
        key: str,
        value: Any,
        state: FromDataAccessObjectState,
    ) -> None:
        """
        Populate a single-valued relationship on the domain object.

        :param domain_object: The domain object.
        :param key: The attribute name.
        :param value: The DAO instance.
        :param state: The conversion state.
        """
        if value is None:
            object.__setattr__(domain_object, key, None)
            return
        instance = self._get_or_allocate_domain_object(dao_instance=value, state=state)
        if isinstance(instance, AlternativeMapping):
            state._alternative_mappings_being_referenced[instance].append(
                (domain_object, Attribute(_attribute_name_=key, _child_=None))
            )
        object.__setattr__(domain_object, key, instance)

    @staticmethod
    def _objects_of_association_row(
        *, row: AssociationDataAccessObject, relationship: CollectionRelationship
    ) -> list[DataAccessObject]:
        """
        :param row: A row of the association table of the relationship.
        :param relationship: A collection relationship holding rows of an association
            table.
        :return: The DAOs of the mapped keys and the mapped value the row refers to.
        """
        single_relationships = [
            level.key_relationship
            for level in relationship.levels
            if level.key_relationship is not None
        ]
        if relationship.association_target_relationship is not None:
            single_relationships.append(relationship.association_target_relationship)
        referenced = [
            read_single_relationship(row, single_relationship)
            for single_relationship in single_relationships
        ]
        return [
            data_access_object
            for data_access_object in referenced
            if data_access_object is not None
        ]

    def _populate_collection_relationship(
        self,
        *,
        domain_object: DomainObject,
        relationship: CollectionRelationship,
        state: FromDataAccessObjectState,
    ) -> None:
        """
        Populate a collection relationship on the domain object.

        :param domain_object: The domain object.
        :param relationship: The collection relationship.
        :param state: The conversion state.
        """
        value = getattr(self, relationship.key)
        if value is None:
            object.__setattr__(domain_object, relationship.key, None)
            return
        if relationship.association_class is not None:
            reader = AssociationRowReader(
                relationship=relationship, holder=domain_object, state=state
            )
            object.__setattr__(
                domain_object, relationship.key, reader.read_rows(rows=value)
            )
            return

        instances = [
            self._get_or_allocate_domain_object(
                dao_instance=data_access_object, state=state
            )
            for data_access_object in value
        ]
        for index, instance in enumerate(instances):
            if isinstance(instance, AlternativeMapping):
                state._alternative_mappings_being_referenced[instance].append(
                    (
                        domain_object,
                        IndexByValue(
                            _key_=index,
                            _child_=Attribute(
                                _attribute_name_=relationship.key, _child_=None
                            ),
                        ),
                    )
                )
        # a fresh list, so the domain object does not alias the DAO's collection
        object.__setattr__(domain_object, relationship.key, list(instances))

    def _get_or_allocate_domain_object(
        self, *, dao_instance: DataAccessObject, state: FromDataAccessObjectState
    ) -> Any:
        """
        Resolve a DAO to a domain object, allocating it if necessary.

        :param dao_instance: The DAO to resolve.
        :param state: The conversion state.
        :return: The corresponding domain object.
        """
        return dao_instance.from_dao(state=state)

    def _alternative_parent(
        self, *, state: FromDataAccessObjectState
    ) -> AlternativeMapping | None:
        """
        The alternative mapping of the alternatively mapped base class that the domain
        object of this DAO inherits from, built from the base's columns of this DAO.

        :param state: The conversion state.
        :return: The alternative mapping, or ``None`` if the DAO has no alternatively
            mapped base class or is an alternative mapping itself.
        """
        plan = get_dao_schema(type(self))
        if plan.alternative_base is None or plan.uses_alternative_mapping:
            return None

        # Keyed by id(self) since a synthetic parent DAO belongs to this DAO only, and
        # cached so discovery and filling see the same parent.
        cache_key = (id(self), plan.alternative_base)
        if cache_key not in state.synthetic_parent_daos:
            state.synthetic_parent_daos[cache_key] = self._create_filled_parent_dao(
                base_clazz=plan.alternative_base
            )
        return state.synthetic_parent_daos[cache_key].from_dao(state=state)

    def _create_filled_parent_dao(
        self, *, base_clazz: type[DataAccessObject]
    ) -> DataAccessObject:
        """
        Create a parent DAO instance populated from the current DAO.

        :param base_clazz: The parent DAO class.
        :return: The populated parent DAO instance.
        """
        parent_dao = base_clazz()
        parent_plan = get_dao_schema(base_clazz)
        for name in parent_plan.data_column_names:
            setattr(parent_dao, name, getattr(self, name))
        for key in parent_plan.relationship_keys:
            setattr(parent_dao, key, getattr(self, key))
        return parent_dao
