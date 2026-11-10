from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, is_dataclass, fields, MISSING
from functools import lru_cache
from inspect import isclass
from typing_extensions import Any, Callable, TYPE_CHECKING, Optional, Union

import rustworkx

from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.ormatic.data_access_objects.base import (
    DataAccessObjectWorkItem,
    DataAccessObjectState,
    InstanceDict,
)

from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.conversion_order import (
    ConversionOrder,
    DeclaredOrder,
)
from krrood.ormatic.data_access_objects.helper import get_data_access_object_class

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import (
        DataAccessObject,
    )


@dataclass(frozen=True)
class _FieldDefault(ABC):
    """
    The default of one field of a domain class, written when the class is allocated.
    """

    name: str
    """
    Name of the dataclass field.
    """

    @abstractmethod
    def produce(self) -> Any:
        """
        :return: The value to assign to the field.
        """


@dataclass(frozen=True)
class _StaticDefault(_FieldDefault):
    """
    A field default that is a fixed value.
    """

    value: Any
    """
    The literal default value to assign.
    """

    def produce(self) -> Any:
        return self.value


@dataclass(frozen=True)
class _DefaultFactory(_FieldDefault):
    """
    A field default produced by a zero-argument factory.
    """

    factory: Callable[[], Any]
    """
    Callable that produces a fresh default value when invoked.
    """

    def produce(self) -> Any:
        return self.factory()


@lru_cache(maxsize=None)
def _get_field_defaults(original_clazz: type) -> tuple[_FieldDefault, ...]:
    """
    Precompute the default values of a domain class for allocation.

    :param original_clazz: The domain class.
    :return: The default of each field that has one, in the order the fields are
        declared, which is the order the class's constructor assigns them in.
    """
    if not is_dataclass(original_clazz):
        return ()

    defaults = []
    for field_ in fields(original_clazz):
        if field_.default is not MISSING:
            defaults.append(_StaticDefault(field_.name, field_.default))
        elif field_.default_factory is not MISSING:
            defaults.append(_DefaultFactory(field_.name, field_.default_factory))
    return tuple(defaults)


@dataclass
class FromDataAccessObjectWorkItem(DataAccessObjectWorkItem):
    """
    Work item for converting a Data Access Object back to a domain object.
    """

    domain_object: Any


@dataclass
class ObjectInheritingFromAlternativeMapping:
    """
    A domain object whose class inherits from an alternatively mapped class, and so
    takes the attributes it inherits from the domain object of that class's mapping.
    """

    domain_object: Any
    """
    The object whose inherited attributes are set.
    """

    parent: AlternativeMapping
    """
    The mapping of the alternatively mapped base class, built from the object's row.
    """

    own_attribute_keys: tuple[str, ...]
    """
    The attributes the object's own columns and relationships set, which the parent does
    not override.
    """

    def copy_inherited_attributes(self, parent_domain_object: Any) -> None:
        """
        Set every field of the object that the parent's domain object has and the
        object's own columns and relationships do not set.

        :param parent_domain_object: The domain object the parent mapping converted to.
        """
        for field_ in fields(self.domain_object):
            if field_.name in self.own_attribute_keys:
                continue
            if not hasattr(parent_domain_object, field_.name):
                continue
            object.__setattr__(
                self.domain_object,
                field_.name,
                getattr(parent_domain_object, field_.name),
            )


@dataclass
class FromDataAccessObjectState(DataAccessObjectState[FromDataAccessObjectWorkItem]):
    """
    State for converting Data Access Objects back to domain objects.
    """

    discovery_mode: bool = False
    """
    Whether the state is currently in discovery mode.
    """

    initialized_ids: set[int] = field(default_factory=set)
    """
    Set of DAO ids that have been fully initialized.
    """

    is_processing: bool = False
    """
    Whether the state is currently in the processing loop.
    """

    synthetic_parent_daos: dict[
        tuple[int, type[DataAccessObject]], DataAccessObject
    ] = field(default_factory=dict)
    """
    Cache for synthetic parent DAOs to maintain identity across discovery and filling
    phases.

    Synthentic DAOs are used when the parent of a DAO uses and AlternativeMapping. In
    this case the, the parent has to be converted using its specialized routine. After
    that, the child can copy its inherited fields from the parent.
    """

    objects_inheriting_from_alternative_mappings: list[
        ObjectInheritingFromAlternativeMapping
    ] = field(default_factory=list)
    """
    The objects of this conversion whose classes inherit from an alternatively mapped
    class, waiting for their parent mappings to be converted.
    """

    _class_dependencies: rustworkx.PyDiGraph = field(
        default_factory=lambda: rustworkx.PyDiGraph(multigraph=False)
    )
    """
    A rustowkrx graph that tracks the dependencies between classes defined in
    `AlternativeMapping.required_pre_build_classes` The nodes are the data access object
    types and the edges represent the dependencies.

    An edge (source, target) means that the class `source` needs to be build before
    `target`.
    """

    _alternative_mappings_being_referenced: dict[
        AlternativeMapping, list[tuple[Any, MappedVariable]]
    ] = field(default_factory=lambda: defaultdict(list))
    """
    A dictionary that maps remembers all occurrences of an alternative mapping in any
    column or relationship of a domain object.

    This is filled during the `_fill_domain_objects` phase in the
    `_populate_relationship` method. The keys are the ids of the instances of the
    alternative mappings and the values are descriptions of how they are referenced. The
    descriptions are MappedVariable instances from EQL.
    """

    _alternative_mappings_being_resolved: set[AlternativeMapping] = field(
        default_factory=set
    )
    """
    The alternative mappings whose conversion has started and not finished, which a
    reference cycle leads back to.
    """

    _converted_alternative_mappings: dict[AlternativeMapping, Any] = field(
        default_factory=dict
    )
    """
    Cache of the final domain objects created from alternative mapping instances.

    This guarantees that `to_domain_object` is called exactly once per alternative
    mapping instance, keeping object identity intact when the same mapping is referenced
    from multiple places (including the root of the conversion).
    """

    def reset_conversion_tracking(self):
        """
        Reset the tracking structures that are only valid for a single top-level
        conversion.

        This has to be called at the start of every top-level conversion, otherwise
        repeated conversions with a shared state accumulate duplicated dependency graph
        nodes and stale alternative mapping references.
        """
        self._class_dependencies = rustworkx.PyDiGraph(multigraph=False)
        self._alternative_mappings_being_referenced = defaultdict(list)

    def resolve_alternative_mapping(
        self, alternative_mapping: AlternativeMapping
    ) -> Any:
        """
        Convert an alternative mapping to its final domain object exactly once per
        instance, after the mappings of the same class its fields hold directly, so that
        its :meth:`~AlternativeMapping.to_domain_object` sees their domain objects. The
        class dependencies order mappings of different classes.

        :param alternative_mapping: The alternative mapping instance to convert.
        :return: The final domain object.
        """
        if alternative_mapping in self._converted_alternative_mappings:
            return self._converted_alternative_mappings[alternative_mapping]

        self._alternative_mappings_being_resolved.add(alternative_mapping)
        for field_ in fields(alternative_mapping):
            value = getattr(alternative_mapping, field_.name)
            if (
                type(value) is type(alternative_mapping)
                and value not in self._alternative_mappings_being_resolved
            ):
                object.__setattr__(
                    alternative_mapping,
                    field_.name,
                    self.resolve_alternative_mapping(value),
                )
        domain_object = alternative_mapping.to_domain_object()
        self._alternative_mappings_being_resolved.discard(alternative_mapping)

        self._converted_alternative_mappings[alternative_mapping] = domain_object
        return domain_object

    def resolve_result(self, result: Any) -> Any:
        """
        Resolve a memoized conversion result to its final domain object.

        :param result: The memoized result, possibly an alternative mapping instance.
        :return: The final domain object.
        """
        if isinstance(result, AlternativeMapping):
            return self.resolve_alternative_mapping(result)
        return result

    def is_initialized(self, dao_instance: DataAccessObject) -> bool:
        """
        Check if the given DAO instance has been fully initialized.

        :param dao_instance: The DAO instance to check.
        :return: True if fully initialized.
        """
        return id(dao_instance) in self.initialized_ids

    def mark_initialized(self, dao_instance: DataAccessObject):
        """
        Mark the given DAO instance as fully initialized.

        :param dao_instance: The DAO instance to mark.
        """
        self.initialized_ids.add(id(dao_instance))

    def push_work_item(self, dao_instance: DataAccessObject, domain_object: Any):
        """
        Add a new work item to the processing queue.

        :param dao_instance: The DAO instance being converted.
        :param domain_object: The domain object being populated.
        """
        self.work_items.append(
            FromDataAccessObjectWorkItem(
                dao_instance=dao_instance, domain_object=domain_object
            )
        )

    def allocate_and_memoize(
        self, dao_instance: DataAccessObject, original_clazz: type
    ) -> Any:
        """
        Allocate a new instance and store it in the memoization dictionary.

        Initializes default values for dataclass fields.

        :param dao_instance: The DAO instance to register.
        :param original_clazz: The domain class to instantiate.
        :return: The uninitialized domain object instance.
        """
        result = original_clazz.__new__(original_clazz)
        for field_default in _get_field_defaults(original_clazz):
            object.__setattr__(result, field_default.name, field_default.produce())
        self.register(dao_instance, result)
        return result

    def _build_class_dependencies(
        self, alternative_mapping_types: list[type[AlternativeMapping]]
    ):
        """
        Build the class dependencies for the given types that can be used to infer the
        built order. This method should only take Alternative Mapping types as input as
        these are the only types that can have order sensitive dependencies.

        :param alternative_mapping_types: The types to build the dependency graph for.
        """
        types_to_index: dict[type, int] = {
            type_: self._class_dependencies.add_node(type_)
            for type_ in alternative_mapping_types
        }  # add all dao types to the dependency graph

        # add all dependencies between the classes defined from the alternative mappings
        for alternative_mapping_type in alternative_mapping_types:
            self._build_dependencies_of_alternative_mapping(
                alternative_mapping_type, alternative_mapping_types, types_to_index
            )

    def _build_dependencies_of_alternative_mapping(
        self,
        alternative_mapping: type[AlternativeMapping],
        concrete_alternative_mappings: list[type[AlternativeMapping]],
        types_to_index: dict[type, int],
    ):
        """
        Builds the dependencies of a given alternative mapping and updates the internal
        class dependency graph.

        :param alternative_mapping: The alternative mapping for which dependencies are
            being resolved.
        :param concrete_alternative_mappings: A list of Alternative Mapping types
            discovered during the discovery phase.
        :param types_to_index: A dictionary mapping Alternative Mapping types to their
            respective indices in the dependency graph.
        """
        # get all concrete types that are affected by the dependencies
        for required_domain_type in alternative_mapping.required_pre_build_classes():
            # for every concrete dao type discovered in the discovery phase
            for concrete_alternative_mapping in concrete_alternative_mappings:
                # get the concrete domain type of the dao current dao type
                concrete_domain_type = concrete_alternative_mapping.original_class()

                if not isclass(
                    concrete_domain_type
                ):  # skip non classes (like generics)
                    continue

                # skip types that are not required
                if not issubclass(concrete_domain_type, required_domain_type):
                    continue

                # add the dependency
                self._class_dependencies.add_edge(
                    types_to_index[concrete_alternative_mapping],
                    types_to_index[alternative_mapping],
                    DeclaredOrder(concrete_alternative_mapping, alternative_mapping),
                )

    def convert_alternative_mappings_to_domain_objects(self):
        """
        Convert all alternative mappings registered in
        `_alternative_mappings_being_referenced` to domain objects.

        Update all the references of other domain objects to the newly created domain
        objects. The class dependency graph determines the conversion order so that the
        alternative mappings respect their dependencies.
        """
        instances_by_type: dict[type, list[AlternativeMapping]] = defaultdict(list)
        for instance in self._alternative_mappings_being_referenced:
            instances_by_type[type(instance)].append(instance)

        # types in conversion order first, then any referenced types not in the graph
        # (e.g. instances that were already converted in a previous conversion)
        ordered_types = ConversionOrder(
            self._class_dependencies, self._alternative_mappings_being_referenced
        ).sort_types()
        ordered_types += [
            type_ for type_ in instances_by_type if type_ not in set(ordered_types)
        ]

        for alternative_mapping_type in ordered_types:
            for instance in instances_by_type[alternative_mapping_type]:
                domain_object = self.resolve_alternative_mapping(instance)
                for (
                    referencing_instance,
                    reference,
                ) in self._alternative_mappings_being_referenced[instance]:
                    reference._set_external_root_instance_value_(
                        referencing_instance, domain_object
                    )

    def copy_attributes_inherited_from_alternative_mappings(self) -> None:
        """
        Convert the parent mapping of every object whose class inherits from an
        alternatively mapped class, and copy the inherited attributes onto the object.
        """
        for inheriting_object in self.objects_inheriting_from_alternative_mappings:
            inheriting_object.copy_inherited_attributes(
                self.resolve_alternative_mapping(inheriting_object.parent)
            )
        self.objects_inheriting_from_alternative_mappings.clear()
