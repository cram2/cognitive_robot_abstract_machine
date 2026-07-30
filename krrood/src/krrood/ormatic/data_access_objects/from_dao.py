from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass, field, is_dataclass, fields, MISSING
from functools import lru_cache
from inspect import isclass
from typing import (
    Any,
    Callable,
    Set,
    Dict,
    NamedTuple,
    Tuple,
    Type,
    List,
    TYPE_CHECKING,
    Optional,
    Union,
)

import rustworkx

from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.ormatic.data_access_objects.base import (
    DataAccessObjectWorkItem,
    DataAccessObjectState,
    InstanceDict,
)

from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.helper import get_dao_class

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import (
        DataAccessObject,
    )


class _StaticDefault(NamedTuple):
    """
    A field name paired with its static default value.
    """

    name: str
    """
    Name of the dataclass field.
    """

    value: Any
    """
    The literal default value to assign.
    """


class _DefaultFactory(NamedTuple):
    """
    A field name paired with its zero-argument factory callable.
    """

    name: str
    """
    Name of the dataclass field.
    """

    factory: Callable[[], Any]
    """
    Callable that produces a fresh default value when invoked.
    """


@dataclass(frozen=True)
class _AllocationPlan:
    """
    Precomputed default-value information for a single domain class.

    Separates fields with literal defaults from fields whose defaults are produced by a
    factory, so the allocation loop can handle each case without re-inspecting the
    dataclass on every call.
    """

    static_defaults: Tuple[_StaticDefault, ...]
    """
    Fields whose default is a fixed value.
    """

    default_factories: Tuple[_DefaultFactory, ...]
    """
    Fields whose default must be produced by calling a factory.
    """


@lru_cache(maxsize=None)
def _get_allocation_plan(original_clazz: Type) -> _AllocationPlan:
    """
    Precompute the default values of a domain class for allocation.

    :param original_clazz: The domain class.
    :return: Static defaults and default factories for each field with a default.
    """
    if not is_dataclass(original_clazz):
        return _AllocationPlan((), ())

    static_defaults = []
    default_factories = []
    for field_ in fields(original_clazz):
        if field_.default is not MISSING:
            static_defaults.append(_StaticDefault(field_.name, field_.default))
        elif field_.default_factory is not MISSING:
            default_factories.append(
                _DefaultFactory(field_.name, field_.default_factory)
            )
    return _AllocationPlan(tuple(static_defaults), tuple(default_factories))


@dataclass
class FromDataAccessObjectWorkItem(DataAccessObjectWorkItem):
    """
    Work item for converting a Data Access Object back to a domain object.
    """

    domain_object: Any


@dataclass
class FromDataAccessObjectState(DataAccessObjectState[FromDataAccessObjectWorkItem]):
    """
    State for converting Data Access Objects back to domain objects.
    """

    discovery_mode: bool = False
    """
    Whether the state is currently in discovery mode.
    """

    initialized_ids: Set[int] = field(default_factory=set)
    """
    Set of DAO ids that have been fully initialized.
    """

    is_processing: bool = False
    """
    Whether the state is currently in the processing loop.
    """

    synthetic_parent_daos: Dict[
        Tuple[int, Type[DataAccessObject]], DataAccessObject
    ] = field(default_factory=dict)
    """
    Cache for synthetic parent DAOs to maintain identity across discovery and filling
    phases.

    Synthentic DAOs are used when the parent of a DAO uses and AlternativeMapping. In
    this case the, the parent has to be converted using its specialized routine. After
    that, the child can copy its inherited fields from the parent.
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

    _alternative_mappings_being_referenced: Dict[
        AlternativeMapping, List[Tuple[Any, MappedVariable]]
    ] = field(default_factory=lambda: defaultdict(list))
    """
    A dictionary that maps remembers all occurrences of an alternative mapping in any
    column or relationship of a domain object.

    This is filled during the `_fill_domain_objects` phase in the
    `_populate_relationship` method. The keys are the ids of the instances of the
    alternative mappings and the values are descriptions of how they are referenced. The
    descriptions are MappedVariable instances from EQL.
    """

    _converted_alternative_mappings: Dict[AlternativeMapping, Any] = field(
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
        instance.

        :param alternative_mapping: The alternative mapping instance to convert.
        :return: The final domain object.
        """
        try:
            return self._converted_alternative_mappings[alternative_mapping]
        except KeyError:
            domain_object = alternative_mapping.to_domain_object()
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
        self, dao_instance: DataAccessObject, original_clazz: Type
    ) -> Any:
        """
        Allocate a new instance and store it in the memoization dictionary.

        Initializes default values for dataclass fields.

        :param dao_instance: The DAO instance to register.
        :param original_clazz: The domain class to instantiate.
        :return: The uninitialized domain object instance.
        """
        result = original_clazz.__new__(original_clazz)
        plan = _get_allocation_plan(original_clazz)
        for entry in plan.static_defaults:
            object.__setattr__(result, entry.name, entry.value)
        for entry in plan.default_factories:
            object.__setattr__(result, entry.name, entry.factory())
        self.register(dao_instance, result)
        return result

    def _build_class_dependencies(
        self, alternative_mapping_types: List[Type[AlternativeMapping]]
    ):
        """
        Build the class dependencies for the given types that can be used to infer the
        built order. This method should only take Alternative Mapping types as input as
        these are the only types that can have order sensitive dependencies.

        :param alternative_mapping_types: The types to build the dependency graph for.
        """
        types_to_index: Dict[Type, int] = {
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
        alternative_mapping: Type[AlternativeMapping],
        concrete_alternative_mappings: List[Type[AlternativeMapping]],
        types_to_index: Dict[Type, int],
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
                    None,
                )

    def convert_alternative_mappings_to_domain_objects(self):
        """
        Convert all alternative mappings registered in
        `_alternative_mappings_being_referenced` to domain objects.

        Update all the references of other domain objects to the newly created domain
        objects. The class dependency graph determines the conversion order so that the
        alternative mappings respect their dependencies.
        """
        instances_by_type: Dict[Type, List[AlternativeMapping]] = defaultdict(list)
        for instance in self._alternative_mappings_being_referenced:
            instances_by_type[type(instance)].append(instance)

        # types in dependency order first, then any referenced types not in the graph
        # (e.g. instances that were already converted in a previous conversion)
        ordered_types = [
            self._class_dependencies[type_index]
            for type_index in rustworkx.topological_sort(self._class_dependencies)
        ]
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
