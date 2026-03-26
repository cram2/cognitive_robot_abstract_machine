from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass, field, is_dataclass, fields, MISSING
from inspect import isclass
from typing import Any, Set, Dict, Tuple, Type, List, TYPE_CHECKING, Optional, Union

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
    Cache for synthetic parent DAOs to maintain identity across discovery and filling phases.
    Synthentic DAOs are used when the parent of a DAO uses and AlternativeMapping.
    In this case the, the parent has to be converted using its specialized routine. After that, the child can copy
    its inherited fields from the parent.
    """

    _class_dependencies: rustworkx.PyDiGraph = field(
        default_factory=lambda: rustworkx.PyDiGraph(multigraph=False)
    )
    """
    A rustowkrx graph that tracks the dependencies between classes defined 
    in `AlternativeMapping.required_pre_build_classes`
    The nodes are the data access object types and the edges represent the dependencies.
    An edge (source, target) means that the class `source` needs to be build before `target`.
    """

    _alternative_mappings_being_referenced: Dict[
        AlternativeMapping, List[Tuple[Any, MappedVariable]]
    ] = field(default_factory=lambda: defaultdict(list))

    """
    A dictionary that maps remembers all occurrences of an alternative mapping in any column or relationship of a 
    domain object. This is filled during the `_fill_domain_objects` phase in the `_populate_relationship` 
    method.
    The keys are the ids of the instances of the alternative mappings and the values are descriptions of how they are 
    referenced. The descriptions are MappedVariable instances from EQL.
    """

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
        if is_dataclass(original_clazz):
            for f in fields(original_clazz):
                if f.default is not MISSING:
                    object.__setattr__(result, f.name, f.default)
                elif f.default_factory is not MISSING:
                    object.__setattr__(result, f.name, f.default_factory())
        self.register(dao_instance, result)
        return result

    def _build_class_dependencies(
        self, alternative_mapping_types: List[Type[AlternativeMapping]]
    ):
        """
        Build the class dependencies for the given types that can be used to infer the built order.
        This method should only take Alternative Mapping types as input as these are the only types that can have
        order sensitive dependencies.

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

        :param alternative_mapping: The alternative mapping for which dependencies
            are being resolved.
        :param concrete_alternative_mappings: A list of Alternative Mapping types discovered during the discovery phase.
        :param types_to_index: A dictionary mapping Alternative Mapping types to their respective
            indices in the dependency graph.
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
        Convert all alternative mappings registered in `_alternative_mappings_being_referenced` to domain objects.
        Update all the references of other domain objects to the newly created domain objects.
        This uses the order from `_order_work_items_by_dependency_graph` to ensure that the alternative mappings are
        respecting their dependencies.
        """

        for type_index in rustworkx.topological_sort(self._class_dependencies):
            alternative_mapping_type = self._class_dependencies[type_index]
            for (
                alternative_mapping_instance,
                references,
            ) in self._alternative_mappings_being_referenced.items():

                if type(alternative_mapping_instance) is not alternative_mapping_type:
                    continue

                domain_object = alternative_mapping_instance.to_domain_object()
                for referencing_instance, reference in references:

                    reference._set_external_root_instance_value_(
                        referencing_instance, domain_object
                    )
