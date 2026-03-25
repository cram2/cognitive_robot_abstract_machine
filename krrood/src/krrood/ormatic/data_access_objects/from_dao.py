from __future__ import annotations

import inspect
from dataclasses import dataclass, field, is_dataclass, fields, MISSING
from inspect import isclass
from typing import Any, Set, Dict, Tuple, Type, List, TYPE_CHECKING

import rustworkx

from krrood.ormatic.data_access_objects.base import (
    DataAccessObjectWorkItem,
    DataAccessObjectState,
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

    resolution_mode: bool = False
    """
    Whether the state is currently in the resolution phase.
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

    def _build_class_dependencies(self, dao_types: List[Type[DataAccessObject]]):
        """
        Build the class dependencies for the given types that can be used to infer the built order.

        :param dao_types: The data access object types to build the dependency graph for.
        """
        types_to_index: Dict[Type, int] = {
            type_: self._class_dependencies.add_node(type_) for type_ in dao_types
        }  # add all dao types to the dependency graph

        # add all dependencies between the classes defined from the alternative mappings
        for dao_type in dao_types:
            alternative_mapping = dao_type.original_class()

            # if it's an alternative mapping, build its dependencies
            if inspect.isclass(alternative_mapping) and issubclass(
                alternative_mapping, AlternativeMapping
            ):
                self._build_dependencies_of_alternative_mapping(
                    alternative_mapping, dao_types, types_to_index
                )

    def _build_dependencies_of_alternative_mapping(
        self,
        alternative_mapping: Type[AlternativeMapping],
        dao_types: List[Type[DataAccessObject]],
        types_to_index: Dict[Type, int],
    ):
        """
        Builds the dependencies of a given alternative mapping and updates the internal
        class dependency graph.

        :param alternative_mapping: The alternative mapping for which dependencies
            are being resolved.
        :param dao_types: A list of DAO types representing the discovered Data Access
            Objects.
        :param types_to_index: A dictionary mapping DAO types to their respective
            indices in the dependency graph.

        """

        dao_of_alternative_mapping = get_dao_class(alternative_mapping)
        # get all concrete types that are affected by the dependencies
        for required_domain_type in alternative_mapping.required_pre_build_classes():
            # for every concrete dao type discovered in the discovery phase
            for concrete_dao_type in dao_types:

                # get the concrete domain type of the dao current dao type
                concrete_domain_type = concrete_dao_type.original_class()

                if not isclass(concrete_domain_type):  # skip non classes for now
                    continue

                if issubclass(concrete_domain_type, AlternativeMapping):
                    concrete_domain_type = concrete_domain_type.original_class()

                # skip types that are not required
                if not issubclass(concrete_domain_type, required_domain_type):
                    continue

                # add the dependency
                self._class_dependencies.add_edge(
                    types_to_index[concrete_dao_type],
                    types_to_index[dao_of_alternative_mapping],
                    None,
                )

    def _order_work_items_by_dependency_graph(
        self, work_items: List[FromDataAccessObjectWorkItem]
    ) -> List[FromDataAccessObjectWorkItem]:
        """
        Order the work items such that dependencies are met first.
        Alternative mappings are moved to the end of the list w. r. t. their dependency order.

        :param work_items: The work items to order.
        :return: The newly sorted work items.
        """

        number_of_work_items = len(work_items)
        result = []
        alternative_mappings = []

        for work_item in work_items:
            if isinstance(work_item.domain_object, AlternativeMapping):
                alternative_mappings.append(work_item)
            else:
                result.append(work_item)

        for type_index in rustworkx.topological_sort(self._class_dependencies):
            dao_type = self._class_dependencies[type_index]

            matching_types = [
                work_item
                for work_item in alternative_mappings
                if type(work_item.dao_instance) is dao_type
            ]
            result.extend(matching_types)

        assert len(result) == number_of_work_items

        return result
