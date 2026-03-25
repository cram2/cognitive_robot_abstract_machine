from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Optional, Type, TYPE_CHECKING
from typing_extensions import get_origin

from krrood.ormatic.data_access_objects import helper
from krrood.ormatic.data_access_objects.base import (
    DataAccessObjectWorkItem,
    DataAccessObjectState,
)

from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.base import InstanceDict

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import (
        DataAccessObject,
    )

@dataclass
class ToDataAccessObjectWorkItem(DataAccessObjectWorkItem):
    """
    Work item for converting an object to a Data Access Object.
    """

    source_object: Any
    alternative_base: Optional[Type[DataAccessObject]] = None


@dataclass
class ToDataAccessObjectState(DataAccessObjectState[ToDataAccessObjectWorkItem]):
    """
    State for converting objects to Data Access Objects.
    """

    keep_alive: InstanceDict = field(default_factory=dict)
    """
    Dictionary that prevents objects from being garbage collected.
    """

    def push_work_item(
        self,
        source_object: Any,
        dao_instance: DataAccessObject,
        alternative_base: Optional[Type[DataAccessObject]] = None,
    ):
        """
        Add a new work item to the processing queue.

        :param source_object: The object being converted.
        :param dao_instance: The DAO instance being populated.
        :param alternative_base: Base class for alternative mapping, if any.
        """
        self.work_items.append(
            ToDataAccessObjectWorkItem(
                dao_instance=dao_instance,
                source_object=source_object,
                alternative_base=alternative_base,
            )
        )

    def apply_alternative_mapping_if_needed(
        self, dao_clazz: Type[DataAccessObject], source_object: Any
    ) -> Any:
        """
        Apply an alternative mapping if the DAO class requires it.

        :param dao_clazz: The DAO class to check.
        :param source_object: The object being converted.
        :return: The source object or the result of alternative mapping.
        """
        original_class = dao_clazz.original_class()
        # Handle GenericAlias which cannot be used with issubclass in some python versions
        # or might not be what we want to check for AlternativeMapping anyway.
        origin = get_origin(original_class) or original_class
        if inspect.isclass(origin) and issubclass(origin, AlternativeMapping):
            return helper.to_dao(source_object, state=self)
        return source_object

    def register(self, source_object: Any, dao_instance: DataAccessObject) -> None:
        """
        Register a partially built DAO in the memoization stores.

        :param source_object: The object being converted.
        :param dao_instance: The partially built DAO.
        """
        super().register(source_object, dao_instance)
        self.keep_alive[id(source_object)] = source_object
