from __future__ import annotations

import abc
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType


from typing_extensions import Optional, Self, List, Type, TypeVar, TYPE_CHECKING

from krrood.ormatic.data_access_objects.base import HasGeneric

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState

T = TypeVar("T")


class AlternativeMapping(HasGeneric[T], abc.ABC):
    """
    Base class for alternative mapping implementations.
    """

    @classmethod
    def to_dao(cls, source_object: T, state: Optional[ToDataAccessObjectState]) -> T:
        """
        Resolve a source object to a DAO.

        :param source_object: The object to convert.
        :param state: The conversion state.
        :return: The converted DAO instance.
        """
        if state.has(source_object):
            return state.get(source_object)
        elif isinstance(source_object, cls):
            return source_object
        else:
            result = cls.from_domain_object(source_object)
            return result

    @classmethod
    @abc.abstractmethod
    def from_domain_object(cls, obj: T) -> Self:
        """
        Create this from a domain object.
        Do not create any DAOs here but the target DAO of `T`.
        The rest of the `to_dao` algorithm will process the fields of the created instance.

        :param obj: The source object.
        :return: A new instance of this mapping class.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_domain_object(self) -> T:
        """
        Create a domain object from this instance.

        :return: The constructed domain object.
        """
        raise NotImplementedError

    @classmethod
    def required_pre_build_classes(cls) -> List[Type]:
        """
        A list of other classes that have to be built before this one in the `from_dao` algorithm.
        The types inside the list are the domain types, not the data access objects nor the alternative mappings.
        """
        return []


@dataclass
class UncallableFunction(NotImplementedError):
    """
    Exception raised when anonymous functions are reconstructed and then called.
    """

    function_mapping: FunctionMapping

    def __post_init__(self):
        super().__init__(
            f"The reconstructed function was a lambda function and hence cannot be called again. "
            f"The function tried to be reconstructed from {self.function_mapping}"
        )


def raise_uncallable_function(function_mapping: FunctionMapping):
    raise UncallableFunction(function_mapping)


@dataclass
class FunctionMapping(AlternativeMapping[FunctionType]):
    """
    Alternative mapping for functions.
    """

    module_name: str
    """
    The module name where the function is defined.
    """

    function_name: str
    """
    The name of the function.
    """

    class_name: Optional[str] = None
    """
    The name of the class if the function is defined by a class.
    """

    @classmethod
    def from_domain_object(cls, obj: Callable) -> Self:

        if "." in obj.__qualname__:
            class_name = obj.__qualname__.split(".")[0]
        else:
            class_name = None
        dao = cls(
            module_name=obj.__module__,
            function_name=obj.__name__,
            class_name=class_name,
        )
        return dao

    def to_domain_object(self) -> T:

        if self.function_name == "<lambda>":
            return lambda *args, **kwargs: raise_uncallable_function(self)

        module = importlib.import_module(self.module_name)

        if self.class_name is not None:
            return getattr(getattr(module, self.class_name), self.function_name)
        else:
            return getattr(module, self.function_name)
