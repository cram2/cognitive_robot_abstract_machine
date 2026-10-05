import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache

from typing_extensions import Callable, Dict, Any, Generic, TypeVar

from krrood.adapters.json_serializer import list_like_classes
from krrood.patterns.exceptions import KeywordNamesNoFactoryParameter

T = TypeVar("T")


@dataclass
class HasFactoryAndKwargs(Generic[T]):
    """
    Mixin containing a hierarchy of factories and their keyword arguments.

    The attributes are underscore-wrapped because hosts of this mixin may hand their
    public attribute namespace to the constructed type (symbolic attribute delegation).
    """

    _factory_: Callable[..., T]
    """
    The factory function to construct `T` with the keyword arguments.
    """

    _kwargs_: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    """
    The keyword arguments to pass to the factory.
    """

    def construct_instance(self):
        """
        Construct a python object from the CallableAndKwargs instance.

        A keyword argument that names no parameter of :attr:`_factory_` is refused,
        unless :attr:`_factory_` accepts arbitrary keywords (a ``**kwargs`` parameter)
        or :meth:`_is_kept_out_of_construction_` says the keyword means something other
        than a constructor argument.

        ..note:: This method may work with ellipsis, but it's not guaranteed to work with all types.

        :return: The constructed object.
        :raises KeywordNamesNoFactoryParameter: If a keyword argument names no parameter
            of :attr:`_factory_` and is not kept out of construction.
        """
        constructed_kwargs = {}
        for key, value in self._kwargs_.items():
            if self._is_kept_out_of_construction_(key, type(value)):
                continue
            if not self._is_factory_parameter_(key):
                raise KeywordNamesNoFactoryParameter(
                    factory=self._factory_, keyword=key
                )
            if isinstance(value, list_like_classes):
                constructed_kwargs[key] = type(value)(
                    self._recurse_construct_instance_and_get_value(element)
                    for element in value
                )
            else:
                constructed_kwargs[key] = (
                    self._recurse_construct_instance_and_get_value(value)
                )
        return self._factory_(**constructed_kwargs)

    def _is_factory_parameter_(self, keyword: str) -> bool:
        """
        :param keyword: A keyword argument.
        :return: Whether :attr:`_factory_` takes that keyword, as a parameter of that
            name or through arbitrary keywords (a ``**kwargs`` parameter).
        """
        return self._factory_takes_keyword_(self._factory_, keyword)

    @staticmethod
    def _factory_takes_keyword_(factory: Callable[..., Any], keyword: str) -> bool:
        """
        :param factory: A factory.
        :param keyword: A keyword argument.
        :return: Whether the factory takes that keyword, as a parameter of that name or
            through arbitrary keywords (a ``**kwargs`` parameter).
        """
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == keyword
            for parameter in HasFactoryAndKwargs._factory_signature_(
                factory
            ).parameters.values()
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _factory_signature_(factory: Callable[..., Any]) -> inspect.Signature:
        """
        :param factory: A factory.
        :return: Its signature, read once per factory.
        """
        return inspect.signature(factory)

    def _is_kept_out_of_construction_(self, keyword: str, value_type: type) -> bool:
        """
        :param keyword: A keyword argument.
        :param value_type: The type of the value given for that keyword.
        :return: Whether that keyword names no parameter of :attr:`_factory_` and means
            something other than a constructor argument, so it is left out of
            construction rather than refused.
        """
        return False

    def _recurse_construct_instance_and_get_value(self, value: Any):
        """
        Recursively construct an instance and return it.

        :param value: The value to construct.
        :return: The constructed instance.
        """
        if isinstance(value, HasFactoryAndKwargs):
            return value.construct_instance()
        return value

    def __deepcopy__(self, memo):
        return self.__class__(
            self._factory_,
            _kwargs_={name: deepcopy(value) for name, value in self._kwargs_.items()},
        )
