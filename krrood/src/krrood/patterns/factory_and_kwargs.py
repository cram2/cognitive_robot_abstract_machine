import inspect
from copy import deepcopy
from dataclasses import dataclass, field

from typing_extensions import Callable, Dict, Any, Generic, TypeVar, Self, Union

from krrood.adapters.json_serializer import list_like_classes

T = TypeVar("T")


@dataclass
class HasFactoryAndKwargs(Generic[T]):
    """
    Mixing containing a hierarchy of factories and their keyword arguments.
    """

    factory: Callable[..., T]
    """
    The factory function to construct `T` with the keyword arguments.
    """

    kwargs: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    """
    The keyword arguments to pass to the factory.
    """

    def construct_instance(self):
        """
        Construct a python object from the CallableAndKwargs instance.

        Keyword arguments that name no parameter of :attr:`factory` are dropped rather
        than passed on, unless :attr:`factory` itself accepts arbitrary keywords (a
        ``**kwargs`` parameter). This lets a keyword carry meaning for something other
        than construction -- for instance a query marker on an aggregate rather than a
        field -- without :attr:`factory` ever seeing it.

        ..note:: This method may work with ellipsis, but it's not guaranteed to work with all types.

        :return: The constructed object.
        """
        parameters = inspect.signature(self.factory).parameters.values()
        accepts_arbitrary_keywords = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        parameter_names = {parameter.name for parameter in parameters}

        constructed_kwargs = {}
        for key, value in self.kwargs.items():
            if not accepts_arbitrary_keywords and key not in parameter_names:
                continue
            if isinstance(value, list_like_classes):
                constructed_kwargs[key] = type(value)(
                    self._recurse_construct_instance_and_get_value(element)
                    for element in value
                )
            else:
                constructed_kwargs[key] = (
                    self._recurse_construct_instance_and_get_value(value)
                )
        return self.factory(**constructed_kwargs)

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
            self.factory,
            kwargs={name: deepcopy(value) for name, value in self.kwargs.items()},
        )
