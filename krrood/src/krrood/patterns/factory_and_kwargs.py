from copy import deepcopy
from dataclasses import dataclass, field
from inspect import isclass, ismethod

from typing_extensions import (
    Callable,
    Any,
    Generic,
    Iterator,
    Self,
    TypeVar,
    get_origin,
)

from krrood.class_diagrams.call_signature import CallSignature
from krrood.class_diagrams.collection_kind import CollectionKind
from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.patterns.exceptions import KeywordNamesNoFactoryParameter

T = TypeVar("T")


@dataclass(kw_only=True)
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

    _kwargs_: dict[str, Any] = field(default_factory=dict, kw_only=True)
    """
    The keyword arguments to pass to the factory.
    """

    def construct_instance(self) -> T:
        """
        Construct a python object from the CallableAndKwargs instance.

        ..note:: This method may work with ellipsis, but it's not guaranteed to work with all types.

        :return: The constructed object.
        :raises KeywordNamesNoFactoryParameter: If a keyword argument names no parameter
            of :attr:`_factory_` and is not kept out of construction.
        """
        constructed_kwargs = {}
        for key, value in self._factory_keyword_arguments_.items():
            if isinstance(value, CollectionKind.ELEMENT_SEQUENCE.value):
                constructed_kwargs[key] = type(value)(
                    self._recurse_construct_instance_and_get_value(value=element)
                    for element in value
                )
            else:
                constructed_kwargs[key] = (
                    self._recurse_construct_instance_and_get_value(value=value)
                )
        return self._factory_(**constructed_kwargs)

    @property
    def _factory_keyword_arguments_(self) -> dict[str, Any]:
        """
        The keyword arguments that are passed to :attr:`_factory_`.

        A keyword argument that names no parameter of :attr:`_factory_` is refused,
        unless :attr:`_factory_` accepts arbitrary keywords (a ``**kwargs`` parameter)
        or :meth:`_is_kept_out_of_construction_` says the keyword means something other
        than a constructor argument, in which case it is left out.

        :raises KeywordNamesNoFactoryParameter: If a keyword argument names no parameter
            of :attr:`_factory_` and is not kept out of construction.
        """
        factory_keyword_arguments = {}
        for key, value in self._kwargs_.items():
            if self._is_kept_out_of_construction_(keyword=key, value_type=type(value)):
                continue
            if not self._is_factory_parameter_(keyword=key):
                raise KeywordNamesNoFactoryParameter(
                    factory=self._factory_, keyword=key
                )
            factory_keyword_arguments[key] = value
        return factory_keyword_arguments

    @staticmethod
    def _type_built_by_(*, factory: Callable[..., Any]) -> type | None:
        """
        :param factory: A class, function or method.
        :return: The class itself; otherwise the class its return annotation names, the
            class a method annotated with :data:`~typing.Self` is bound to, the origin
            of a parameterized generic it names (``list`` for ``list[int]``), or
            :class:`object` for :data:`~typing.Any`. ``None`` when it has no return
            annotation or the annotation names no class.
        """
        if isclass(factory):
            return factory
        return_annotation = get_type_hints_of_object(factory).get("return")
        if return_annotation is Self and ismethod(factory):
            owner = factory.__self__
            return owner if isclass(owner) else type(owner)
        if return_annotation is Any:
            return object
        if isclass(return_annotation):
            return return_annotation
        if isclass(get_origin(return_annotation)):
            return get_origin(return_annotation)
        return None

    def _stated_values_(self, *, keyword: str) -> Iterator[Any]:
        """
        :param keyword: A keyword argument.
        :return: The value given for that keyword, or nothing when none was given.
        """
        if keyword in self._kwargs_:
            yield self._kwargs_[keyword]

    def _is_factory_parameter_(self, *, keyword: str) -> bool:
        """
        :param keyword: A keyword argument.
        :return: Whether :attr:`_factory_` takes that keyword, as a parameter of that
            name or through arbitrary keywords (a ``**kwargs`` parameter).
        """
        return CallSignature.of(callee=self._factory_).accepts_keyword(keyword=keyword)

    def _is_kept_out_of_construction_(self, *, keyword: str, value_type: type) -> bool:
        """
        :param keyword: A keyword argument.
        :param value_type: The type of the value given for that keyword.
        :return: Whether that keyword names no parameter of :attr:`_factory_` and means
            something other than a constructor argument, so it is left out of
            construction rather than refused.
        """
        return False

    def _recurse_construct_instance_and_get_value(self, *, value: Any):
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
            _factory_=self._factory_,
            _kwargs_={name: deepcopy(value) for name, value in self._kwargs_.items()},
        )
