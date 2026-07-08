from __future__ import annotations

from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

from typing_extensions import TYPE_CHECKING

from krrood.singleton import SingletonMeta

if TYPE_CHECKING:
    from typing_extensions import Set, Type


@dataclass
class FactoryMethodRegistry(metaclass=SingletonMeta):
    """
    Process-wide index of methods explicitly marked as factories by ``@factory_method``.

    The registry is populated eagerly at class-definition time: the marker descriptor records its
    owner class and attribute name from ``__set_name__``. Classification then queries this index
    instead of reading a marker attribute off the wrapped function, keeping the factory fact in a
    single owned store rather than scattered across foreign function objects.
    """

    names_by_owner: WeakKeyDictionary[Type, Set[str]] = field(
        default_factory=WeakKeyDictionary
    )
    """Maps each declaring class to the factory-method names marked on its own body."""

    def register(self, owner: Type, name: str) -> None:
        """
        Record that ``owner`` declares ``name`` as a factory method.

        :param owner: The class on whose body the factory method is defined.
        :param name: The attribute name of the factory method.
        """
        self.names_by_owner.setdefault(owner, set()).add(name)

    def is_registered(self, cls: Type, name: str) -> bool:
        """
        :param cls: The class to look the method up on.
        :param name: The attribute name to check.
        :return: Whether ``name`` was marked as a factory on ``cls`` or any of its bases.
        """
        return any(name in self.names_by_owner.get(klass, ()) for klass in cls.__mro__)

    def names_for(self, cls: Type) -> Set[str]:
        """
        :param cls: The class to inspect.
        :return: Every factory-method name marked on ``cls`` or any of its bases.
        """
        names: Set[str] = set()
        for klass in cls.__mro__:
            names |= self.names_by_owner.get(klass, set())
        return names
