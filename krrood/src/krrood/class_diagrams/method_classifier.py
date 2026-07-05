"""
Classification of class methods for the class diagram.

This module identifies *factory methods*: classmethods that construct and return an instance
of their owning class. The classification is reused by the class diagram (so consumers can ask
"which methods of this class are factories?") and by the role pattern (so a factory method is
not silently delegated through a role, which would return a bare role taker and drop the role).

A method is a factory method when it is a ``@classmethod`` and either:

* its return annotation resolves to ``Self`` (or the owning class), or
* it is explicitly marked with the :func:`factory_method` decorator.

Only classmethods are considered; instance and static methods are ignored.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache

from typing_extensions import Any, Callable, Self, Tuple, Type

from krrood.class_diagrams.exceptions import FactoryMethodDecoratorError
from krrood.class_diagrams.factory_method_registry import FactoryMethodRegistry


@dataclass
class FactoryMethodMarker:
    """
    Descriptor that marks the classmethod it wraps as a factory and registers it eagerly.

    Python calls :meth:`__set_name__` once, when the owning class body finishes executing, and
    hands over both the owner class and the attribute name. The descriptor uses that single event
    to record itself in :class:`FactoryMethodRegistry`, so no marker attribute is written onto the
    wrapped function. Attribute access is forwarded transparently to the wrapped ``classmethod``,
    preserving its original call semantics.

    ..note::
        The descriptor must be the **outermost** decorator (applied above ``@classmethod``):
        ``classmethod`` does not forward ``__set_name__`` to the object it wraps, so wrapping the
        descriptor inside ``classmethod`` would skip registration silently. :func:`factory_method`
        enforces this by rejecting anything that is not a ``classmethod``.
    """

    wrapped: classmethod
    """The ``classmethod`` being marked as a factory."""

    def __set_name__(self, owner: Type, name: str) -> None:
        """Register the factory with :class:`FactoryMethodRegistry` once the owner is known."""
        FactoryMethodRegistry().register(owner, name)

    def __get__(self, instance: Any, owner: Type = None) -> Any:
        """Bind the wrapped classmethod so the factory stays callable on the class and instances."""
        return self.wrapped.__get__(instance, owner)


def factory_method(method: classmethod) -> FactoryMethodMarker:
    """
    Mark a classmethod as a factory method explicitly, regardless of its return annotation.

    :param method: The ``classmethod`` to mark.
    :return: A descriptor that registers the factory at class-definition time.
    :raises FactoryMethodDecoratorError: If ``method`` is not a ``classmethod`` (for example when
        the marker is applied below ``@classmethod`` instead of above it).
    """
    if not isinstance(method, classmethod):
        raise FactoryMethodDecoratorError(decorated_name=_decorated_name(method))
    return FactoryMethodMarker(method)


def _decorated_name(method: Any) -> str:
    """
    :param method: The object ``@factory_method`` was applied to.
    :return: A readable name for the object, used only in the rejection error message.
    """
    if hasattr(method, "__name__"):
        return method.__name__
    return repr(method)


def _return_annotation_is_self_or_owner(func: Callable, cls: Type) -> bool:
    """
    :param func: The underlying function of a classmethod.
    :param cls: The class on which the method is looked up.
    :return: Whether the function's return annotation resolves to ``Self`` or to ``cls`` itself.
    """
    annotation = getattr(func, "__annotations__", {}).get("return", None)
    if annotation is None:
        return False
    if isinstance(annotation, str):
        name = annotation.strip().strip("\"'")
        return name in ("Self", cls.__name__) or name.endswith("." + cls.__name__)
    return annotation is Self or annotation is cls


@lru_cache(maxsize=None)
def is_factory_method(cls: Type, name: str) -> bool:
    """
    :param cls: The class to look the method up on.
    :param name: The attribute name to classify.
    :return: Whether ``cls.<name>`` is a factory classmethod.
    """
    if FactoryMethodRegistry().is_registered(cls, name):
        return True
    attribute = inspect.getattr_static(cls, name, None)
    if not isinstance(attribute, classmethod):
        return False
    return _return_annotation_is_self_or_owner(attribute.__func__, cls)


def factory_method_names(cls: Type) -> Tuple[str, ...]:
    """
    :param cls: The class to inspect.
    :return: The names of all factory classmethods reachable on ``cls`` (including inherited ones).
    """
    names = []
    seen = set()
    for klass in cls.__mro__:
        for name, member in vars(klass).items():
            if name in seen:
                continue
            seen.add(name)
            if isinstance(member, (classmethod, FactoryMethodMarker)) and is_factory_method(
                cls, name
            ):
                names.append(name)
    return tuple(names)
