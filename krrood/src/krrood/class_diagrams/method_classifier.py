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

import inspect
from functools import lru_cache

from typing_extensions import Any, Callable, Self, Tuple, Type

try:  # ``typing.Self`` exists on Python 3.11+; ``typing_extensions`` backfills the rest.
    from typing import Self as _TypingSelf
except ImportError:  # pragma: no cover - depends on the interpreter version
    _TypingSelf = None

FACTORY_METHOD_MARKER = "__krrood_factory_method__"
"""Attribute set on a function to mark it (and any ``classmethod`` wrapping it) as a factory."""

_SELF_ANNOTATIONS = {
    annotation for annotation in (Self, _TypingSelf) if annotation is not None
}


def factory_method(func: Callable) -> Callable:
    """
    Mark a classmethod as a factory method explicitly, regardless of its return annotation.

    The marker is written on the underlying function so it survives ``classmethod`` wrapping and
    is inherited by subclasses. The decorator may be stacked in either order relative to
    ``@classmethod``.

    :param func: The function or ``classmethod``/``staticmethod`` descriptor to mark.
    :return: The same object, marked as a factory method.
    """
    target = func.__func__ if isinstance(func, (classmethod, staticmethod)) else func
    setattr(target, FACTORY_METHOD_MARKER, True)
    return func


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
    return annotation in _SELF_ANNOTATIONS or annotation is cls


@lru_cache(maxsize=None)
def is_factory_method(cls: Type, name: str) -> bool:
    """
    :param cls: The class to look the method up on.
    :param name: The attribute name to classify.
    :return: Whether ``cls.<name>`` is a factory classmethod.
    """
    attribute = inspect.getattr_static(cls, name, None)
    if not isinstance(attribute, classmethod):
        return False
    func = attribute.__func__
    if getattr(func, FACTORY_METHOD_MARKER, False):
        return True
    return _return_annotation_is_self_or_owner(func, cls)


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
            if isinstance(member, classmethod) and is_factory_method(cls, name):
                names.append(name)
    return tuple(names)
