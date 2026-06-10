"""
Isolated monitoring infrastructure for EQL object creation tracking.

This module intentionally has no intra-package EQL imports so that it can be
safely imported by modules (variable.py, query.py) that are themselves
dependencies of the main explanation module, without triggering circular
imports.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import wraps
from typing_extensions import Any, Optional, Type, Callable, Union

from krrood.entity_query_language._stack import CallStack, StackFrame
from krrood.singleton import SingletonMeta


@dataclass
class MonitoredRegistry(metaclass=SingletonMeta):
    """
    Registry for monitoring EQL object creation stacks.
    Acts as a class decorator and provides lookup methods.
    """

    _monitored: set[type] = field(default_factory=set)
    """
    Set of classes that are currently being monitored.
    """

    def __call__(self, cls: Type) -> Type:
        """Decorate a class to automatically record its creation stack as a :class:`CallStack`."""
        cls._is_monitored_ = True
        self._monitored.add(cls)

        original_post_init = getattr(cls, "__post_init__", lambda self: None)

        @wraps(original_post_init)
        def new_post_init(self, *args, **kwargs):
            raw_frames = inspect.stack()[1:]
            stack = CallStack(
                [StackFrame.from_frame_info(frame_info) for frame_info in raw_frames]
            )
            self._creation_stack = stack.filter()  # drop site-packages immediately
            original_post_init(self, *args, **kwargs)

        cls.__post_init__ = new_post_init
        return cls

    def get_stack(self, instance: Any) -> Optional[CallStack]:
        """Retrieve the creation stack for a monitored instance."""
        if not self.is_monitored(type(instance)):
            return None
        return instance._creation_stack

    def is_monitored(self, target: Union[Type, Callable]) -> bool:
        """Check whether a class or callable is monitored."""
        return target in self._monitored or bool(
            getattr(target, "_is_monitored_", False)
        )

    def unregister(self, cls: Type) -> None:
        """Remove a class from monitoring."""
        self._monitored.discard(cls)
        if hasattr(cls, "_is_monitored_"):
            del cls._is_monitored_

    @property
    def monitored_classes(self) -> tuple[type, ...]:
        """
        :return: An immutable snapshot of all currently monitored classes as a tuple.
        """
        return tuple(self._monitored)


# The decorator to use for classes to be monitored.
monitored = MonitoredRegistry()
