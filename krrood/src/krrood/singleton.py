from __future__ import annotations

import threading
from typing_extensions import ClassVar, Dict, Type, Any, List


class SingletonMeta(type):
    """
    A metaclass for creating singleton classes.
    """

    _instances: ClassVar[Dict[Type, Any]] = {}
    """
    The available instances of the singleton classes.
    """

    _is_being_created: ClassVar[List[Type]] = []
    """
    A list of classes that are currently being created.
    """

    _construction_lock: ClassVar[threading.RLock] = threading.RLock()
    """
    Serialises singleton construction so a different thread racing to construct the same
    class waits for it instead of seeing the in-progress construction as a circular
    dependency.

    Reentrant so the same thread recursing into its own construction is still caught by
    ``_is_being_created``.
    """

    def __call__(cls, *args, **kwargs):
        """
        Intercept the initialization of every class using this metaclass to check if
        there is an instance registered already.
        """
        with cls._construction_lock:
            if cls in cls._instances:
                return cls._instances[cls]
            if cls in cls._is_being_created:
                raise RuntimeError(
                    f"Circular dependency detected for class {cls.__name__},"
                    f"Some part of the code is trying to create an instance of {cls.__name__} while it is "
                    f"already being created."
                )
            cls._is_being_created.append(cls)
            try:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            finally:
                cls._is_being_created.remove(cls)
            return cls._instances[cls]

    def clear_instance(cls):
        """
        Removes the single, stored instance of this class, allowing a new one to be
        created on the next call.
        """
        if cls in cls._instances:
            del cls._instances[cls]
