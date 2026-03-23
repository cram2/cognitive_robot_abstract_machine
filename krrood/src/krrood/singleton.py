from __future__ import annotations

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

    def __call__(cls, *args, **kwargs):
        """
        Intercept the initialization of every class using this metaclass to check if there is an instance registered
        already.
        """
        if cls in cls._is_being_created:
            raise RuntimeError(
                f"Circular dependency detected for class {cls.__name__},"
                f"Some part of the code is trying to create an instance of {cls.__name__} while it is "
                f"already being created."
            )
        if cls not in cls._instances:
            cls._is_being_created.append(cls)
            cls._instances[cls] = super().__call__(*args, **kwargs)
            cls._is_being_created.remove(cls)
        return cls._instances[cls]

    def clear_instance(cls):
        """
        Removes the single, stored instance of this class, allowing a new one
        to be created on the next call.
        """
        if cls in cls._instances:
            del cls._instances[cls]
