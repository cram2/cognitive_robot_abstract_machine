from __future__ import annotations

from typing_extensions import ClassVar, Dict, Type, Any


class SingletonMeta(type):
    """
    A metaclass for creating singleton classes.
    """

    _instances: ClassVar[Dict[Type, Any]] = {}
    """
    The available instances of the singleton classes.
    """

    def __call__(cls, *args, **kwargs):
        """
        Intercept the initialization of every class using this metaclass to check if there is an instance registered
        already.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def clear_instance(cls):
        """
        Removes the single, stored instance of this class, allowing a new one
        to be created on the next call.
        """
        if cls in cls._instances:
            del cls._instances[cls]
