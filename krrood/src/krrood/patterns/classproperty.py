"""
A property read on the class rather than on an instance.
"""

from __future__ import annotations

from typing_extensions import Any, Callable, Generic

from krrood.utils import T

# %% class property


class classproperty(Generic[T]):
    """
    A decorator that allows a class method to be accessed as a property.
    """

    def __init__(self, fget: Callable[[type[Any]], T]) -> None:
        """
        Stores the accessor function invoked when the property is read on the owning
        class.
        """
        self.fget = fget
        """
        The underlying accessor; receives the owning class and returns the property
        value.
        """

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> T:
        """
        Returns the property value by invoking the accessor with the owning class.
        """
        return self.fget(owner)
