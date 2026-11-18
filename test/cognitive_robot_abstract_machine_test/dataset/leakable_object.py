"""
A stand-in for the objects the leak guard watches.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LeakableObject:
    """
    Something a test creates and may leave in memory after it finished.

    Cheap to create and to copy, so that a test can leave hundreds behind without paying
    for what a real world costs.
    """

    name: str = ""
    """
    Tells one of them from another.
    """


@dataclass
class ObjectMakingItsOwnInstances:
    """
    Something that takes creating its own instances over, as a type does that inherits an
    interface.
    """

    made_its_own_way: bool = field(init=False)
    """
    Whether the instance was made by the type's own way of creating it.
    """

    def __new__(cls, *arguments, **keyword_arguments) -> ObjectMakingItsOwnInstances:
        """
        Make an instance its own way, which a record watching the type must keep.
        """
        instance = super().__new__(cls)
        instance.made_its_own_way = True
        return instance
