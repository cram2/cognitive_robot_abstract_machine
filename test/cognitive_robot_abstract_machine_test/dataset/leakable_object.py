"""
A stand-in for the objects the leak guard watches.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    Something that takes creating its own instances over, which the record can only watch
    by replacing what it does.
    """

    def __new__(cls, *arguments, **keyword_arguments) -> ObjectMakingItsOwnInstances:
        """
        Make an instance the way the record must not silently take over.
        """
        return super().__new__(cls)
