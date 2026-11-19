"""
Roles whose role takers are of a subclass, to check that a role for a class is also a
role for its superclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.predicate import Symbol
from krrood.patterns.role import Role

# %% role takers


@dataclass(kw_only=True, eq=False)
class Animal(Symbol):
    """
    The superclass of the role taker.
    """


@dataclass(kw_only=True, eq=False)
class Tiger(Animal):
    """
    The role taker.
    """


@dataclass(kw_only=True, eq=False)
class Vehicle(Symbol):
    """
    A class no role taker here inherits from.
    """


# %% roles


@dataclass(kw_only=True, eq=False)
class CircusPerformer(Role[Tiger]):
    """
    A role taken by a tiger.
    """


@dataclass(kw_only=True, eq=False)
class HeadlineAct(Role[CircusPerformer]):
    """
    A role taken by another role, so its taker chain ends at a tiger.
    """
