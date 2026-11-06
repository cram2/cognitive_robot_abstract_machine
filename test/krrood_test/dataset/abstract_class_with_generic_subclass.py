"""
An abstract class that is not generic, and a subclass that inherits it first and binds a
generic mixin second, in the shape of a goal whose subclasses bind the type of their
goal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Generic

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import T

# %% the abstract class and the generic mixin


@dataclass(eq=False)
class AbstractGoal(ABC):
    """
    A goal that leaves how it is reached to its subclasses.
    """

    @abstractmethod
    def describe(self) -> str:
        """
        :return: What the goal asks for.
        """


@dataclass(eq=False)
class HasTarget(Generic[T], SubClassSafeGeneric):
    """
    Something that carries a target of a type its subclasses bind.
    """

    target: T
    """
    What is aimed for.
    """


# %% the subclass inheriting both


@dataclass(eq=False)
class GoalWithNumericTarget(AbstractGoal, HasTarget[int]):
    """
    A goal aiming for a number.
    """

    def describe(self) -> str:
        return f"reach {self.target}"
