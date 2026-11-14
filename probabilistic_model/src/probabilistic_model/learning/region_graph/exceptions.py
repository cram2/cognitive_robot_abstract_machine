"""
Exceptions of building region graphs and the circuits they describe.
"""

from __future__ import annotations

from dataclasses import dataclass

from random_events.variable import Variable
from typing_extensions import TYPE_CHECKING

from probabilistic_model.exceptions import ProbabilisticModelException

if TYPE_CHECKING:
    from probabilistic_model.learning.region_graph.region_graph import Region

# %% exceptions


@dataclass(kw_only=True)
class MultipleRootsError(ProbabilisticModelException, ValueError):
    """
    Exception raised when a region graph has more than one node without parents.
    """

    roots: list[Region]
    """
    The nodes without parents.
    """

    def error_message(self) -> str:
        return f"The region graph has more than one root: {self.roots}."

    def suggest_correction(self) -> str:
        return "Connect every region to the single root region of the graph."


@dataclass(kw_only=True)
class UnsupportedRegionVariableError(ProbabilisticModelException, ValueError):
    """
    Exception raised when a leaf region holds a variable that no input layer of a region
    graph circuit models.
    """

    variable: Variable
    """
    The variable of the leaf region.
    """

    def error_message(self) -> str:
        return (
            f"The variable {self.variable.name} of type {type(self.variable).__name__} "
            f"has no input layer in a region graph circuit."
        )

    def suggest_correction(self) -> str:
        return "Use continuous or symbolic variables."


@dataclass(kw_only=True)
class ClassVariableMismatchError(ProbabilisticModelException, ValueError):
    """
    Exception raised when the class variable of a classifying region graph is missing or
    has another number of classes than the graph.
    """

    classes: int
    """
    The number of classes of the region graph.
    """

    class_variable: Variable | None
    """
    The class variable that was given, if any.
    """

    def error_message(self) -> str:
        return (
            f"The region graph has {self.classes} classes, but the class variable is "
            f"{self.class_variable}."
        )

    def suggest_correction(self) -> str:
        return (
            "Pass a symbolic class variable with one domain element per class, which is "
            "not among the variables of the region graph."
        )
