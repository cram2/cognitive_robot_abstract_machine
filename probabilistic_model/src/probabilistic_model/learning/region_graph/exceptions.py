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


@dataclass
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


@dataclass
class UnequalFactorSizesError(ProbabilisticModelException, ValueError):
    """
    Exception raised when the regions a partition multiplies have different numbers of
    nodes, so that their layers cannot be multiplied node by node.
    """

    numbers_of_nodes: list[int]
    """
    The number of nodes of the layer of every region the partition multiplies.
    """

    def error_message(self) -> str:
        return (
            f"The regions of a partition have layers of {self.numbers_of_nodes} nodes, "
            f"but they must all have the same number."
        )

    def suggest_correction(self) -> str:
        return "Give the input and sum layers of a partition's regions equal sizes."


@dataclass
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
