from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING

from probabilistic_model.exceptions import ProbabilisticModelException

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
        Layer,
    )


@dataclass
class UndefinedCumulativeDistributionError(ProbabilisticModelException, TypeError):
    """
    Exception raised when the cumulative distribution function of a layer over an
    unordered variable is queried.
    """

    layer_type: type
    """
    The type of the layer.
    """

    def error_message(self) -> str:
        return (
            f"{self.layer_type.__name__} has no cumulative distribution function, "
            f"since the states of its variable are not ordered."
        )

    def suggest_correction(self) -> str:
        return "Query the cumulative distribution only over ordered variables."


@dataclass
class NonPositiveScaleError(ProbabilisticModelException, ValueError):
    """
    Exception raised when a layer has a scale that is not positive.
    """

    scale: np.ndarray
    """
    The scales that are not positive.
    """

    def error_message(self) -> str:
        return f"The scales {self.scale.tolist()} are not positive."

    def suggest_correction(self) -> str:
        return "Give every node a positive scale."


@dataclass
class CopiesNotAlignedError(ProbabilisticModelException, ValueError):
    """
    Exception raised when copies of a layer graph that are to be stacked differ in their
    structure.
    """

    copies: tuple[Layer, ...]
    """
    The layer at the same position in every copy of the layer graph, in the order of
    the copies.
    """

    path: tuple[int, ...]
    """
    The index of the child layer taken at every step from the roots of the copies down
    to :attr:`copies`.

    Empty if the roots themselves differ.
    """

    @property
    def misaligned_copies(self) -> tuple[int, ...]:
        """
        :return: The index of every copy whose layer differs from the one of the first
            copy in its type, its number of nodes or its number of child layers.
        """
        structures = [
            (type(copy), copy.number_of_nodes, len(copy.child_layers))
            for copy in self.copies
        ]
        return tuple(
            index
            for index, structure in enumerate(structures)
            if structure != structures[0]
        )

    def structure_of_copy(self, index: int) -> str:
        """
        :param index: The index of a copy.
        :return: The type, number of nodes and number of child layers of the layer in
            that copy.
        """
        copy = self.copies[index]
        return (
            f"copy {index} has a {type(copy).__name__} with "
            f"{copy.number_of_nodes} nodes and {len(copy.child_layers)} child layers"
        )

    def error_message(self) -> str:
        return (
            f"The layers reached from the roots through the child layers "
            f"{list(self.path)} differ in their structure: "
            + "; ".join(
                self.structure_of_copy(index) for index in (0, *self.misaligned_copies)
            )
            + "."
        )

    def suggest_correction(self) -> str:
        return (
            "Only stack copies that a structural pass made from the same layer graph "
            "without pruning it."
        )
