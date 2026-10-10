"""
Exceptions of progressive probabilistic circuits.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from probabilistic_model.learning.progressive.progressive_circuit import (
        CircuitColumn,
    )
    from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import Unit
    from probabilistic_model.probabilistic_model import ProbabilisticModel


@dataclass
class ColumnStructureMismatchError(DataclassException, ValueError):
    """
    Exception raised when two columns of a progressive circuit differ in structure.
    """

    left: Unit
    """
    The unit of the first column where the columns differ.
    """

    right: Unit
    """
    The unit of the second column at the same position.
    """

    def error_message(self) -> str:
        return (
            f"The columns differ at {type(self.left).__name__} over "
            f"{[variable.name for variable in self.left.variables]} and "
            f"{type(self.right).__name__} over "
            f"{[variable.name for variable in self.right.variables]}."
        )

    def suggest_correction(self) -> str:
        return "Build every column from the same template circuit."


@dataclass
class UnregisteredColumnError(DataclassException, ValueError):
    """
    Exception raised when a column is used with a progressive circuit it does not belong
    to.
    """

    column: CircuitColumn
    """
    The column that is not part of the progressive circuit.
    """

    def error_message(self) -> str:
        return (
            f"Column {self.column.task_name} is not part of this progressive circuit."
        )

    def suggest_correction(self) -> str:
        return "Create columns with ProgressiveProbabilisticCircuit.add_column."


@dataclass
class UnsupportedLeafDistributionError(DataclassException, TypeError):
    """
    Exception raised when a leaf distribution cannot be learned from weighted rows.
    """

    distribution: ProbabilisticModel
    """
    The distribution that cannot be learned.
    """

    def error_message(self) -> str:
        return (
            f"Leaf distributions of type {type(self.distribution).__name__} cannot be "
            f"learned from weighted rows."
        )

    def suggest_correction(self) -> str:
        return "Use Gaussian, discrete or symbolic leaf distributions."
