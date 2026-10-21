"""
Expectation maximization that learns one column of a progressive probabilistic circuit
at a time.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field

import numpy as np

from probabilistic_model.learning.expectation_maximization import (
    ExpectationMaximization,
    LearnableUnits,
)
from probabilistic_model.learning.progressive.progressive_circuit import (
    CircuitColumn,
    ProgressiveProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    SampleArray,
)


# %% expectation maximization
@dataclass
class ProgressiveExpectationMaximization:
    """
    Expectation maximization for one column of a progressive probabilistic circuit.

    It is ordinary expectation maximization restricted to the units of the column, so
    the earlier columns stay unchanged while the column learns how much to reuse them
    through its edges to them. While the column is learned, the root gives it the whole
    weight, so it is trained as the only model of its task. Afterwards the root weights
    every column by its share of all rows the columns were learned from.

    .. warning::

        Later columns read from earlier ones, so learning an earlier column again also
        changes the later columns.
    """

    progressive_circuit: ProgressiveProbabilisticCircuit
    """
    The circuit whose columns are learned.
    """

    expectation_maximization: ExpectationMaximization = field(
        default_factory=ExpectationMaximization
    )
    """
    The expectation maximization the parameters of a column are learned with.
    """

    def learn(
        self, data: SampleArray, column: CircuitColumn, iterations: int = 1
    ) -> list[float]:
        """
        Learn the parameters of a column.

        :param data: One row per sample, columns ordered like the circuit's variables.
        :param column: The column to learn.
        :param iterations: Number of expectation maximization iterations.
        :return: The average log-likelihood of the rows under the column before every
            iteration.
        :raises UnregisteredColumnError: If the column belongs to another progressive
            circuit.
        :raises UnsupportedLeafDistributionError: If the column has a leaf that cannot
            be learned.
        """
        self.progressive_circuit.validate_column(column)
        data = np.asarray(data)
        if len(data) == 0:
            return []

        learnable_units = self.learnable_units(column)
        self.progressive_circuit.restrict_root_to(column)
        history = self.expectation_maximization.learn(
            self.progressive_circuit.circuit, data, learnable_units, iterations
        )
        column.sample_count = len(data)
        self.progressive_circuit.weight_root_by_sample_count()
        return history

    def learnable_units(self, column: CircuitColumn) -> LearnableUnits:
        """
        Collect the units whose parameters learning a column updates.

        :param column: The column to learn.
        :return: The sum and leaf units of the column.
        :raises UnsupportedLeafDistributionError: If the column has a leaf that cannot
            be learned.
        """
        return LearnableUnits.from_units(self.progressive_circuit.units_of(column))
