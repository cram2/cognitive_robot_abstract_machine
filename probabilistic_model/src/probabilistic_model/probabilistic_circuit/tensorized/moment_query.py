from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sortedcontainers import SortedSet
from typing_extensions import Self

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    VariableMask,
    VariableValues,
)
from probabilistic_model.probabilistic_model import CenterType, OrderType


@dataclass
class MomentQuery:
    """
    The order and center of a moment, per variable of a circuit.
    """

    order: VariableValues
    """
    The order of the moment of every variable.
    """

    center: VariableValues
    """
    The center of the moment of every variable.
    """

    requested: VariableMask
    """
    Which variables the moment is requested for.
    """

    @classmethod
    def from_maps(
        cls, order: OrderType, center: CenterType, variables: SortedSet
    ) -> Self:
        """
        :param order: The order per requested variable.
        :param center: The center per variable. Missing variables are centered at 0.
        :param variables: The variables of the circuit.
        :return: The query.
        """
        order_array = np.zeros(len(variables), dtype=np.int64)
        center_array = np.zeros(len(variables))
        requested = np.zeros(len(variables), dtype=bool)
        for variable, value in order.items():
            index = variables.index(variable)
            order_array[index] = value
            requested[index] = True
        for variable, value in center.items():
            center_array[variables.index(variable)] = value
        return cls(order_array, center_array, requested)

    @property
    def number_of_variables(self) -> int:
        return len(self.order)
