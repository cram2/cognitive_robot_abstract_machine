from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List

from krrood.symbolic_math.symbolic_math import FloatVariable, SymbolicMathType


@dataclass
class FloatVariableData:
    """
    Stores float variables and their values in a single flat numpy array.
    """

    variables: List[FloatVariable] = field(default_factory=list)
    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def add_variable(self, variable: FloatVariable) -> int:
        self.variables.append(variable)
        self.data = np.append(self.data, 0.0)
        index = len(self.variables) - 1
        variable.resolve = lambda: self.data[index]
        return index

    def add_variables_of_expression(self, expression: SymbolicMathType) -> int:
        index = len(self.variables)
        for variable in expression.free_variables():
            self.add_variable(variable)
        return index

    def set_value(self, variable_index: int, value: float):
        self.data[variable_index] = value

    @property
    def mapping(self) -> dict[FloatVariable, float]:
        return {variable: data for variable, data in zip(self.variables, self.data)}
