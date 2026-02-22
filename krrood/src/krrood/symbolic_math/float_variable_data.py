from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List

from .symbolic_math import FloatVariable, SymbolicMathType


@dataclass
class FloatVariableData:
    """
    Stores float variables and their values in a single flat numpy array.
    """

    variables: List[FloatVariable] = field(default_factory=list)
    """
    All FloatVariables managed by this data object.
    """
    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    """
    Flat array of values for all `variables`.
    """

    def add_variable(self, variable: FloatVariable) -> int:
        """
        Add a new variable to the data.
        :param variable: The new variable.
        :return: The index in data for the variable
        """
        self.variables.append(variable)
        self.data = np.append(self.data, 0.0)
        index = len(self.variables) - 1
        variable.resolve = lambda: self.data[index]
        return index

    def add_variables_of_expression(self, expression: SymbolicMathType) -> int:
        """
        Add variables from an expression to the data.
        :param expression: The expression to add variables from.
        :return: The index in data for the first added variable
        """
        index = len(self.variables)
        for variable in expression.free_variables():
            self.add_variable(variable)
        return index

    def set_value(self, variable_index: int, value: float):
        """
        Set the value of a variable.
        """
        self.data[variable_index] = value

    def set_values(self, variable_index: int, values: List[float] | np.ndarray):
        """
        Set the values of multiple variables which are contiguous in the data array.
        :param variable_index: The index of the first variable to set.
        :param values: The values to set for the variables.
        """
        self.data[variable_index : variable_index + len(values)] = values

    @property
    def mapping(self) -> dict[FloatVariable, float]:
        """
        :return: Mapping from variables to their values.
        """
        return {variable: data for variable, data in zip(self.variables, self.data)}
