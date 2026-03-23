from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List

from krrood.symbolic_math.exceptions import (
    SymbolicMathExpressionNotRegisteredError,
    NoFreeVariablesError,
    SymbolicMathExpressionAlreadyRegisteredError,
    FloatVariableAlreadyHasResolveError,
)
from krrood.symbolic_math.symbolic_math import FloatVariable, SymbolicMathType


hidden_index_name = "__FLOAT_VARIABLE_INDEX__"


@dataclass
class FloatVariableData:
    """
    Stores float variables and their values in a single flat numpy array.
    The purpose of this class is to store data in a single numpy array for efficient evaluation of compiled casadi functions.
    """

    variables: List[FloatVariable] = field(default_factory=list)
    """
    All FloatVariables managed by this data object.
    """
    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    """
    Flat array of values for all `variables`.
    """

    def register_expression(self, expression: SymbolicMathType):
        """
        Add an expression to the data.
        Adds a `hidden_index_name` attribute to the expression to keep track of its index in the data array.
        .. warning:: You can only use expressions with free variables that have no resolve function defined.
            This is a safeguard to prevent accidentally registering, e.g., degree of freedom variables.
        .. warning:: this class is not thread-safe.
        :param expression: The expression to be tracked.
        """
        free_variables = expression.free_variables()
        if len(free_variables) == 0:
            raise NoFreeVariablesError()

        if hasattr(expression, hidden_index_name):
            raise SymbolicMathExpressionAlreadyRegisteredError(expression)

        for variable in free_variables:
            if variable.resolve is not None:
                raise FloatVariableAlreadyHasResolveError(variable=variable)

        index = len(self.variables)
        # save the data index at the expression
        setattr(expression, hidden_index_name, index)

        self.variables.extend(free_variables)
        self.data = np.concatenate((self.data, np.zeros(len(free_variables))))

        # define resolvers for the variables to make `.evaluate()` work
        for i, variable in enumerate(free_variables):

            def resolve_variable(data_index=index + i):
                # this is a workaround to hardcode the "i"
                return self.data[data_index]

            variable.resolve = resolve_variable

    def set_value(
        self, expression: SymbolicMathType, value: float | list[float] | np.ndarray
    ):
        """
        Set the managed values of free variables in an expression.
        Only works if the expression was registered before.
        :param expression: The expression to set the values for.
        :param value: The new value(s) for the expression's free variables.
        """
        if not hasattr(expression, hidden_index_name):
            raise SymbolicMathExpressionNotRegisteredError(expression)

        variable_index = getattr(expression, hidden_index_name)
        if isinstance(value, (int, float)):
            self.data[variable_index] = value
        else:
            self.data[variable_index : variable_index + len(value)] = value

    @property
    def mapping(self) -> dict[FloatVariable, float]:
        """
        :return: Mapping from variables to their values.
        """
        return {variable: data for variable, data in zip(self.variables, self.data)}
