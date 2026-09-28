from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from krrood.symbolic_math.exceptions import (
    SymbolicMathExpressionNotRegisteredError,
    NoFreeVariablesError,
    SymbolicMathExpressionAlreadyRegisteredError,
    FloatVariableAlreadyHasResolveError,
)
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    FloatVariable,
    SymbolicMathType,
)

hidden_index_name = "__FLOAT_VARIABLE_INDEX__"


@dataclass
class BoundArgument:
    """
    One argument of a compiled function that reads a managed value array.
    """

    function: CompiledFunction
    """
    The function reading the values.
    """

    argument_index: int
    """
    Index of the argument holding the values.
    """

    def point_at(self, data: npt.NDArray) -> None:
        """
        Make the argument read `data` from now on.

        :param data: The array holding the values.
        """
        self.function.bind_args_to_memory_view(self.argument_index, data)


@dataclass
class FloatVariableData:
    """
    Stores float variables and their values in a single flat numpy array.

    The purpose of this class is to store data in a single numpy array for efficient
    evaluation of compiled casadi functions.
    """

    variables: list[FloatVariable] = field(default_factory=list)
    """
    All FloatVariables managed by this data object.
    """

    data: npt.NDArray = field(default_factory=lambda: np.array([], dtype=np.float64))
    """
    Flat array of values for all `variables`.

    .. warning:: Registering an expression replaces this array. Read it through
        :meth:`bind_argument` instead of holding on to it, or the values read go stale.
    """

    _bound_arguments: list[BoundArgument] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Compiled function arguments that read `data` and are re-pointed at it when it grows.
    """

    def bind_argument(
        self, compiled_function: CompiledFunction, argument_index: int
    ) -> None:
        """
        Let one argument of a compiled function read the managed values.

        The argument keeps reading them for the lifetime of this data object, including
        across later registrations.
        :param compiled_function: The function reading the values.
        :param argument_index: Index of the argument holding the managed values.
        """
        bound_argument = BoundArgument(
            function=compiled_function, argument_index=argument_index
        )
        self._bound_arguments.append(bound_argument)
        bound_argument.point_at(self.data)

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
        for bound_argument in self._bound_arguments:
            bound_argument.point_at(self.data)

        # define resolvers for the variables to make `.evaluate()` work
        for i, variable in enumerate(free_variables):

            def resolve_variable(data_index=index + i):
                # this is a workaround to hardcode the "i"
                return self.data[data_index]

            variable.resolve = resolve_variable

    def set_value(
        self, expression: SymbolicMathType, value: float | list[float] | npt.NDArray
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

    def get_value(self, variable: FloatVariable) -> float:
        """
        Read the managed value of a single free variable.

        Unlike :meth:`SymbolicMathType.evaluate`, this reads the stored value directly
        instead of compiling and evaluating the expression.
        :param variable: The variable to read the value of.
        :return: The value currently stored for the variable.
        """
        if not hasattr(variable, hidden_index_name):
            raise SymbolicMathExpressionNotRegisteredError(variable)

        return self.data[getattr(variable, hidden_index_name)]

    @property
    def mapping(self) -> dict[FloatVariable, float]:
        """
        :return: Mapping from variables to their values.
        """
        return {variable: data for variable, data in zip(self.variables, self.data)}
