"""
Symbolic math utilities built on top of CasADi.

This module provides small, object oriented wrappers around symbolic arrays
and functions. It aims to make operations on scalars, vectors and matrices
feel similar to NumPy, while keeping expressions symbolic so they can be
compiled and evaluated efficiently.

The public API centers around the following types:

- Scalar: symbolic scalar values
- Vector: symbolic equivalent to numpy 1d arrays
- Matrix: symbolic matrices of arbitrary 2d shape

There are helpers to create variables, to compile expressions for fast
numerical evaluation, and to perform common operations such as stacking,
logical composition, and conditional selection.
"""

from __future__ import annotations

import builtins
import copy
import math
import operator
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import field, dataclass
from enum import IntEnum
from functools import partial

import casadi as ca
import numpy as np
from scipy import sparse as sp
from typing_extensions import (
    ClassVar,
    Iterable,
    Tuple,
    Sequence,
    Self,
    Callable,
    List,
    Optional,
    Dict,
    TypeVar,
    Type,
)

from .exceptions import (
    HasFreeVariablesError,
    DuplicateVariablesError,
    WrongNumberOfArgsError,
    NotSquareMatrixError,
    SymbolicMathError,
    NotScalerError,
    UnsupportedOperationError,
    WrongDimensionsError,
)

EPS: float = sys.float_info.epsilon * 4.0


@dataclass
class VariableGroup:
    """
    A homogeneous, ordered group of variables that forms one input block.
    """

    variables: Tuple[FloatVariable, ...]

    def __len__(self) -> int:
        return len(self.variables)


@dataclass
class VariableParameters:
    """
    A collection of variable groups that define the input blocks of a compiled function.
    """

    groups: Tuple[VariableGroup, ...]

    def __len__(self) -> int:
        return len(self.groups)

    def flatten(self) -> Tuple[FloatVariable, ...]:
        return tuple(v for g in self.groups for v in g.variables)

    @classmethod
    def from_lists(cls, *args: List[FloatVariable]) -> VariableParameters:
        """
        Creates a new instance of VariableParameters from multiple lists.

        :param args: A variable number of lists, where each list contains
            FloatVariable instances.
        :return: A new instance of VariableParameters created from the provided lists.
        """
        return cls(groups=tuple(VariableGroup(tuple(g)) for g in args))

    def to_casadi_parameters(self) -> List[ca.SX]:
        casadi_parameters = []
        if len(self) > 0:
            # create an array for each VariableGroup
            casadi_parameters = [to_sx(list(group.variables)) for group in self.groups]
        return casadi_parameters


@dataclass
class _Layout(ABC):
    """
    Represents an abstract base class for layout handling and compilation.
    """

    compiled: CompiledFunction
    """
    Reference to the CompiledFunction that is being compiled.
    """

    def is_empty_result(self) -> bool:
        return len(self.compiled.expression) == 0

    @abstractmethod
    def compile(self, casadi_parameters: List[ca.SX]) -> None: ...

    @abstractmethod
    def setup_output(self) -> None: ...


@dataclass
class _DenseLayout(_Layout):
    """Strategy for dense compiled function setup."""

    def compile(self, casadi_parameters: List[ca.SX]) -> None:
        self.compiled.expression.casadi_sx = ca.densify(
            self.compiled.expression.casadi_sx
        )
        self.compiled._compiled_casadi_function = ca.Function(
            "f", casadi_parameters, [self.compiled.expression.casadi_sx]
        )
        self.compiled._function_buffer, self.compiled._function_evaluator = (
            self.compiled._compiled_casadi_function.buffer()
        )

    def setup_output(self) -> None:
        expr = self.compiled.expression
        if expr.shape[1] <= 1:
            shape = expr.shape[0]
        else:
            shape = expr.shape
        self.compiled._out = np.zeros(shape, order="F")
        self.compiled._function_buffer.set_res(0, memoryview(self.compiled._out))


@dataclass
class _SparseLayout(_Layout):
    """Strategy for sparse compiled function setup."""

    def compile(self, casadi_parameters: List[ca.SX]) -> None:
        self.compiled.expression.casadi_sx = ca.sparsify(
            self.compiled.expression.casadi_sx
        )
        self.compiled._compiled_casadi_function = ca.Function(
            "f", casadi_parameters, [self.compiled.expression.casadi_sx]
        )
        self.compiled._function_buffer, self.compiled._function_evaluator = (
            self.compiled._compiled_casadi_function.buffer()
        )
        self.csc_indices, self.csc_indptr = (
            self.compiled.expression.casadi_sx.sparsity().get_ccs()
        )
        self.zeroes = np.zeros(self.compiled.expression.casadi_sx.nnz())

    def setup_output(self) -> None:
        expr = self.compiled.expression
        out = sp.csc_matrix(
            arg1=(
                self.zeroes,
                self.csc_indptr,
                self.csc_indices,
            ),
            shape=expr.shape,
        )
        self.compiled._out = out
        self.compiled._function_buffer.set_res(0, memoryview(out.data))


@dataclass
class CompiledFunction:
    """
    A compiled symbolic function that can be efficiently evaluated with CasADi.

    This class compiles symbolic expressions into optimized CasADi functions that can be
    evaluated efficiently. It supports both sparse and dense matrices and handles
    parameter substitution automatically.
    """

    expression: SymbolicMathType
    """
    The symbolic expression to compile.
    """
    variable_parameters: Optional[VariableParameters] = None
    """
    The input parameters for the compiled symbolic expression.
    """
    sparse: bool = False
    """
    Whether to return a sparse matrix or a dense numpy matrix
    """
    _layout: _Layout = field(init=False)
    """
    The layout strategy to use for the compiled function.
    """

    _compiled_casadi_function: ca.Function = field(init=False)

    _function_buffer: ca.FunctionBuffer = field(init=False)
    _function_evaluator: partial = field(init=False)
    """
    Helpers to avoid new memory allocation during function evaluation
    """

    _out: np.ndarray | sp.csc_matrix = field(init=False)
    """
    The result of a function evaluation is stored in this variable.
    """

    _is_constant: bool = False
    """
    Used to memorize if the result must be recomputed every time.
    """

    def __post_init__(self):
        # Normalize variable_parameters to VariableParameters
        if self.variable_parameters is None:
            free_vars = self.expression.free_variables()
            if len(free_vars) == 0:
                vp = VariableParameters(groups=tuple())
            else:
                vp = VariableParameters(groups=(VariableGroup(tuple(free_vars)),))
            self.variable_parameters = vp

        # Validate variables
        self._validate_variables()

        if len(self.expression) == 0:
            self._setup_empty_result()
            return

        self._setup_compiled_function()
        self._layout.setup_output()
        if len(self.variable_parameters) == 0:
            self._setup_constant_result()

    def _validate_variables(self):
        """Validates variables for both missing and duplicate issues."""
        variables = list(self.variable_parameters.flatten())
        variables_set = set(variables)

        # Check for missing variables
        missing_variables = set(self.expression.free_variables()).difference(
            variables_set
        )
        if missing_variables:
            raise HasFreeVariablesError(list(missing_variables))

        # Check for duplicate variables
        if len(variables_set) != len(variables):
            variable_counts = Counter(variables)
            all_duplicates = [
                variable
                for variable, count in variable_counts.items()
                if count > 1
                for _ in range(count)
            ]
            raise DuplicateVariablesError(all_duplicates)

    def _setup_empty_result(self) -> None:
        """
        Setup result for empty expressions.
        """
        if self.sparse:
            self._out = sp.csc_matrix(np.empty(self.expression.shape))
        else:
            self._out = np.empty(self.expression.shape)
        self._is_constant = True

    def _setup_compiled_function(self) -> None:
        """
        Setup the CasADi compiled function.
        """
        self._layout = _SparseLayout(self) if self.sparse else _DenseLayout(self)
        self._layout.compile(self.variable_parameters.to_casadi_parameters())

    def _setup_constant_result(self) -> None:
        """
        Setup result for constant expressions (no parameters).

        For expressions with no free parameters, we can evaluate once and return
        the constant result for all future calls.
        """
        self._function_evaluator()
        self._is_constant = True

    def bind_args_to_memory_view(self, arg_idx: int, numpy_array: np.ndarray) -> None:
        """
        Binds the arg at index arg_idx to the memoryview of a numpy_array.
        If your args keep the same memory across calls, you only need to bind them once.
        """
        self._function_buffer.set_arg(arg_idx, memoryview(numpy_array))

    def evaluate(self) -> np.ndarray | sp.csc_matrix:
        """
        Evaluate the compiled function with the current args.
        """
        self._function_evaluator()
        return self._out

    def __call__(self, *args: np.ndarray) -> np.ndarray | sp.csc_matrix:
        """
        Efficiently evaluate the compiled function with positional arguments by directly writing the memory of the
        numpy arrays to the memoryview of the compiled function.
        Similarly, the result will be written to the output buffer and does not allocate new memory on each eval.

        :param args: A numpy array for each VariableGroup in self.variable_parameters.
            .. warning:: Make sure the numpy array is of type float! (check is too expensive)
        :return: The evaluated result as numpy array or sparse matrix
        """
        if self._is_constant:
            return self._out
        expected_number_of_args = len(self.variable_parameters)
        actual_number_of_args = len(args)
        if expected_number_of_args != actual_number_of_args:
            raise WrongNumberOfArgsError(
                expected_number_of_args,
                actual_number_of_args,
            )
        for arg_idx, arg in enumerate(args):
            self.bind_args_to_memory_view(arg_idx, arg)
        return self.evaluate()

    def call_with_kwargs(self, **kwargs: float) -> np.ndarray:
        """
        Call the object instance with the provided keyword arguments. This method retrieves
        the required arguments from the keyword arguments based on the defined
        `variable_parameters`, compiles them into an array, and then calls the instance
        with the constructed array.

        :param kwargs: A dictionary of keyword arguments containing the parameters
            that match the variables defined in `variable_parameters`.
        :return: A NumPy array resulting from invoking the callable object instance
            with the filtered arguments.
        """
        args = []
        for group in self.variable_parameters.groups:
            for param in group.variables:
                args.append(kwargs[str(param)])
        filtered_args = np.array(args, dtype=float)
        return self(filtered_args)


@dataclass
class CompiledFunctionWithViews:
    """
    A wrapper for CompiledFunction which automatically splits the result array into multiple views, with minimal
    overhead.
    Useful, when many arrays must be evaluated at the same time, especially when they depend on the same variables.
    __call__ returns first a list of expressions, followed by additional_views.
    e.g. CompiledFunctionWithViews(expressions=[expr1, expr2], additional_views=[(start, end)])
        returns [expr1_result, expr2_result, np.concatenate([expr1_result, expr2_result])[start:end]]
    """

    expressions: List[SymbolicMathType]
    """
    The list of expressions to be compiled.
    """

    parameters: VariableParameters
    """
    The input parameters for the compiled symbolic expression.
    """

    additional_views: Optional[List[slice]] = field(default_factory=list)
    """
    If additional views are required that don't correspond to the expressions directly.
    """

    compiled_function: CompiledFunction = field(init=False)
    """
    Reference to the compiled function.
    """

    split_out_view: List[np.ndarray] = field(init=False)
    """
    Views to the out buffer of the compiled function.
    """

    def __post_init__(self):
        combined_expression = Matrix.vstack(self.expressions)
        self.compiled_function = combined_expression.compile(
            parameters=self.parameters, sparse=False
        )
        slices = []
        start = 0
        for expression in self.expressions[:-1]:
            end = start + expression.shape[0]
            slices.append(end)
            start = end
        self.split_out_view = np.split(self.compiled_function._out, slices)
        for expression_slice in self.additional_views:
            self.split_out_view.append(self.compiled_function._out[expression_slice])

    def __call__(self, *args: np.ndarray) -> List[np.ndarray]:
        """
        :param args: A numpy array for each List[FloatVariable] in self.variable_parameters.
        :return: A np array for each expression, followed by arrays corresponding to the additional views.
            They are all views on self.compiled_function.out.
        """
        self.compiled_function(*args)
        return self.split_out_view


@dataclass(eq=False)
class SymbolicMathType(ABC):
    """
    A wrapper around CasADi's ca.SX, with better usability
    """

    _casadi_sx: ca.SX = field(kw_only=True, default_factory=ca.SX, repr=False)
    """
    Reference to the casadi data structure of type casadi.SX
    """

    @classmethod
    def from_casadi_sx(cls, casadi_sx: ca.SX) -> Self:
        result = cls()
        result.casadi_sx = casadi_sx
        return result

    @property
    def casadi_sx(self) -> ca.SX:
        return self._casadi_sx

    @casadi_sx.setter
    def casadi_sx(self, casadi_sx: ca.SX) -> None:
        self._casadi_sx = casadi_sx
        self._verify_type()

    @abstractmethod
    def _verify_type(self):
        """
        Called after the casadi_sx is set. Checks that the casadi_sx has the correct properties for this subclass.
        """

    def __str__(self):
        return str(self.casadi_sx)

    def __hash__(self) -> int:
        return hash(self.casadi_sx)

    def pretty_str(self) -> List[List[str]]:
        """
        Turns a symbolic type into a more or less readable string.
        """
        result_list = np.zeros(self.shape).tolist()
        for x_index in range(self.shape[0]):
            for y_index in range(self.shape[1]):
                s = str(self[x_index, y_index])
                parts = s.split(", ")
                result = parts[-1]
                for x in reversed(parts[:-1]):
                    equal_position = len(x.split("=")[0])
                    index = x[:equal_position]
                    sub = x[equal_position + 1 :]
                    result = result.replace(index, sub)
                result_list[x_index][y_index] = result
        return result_list

    def is_scalar(self) -> bool:
        return self._casadi_sx.is_scalar()

    def __array__(self):
        return self.to_np()

    def __repr__(self):
        return repr(self.casadi_sx)

    def __getitem__(
        self, item: np.ndarray | int | slice | Tuple[int | slice, int | slice]
    ) -> Scalar | Vector:
        """
        Gives this class the getitem behavior of numpy.
        """
        if isinstance(item, np.ndarray) and item.dtype == bool:
            item = (np.where(item)[0], slice(None, None))
        item_sx = self.casadi_sx[item]
        if item_sx.shape == (1, 1):
            return Scalar.from_casadi_sx(item_sx)
        if item_sx.shape[0] == 1 or item_sx.shape[1] == 1:
            return Vector.from_casadi_sx(item_sx)
        return Matrix.from_casadi_sx(item_sx)

    def __setitem__(
        self,
        key: int | slice | Tuple[int | slice, int | slice],
        value: ScalarData,
    ):
        """
        Gives this class the setitem behavior of numpy.
        """
        self.casadi_sx[key] = value.casadi_sx if hasattr(value, "casadi_sx") else value

    @property
    def shape(self) -> Tuple[int, int]:
        return self.casadi_sx.shape

    def __len__(self) -> int:
        return self.shape[0]

    def free_variables(self) -> List[FloatVariable]:
        return [FloatVariable._registry[s] for s in ca.symvar(self.casadi_sx)]

    def is_constant(self) -> bool:
        return self.casadi_sx.is_constant()

    def to_np(self) -> np.ndarray:
        """
        Transforms the data into a numpy array.
        Only works if the expression has no free variables.
        """
        if not self.is_constant():
            raise HasFreeVariablesError(self.free_variables())

        dm = ca.DM(self.casadi_sx)
        out = dm.full()  # numpy.ndarray (2-D)

        if out.shape[0] == 1 or out.shape[1] == 1:
            # CasADi uses column-major internally; preserve that on flatten
            return out.ravel(order="F")

        return out

    def to_list(self) -> list:
        """
        Converts the symbolic expression into a nested Python list, like numpy.tolist.

        The expression must be constant; otherwise a HasFreeVariablesError is raised.
        """
        return self.to_np().tolist()

    def safe_division(
        self,
        other: GenericSymbolicType,
        if_nan: Optional[ScalarData] = None,
    ) -> GenericSymbolicType:
        """
        A version of division where no sub-expression is ever NaN. The expression would evaluate to 'if_nan', but
        you should probably never work with the 'if_nan' result. However, if one sub-expressions is NaN, the whole expression
        evaluates to NaN, even if it is only in a branch of an if-else, that is not returned.
        This method is a workaround for such cases.
        """
        if if_nan is None:
            if_nan = 0
        save_denominator = if_eq_zero(
            condition=other, if_result=Scalar(data=1), else_result=other
        )
        return if_eq_zero(other, if_result=if_nan, else_result=self / save_denominator)

    def compile(
        self,
        parameters: Optional[VariableParameters] = None,
        sparse: bool = False,
    ) -> CompiledFunction:
        """
        Compiles the function into a representation that can be executed efficiently. This method
        allows for optional parameterization and the ability to specify whether the compilation
        should consider a sparse representation.

        :param parameters: A list of parameter sets, where each set contains variables that define
            the configuration for the compiled function. If set to None, no parameters are applied.
        :param sparse: A boolean that determines whether the compiled function should use a
            sparse representation. Defaults to False.
        :return: The compiled function as an instance of CompiledFunction.
        """
        return CompiledFunction(self, parameters, sparse)

    def evaluate(self) -> np.ndarray:
        """
        Substitutes the free variables in this expression using their `resolve` method and compute the result.
        :return: The evaluated value of this expression.
        """
        f = self.compile(
            VariableParameters.from_lists(self.free_variables()), sparse=False
        )
        return f(
            np.array([s.resolve() for s in self.free_variables()], dtype=np.float64)
        )

    def substitute(
        self,
        old_variables: List[FloatVariable],
        new_variables: List[ScalarData] | Vector,
    ) -> Self:
        """
        Replace variables in an expression with new variables or expressions.

        This function substitutes variables in the given expression with the provided
        new variables or expressions. It ensures that the original expression remains
        unaltered and creates a new instance with the substitutions applied.

        :param old_variables: A list of variables in the expression which need to be replaced.
        :param new_variables: A list of new variables or expressions which will replace the old variables.
            The length of this list must correspond to the `old_variables` list.
        :return: A new expression with the specified variables replaced.
        """
        old_variables = to_sx(old_variables)
        new_variables = ca.densify(to_sx(new_variables))
        result = copy.copy(self)
        result.casadi_sx = ca.substitute(result.casadi_sx, old_variables, new_variables)
        return result

    def equivalent(self, other: ScalarData) -> bool:
        """
        Determines whether two scalar expressions are mathematically equivalent by simplifying
        and comparing them.

        :param other: Second scalar expression to compare
        :return: True if the two expressions are equivalent, otherwise False
        """
        other_expression = to_sx(other)
        return ca.is_equal(
            ca.simplify(self.casadi_sx), ca.simplify(other_expression), 5
        )

    def __copy__(self) -> Self:
        return self.from_casadi_sx(copy.copy(self.casadi_sx))

    def __neg__(self) -> Self:
        return self.from_casadi_sx(self.casadi_sx.__neg__())

    def __abs__(self) -> Self:
        return self.from_casadi_sx(ca.fabs(self.casadi_sx))

    def jacobian(self, variables: Iterable[FloatVariable]) -> Matrix:
        """
        Compute the Jacobian matrix of a vector of expressions with respect to a vector of variables.

        This function calculates the Jacobian matrix, which is a matrix of all first-order
        partial derivatives of a vector of functions with respect to a vector of variables.

        :param variables: The variables with respect to which the partial derivatives are taken.
        :return: The Jacobian matrix as an SymbolicMathType.
        """
        return Matrix.from_casadi_sx(ca.jacobian(self.casadi_sx, to_sx(variables)))

    def jacobian_dot(
        self,
        variables: Iterable[FloatVariable],
        variables_dot: Iterable[FloatVariable],
    ) -> Matrix:
        """
        Compute the total derivative of the Jacobian matrix.

        This function calculates the time derivative of a Jacobian matrix given
        a set of expressions and variables, along with their corresponding
        derivatives. For each element in the Jacobian matrix, this method
        computes the total derivative based on the provided variables and
        their time derivatives.

        :param variables: Iterable containing the variables with respect to which
            the Jacobian is calculated.
        :param variables_dot: Iterable containing the time derivatives of the
            corresponding variables in `variables`.
        :return: The time derivative of the Jacobian matrix.
        """
        Jd = self.jacobian(variables)
        for i in range(Jd.shape[0]):
            for j in range(Jd.shape[1]):
                Jd[i, j] = Jd[i, j].total_derivative(variables, variables_dot)
        return Jd

    def jacobian_ddot(
        self,
        variables: Iterable[FloatVariable],
        variables_dot: Iterable[FloatVariable],
        variables_ddot: Iterable[FloatVariable],
    ) -> Matrix:
        """
        Compute the second-order total derivative of the Jacobian matrix.

        This function computes the Jacobian matrix of the given expressions with
        respect to specified variables and further calculates the second-order
        total derivative for each element in the Jacobian matrix with respect to
        the provided variables, their first-order derivatives, and their second-order
        derivatives.

        :param variables: An iterable of symbolic variables representing the
            primary variables with respect to which the Jacobian and derivatives
            are calculated.
        :param variables_dot: An iterable of symbolic variables representing the
            first-order derivatives of the primary variables.
        :param variables_ddot: An iterable of symbolic variables representing the
            second-order derivatives of the primary variables.
        :return: A symbolic matrix representing the second-order total derivative
            of the Jacobian matrix of the provided expressions.
        """
        Jdd = self.jacobian(variables)
        for i in range(Jdd.shape[0]):
            for j in range(Jdd.shape[1]):
                Jdd[i, j] = Jdd[i, j].second_order_total_derivative(
                    variables, variables_dot, variables_ddot
                )
        return Jdd

    def total_derivative(
        self,
        variables: Iterable[FloatVariable],
        variables_dot: Iterable[FloatVariable],
    ) -> Vector:
        """
        Compute the total derivative of an expression with respect to given variables and their derivatives
        (dot variables).

        The total derivative accounts for a dependent relationship where the specified variables represent
        the variables of interest, and the dot variables represent the time derivatives of those variables.

        :param variables: Iterable of variables with respect to which the derivative is computed.
        :param variables_dot: Iterable of dot variables representing the derivatives of the variables.
        :return: The expression resulting from the total derivative computation.
        """
        return Vector.from_casadi_sx(
            ca.jtimes(self.casadi_sx, to_sx(variables), to_sx(variables_dot))
        )

    def second_order_total_derivative(
        self,
        variables: Iterable[FloatVariable],
        variables_dot: Iterable[FloatVariable],
        variables_ddot: Iterable[FloatVariable],
    ) -> Vector:
        """
        Computes the second-order total derivative of an expression with respect to a set of variables.

        This function takes an expression and computes its second-order total derivative
        using provided variables, their first-order derivatives, and their second-order
        derivatives. The computation internally constructs a Hessian matrix of the
        expression and multiplies it by a vector that combines the provided derivative
        data.

        :param variables: Iterable containing the variables with respect to which the derivative is calculated.
        :param variables_dot: Iterable containing the first-order derivatives of the variables.
        :param variables_ddot: Iterable containing the second-order derivatives of the variables.
        :return: The computed second-order total derivative, returned as an `SymbolicMathType`.
        """
        variables = to_sx(variables)
        variables_dot = to_sx(variables_dot)
        variables_ddot = to_sx(variables_ddot)
        v = []
        for i in range(variables.shape[0]):
            for j in range(variables.shape[0]):
                if i == j:
                    v.append(variables_ddot[i])
                else:
                    v.append(variables_dot[i] * variables_dot[j])
        v = Vector(v)
        H = Matrix(ca.hessian(to_sx(self), variables)[0])
        H = H.reshape((1, len(H) ** 2))
        return H.dot(v)


@dataclass(eq=False, init=False)
class Scalar(SymbolicMathType):
    """
    A symbolic type representing a scalar value.
    """

    def __init__(self, data: ScalarData = 0):
        self.casadi_sx = to_sx(data)

    def _verify_type(self):
        if self.casadi_sx.shape != (1, 1):
            raise NotScalerError(self.casadi_sx.shape)

    # %% Boolean operations
    @classmethod
    def const_false(cls) -> Self:
        return cls(False)

    @classmethod
    def const_trinary_unknown(cls) -> Self:
        return cls(0.5)

    @classmethod
    def const_true(cls) -> Self:
        return cls(True)

    def is_const_true(self):
        return self.is_constant() and self == True

    def is_const_unknown(self):
        return self.is_constant() and self == 0.5

    def is_const_false(self):
        return self.is_constant() and self == False

    def __bool__(self) -> bool:
        """
        Evaluates the object as a boolean value, implementing the `__bool__` special method.
        If the expression is scalar and constant, it is evaluated as a python bool.
            This allows comparisons to work as expected, e.g. `if x > 0:`
        If the expression is scaler, non-constant and an ==, we use casadi's equivalent.
            This allows `2*FloatVariable("a") == FloatVariable("a")*2` to work as expected.
        """
        if self.is_constant():
            return bool(self.to_np())
        elif self.casadi_sx.op() == ca.OP_EQ:
            # not evaluating bool would cause all expressions containing == to be evaluated to True, because they are not None
            # this can cause a lot of unintended bugs, therefore we try to evaluate it
            left = self.casadi_sx.dep(0)
            right = self.casadi_sx.dep(1)
            return ca.is_equal(ca.simplify(left), ca.simplify(right), 5)
        raise HasFreeVariablesError(self.free_variables())

    def __neg__(self) -> Scalar:
        return Scalar.from_casadi_sx(self.casadi_sx.__neg__())

    def __invert__(self) -> Scalar:
        return Scalar.from_casadi_sx(ca.logic_not(self.casadi_sx))

    def __and__(self, other: Scalar | FloatVariable) -> Scalar:
        if is_const_false(self):
            return self
        if is_const_false(other):
            return other
        return Scalar.from_casadi_sx(ca.logic_and(to_sx(self), to_sx(other)))

    def __or__(self, other: Scalar | FloatVariable) -> Scalar:
        if is_const_true(self):
            return self
        if is_const_true(other):
            return other
        return Scalar.from_casadi_sx(ca.logic_or(to_sx(self), to_sx(other)))

    # %% Comparison operations
    def _compare(
        self, other: Scalar | FloatVariable | NumericalScalar | bool, op_f: Callable
    ) -> Scalar | bool:
        left = to_sx(self)
        right = to_sx(other)
        result = op_f(left, right)
        if result.is_constant():
            return bool(result)
        return Scalar.from_casadi_sx(result)

    def __eq__(
        self, other: Scalar | FloatVariable | NumericalScalar | bool
    ) -> Scalar | bool:
        return self._compare(other, operator.eq)

    def __le__(self, other: Scalar | FloatVariable) -> Scalar | bool:
        return self._compare(other, operator.le)

    def __lt__(self, other: Scalar | FloatVariable) -> Scalar | bool:
        return self._compare(other, operator.lt)

    def __ge__(self, other: Scalar | FloatVariable) -> Scalar | bool:
        return self._compare(other, operator.ge)

    def __gt__(self, other: Scalar | FloatVariable) -> Scalar | bool:
        return self._compare(other, operator.gt)

    # %% Arithmatic operations
    def __float__(self):
        if not self.is_constant():
            raise HasFreeVariablesError(self.free_variables())
        return float(self._casadi_sx)

    def hessian(self, variables: Iterable[FloatVariable]) -> Matrix:
        """
        Calculate the Hessian matrix of a given expression with respect to specified variables.

        The function computes the second-order partial derivatives (Hessian matrix) for a
        provided mathematical expression using the specified variables. It utilizes a symbolic
        library for the internal operations to generate the Hessian.

        :param variables: An iterable containing the variables with respect to which the derivatives
            are calculated.
        :return: The resulting Hessian matrix as an expression.
        """
        expressions = self.casadi_sx
        return Matrix(ca.hessian(expressions, to_sx(variables))[0])

    def _binary(self, other, op: Callable[[ca.SX, ca.SX], ca.SX]):
        if not isinstance(other, ScalarData):
            return NotImplemented
        a = to_sx(self)
        b = to_sx(other)
        return Scalar.from_casadi_sx(op(a, b))

    def _rbinary(self, other, op: Callable[[ca.SX, ca.SX], ca.SX]):
        if not isinstance(other, ScalarData):
            return NotImplemented
        a = to_sx(other)
        b = to_sx(self)
        return Scalar.from_casadi_sx(op(a, b))

    def __add__(self, other: ScalarData) -> Scalar:
        return self._binary(other, operator.add)

    def __radd__(self, other: ScalarData) -> Scalar:
        return self._rbinary(other, operator.add)

    def __sub__(self, other: ScalarData) -> Scalar:
        return self._binary(other, operator.sub)

    def __rsub__(self, other: ScalarData) -> Scalar:
        return self._rbinary(other, operator.sub)

    def __mul__(self, other: ScalarData) -> Scalar:
        return self._binary(other, operator.mul)

    def __rmul__(self, other: ScalarData) -> Scalar:
        return self._rbinary(other, operator.mul)

    def __truediv__(self, other: ScalarData) -> Scalar:
        return self._binary(other, operator.truediv)

    def __rtruediv__(self, other: ScalarData) -> Scalar:
        return self._rbinary(other, operator.truediv)

    def __pow__(self, other: ScalarData) -> Scalar:
        return self._binary(other, operator.pow)

    def __rpow__(self, other: ScalarData) -> Scalar:
        return self._rbinary(other, operator.pow)

    def __floordiv__(self, other: ScalarData) -> Scalar:
        return self._binary(other, lambda a, b: ca.floor(to_sx(a) / to_sx(b)))

    def __rfloordiv__(self, other: ScalarData) -> Scalar:
        return self._rbinary(other, lambda a, b: ca.floor(to_sx(a) / to_sx(b)))

    def __mod__(self, other: ScalarData) -> Scalar:
        return self._binary(other, ca.fmod)

    def __rmod__(self, other: ScalarData) -> Scalar:
        return self._rbinary(other, ca.fmod)


@dataclass(eq=False, init=False)
class FloatVariable(Scalar):
    """
    A symbolic expression representing a single float variable.
    Applying any operation on a FloatVariable results in a Scalar.
    """

    name: str = field(kw_only=True)

    _registry: ClassVar[Dict[ca.SX, FloatVariable]] = {}
    """
    Keeps track of which FloatVariable instances are associated with which which casadi.SX instances.
    Needed to recreate the FloatVariables from a casadi expression.
    .. warning:: Does not ensure that two FloatVariable instances are identical.
    """

    def __init__(self, name: str):
        self.name = name
        casadi_sx = ca.SX.sym(self.name)
        self._registry[casadi_sx] = self
        super().__init__(casadi_sx)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return f"Variable({self})"

    def __hash__(self):
        return hash(self.casadi_sx)

    def resolve(self) -> float:
        """
        This method is called by SymbolicType.evaluate().
        Subclasses should override this method to return the current float value for this variable.
        :return: This variables' current value.
        """
        return np.nan


@dataclass(eq=False)
class Vector(SymbolicMathType):
    """
    A vector of symbolic expressions.
    Should behave like a numpy array with one dimension.
    """

    def __init__(
        self,
        data: Optional[VectorData] = None,
    ):
        if data is None:
            data = []
        self.casadi_sx = to_sx(data)

    def _verify_type(self):
        """
        In numpy a 1d array acts like a matrix with a single column.
        Since casadi is always 2d, we reshape the vector to be a single column matrix.
        """
        if self.shape[0] == 1:
            self._casadi_sx = self._casadi_sx.T
        assert self.shape[1] == 1 or self.shape == (0, 0)

    @classmethod
    def zeros(cls, size: int) -> Self:
        return cls(np.zeros(size))

    @classmethod
    def ones(cls, size: int) -> Self:
        return cls(np.ones(size))

    def __iter__(self):
        """
        Iterate over the elements of the vector, yielding Scalar objects.

        This mirrors NumPy's behavior for 1D arrays where iteration returns
        individual scalar elements in order of the first axis.
        """
        for i in range(self.shape[0]):
            yield self[i]

    def __add__(self, other: ScalarData | VectorData) -> Self:
        return Vector.from_casadi_sx(to_sx(self) + to_sx(other))

    def __sub__(self, other: ScalarData | VectorData) -> Self:
        return Vector.from_casadi_sx(to_sx(self) - to_sx(other))

    def __mul__(self, other: ScalarData | VectorData) -> Self:
        return Vector.from_casadi_sx(to_sx(self) * to_sx(other))

    def __truediv__(self, other: ScalarData | VectorData) -> Self:
        return Vector.from_casadi_sx(to_sx(self) / to_sx(other))

    def __pow__(self, other: ScalarData | VectorData) -> Self:
        return Vector.from_casadi_sx(to_sx(self) ** to_sx(other))

    def __floordiv__(self, other: ScalarData | VectorData) -> Self:
        return Vector.from_casadi_sx(ca.floor(to_sx(self) / to_sx(other)))

    def __mod__(self, other: ScalarData | VectorData) -> Self:
        return fmod(self, other)

    def dot(self, other: GenericSymbolicType) -> Scalar | Vector:
        """
        Same as numpy dot.
        """
        if isinstance(other, Matrix):  # copy numpy logic, where vectors only have 1 dim
            return Vector.from_casadi_sx(ca.mtimes(to_sx(self).T, to_sx(other)))
        if isinstance(other, Vector):
            return Scalar.from_casadi_sx(ca.mtimes(to_sx(self).T, to_sx(other)))
        raise UnsupportedOperationError("dot", self, other)

    def __matmul__(self, other: GenericSymbolicType) -> GenericSymbolicType:
        return self.dot(other)

    # %% Comparison operations
    def __eq__(self, other: Vector) -> Vector | bool:
        if self.is_constant() and other.is_constant():
            return self.to_np() == other.to_np()
        return Vector.from_casadi_sx(to_sx(self).__eq__(to_sx(other)))

    def __le__(self, other: Vector) -> Self:
        return self.from_casadi_sx(to_sx(self).__le__(to_sx(other)))

    def __lt__(self, other: Vector) -> Self:
        return self.from_casadi_sx(to_sx(self).__lt__(to_sx(other)))

    def __ge__(self, other: Vector) -> Self:
        return self.from_casadi_sx(to_sx(self).__ge__(to_sx(other)))

    def __gt__(self, other: Vector) -> Self:
        return self.from_casadi_sx(to_sx(self).__gt__(to_sx(other)))

    def euclidean_distance(self, other: Self) -> Scalar:
        difference = self - other
        distance = difference.norm()
        return distance

    def norm(self) -> Scalar:
        """
        Computes the 2-norm (Euclidean norm) of the current object.

        :return: The 2-norm of the object, represented as a `Scalar` type.
        """
        return Scalar.from_casadi_sx(ca.norm_2(to_sx(self)))

    def scale(self, a: ScalarData) -> Vector:
        """
        Scales the current vector proportionally based on the provided scalar value.

        :param a: A scalar value used to scale the vector
        :return: A new vector resulting from the scaling operation
        """
        return self.safe_division(self.norm()) * a

    def concatenate(self, other: Vector) -> Vector:
        """
        Concatenates the calling vector object with another vector, resulting in
        a single unified vector.

        :param other: The vector to concatenate with the current vector.
        :return: A new vector object representing the combined result of the two vectors.
        """
        return Vector.from_casadi_sx(ca.vertcat(to_sx(self), to_sx(other)))


@dataclass(eq=False)
class Matrix(SymbolicMathType):
    """
    A matrix of symbolic expressions.
    Should behave like a 2d numpy array.
    """

    def __init__(
        self,
        data: Optional[VectorData | MatrixData] = None,
    ):
        if data is None:
            data = []
        self.casadi_sx = to_sx(data)

    @classmethod
    def create_filled_with_variables(cls, shape: Tuple[int, int], name: str) -> Self:
        data = []
        for row in range(shape[0]):
            row_data = []
            for col in range(shape[1]):
                row_data.append(FloatVariable(f"{name}_{row}_{col}"))
            data.append(row_data)
        return cls(data)

    def _verify_type(self):
        """
        Any shape is fine for a matrix.
        """

    def __iter__(self):
        """
        Iterate over the first axis of the matrix, yielding Vector rows.

        This mirrors NumPy's behavior for 2D arrays where iteration returns
        1D row views along axis 0.
        """
        for i in range(self.shape[0]):
            yield Vector.from_casadi_sx(self.casadi_sx[i, :])

    def __add__(self, other: Scalar | Vector | Matrix) -> Self:
        other_sx = self._broadcast_like_self(other)
        return Matrix.from_casadi_sx(self.casadi_sx + other_sx)

    def __radd__(self, other: Scalar | Vector | Matrix) -> Self:
        other_sx = self._broadcast_like_self(other)
        return Matrix.from_casadi_sx(self.casadi_sx + other_sx)

    def __sub__(self, other: Scalar | Vector | Matrix) -> Self:
        other_sx = self._broadcast_like_self(other)
        return Matrix.from_casadi_sx(self.casadi_sx - other_sx)

    def __rsub__(self, other: Scalar | Vector | Matrix) -> Self:
        other_sx = self._broadcast_like_self(other)
        return Matrix.from_casadi_sx(self.casadi_sx - other_sx)

    def __mul__(self, other: Scalar | Vector | Matrix) -> Self:
        other_sx = self._broadcast_like_self(other)
        return Matrix.from_casadi_sx(self.casadi_sx * other_sx)

    def __rmul__(self, other: Scalar | Vector | Matrix) -> Self:
        other_sx = self._broadcast_like_self(other)
        return Matrix.from_casadi_sx(self.casadi_sx * other_sx)

    def __truediv__(self, other: Scalar | Vector | Matrix) -> Self:
        other_sx = self._broadcast_like_self(other)
        num = self.casadi_sx
        den = other_sx
        zero_den = ca.eq(den, 0)
        zero_num = ca.eq(num, 0)
        signed_inf = ca.sign(num) * ca.SX(np.inf)
        nan_const = ca.SX(np.nan)
        base_div = num / den
        # Where denominator is zero, use signed infinity; where both numerator and denominator are zero, use NaN
        result = ca.if_else(
            zero_den, ca.if_else(zero_num, nan_const, signed_inf), base_div
        )
        return Matrix.from_casadi_sx(result)

    def __pow__(self, other: Scalar | Vector | Matrix) -> Self:
        other_sx = self._broadcast_like_self(other)
        return Matrix.from_casadi_sx(self.casadi_sx**other_sx)

    def __floordiv__(self, other: Scalar | Vector | Matrix) -> Self:
        other_sx = self._broadcast_like_self(other)
        num = self.casadi_sx
        den = other_sx
        zero_den = ca.eq(den, 0)
        # numpy: floor_divide by zero yields 0 and issues a warning; we mimic the value semantics
        div = ca.floor(num / den)
        result = ca.if_else(zero_den, ca.SX.zeros(*self.shape), div)
        return Matrix.from_casadi_sx(result)

    def __mod__(self, other: Scalar | Vector | Matrix) -> Self:
        other_sx = self._broadcast_like_self(other)
        num = self.casadi_sx
        den = other_sx
        zero_den = ca.eq(den, 0)
        mod_val = ca.fmod(num, den)
        result = ca.if_else(zero_den, ca.SX.zeros(*self.shape), mod_val)
        return Matrix.from_casadi_sx(result)

    def dot(self, other: GenericSymbolicType) -> GenericSymbolicType:
        """
        Same as numpy dot.
        """
        return _create_return_type(other).from_casadi_sx(
            ca.mtimes(to_sx(self), to_sx(other))
        )

    def __matmul__(self, other: GenericSymbolicType) -> GenericSymbolicType:
        return self.dot(other)

    # %% Comparison operations
    def __eq__(self, other: Matrix) -> Self | bool:
        if self.is_constant() and other.is_constant():
            return self.to_np() == other.to_np()
        return Matrix.from_casadi_sx(self.casadi_sx.__eq__(other.casadi_sx))

    def __le__(self, other: Scalar | FloatVariable) -> Scalar:
        return Scalar.from_casadi_sx(self.casadi_sx.__le__(other.casadi_sx))

    def __lt__(self, other: Scalar | FloatVariable) -> Scalar:
        return Scalar.from_casadi_sx(self.casadi_sx.__lt__(other.casadi_sx))

    def __ge__(self, other: Scalar | FloatVariable) -> Scalar:
        return Scalar.from_casadi_sx(self.casadi_sx.__ge__(other.casadi_sx))

    def __gt__(self, other: Scalar | FloatVariable) -> Scalar:
        return Scalar.from_casadi_sx(self.casadi_sx.__gt__(other.casadi_sx))

    def _broadcast_like_self(self, other: SymbolicMathType) -> ca.SX:
        """
        Broadcast the other operand to match this matrix's shape for element-wise operations.

        Rules:
        - Scalar: allowed without change.
        - Vector: broadcast across rows, mirroring numpy behavior.
        - Matrix with identical shape: allowed.
        Otherwise, raise WrongDimensionsError.
        """
        other_sx = to_sx(other)
        if other_sx.is_scalar():
            return other_sx
        # Exact same shape → no broadcasting required
        if self.shape == other_sx.shape:
            return other_sx
        # Vector column broadcasting: 1d vectors in numpy behave like row vectors for broadcasting
        if isinstance(other, Vector) and other_sx.shape[0] == self.shape[0]:
            return ca.repmat(other_sx.T, self.shape[0], 1)
        # If we reach here, shapes are incompatible
        raise WrongDimensionsError(self.shape, other_sx.shape)

    @classmethod
    def zeros(cls, rows: int, columns: int) -> Self:
        """
        See numpy.zeros.
        """
        return cls.from_casadi_sx(casadi_sx=ca.SX.zeros(rows, columns))

    @classmethod
    def ones(cls, x: int, y: int) -> Self:
        """
        See numpy.ones.
        """
        return cls.from_casadi_sx(casadi_sx=ca.SX.ones(x, y))

    @classmethod
    def tri(cls, dimension: int) -> Self:
        """
        See numpy.tri.
        """
        return cls(data=np.tri(dimension))

    @classmethod
    def eye(cls, size: int) -> Self:
        """
        See numpy.eye.
        """
        return cls.from_casadi_sx(casadi_sx=ca.SX.eye(size))

    @classmethod
    def diag(cls, args: VectorData) -> Self:
        """
        See numpy.diag.
        """
        return cls.from_casadi_sx(casadi_sx=ca.diag(to_sx(args)))

    @classmethod
    def vstack(
        cls,
        list_of_matrices: VectorData | MatrixData,
    ) -> Self:
        """
        See numpy.vstack.
        """
        if len(list_of_matrices) == 0:
            return cls(data=[])
        return cls.from_casadi_sx(
            casadi_sx=ca.vertcat(*[to_sx(x) for x in list_of_matrices])
        )

    @classmethod
    def hstack(
        cls,
        list_of_matrices: VectorData | MatrixData,
    ) -> Self:
        """
        See numpy.hstack.
        """
        if len(list_of_matrices) == 0:
            return cls(data=[])
        return cls.from_casadi_sx(
            casadi_sx=ca.horzcat(*[to_sx(x) for x in list_of_matrices])
        )

    @classmethod
    def diag_stack(
        cls,
        list_of_matrices: VectorData | MatrixData,
    ) -> Self:
        """
        See numpy.diag_stack.
        """
        num_rows = int(math.fsum(e.shape[0] for e in list_of_matrices))
        num_columns = int(math.fsum(e.shape[1] for e in list_of_matrices))
        combined_matrix = Matrix.zeros(num_rows, num_columns)
        row_counter = 0
        column_counter = 0
        for matrix in list_of_matrices:
            combined_matrix[
                row_counter : row_counter + matrix.shape[0],
                column_counter : column_counter + matrix.shape[1],
            ] = matrix
            row_counter += matrix.shape[0]
            column_counter += matrix.shape[1]
        return combined_matrix

    def remove(self, rows: List[int], columns: List[int]):
        """
        Removes the specified rows and columns from the matrix.
        :param rows: Row ids to be removed
        :param columns: Column ids to be removed
        """
        self.casadi_sx.remove(rows, columns)

    def sum(self) -> Scalar:
        """
        the equivalent to _np.sum(matrix)
        """
        return Scalar.from_casadi_sx(ca.sum1(ca.sum2(self.casadi_sx)))

    def sum_row(self) -> Self:
        """
        the equivalent to _np.sum(matrix, axis=0)
        """
        return Vector.from_casadi_sx(ca.sum1(self.casadi_sx))

    def sum_column(self) -> Self:
        """
        the equivalent to _np.sum(matrix, axis=1)
        """
        return Vector.from_casadi_sx(ca.sum2(self.casadi_sx))

    def trace(self) -> Scalar:
        """
        See numpy.trace.
        """
        if not self.is_square():
            raise NotSquareMatrixError(actual_dimensions=self.casadi_sx.shape)
        s = 0
        for i in range(self.casadi_sx.shape[0]):
            s += self.casadi_sx[i, i]
        return Scalar(s)

    def det(self) -> Scalar:
        """
        See numpy.linalg.det.
        """
        if not self.is_square():
            raise NotSquareMatrixError(actual_dimensions=self.casadi_sx.shape)
        return Scalar(ca.det(self.casadi_sx))

    def is_square(self) -> bool:
        return self.casadi_sx.shape[0] == self.casadi_sx.shape[1]

    @property
    def T(self) -> Self:
        """
        :return: the Transpose of the matrix.
        """
        return Matrix(self.casadi_sx.T)

    def reshape(self, new_shape: Tuple[int, int]) -> Self:
        """
        See numpy.reshape.
        """
        rows, cols = new_shape
        # CasADi uses column-major ordering for reshape, whereas NumPy use row-major ordering. To emulate NumPy's
        # behavior, we transpose before and after reshaping with swapped dims.
        reshaped = ca.reshape(self.casadi_sx.T, cols, rows).T
        return Matrix.from_casadi_sx(reshaped)

    def inverse(self) -> Matrix:
        """
        Computes the matrix inverse. Only works if the expression is square.
        """
        assert self.shape[0] == self.shape[1]
        return Matrix(ca.inv(self.casadi_sx))

    def flatten(self) -> Vector:
        """
        Returns a row-major flattened Vector, matching numpy.ndarray.flatten(order='C').
        """
        rows, cols = self.shape
        if rows == 0 or cols == 0:
            return Vector([])
        flat = ca.reshape(self.casadi_sx.T, rows * cols, 1)
        return Vector.from_casadi_sx(flat)

    def kron(self, other: Matrix) -> Self:
        """
        Compute the Kronecker product of two given matrices.

        The Kronecker product is a block matrix construction, derived from the
        direct product of two matrices. It combines the entries of the first
        matrix (`m1`) with each entry of the second matrix (`m2`) by a rule
        of scalar multiplication. This operation extends to any two matrices
        of compatible shapes.

        :param other: The second matrix to be used in calculating the Kronecker product.
                   Supports symbolic or numerical matrix types.
        :return: An SymbolicMathType representing the resulting Kronecker product as a
                 symbolic or numerical matrix of appropriate size.
        """
        m1 = to_sx(self)
        m2 = to_sx(other)
        return Matrix(ca.kron(m1, m2))


def _create_return_type(input_type: SymbolicMathType) -> Type[SymbolicMathType]:
    """
    Determines the return type based on the given input type.

    This function analyzes the `input_type` parameter to decide the appropriate
    return type. If `input_type` is an instance of specific types such as
    `FloatVariable`, `int`, `float`, `bool`, or `IntEnum`, the function
    returns the `Scalar` type. For any other type, it returns the
    type of the given `input_type`.

    :param input_type: The input symbolic math type object to determine the
        corresponding return type.
    :return: The determined return type, either `Scalar` for specific
        input types or the same type as `input_type` for others.
    """
    if isinstance(input_type, (FloatVariable, int, float, bool, IntEnum)):
        return Scalar
    else:
        return type(input_type)


def to_sx(
    data: (
        NumericalScalar
        | NumericalVector
        | NumericalMatrix
        | Iterable[FloatVariable]
        | SymbolicMathType
    ),
) -> ca.SX:
    """
    Tries to turn anything into a casadi SX object.
    :param data: input data to be converted to SX
    :return: casadi SX object
    """
    if isinstance(data, ca.SX):
        return data
    if isinstance(data, SymbolicMathType):
        return data.casadi_sx
    if isinstance(data, NumericalScalar):
        return ca.SX(data)
    return array_like_to_casadi_sx(data)


def array_like_to_casadi_sx(data: VectorData) -> ca.SX:
    """
    Converts a given array-like data structure into a CasADi SX matrix. The input
    data can be a list, tuple, or numpy array. Based on the structure of the input
    data, the function determines the dimensions of the resulting CasADi SX object
    and populates it with values using the `to_sx` function.

    :param data: Input array-like data. It can be a 1D or 2D array-like structure,
        such as a list, tuple, or numpy array.
    :return: A CasADi SX object representation of the input data.
    """
    x = len(data)
    if x == 0:
        return ca.SX()
    first = data[0]
    is_row_like = isinstance(first, (list, tuple, np.ndarray))
    y = len(first) if is_row_like else 1
    try:
        # Attempt to convert to a numeric numpy array. This raises if mixed/object.
        # Note: np.array(..., dtype=float) will fail on symbolic entries.
        if is_row_like:
            arr = np.array(data, dtype=float)
            if arr.ndim != 2:
                # Fall back if irregular nesting
                raise ValueError
        else:
            arr = np.array(data, dtype=float).reshape((-1, 1))
        # Verify that conversion did not introduce non-finite values (e.g., from symbolic SX)
        if not np.isfinite(arr).all():
            raise ValueError
        # CasADi fast conversion: DM -> SX
        return ca.SX(ca.DM(arr))
    except Exception:
        # Mixed/symbolic path
        pass
    if is_row_like:
        # Flatten in column-major order to align with CasADi reshape semantics
        # This ensures that element [i][j] from input ends up at (i, j) in the SX matrix.
        flat = []
        for j in range(y):
            for i in range(x):
                flat.append(to_sx(data[i][j]))
        stacked = ca.vertcat(*flat) if flat else ca.SX()
        return ca.reshape(stacked, x, y)
    else:
        # Column vector
        flat = [to_sx(data[i]) for i in range(x)]
        return ca.vertcat(*flat) if flat else ca.SX(x, 1)


def _unary_function_wrapper(
    casadi_fn: Callable[[ca.SX], ca.SX],
) -> Callable[[GenericSymbolicType], GenericSymbolicType]:
    """
    Wraps a unary CasADi function to allow it to operate on symbolic types while maintaining
    compatibility with the associated symbolic structure. This function converts the input
    to a CasADi symbolic expression, applies the provided CasADi function, and reconverts
    the result back to the symbolic type of the input.

    :param casadi_fn: A CasADi function that transforms a symbolic expression (ca.SX)
        into another symbolic expression of the same type.
    :return: A callable function that applies the CasADi function to an input of type
        GenericSymbolicType and returns an output of the same type.
    """

    def f(x: GenericSymbolicType) -> GenericSymbolicType:
        return _create_return_type(x).from_casadi_sx(casadi_fn(to_sx(x)))

    f.__name__ = casadi_fn.__name__
    f.__doc__ = (
        f"Applies {f.__name__} to the expression. Look at numpy for documentation."
    )
    return f


def _binary_function_wrapper(
    casadi_fn: Callable[[ca.SX, ca.SX], ca.SX],
) -> Callable[[GenericSymbolicType, GenericSymbolicType], GenericSymbolicType]:
    """
    Wraps a CasADi callable into a function that operates with a given symbolic type.
    The returned function applies the CasADi operation to two symbolic arguments,
    while handling their conversion to and from CasADi types as needed.

    :param casadi_fn: The CasADi function to be wrapped. It should accept two CasADi SX
                      symbolic expressions and return a CasADi SX symbolic expression.
    :return: A callable function that accepts two symbolic arguments of a generic
             symbolic type and returns a result of the same type after applying the
             wrapped CasADi operation.
    """

    def f(x: GenericSymbolicType, y: GenericSymbolicType) -> GenericSymbolicType:
        return _create_return_type(x).from_casadi_sx(casadi_fn(to_sx(x), to_sx(y)))

    f.__name__ = casadi_fn.__name__
    f.__doc__ = (
        f"Applies {f.__name__} to the expression. Look at numpy for documentation."
    )
    return f


def create_float_variables(
    names: List[str] | int,
) -> List[FloatVariable]:
    """
    Generates a list of symbolic objects based on the input names or an integer value.

    This function takes either a list of names or an integer. If an integer is
    provided, it generates symbolic objects with default names in the format
    `s_<index>` for numbers up to the given integer. If a list of names is
    provided, it generates symbolic objects for each name in the list.

    :param names: A list of strings representing names of variables or an integer
        specifying the number of variables to generate.
    :return: A list of symbolic objects created based on the input.
    """
    if isinstance(names, int):
        names = [f"s_{i}" for i in range(names)]
    return [FloatVariable(name=x) for x in names]


def diag(args: VectorData | MatrixData) -> Matrix:
    """
    Places the input along the diagonal of a matrix.

    :param args: A vector, list of scalars, or nested lists representing a matrix.
    :return: A square matrix with the input values on its main diagonal.
    """
    return Matrix.diag(args)


def vstack(args: VectorData | MatrixData) -> Matrix:
    """
    Stacks vectors or matrices vertically into a single matrix.

    :param args: A sequence of vectors or matrices with matching column sizes.
    :return: A new matrix containing the inputs stacked by rows.
    """
    return Matrix.vstack(args)


def hstack(args: VectorData | MatrixData) -> Matrix:
    """
    Stacks vectors or matrices horizontally into a single matrix.

    :param args: A sequence of vectors or matrices with matching row sizes.
    :return: A new matrix containing the inputs stacked by columns.
    """
    return Matrix.hstack(args)


def diag_stack(args: VectorData | MatrixData) -> Matrix:
    """
    Builds a block diagonal matrix from the provided inputs.

    :param args: A sequence of vectors or matrices to place on the block diagonal.
    :return: A block diagonal matrix.
    """
    return Matrix.diag_stack(args)


def concatenate(*vectors: Vector) -> Vector:
    """
    Concatenates multiple vectors into a single vector.

    :param vectors: The vectors to concatenate in order.
    :return: A new vector with all inputs concatenated.
    """
    return Vector.from_casadi_sx(ca.vertcat(*[to_sx(v) for v in vectors]))


# %% basic math
abs = _unary_function_wrapper(ca.fabs)


def max(
    arg1: GenericSymbolicType, arg2: Optional[GenericSymbolicType] = None
) -> GenericSymbolicType:
    """
    Returns the maximum element-wise value.

    - With one argument, returns the maximum value across all elements.
    - With two arguments, returns the element-wise maximum.

    :param arg1: The first expression.
    :param arg2: Optional second expression.
    :return: The resulting expression with maximum values.
    """
    if arg2 is None:
        return Scalar.from_casadi_sx(ca.mmax(to_sx(arg1)))
    return _create_return_type(arg1).from_casadi_sx(ca.fmax(to_sx(arg1), to_sx(arg2)))


def min(
    arg1: GenericSymbolicType, arg2: Optional[GenericSymbolicType] = None
) -> GenericSymbolicType:
    """
    Returns the minimum element-wise value.

    - With one argument, returns the minimum value across all elements.
    - With two arguments, returns the element-wise minimum.

    :param arg1: The first expression.
    :param arg2: Optional second expression.
    :return: The resulting expression with minimum values.
    """
    if arg2 is None:
        return Scalar.from_casadi_sx(ca.mmin(to_sx(arg1)))
    return _create_return_type(arg1).from_casadi_sx(ca.fmin(to_sx(arg1), to_sx(arg2)))


def limit(
    x: GenericSymbolicType, lower_limit: ScalarData, upper_limit: ScalarData
) -> GenericSymbolicType:
    """
    Clamps values to the closed interval [lower_limit, upper_limit].

    :param x: The expression to clamp.
    :param lower_limit: The lower bound.
    :param upper_limit: The upper bound.
    :return: The clamped expression.
    """
    return max(lower_limit, min(upper_limit, x))


def dot(
    e1: GenericVectorOrMatrixType, e2: GenericVectorOrMatrixType
) -> GenericVectorOrMatrixType:
    """
    Computes the dot product following NumPy semantics.

    :param e1: The left vector or matrix.
    :param e2: The right vector or matrix.
    :return: The dot product result.
    """
    return e1.dot(e2)


def sum(*expressions: ScalarData) -> Scalar:
    """
    Sums the provided scalar expressions.

    :param expressions: The values to add.
    :return: The total as a scalar expression.
    """
    return Scalar(ca.sum(to_sx(expressions)))


floor = _unary_function_wrapper(ca.floor)
ceil = _unary_function_wrapper(ca.ceil)
sign = _unary_function_wrapper(ca.sign)
exp = _unary_function_wrapper(ca.exp)
log = _unary_function_wrapper(ca.log)
sqrt = _unary_function_wrapper(ca.sqrt)

fmod = _binary_function_wrapper(ca.fmod)


# %% trigonometry
def normalize_angle_positive(angle: ScalarData) -> Scalar:
    """
    Normalizes the angle to be 0 to 2*pi
    It takes and returns radians.
    """
    return fmod(fmod(angle, 2.0 * ca.pi) + 2.0 * ca.pi, 2.0 * ca.pi)


def normalize_angle(angle: ScalarData) -> Scalar:
    """
    Normalizes the angle to be -pi to +pi
    It takes and returns radians.
    """
    a = normalize_angle_positive(angle)
    return if_greater(a, ca.pi, a - 2.0 * ca.pi, a)


def shortest_angular_distance(from_angle: ScalarData, to_angle: ScalarData) -> Scalar:
    """
    Given 2 angles, this returns the shortest angular
    difference.  The inputs and outputs are radians.

    The result would always be -pi <= result <= pi. Adding the result
    to "from" will always get you an equivalent angle to "to".
    """
    return normalize_angle(to_angle - from_angle)


def safe_acos(angle: GenericSymbolicType) -> GenericSymbolicType:
    """
    Limits the angle between -1 and 1 to avoid acos becoming NaN.
    """
    angle = limit(angle, -1, 1)
    return acos(angle)


cos = _unary_function_wrapper(ca.cos)
sin = _unary_function_wrapper(ca.sin)
tan = _unary_function_wrapper(ca.tan)
cosh = _unary_function_wrapper(ca.cosh)
sinh = _unary_function_wrapper(ca.sinh)
acos = _unary_function_wrapper(ca.acos)
atan2 = _binary_function_wrapper(ca.atan2)


# %% other


def solve_for(
    expression: SymbolicMathType,
    target_value: float,
    start_value: float = 0.0001,
    max_tries: int = 10000,
    eps: float = 1e-10,
    max_step: float = 1,
) -> float:
    """
    Solves for a value `x` such that the given mathematical expression, when evaluated at `x`,
    is approximately equal to the target value. The solver iteratively adjusts the value of `x`
    using a numerical approach based on the derivative of the expression.

    :param expression: The mathematical expression to solve. It is assumed to be differentiable.
    :param target_value: The value that the expression is expected to approximate.
    :param start_value: The initial guess for the iterative solver. Defaults to 0.0001.
    :param max_tries: The maximum number of iterations the solver will perform. Defaults to 10000.
    :param eps: The maximum tolerated absolute error for the solution. If the difference
        between the computed value and the target value is less than `eps`, the solution is considered valid. Defaults to 1e-10.
    :param max_step: The maximum adjustment to the value of `x` at each iteration step. Defaults to 1.
    :return: The estimated value of `x` that solves the equation for the given expression and target value.
    :raises ValueError: If no solution is found within the allowed number of steps or if convergence criteria are not met.
    """
    f_dx = expression.jacobian(expression.free_variables()).compile()
    f = expression.compile()
    x = start_value
    for tries in range(max_tries):
        err = f(np.array([x]))[0] - target_value
        if builtins.abs(err) < eps:
            return x
        slope = f_dx(np.array([x]))[0]
        if slope == 0:
            if start_value > 0:
                slope = -0.001
            else:
                slope = 0.001
        x -= builtins.max(builtins.min(err / slope, max_step), -max_step)
    raise ValueError("no solution found")


def gauss(n: ScalarData) -> Scalar:
    """
    Calculate the sum of the first `n` natural numbers using the Gauss formula.

    This function computes the sum of an arithmetic series where the first term
    is 1, the last term is `n`, and the total count of the terms is `n`. The
    result is derived from the formula `(n * (n + 1)) / 2`, which simplifies
    to `(n ** 2 + n) / 2`.

    :param n: The upper limit of the sum, representing the last natural number
              of the series to include.
    :return: The sum of the first `n` natural numbers.
    """
    return (n**2 + n) / 2


# %% binary logic
def is_const_true(expression: Scalar) -> bool:
    """
    Checks whether a scalar expression is the constant truth value.

    :param expression: The scalar expression to test.
    :return: True if the expression is exactly the constant 1, otherwise False.
    """
    return bool(expression == 1)


def is_const_false(expression: Scalar) -> bool:
    """
    Checks whether a scalar expression is the constant false value.

    :param expression: The scalar expression to test.
    :return: True if the expression is exactly the constant 0, otherwise False.
    """
    return bool(expression == 0)


def logic_and(left: ScalarData, right: ScalarData) -> Scalar:
    """
    Logical conjunction on symbolic scalars.

    :param left: The left operand.
    :param right: The right operand.
    :return: The symbolic result of left AND right.
    """
    return left & right


def logic_or(left: ScalarData, right: ScalarData) -> Scalar:
    """
    Logical disjunction on symbolic scalars.

    :param left: The left operand.
    :param right: The right operand.
    :return: The symbolic result of left OR right.
    """
    return left | right


def logic_not(expression: ScalarData) -> Scalar:
    """
    Logical negation on a symbolic scalar.

    :param expression: The operand to negate.
    :return: The symbolic result of NOT expression.
    """
    return ~expression


def logic_any(args: VectorData | MatrixData) -> Scalar:
    """
    Returns True if any element evaluates to True.

    :param args: A vector or matrix of logical scalars.
    :return: A scalar truth value.
    """
    return Scalar.from_casadi_sx(ca.logic_any(args.casadi_sx))


def logic_all(args: GenericVectorOrMatrixType) -> Scalar:
    """
    Returns True if all elements evaluate to True.

    :param args: A vector or matrix of logical scalars.
    :return: A scalar truth value.
    """
    return Scalar.from_casadi_sx(ca.logic_all(to_sx(args)))


# %% trinary logic
def trinary_logic_not(expression: FloatVariable | Scalar) -> Scalar:
    """
            |   Not
    ------------------
    True    |  False
    Unknown | Unknown
    False   |  True
    """
    return Scalar.from_casadi_sx(to_sx(1) - to_sx(expression))


def trinary_logic_and(*args: FloatVariable | Scalar) -> Scalar:
    """
      AND   |  True   | Unknown | False
    ------------------+---------+-------
    True    |  True   | Unknown | False
    Unknown | Unknown | Unknown | False
    False   |  False  |  False  | False
    """
    assert len(args) >= 2, "and must be called with at least 2 arguments"
    # if there is any False, return False
    if any(x for x in args if x.is_const_false()):
        return Scalar.const_false()
    # filter all True
    args = [x for x in args if not x.is_const_true()]
    if len(args) == 0:
        return Scalar.const_true()
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        return min(args[0], args[1])
    else:
        return trinary_logic_and(args[0], trinary_logic_and(*args[1:]))


def trinary_logic_or(*args: FloatVariable | Scalar) -> Scalar:
    """
       OR   |  True   | Unknown | False
    ------------------+---------+-------
    True    |  True   |  True   | True
    Unknown |  True   | Unknown | Unknown
    False   |  True   | Unknown | False
    """
    assert len(args) >= 2, "and must be called with at least 2 arguments"
    # if there is any False, return False
    if any(x for x in args if x.is_const_true()):
        return Scalar.const_true()
    # filter all True
    args = [x for x in args if not x.is_const_true()]
    if len(args) == 0:
        return Scalar.const_false()
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        return max(args[0], args[1])
    else:
        return trinary_logic_or(args[0], trinary_logic_or(*args[1:]))


def trinary_logic_to_str(expression: Scalar) -> str:
    """
    Converts a trinary logic expression into its string representation.

    This function processes an expression with trinary logic values (True, False,
    Unknown) and translates it into a comprehensible string format. It takes into
    account the logical operations involved and recursively evaluates the components
    if necessary. The function handles variables representing trinary logic values,
    as well as logical constructs such as "and", "or", and "not". If the expression
    cannot be evaluated, an exception is raised.

    :param expression: The trinary logic expression to be converted into a string
        representation.
    :return: A string representation of the trinary logic expression, displaying
        the appropriate logical variables and structure.
    :raises SpatialTypesError: If the provided expression cannot be converted
        into a string representation.
    """
    cas_expr = to_sx(expression)

    # Constant case
    if cas_expr.n_dep() == 0:
        if not cas_expr.is_constant():
            return f'"{expression}"'
        if float(expression) == 1.0:
            return "True"
        if float(expression) == 0.0:
            return "False"
        if float(expression) == 0.5:
            return "Unknown"

    match cas_expr.op():
        case ca.OP_SUB:  # trinary "not" is 1-x
            return f"not {trinary_logic_to_str(cas_expr.dep(1))}"
        case ca.OP_FMIN:  # trinary "and" is min(left, right)
            left = trinary_logic_to_str(cas_expr.dep(0))
            right = trinary_logic_to_str(cas_expr.dep(1))
            return f"({left} and {right})"
        case ca.OP_FMAX:  # trinary "or" is max(left, right)
            left = trinary_logic_to_str(cas_expr.dep(0))
            right = trinary_logic_to_str(cas_expr.dep(1))
            return f"({left} or {right})"
        case _:
            raise SymbolicMathError(message=f"cannot convert {expression} to a string")


# %% ifs


def if_else(
    condition: ScalarData,
    if_result: GenericSymbolicType,
    else_result: GenericSymbolicType,
) -> GenericSymbolicType:
    """
    Creates an expression that represents:
    if condition:
        return if_result
    else:
        return else_result
    """
    condition = to_sx(condition)
    if_result_sx = to_sx(if_result)
    else_result_sx = to_sx(else_result)
    return_type = _create_return_type(if_result)
    return return_type.from_casadi_sx(
        ca.if_else(condition, if_result_sx, else_result_sx)
    )


def if_greater(
    a: ScalarData,
    b: ScalarData,
    if_result: GenericSymbolicType,
    else_result: GenericSymbolicType,
) -> GenericSymbolicType:
    """
    Creates an expression that represents:
    if a > b:
        return if_result
    else:
        return else_result
    """
    a = to_sx(a)
    b = to_sx(b)
    return if_else(ca.gt(a, b), if_result, else_result)


def if_less(
    a: ScalarData,
    b: ScalarData,
    if_result: GenericSymbolicType,
    else_result: GenericSymbolicType,
) -> GenericSymbolicType:
    """
    Creates an expression that represents:
    if a < b:
        return if_result
    else:
        return else_result
    """
    a = to_sx(a)
    b = to_sx(b)
    return if_else(ca.lt(a, b), if_result, else_result)


def if_greater_zero(
    condition: ScalarData,
    if_result: GenericSymbolicType,
    else_result: GenericSymbolicType,
) -> GenericSymbolicType:
    """
    Creates an expression that represents:
    if condition > 0:
        return if_result
    else:
        return else_result
    """
    condition = to_sx(condition)
    return if_else(ca.gt(condition, 0), if_result, else_result)


def if_greater_eq_zero(
    condition: ScalarData,
    if_result: GenericSymbolicType,
    else_result: GenericSymbolicType,
) -> GenericSymbolicType:
    """
    Creates an expression that represents:
    if condition >= 0:
        return if_result
    else:
        return else_result
    """
    return if_greater_eq(condition, 0, if_result, else_result)


def if_greater_eq(
    a: ScalarData,
    b: ScalarData,
    if_result: GenericSymbolicType,
    else_result: GenericSymbolicType,
) -> GenericSymbolicType:
    """
    Creates an expression that represents:
    if a >= b:
        return if_result
    else:
        return else_result
    """
    a = to_sx(a)
    b = to_sx(b)
    return if_else(ca.ge(a, b), if_result, else_result)


def if_less_eq(
    a: ScalarData,
    b: ScalarData,
    if_result: GenericSymbolicType,
    else_result: GenericSymbolicType,
) -> GenericSymbolicType:
    """
    Creates an expression that represents:
    if a <= b:
        return if_result
    else:
        return else_result
    """
    return if_greater_eq(b, a, if_result, else_result)


def if_eq_zero(
    condition: ScalarData,
    if_result: GenericSymbolicType,
    else_result: GenericSymbolicType,
) -> GenericSymbolicType:
    """
    Creates an expression that represents:
    if condition == 0:
        return if_result
    else:
        return else_result
    """
    return if_else(condition, else_result, if_result)


def if_eq(
    a: ScalarData,
    b: ScalarData,
    if_result: GenericSymbolicType,
    else_result: GenericSymbolicType,
) -> GenericSymbolicType:
    """
    Creates an expression that represents:
    if a == b:
        return if_result
    else:
        return else_result
    """
    a = to_sx(a)
    b = to_sx(b)
    return if_else(ca.eq(a, b), if_result, else_result)


def if_eq_cases(
    a: ScalarData,
    b_result_cases: Iterable[Tuple[ScalarData, GenericSymbolicType]],
    else_result: GenericSymbolicType,
) -> GenericSymbolicType:
    """
    if a == b_result_cases[0][0]:
        return b_result_cases[0][1]
    elif a == b_result_cases[1][0]:
        return b_result_cases[1][1]
    ...
    else:
        return else_result
    """
    a_sx = to_sx(a)
    result_sx = to_sx(else_result)
    for b, b_result in b_result_cases:
        b_sx = to_sx(b)
        b_result_sx = to_sx(b_result)
        result_sx = ca.if_else(ca.eq(a_sx, b_sx), b_result_sx, result_sx)
    return _create_return_type(else_result).from_casadi_sx(result_sx)


def if_cases(
    cases: Sequence[Tuple[ScalarData, GenericSymbolicType]],
    else_result: GenericSymbolicType,
) -> GenericSymbolicType:
    """
    if cases[0][0]:
        return cases[0][1]
    elif cases[1][0]:
        return cases[1][1]
    ...
    else:
        return else_result
    """
    result_sx = to_sx(else_result)
    for i in reversed(range(len(cases))):
        case = to_sx(cases[i][0])
        case_result = to_sx(cases[i][1])
        result_sx = ca.if_else(case, case_result, result_sx)
    return _create_return_type(else_result).from_casadi_sx(result_sx)


def if_less_eq_cases(
    a: ScalarData,
    b_result_cases: Sequence[Tuple[ScalarData, GenericSymbolicType]],
    else_result: GenericSymbolicType,
) -> GenericSymbolicType:
    """
    This only works if b_result_cases is sorted in ascending order.
    if a <= b_result_cases[0][0]:
        return b_result_cases[0][1]
    elif a <= b_result_cases[1][0]:
        return b_result_cases[1][1]
    ...
    else:
        return else_result
    """
    a_sx = to_sx(a)
    result_sx = to_sx(else_result)
    for i in reversed(range(len(b_result_cases))):
        b_sx = to_sx(b_result_cases[i][0])
        b_result_sx = to_sx(b_result_cases[i][1])
        result_sx = ca.if_else(ca.le(a_sx, b_sx), b_result_sx, result_sx)
    return _create_return_type(else_result).from_casadi_sx(result_sx)


_substitution_cache: Dict[str, Tuple[SymbolicMathType, List[FloatVariable]]] = {}


def substitution_cache(method):
    """
    This decorator allows you to speed up complex symbolic math operations.
    The operator computes the expression once with variables and stores it in a cache.
    On subsequent calls, the cached expression is used and the args are substituted into the variables,
    avoiding rebuilding of the computation graph.
    """

    def wrapper(*args, **kwargs):
        if len(kwargs) > 0:
            raise SymbolicMathError(
                message="substitution_cache does not support kwargs"
            )
        global _substitution_cache
        cache_key = method.__name__
        if not cache_key in _substitution_cache:
            variable_args = []
            for i, arg in enumerate(args):
                variable_args.append(
                    Matrix.create_filled_with_variables(arg.shape, name=f"arg{i}")
                )
            variables = [
                item.free_variables()[0]
                for arg in variable_args
                for item in arg.flatten()
            ]
            result = method(*variable_args)
            _substitution_cache[cache_key] = (result, variables)
        expr, variables = _substitution_cache[cache_key]
        substitutions = [item for arg in args for item in arg.flatten()]
        return expr.substitute(variables, substitutions)

    return wrapper


# %% type hints

NumericalScalar = bool | int | np.int64 | float | np.float64 | IntEnum
NumericalVector = np.ndarray | Iterable[NumericalScalar]
NumericalMatrix = np.ndarray | Iterable[Iterable[NumericalScalar]]

SymbolicScalar = FloatVariable | Scalar

ScalarData = NumericalScalar | SymbolicScalar
VectorData = NumericalVector | Vector
MatrixData = NumericalMatrix | Matrix

GenericSymbolicType = TypeVar(
    "GenericSymbolicType",
    bound=SymbolicMathType,
)

GenericVectorOrMatrixType = TypeVar(
    "GenericVectorOrMatrixType",
    Vector,
    Matrix,
)
