from __future__ import annotations

import builtins as _builtins
import copy as _copy
import functools as _functools
import math as _math
import operator as _operator
import sys as _sys
from collections import Counter as _Counter
import dataclasses as _dataclasses
from enum import IntEnum as _IntEnum

import casadi as _ca
import numpy as _np
from scipy import sparse as _sp
import typing_extensions as _te
from typing_extensions import ClassVar as _ClassVar

from krrood.entity_query_language.predicate import Symbol
from krrood.symbolic_math.exceptions import (
    HasFreeVariablesError,
    DuplicateVariablesError,
    WrongNumberOfArgsError,
    NotSquareMatrixError,
    SymbolicMathError,
    NotScalerError,
    UnsupportedOperationError,
)

EPS: float = _sys.float_info.epsilon * 4.0


@_dataclasses.dataclass
class CompiledFunction:
    """
    A compiled symbolic function that can be efficiently evaluated with CasADi.

    This class compiles symbolic expressions into optimized CasADi functions that can be
    evaluated efficiently. It supports both sparse and dense matrices and handles
    parameter substitution automatically.
    """

    expression: SymbolicType
    """
    The symbolic expression to compile.
    """
    variable_parameters: _te.Optional[_te.List[_te.List[FloatVariable]]] = None
    """
    The input parameters for the compiled symbolic expression.
    """
    sparse: bool = False
    """
    Whether to return a sparse matrix or a dense numpy matrix
    """

    _compiled_casadi_function: _ca.Function = _dataclasses.field(init=False)

    _function_buffer: _ca.FunctionBuffer = _dataclasses.field(init=False)
    _function_evaluator: _functools.partial = _dataclasses.field(init=False)
    """
    Helpers to avoid new memory allocation during function evaluation
    """

    _out: _te.Union[_np.ndarray, _sp.csc_matrix] = _dataclasses.field(init=False)
    """
    The result of a function evaluation is stored in this variable.
    """

    _is_constant: bool = False
    """
    Used to memorize if the result must be recomputed every time.
    """

    def __post_init__(self):
        if self.variable_parameters is None:
            self.variable_parameters = [self.expression.free_variables()]
        else:
            self._validate_variables()

        if len(self.variable_parameters) == 1 and len(self.variable_parameters[0]) == 0:
            self.variable_parameters = []

        if len(self.expression) == 0:
            self._setup_empty_result()
            return

        self._setup_compiled_function()
        self._setup_output_buffer()
        if len(self.variable_parameters) == 0:
            self._setup_constant_result()

    def _validate_variables(self):
        """Validates variables for both missing and duplicate issues."""
        variables = []
        for variable_parameter in self.variable_parameters:
            variables.extend(variable_parameter)

        variables_set = set(variables)

        # Check for missing variables
        missing_variables = set(self.expression.free_variables()).difference(
            variables_set
        )
        if missing_variables:
            raise HasFreeVariablesError(list(missing_variables))

        # Check for duplicate variables
        if len(variables_set) != len(variables):
            variable_counts = _Counter(variables)
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
            self._out = _sp.csc_matrix(_np.empty(self.expression.shape))
        else:
            self._out = _np.empty(self.expression.shape)
        self._is_constant = True

    def _setup_compiled_function(self) -> None:
        """
        Setup the CasADi compiled function.
        """
        casadi_parameters = []
        if len(self.variable_parameters) > 0:
            # create an array for each _te.List[FloatVariable]
            casadi_parameters = [
                Expression(data=p).casadi_sx for p in self.variable_parameters
            ]

        if self.sparse:
            self._compile_sparse_function(casadi_parameters)
        else:
            self._compile_dense_function(casadi_parameters)

    def _compile_sparse_function(self, casadi_parameters: _te.List[Expression]) -> None:
        """
        Compile function for sparse matrices.
        """
        self.expression.casadi_sx = _ca.sparsify(self.expression.casadi_sx)
        self._compiled_casadi_function = _ca.Function(
            "f", casadi_parameters, [self.expression.casadi_sx]
        )

        self._function_buffer, self._function_evaluator = (
            self._compiled_casadi_function.buffer()
        )
        self.csc_indices, self.csc_indptr = (
            self.expression.casadi_sx.sparsity().get_ccs()
        )
        self.zeroes = _np.zeros(self.expression.casadi_sx.nnz())

    def _compile_dense_function(
        self, casadi_parameters: _te.List[FloatVariable]
    ) -> None:
        """
        Compile function for dense matrices.

        :param casadi_parameters: _te.List of CasADi parameters for the function
        """
        self.expression.casadi_sx = _ca.densify(self.expression.casadi_sx)
        self._compiled_casadi_function = _ca.Function(
            "f", casadi_parameters, [self.expression.casadi_sx]
        )

        self._function_buffer, self._function_evaluator = (
            self._compiled_casadi_function.buffer()
        )

    def _setup_output_buffer(self) -> None:
        """
        Setup the output buffer for the compiled function.
        """
        if self.sparse:
            self._setup_sparse_output_buffer()
        else:
            self._setup_dense_output_buffer()

    def _setup_sparse_output_buffer(self) -> None:
        """
        Setup output buffer for sparse matrices.
        """
        self._out = _sp.csc_matrix(
            arg1=(
                self.zeroes,
                self.csc_indptr,
                self.csc_indices,
            ),
            shape=self.expression.shape,
        )
        self._function_buffer.set_res(0, memoryview(self._out.data))

    def _setup_dense_output_buffer(self) -> None:
        """
        Setup output buffer for dense matrices.
        """
        if self.expression.shape[1] <= 1:
            shape = self.expression.shape[0]
        else:
            shape = self.expression.shape
        self._out = _np.zeros(shape, order="F")
        self._function_buffer.set_res(0, memoryview(self._out))

    def _setup_constant_result(self) -> None:
        """
        Setup result for constant expressions (no parameters).

        For expressions with no free parameters, we can evaluate once and return
        the constant result for all future calls.
        """
        self._function_evaluator()
        self._is_constant = True

    def bind_args_to_memory_view(self, arg_idx: int, numpy_array: _np.ndarray) -> None:
        """
        Binds the arg at index arg_idx to the memoryview of a numpy_array.
        If your args keep the same memory across calls, you only need to bind them once.
        """
        self._function_buffer.set_arg(arg_idx, memoryview(numpy_array))

    def evaluate(self) -> _te.Union[_np.ndarray, _sp.csc_matrix]:
        """
        Evaluate the compiled function with the current args.
        """
        self._function_evaluator()
        return self._out

    def __call__(self, *args: _np.ndarray) -> _te.Union[_np.ndarray, _sp.csc_matrix]:
        """
        Efficiently evaluate the compiled function with positional arguments by directly writing the memory of the
        numpy arrays to the memoryview of the compiled function.
        Similarly, the result will be written to the output buffer and doesn't allocate new memory on each eval.

        (Yes, this makes a significant speed different.)

        :param args: A numpy array for each _te.List[FloatVariable] in self.variable_parameters.
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

    def call_with_kwargs(self, **kwargs: float) -> _np.ndarray:
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
        for params in self.variable_parameters:
            for param in params:
                args.append(kwargs[str(param)])
        filtered_args = _np.array(args, dtype=float)
        return self(filtered_args)


@_dataclasses.dataclass
class CompiledFunctionWithViews:
    """
    A wrapper for CompiledFunction which automatically splits the result array into multiple views, with minimal
    overhead.
    Useful, when many arrays must be evaluated at the same time, especially when they depend on the same variables.
    __call__ returns first a list of expressions, followed by additional_views.
    e.g. CompiledFunctionWithViews(expressions=[expr1, expr2], additional_views=[(start, end)])
        returns [expr1_result, expr2_result, np.concatenate([expr1_result, expr2_result])[start:end]]
    """

    expressions: _te.List[Expression]
    """
    The list of expressions to be compiled.
    """

    variable_parameters: _te.List[_te.List[FloatVariable]]
    """
    The input parameters for the compiled symbolic expression.
    """

    additional_views: _te.Optional[_te.List[slice]] = None
    """
    If additional views are required that don't correspond to the expressions directly.
    """

    compiled_function: CompiledFunction = _dataclasses.field(init=False)
    """
    Reference to the compiled function.
    """

    split_out_view: _te.List[_np.ndarray] = _dataclasses.field(init=False)
    """
    Views to the out buffer of the compiled function.
    """

    def __post_init__(self):
        combined_expression = Expression.vstack(self.expressions)
        self.compiled_function = combined_expression.compile(
            parameters=self.variable_parameters, sparse=False
        )
        slices = []
        start = 0
        for expression in self.expressions[:-1]:
            end = start + expression.shape[0]
            slices.append(end)
            start = end
        self.split_out_view = _np.split(self.compiled_function._out, slices)
        if self.additional_views is not None:
            for expression_slice in self.additional_views:
                self.split_out_view.append(
                    self.compiled_function._out[expression_slice]
                )

    def __call__(self, *args: _np.ndarray) -> _te.List[_np.ndarray]:
        """
        :param args: A numpy array for each _te.List[FloatVariable] in self.variable_parameters.
        :return: A np array for each expression, followed by arrays corresponding to the additional views.
            They are all views on self.compiled_function.out.
        """
        self.compiled_function(*args)
        return self.split_out_view


@_dataclasses.dataclass(eq=False)
class SymbolicType(Symbol):
    """
    A wrapper around CasADi's _ca.SX, with better usability
    """

    casadi_sx: _ca.SX = _dataclasses.field(kw_only=True, default_factory=_ca.SX)
    """
    Reference to the casadi data structure of type casadi.SX
    """

    @classmethod
    def from_casadi_sx(cls, casadi_sx: _ca.SX) -> _te.Self:
        result = cls()
        result.casadi_sx = casadi_sx
        return result

    def __str__(self):
        return str(self.casadi_sx)

    def pretty_str(self) -> _te.List[_te.List[str]]:
        """
        Turns a symbolic type into a more or less readable string.
        """
        result_list = _np.zeros(self.shape).tolist()
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
        return self.shape == (1, 1)

    def __bool__(self) -> bool:
        if self.is_scalar():
            if self.is_constant():
                return bool(self.to_np())
            elif self.casadi_sx.op() == _ca.OP_EQ:
                # not evaluating bool would cause all expressions containing == to be evaluated to True, because they are not None
                # this can cause a lot of unintended bugs, therefore we try to evaluate it
                left = self.casadi_sx.dep(0)
                right = self.casadi_sx.dep(1)
                return _ca.is_equal(_ca.simplify(left), _ca.simplify(right), 5)
        # it's not evaluatable as a bool, so we revert to the normal behavior, and a not None python thing is true
        return True

    def __array__(self):
        return self.to_np()

    def __float__(self):
        if not self.is_scalar():
            raise NotScalerError(actual_dimensions=self.shape)
        if not self.is_constant():
            raise HasFreeVariablesError(self.free_variables())
        return float(self.to_np())

    def __repr__(self):
        return repr(self.casadi_sx)

    def __hash__(self) -> int:
        return self.casadi_sx.__hash__()

    def __getitem__(
        self,
        item: _te.Union[
            _np.ndarray,
            _te.Union[int, slice],
            _te.Tuple[_te.Union[int, slice], _te.Union[int, slice]],
        ],
    ) -> Expression:
        if isinstance(item, _np.ndarray) and item.dtype == bool:
            item = (_np.where(item)[0], slice(None, None))
        return Expression(self.casadi_sx[item])

    def __setitem__(
        self,
        key: _te.Union[
            _te.Union[int, slice],
            _te.Tuple[_te.Union[int, slice], _te.Union[int, slice]],
        ],
        value: ScalarData,
    ):
        self.casadi_sx[key] = value.casadi_sx if hasattr(value, "casadi_sx") else value

    @property
    def shape(self) -> _te.Tuple[int, int]:
        return self.casadi_sx.shape

    def __len__(self) -> int:
        return self.shape[0]

    def free_variables(self) -> _te.List[FloatVariable]:
        return [FloatVariable._registry[s] for s in _ca.symvar(self.casadi_sx)]

    def is_constant(self) -> bool:
        return len(self.free_variables()) == 0

    def to_np(self) -> _np.ndarray:
        """
        Transforms the data into a numpy array.
        Only works if the expression has no free variables.
        """
        if not self.is_constant():
            raise HasFreeVariablesError(self.free_variables())
        if self.shape[0] == self.shape[1] == 0:
            return _np.eye(0)
        elif self.casadi_sx.shape[0] == 1 or self.casadi_sx.shape[1] == 1:
            return _np.array(_ca.evalf(self.casadi_sx)).ravel()
        else:
            return _np.array(_ca.evalf(self.casadi_sx))

    def compile(
        self,
        parameters: _te.Optional[_te.List[_te.List[FloatVariable]]] = None,
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

    def evaluate(self) -> _np.ndarray:
        """
        Substitutes the free variables in this expression using their `resolve` method and compute the result.
        :return: The evaluate value of this expression.
        """
        f = self.compile([self.free_variables()], sparse=False)
        return f(
            _np.array([s.resolve() for s in self.free_variables()], dtype=_np.float64)
        )

    def substitute(
        self,
        old_variables: _te.List[FloatVariable],
        new_variables: _te.List[_te.Union[FloatVariable, Expression]],
    ) -> _te.Self:
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
        old_variables = Expression(data=[to_sx(s) for s in old_variables]).casadi_sx
        new_variables = Expression(data=[to_sx(s) for s in new_variables]).casadi_sx
        result = _copy.copy(self)
        result.casadi_sx = _ca.substitute(self.casadi_sx, old_variables, new_variables)
        return result

    def norm(self) -> Expression:
        return Expression(_ca.norm_2(self.casadi_sx))

    def equivalent(self, other: ScalarData) -> bool:
        """
        Determines whether two scalar expressions are mathematically equivalent by simplifying
        and comparing them.

        :param other: Second scalar expression to compare
        :return: True if the two expressions are equivalent, otherwise False
        """
        other_expression = to_sx(other)
        return _ca.is_equal(
            _ca.simplify(self.casadi_sx), _ca.simplify(other_expression), 5
        )


class BasicOperatorMixin:
    """
    Base class providing arithmetic operations for symbolic types.
    """

    casadi_sx: _ca.SX
    """
    Reference to the casadi data structure of type casadi.SX
    """

    def _binary_operation(
        self, other: ScalarData, operation: _te.Callable, reverse: bool = False
    ) -> Expression:
        """
        Performs a binary operation between the current instance and another operand.

        FloatVariable only allows ScalarData on the righthand sight and implements the reverse version only for NumericalScalaer

        :param other: The operand to be used in the binary operation. Either `ScalarData`
            or `NumericalScalar` types are expected, depending on the context.
        :param reverse: A boolean indicating whether the operation is a reverse operation.
            Defaults to `False`.
        :return: An `Expression` instance resulting from the binary operation, or
            `NotImplemented` if the operand type does not match the expected type.
        """
        if reverse:
            # For reverse operations, check if other is NumericalScalar
            if not isinstance(other, NumericalScalar):
                return NotImplemented
            return Expression(operation(other, self.casadi_sx))
        else:
            # For regular operations, check if other is ScalarData
            if isinstance(other, SymbolicScalar):
                other = other.casadi_sx
            elif not isinstance(other, NumericalScalar):
                return NotImplemented
            return Expression(operation(self.casadi_sx, other))

    # %% arthimetic operators
    def __neg__(self) -> Expression:
        return Expression(self.casadi_sx.__neg__())

    def __add__(self, other: ScalarData) -> Expression:
        return self._binary_operation(other, _operator.add)

    def __radd__(self, other: NumericalScalar) -> Expression:
        return self._binary_operation(other, _operator.add, reverse=True)

    def __sub__(self, other: ScalarData) -> Expression:
        return self._binary_operation(other, _operator.sub)

    def __rsub__(self, other: NumericalScalar) -> Expression:
        return self._binary_operation(other, _operator.sub, reverse=True)

    def __mul__(self, other: ScalarData) -> Expression:
        return self._binary_operation(other, _operator.mul)

    def __rmul__(self, other: NumericalScalar) -> Expression:
        return self._binary_operation(other, _operator.mul, reverse=True)

    def __truediv__(self, other: ScalarData) -> Expression:
        return self._binary_operation(other, _operator.truediv)

    def __rtruediv__(self, other: NumericalScalar) -> Expression:
        return self._binary_operation(other, _operator.truediv, reverse=True)

    def __pow__(self, other: ScalarData) -> Expression:
        return self._binary_operation(other, _operator.pow)

    def __rpow__(self, other: NumericalScalar) -> Expression:
        return self._binary_operation(other, _operator.pow, reverse=True)

    def __floordiv__(self, other: ScalarData) -> Expression:
        return floor(self / other)

    def __rfloordiv__(self, other: ScalarData) -> Expression:
        return floor(other / self)

    def __mod__(self, other: ScalarData) -> Expression:
        return fmod(self.casadi_sx, other)

    def __rmod__(self, other: ScalarData) -> Expression:
        return fmod(other, self.casadi_sx)

    def __divmod__(self, other: ScalarData) -> _te.Tuple[Expression, Expression]:
        return self // other, self % other

    def __rdivmod__(self, other: ScalarData) -> _te.Tuple[Expression, Expression]:
        return other // self, other % self

    # %% logical operators

    def __invert__(self) -> Expression:
        return logic_not(self.casadi_sx)

    def __eq__(self, other: ScalarData) -> Expression:
        if isinstance(other, SymbolicType):
            other = other.casadi_sx
        return Expression(self.casadi_sx.__eq__(other))

    def __ne__(self, other):
        if isinstance(other, SymbolicType):
            other = other.casadi_sx
        return Expression(self.casadi_sx.__ne__(other))

    def __or__(self, other: ScalarData) -> Expression:
        return logic_or(self.casadi_sx, other)

    def __and__(self, other: ScalarData) -> Expression:
        return logic_and(self.casadi_sx, other)

    def __lt__(self, other: ScalarData) -> Expression:
        return self._binary_operation(other, _operator.lt)

    def __le__(self, other: ScalarData) -> Expression:
        return self._binary_operation(other, _operator.le)

    def __gt__(self, other: ScalarData) -> Expression:
        return self._binary_operation(other, _operator.gt)

    def __ge__(self, other: ScalarData) -> Expression:
        return self._binary_operation(other, _operator.ge)

    def safe_division(
        self,
        other: ScalarData,
        if_nan: _te.Optional[ScalarData] = None,
    ) -> Expression:
        """
        A version of division where no sub-expression is ever NaN. The expression would evaluate to 'if_nan', but
        you should probably never work with the 'if_nan' result. However, if one sub-expressions is NaN, the whole expression
        evaluates to NaN, even if it is only in a branch of an if-else, that is not returned.
        This method is a workaround for such cases.
        """
        other = Expression(data=other)
        if if_nan is None:
            if_nan = 0
        if_nan = Expression(data=if_nan)
        save_denominator = if_eq_zero(
            condition=other, if_result=Expression(data=1), else_result=other
        )
        return if_eq_zero(other, if_result=if_nan, else_result=self / save_denominator)


class VectorOperationsMixin:
    casadi_sx: _ca.SX
    """
    Reference to the casadi data structure of type casadi.SX
    """

    def __sub__(self, other: _te.Self) -> Expression: ...

    def euclidean_distance(self, other: _te.Self) -> Expression:
        difference = self - other
        distance = difference.norm()
        return distance


class MatrixOperationsMixin:
    casadi_sx: _ca.SX
    """
    Reference to the casadi data structure of type casadi.SX
    """
    shape: _te.Tuple[int, int]

    def sum(self) -> Expression:
        """
        the equivalent to _np.sum(matrix)
        """
        return Expression(_ca.sum1(_ca.sum2(self.casadi_sx)))

    def sum_row(self) -> Expression:
        """
        the equivalent to _np.sum(matrix, axis=0)
        """
        return Expression(_ca.sum1(self.casadi_sx))

    def sum_column(self) -> Expression:
        """
        the equivalent to _np.sum(matrix, axis=1)
        """
        return Expression(_ca.sum2(self.casadi_sx))

    def trace(self) -> Expression:
        if not self.is_square():
            raise NotSquareMatrixError(actual_dimensions=self.casadi_sx.shape)
        s = 0
        for i in range(self.casadi_sx.shape[0]):
            s += self.casadi_sx[i, i]
        return Expression(s)

    def det(self) -> Expression:
        """
        Calculate the determinant of the given expression.

        This function computes the determinant of the provided mathematical expression.
        The input can be an instance of either `Expression`, `RotationMatrix`, or
        `TransformationMatrix`. The result is returned as an `Expression`.

        :return: An `Expression` representing the determinant of the input.
        """
        if not self.is_square():
            raise NotSquareMatrixError(actual_dimensions=self.casadi_sx.shape)
        return Expression(_ca.det(self.casadi_sx))

    def is_square(self):
        return self.casadi_sx.shape[0] == self.casadi_sx.shape[1]

    def entrywise_product(self, other: Expression) -> Expression:
        """
        Computes the entrywise (element-wise) product of two matrices, assuming they have the same dimensions. The
        operation multiplies each corresponding element of the input matrices and stores the result in a new matrix
        of the same shape.

        :param other: The second matrix, represented as an object of type `Expression`, whose shape
                        must match the shape of `matrix1`.
        :return: A new matrix of type `Expression` containing the entrywise product of `matrix1` and `matrix2`.
        """
        assert self.shape == other.shape
        result = Expression.zeros(*self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = self[i, j] * other[i, j]
        return result


@_dataclasses.dataclass(eq=False)
class FloatVariable(SymbolicType, BasicOperatorMixin):
    """
    A symbolic expression representing a single float variable.
    No matrix and no numbers.
    """

    name: str = _dataclasses.field(kw_only=True)

    casadi_sx: _ca.SX = _dataclasses.field(kw_only=True, init=False, default=None)

    _registry: _ClassVar[_te.Dict[_ca.SX, FloatVariable]] = {}
    """
    Keeps track of which FloatVariable instances are associated with which which casadi.SX instances.
    Needed to recreate the FloatVariables from a casadi expression.
    .. warning:: Does not ensure that two FloatVariable instances are identical.
    """

    def __post_init__(self):
        self.casadi_sx = _ca.SX.sym(str(self.name))
        self._registry[self.casadi_sx] = self

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
        return _np.nan


@_dataclasses.dataclass(eq=False)
class Expression(
    SymbolicType, BasicOperatorMixin, VectorOperationsMixin, MatrixOperationsMixin
):
    """
    Represents symbolic expressions with rich mathematical capabilities, including matrix
    operations, derivatives, and manipulation of symbolic representations.

    This class is designed to encapsulate symbolic mathematical expressions and provide a wide
    range of features for computations, including matrix constructions (zeros, ones, identity),
    derivative computations (Jacobian, total derivatives, Hessian), reshaping, and scaling.
    It is essential to symbolic computation workflows in applications that require gradient
    analysis, second-order derivatives, or other advanced mathematical operations. The class
    leverages symbolic computation libraries for handling low-level symbolic details efficiently.
    """

    casadi_sx: _ca.SX = _dataclasses.field(
        kw_only=True, default_factory=lambda: _ca.SX()
    )

    data: _dataclasses.InitVar[
        _te.Optional[
            _te.Union[
                NumericalScalar,
                NumericalArray,
                Numerical2dMatrix,
                SymbolicType,
                _te.Iterable[FloatVariable],
                _te.Iterable[SymbolicType],
                _te.Iterable[_te.Iterable[SymbolicType]],
            ]
        ]
    ] = None

    def __post_init__(
        self,
        data: _te.Optional[
            _te.Union[
                _ca.SX,
                NumericalScalar,
                NumericalArray,
                Numerical2dMatrix,
                _te.Iterable[FloatVariable],
            ]
        ],
    ):
        if data is None:
            return
        if isinstance(data, _ca.SX):
            self.casadi_sx = data
        elif isinstance(data, SymbolicType):
            self.casadi_sx = data.casadi_sx
        elif isinstance(data, _te.Iterable):
            self._from_iterable(data)
        else:
            self.casadi_sx = _ca.SX(data)

    def _from_iterable(
        self,
        data: _te.Union[NumericalArray, Numerical2dMatrix, _te.Iterable[FloatVariable]],
    ):
        x = len(data)
        if x == 0:
            self.casadi_sx = _ca.SX()
            return
        if (
            isinstance(data[0], list)
            or isinstance(data[0], tuple)
            or isinstance(data[0], _np.ndarray)
        ):
            y = len(data[0])
        else:
            y = 1
        casadi_sx = _ca.SX(x, y)
        for i in range(casadi_sx.shape[0]):
            if y > 1:
                for j in range(casadi_sx.shape[1]):
                    casadi_sx[i, j] = to_sx(data[i][j])
            else:
                casadi_sx[i] = to_sx(data[i])
        self.casadi_sx = casadi_sx

    @classmethod
    def zeros(cls, rows: int, columns: int) -> Expression:
        return cls(casadi_sx=_ca.SX.zeros(rows, columns))

    @classmethod
    def ones(cls, x: int, y: int) -> Expression:
        return cls(casadi_sx=_ca.SX.ones(x, y))

    @classmethod
    def tri(cls, dimension: int) -> Expression:
        return cls(data=_np.tri(dimension))

    @classmethod
    def eye(cls, size: int) -> Expression:
        return cls(casadi_sx=_ca.SX.eye(size))

    @classmethod
    def diag(cls, args: _te.Union[_te.List[ScalarData], Expression]) -> Expression:
        return cls(casadi_sx=_ca.diag(to_sx(args)))

    @classmethod
    def vstack(
        cls,
        list_of_matrices: _te.List[Expression],
    ) -> _te.Self:
        if len(list_of_matrices) == 0:
            return cls(data=[])
        return cls(casadi_sx=_ca.vertcat(*[to_sx(x) for x in list_of_matrices]))

    @classmethod
    def hstack(
        cls,
        list_of_matrices: _te.Union[_te.List[Expression]],
    ) -> _te.Self:
        if len(list_of_matrices) == 0:
            return cls(data=[])
        return cls(casadi_sx=_ca.horzcat(*[to_sx(x) for x in list_of_matrices]))

    @classmethod
    def diag_stack(
        cls,
        list_of_matrices: _te.Union[_te.List[Expression]],
    ) -> Expression:
        num_rows = int(_math.fsum(e.shape[0] for e in list_of_matrices))
        num_columns = int(_math.fsum(e.shape[1] for e in list_of_matrices))
        combined_matrix = Expression.zeros(num_rows, num_columns)
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

    def remove(self, rows: _te.List[int], columns: _te.List[int]):
        self.casadi_sx.remove(rows, columns)

    def split(self) -> _te.List[Expression]:
        assert self.shape[0] == 1 and self.shape[1] == 1
        parts = [
            Expression(self.casadi_sx.dep(i)) for i in range(self.casadi_sx.n_dep())
        ]
        return parts

    def __copy__(self) -> Expression:
        return Expression(_copy.copy(self.casadi_sx))

    def dot(self, other: Expression) -> Expression:
        if isinstance(other, Expression):
            if self.shape[1] == 1 and other.shape[1] == 1:
                return Expression(_ca.mtimes(self.T.casadi_sx, other.casadi_sx))
            return Expression(_ca.mtimes(self.casadi_sx, other.casadi_sx))
        raise UnsupportedOperationError("dot", self, other)

    def __matmul__(self, other: Expression) -> Expression:
        return self.dot(other)

    @property
    def T(self) -> Expression:
        return Expression(self.casadi_sx.T)

    def reshape(self, new_shape: _te.Tuple[int, int]) -> Expression:
        return Expression(self.casadi_sx.reshape(new_shape))

    def jacobian(self, variables: _te.Iterable[FloatVariable]) -> Expression:
        """
        Compute the Jacobian matrix of a vector of expressions with respect to a vector of variables.

        This function calculates the Jacobian matrix, which is a matrix of all first-order
        partial derivatives of a vector of functions with respect to a vector of variables.

        :param variables: The variables with respect to which the partial derivatives are taken.
        :return: The Jacobian matrix as an Expression.
        """
        return Expression(
            _ca.jacobian(self.casadi_sx, Expression(data=variables).casadi_sx)
        )

    def jacobian_dot(
        self,
        variables: _te.Iterable[FloatVariable],
        variables_dot: _te.Iterable[FloatVariable],
    ) -> Expression:
        """
        Compute the total derivative of the Jacobian matrix.

        This function calculates the time derivative of a Jacobian matrix given
        a set of expressions and variables, along with their corresponding
        derivatives. For each element in the Jacobian matrix, this method
        computes the total derivative based on the provided variables and
        their time derivatives.

        :param variables: _te.Iterable containing the variables with respect to which
            the Jacobian is calculated.
        :param variables_dot: _te.Iterable containing the time derivatives of the
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
        variables: _te.Iterable[FloatVariable],
        variables_dot: _te.Iterable[FloatVariable],
        variables_ddot: _te.Iterable[FloatVariable],
    ) -> Expression:
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
        variables: _te.Iterable[FloatVariable],
        variables_dot: _te.Iterable[FloatVariable],
    ) -> Expression:
        """
        Compute the total derivative of an expression with respect to given variables and their derivatives
        (dot variables).

        The total derivative accounts for a dependent relationship where the specified variables represent
        the variables of interest, and the dot variables represent the time derivatives of those variables.

        :param variables: _te.Iterable of variables with respect to which the derivative is computed.
        :param variables_dot: _te.Iterable of dot variables representing the derivatives of the variables.
        :return: The expression resulting from the total derivative computation.
        """
        variables = Expression(data=variables)
        variables_dot = Expression(data=variables_dot)
        return Expression(
            _ca.jtimes(self.casadi_sx, variables.casadi_sx, variables_dot.casadi_sx)
        )

    def second_order_total_derivative(
        self,
        variables: _te.Iterable[FloatVariable],
        variables_dot: _te.Iterable[FloatVariable],
        variables_ddot: _te.Iterable[FloatVariable],
    ) -> Expression:
        """
        Computes the second-order total derivative of an expression with respect to a set of variables.

        This function takes an expression and computes its second-order total derivative
        using provided variables, their first-order derivatives, and their second-order
        derivatives. The computation internally constructs a Hessian matrix of the
        expression and multiplies it by a vector that combines the provided derivative
        data.

        :param variables: _te.Iterable containing the variables with respect to which the derivative is calculated.
        :param variables_dot: _te.Iterable containing the first-order derivatives of the variables.
        :param variables_ddot: _te.Iterable containing the second-order derivatives of the variables.
        :return: The computed second-order total derivative, returned as an `Expression`.
        """
        variables = Expression(data=variables)
        variables_dot = Expression(data=variables_dot)
        variables_ddot = Expression(data=variables_ddot)
        v = []
        for i in range(len(variables)):
            for j in range(len(variables)):
                if i == j:
                    v.append(variables_ddot[i].casadi_sx)
                else:
                    v.append(variables_dot[i].casadi_sx * variables_dot[j].casadi_sx)
        v = Expression(data=v)
        H = Expression(_ca.hessian(self.casadi_sx, variables.casadi_sx)[0])
        H = H.reshape((1, len(H) ** 2))
        return H.dot(v)

    def hessian(self, variables: _te.Iterable[FloatVariable]) -> Expression:
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
        return Expression(
            _ca.hessian(expressions, Expression(data=variables).casadi_sx)[0]
        )

    def inverse(self) -> Expression:
        """
        Computes the matrix inverse. Only works if the expression is square.
        """
        assert self.shape[0] == self.shape[1]
        return Expression(_ca.inv(self.casadi_sx))

    def scale(self, a: ScalarData) -> Expression:
        return self.safe_division(self.norm()) * a

    def kron(self, other: Expression) -> Expression:
        """
        Compute the Kronecker product of two given matrices.

        The Kronecker product is a block matrix construction, derived from the
        direct product of two matrices. It combines the entries of the first
        matrix (`m1`) with each entry of the second matrix (`m2`) by a rule
        of scalar multiplication. This operation extends to any two matrices
        of compatible shapes.

        :param other: The second matrix to be used in calculating the Kronecker product.
                   Supports symbolic or numerical matrix types.
        :return: An Expression representing the resulting Kronecker product as a
                 symbolic or numerical matrix of appropriate size.
        """
        m1 = to_sx(self)
        m2 = to_sx(other)
        return Expression(_ca.kron(m1, m2))


def create_float_variables(
    names: _te.Union[_te.List[str], int],
) -> _te.List[FloatVariable]:
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


def diag(args: _te.Union[_te.List[ScalarData], Expression]) -> Expression:
    return Expression.diag(args)


def vstack(args: _te.Union[_te.List[Expression], Expression]) -> Expression:
    return Expression.vstack(args)


def hstack(args: _te.Union[_te.List[Expression], Expression]) -> Expression:
    return Expression.hstack(args)


def diag_stack(args: _te.Union[_te.List[Expression], Expression]) -> Expression:
    return Expression.diag_stack(args)


def to_sx(thing: _te.Union[_ca.SX, SymbolicType]) -> _ca.SX:
    if isinstance(thing, SymbolicType):
        return thing.casadi_sx
    if isinstance(thing, _ca.SX):
        return thing
    return _ca.SX(thing)


# %% basic math
def abs(x: SymbolicType) -> Expression:
    x_sx = to_sx(x)
    result = _ca.fabs(x_sx)
    return Expression(result)


def max(x: ScalarData, y: ScalarData) -> Expression:
    x = to_sx(x)
    y = to_sx(y)
    return Expression(_ca.fmax(x, y))


def min(x: ScalarData, y: ScalarData) -> Expression:
    x = to_sx(x)
    y = to_sx(y)
    return Expression(_ca.fmin(x, y))


def limit(
    x: ScalarData, lower_limit: ScalarData, upper_limit: ScalarData
) -> Expression:
    return Expression(data=max(lower_limit, min(upper_limit, x)))


def dot(e1: Expression, e2: Expression) -> Expression:
    return e1.dot(e2)


def fmod(a: ScalarData, b: ScalarData) -> Expression:
    a = to_sx(a)
    b = to_sx(b)
    return Expression(_ca.fmod(a, b))


def sum(*expressions: ScalarData) -> Expression:
    return Expression(_ca.sum(to_sx(Expression(expressions))))


def floor(x: ScalarData) -> Expression:
    x = to_sx(x)
    return Expression(_ca.floor(x))


def ceil(x: ScalarData) -> Expression:
    x = to_sx(x)
    return Expression(_ca.ceil(x))


def sign(x: ScalarData) -> Expression:
    x = to_sx(x)
    return Expression(_ca.sign(x))


def exp(x: ScalarData) -> Expression:
    x = to_sx(x)
    return Expression(_ca.exp(x))


def log(x: ScalarData) -> Expression:
    x = to_sx(x)
    return Expression(_ca.log(x))


def sqrt(x: ScalarData) -> Expression:
    x = to_sx(x)
    return Expression(_ca.sqrt(x))


# %% trigonometry
def normalize_angle_positive(angle: ScalarData) -> Expression:
    """
    Normalizes the angle to be 0 to 2*pi
    It takes and returns radians.
    """
    return fmod(fmod(angle, 2.0 * _ca.pi) + 2.0 * _ca.pi, 2.0 * _ca.pi)


def normalize_angle(angle: ScalarData) -> Expression:
    """
    Normalizes the angle to be -pi to +pi
    It takes and returns radians.
    """
    a = normalize_angle_positive(angle)
    return if_greater(a, _ca.pi, a - 2.0 * _ca.pi, a)


def shortest_angular_distance(
    from_angle: ScalarData, to_angle: ScalarData
) -> Expression:
    """
    Given 2 angles, this returns the shortest angular
    difference.  The inputs and outputs are of course radians.

    The result would always be -pi <= result <= pi. Adding the result
    to "from" will always get you an equivalent angle to "to".
    """
    return normalize_angle(to_angle - from_angle)


def safe_acos(angle: ScalarData) -> Expression:
    """
    Limits the angle between -1 and 1 to avoid acos becoming NaN.
    """
    angle = limit(angle, -1, 1)
    return acos(angle)


def cos(x: ScalarData) -> Expression:
    x = to_sx(x)
    return Expression(_ca.cos(x))


def sin(x: ScalarData) -> Expression:
    x = to_sx(x)
    return Expression(_ca.sin(x))


def tan(x: ScalarData) -> Expression:
    x = to_sx(x)
    return Expression(_ca.tan(x))


def cosh(x: ScalarData) -> Expression:
    x = to_sx(x)
    return Expression(_ca.cosh(x))


def sinh(x: ScalarData) -> Expression:
    x = to_sx(x)
    return Expression(_ca.sinh(x))


def acos(x: ScalarData) -> Expression:
    x = to_sx(x)
    return Expression(_ca.acos(x))


def atan2(x: ScalarData, y: ScalarData) -> Expression:
    x = to_sx(x)
    y = to_sx(y)
    return Expression(_ca.atan2(x, y))


# %% other


def solve_for(
    expression: Expression,
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
        err = f(_np.array([x]))[0] - target_value
        if _builtins.abs(err) < eps:
            return x
        slope = f_dx(_np.array([x]))[0]
        if slope == 0:
            if start_value > 0:
                slope = -0.001
            else:
                slope = 0.001
        x -= _builtins.max(_builtins.min(err / slope, max_step), -max_step)
    raise ValueError("no solution found")


def gauss(n: ScalarData) -> Expression:
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
BinaryTrue = Expression(data=True)
BinaryFalse = Expression(data=False)


def is_const_binary_false(expression: Expression) -> bool:
    try:
        return bool((expression == BinaryFalse).to_np())
    except Exception as e:
        return False


def logic_and(*args: ScalarData) -> ScalarData:
    assert len(args) >= 2, "and must be called with at least 2 arguments"
    # if there is any False, return False
    # not x because all x that are found are False
    if any(not x for x in args if is_const_binary_false(x)):
        return BinaryFalse
    # filter all True
    args = [x for x in args if not is_const_binary_true(x)]
    if len(args) == 0:
        return BinaryTrue
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        cas_a = to_sx(args[0])
        cas_b = to_sx(args[1])
        return Expression(_ca.logic_and(cas_a, cas_b))
    else:
        return Expression(
            _ca.logic_and(args[0].casadi_sx, logic_and(*args[1:]).casadi_sx)
        )


def logic_not(expression: ScalarData) -> Expression:
    cas_expr = to_sx(expression)
    return Expression(_ca.logic_not(cas_expr))


def logic_any(args: Expression) -> ScalarData:
    return Expression(_ca.logic_any(args.casadi_sx))


def logic_all(args: Expression) -> ScalarData:
    return Expression(_ca.logic_all(args.casadi_sx))


def logic_or(*args: ScalarData, simplify: bool = True) -> ScalarData:
    assert len(args) >= 2, "and must be called with at least 2 arguments"
    # if there is any True, return True
    if simplify and any(x for x in args if is_const_binary_true(x)):
        return BinaryTrue
    # filter all False
    if simplify:
        args = [x for x in args if not is_const_binary_false(x)]
    if len(args) == 0:
        return BinaryFalse
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        return Expression(_ca.logic_or(to_sx(args[0]), to_sx(args[1])))
    else:
        return Expression(
            _ca.logic_or(to_sx(args[0]), to_sx(logic_or(*args[1:], False)))
        )


def is_const_binary_true(expression: Expression) -> bool:
    try:
        equality_expr = expression == BinaryTrue
        return bool(equality_expr.to_np())
    except Exception as e:
        return False


# %% trinary logic
TrinaryFalse: Expression = Expression(data=0.0)
TrinaryUnknown: Expression = Expression(data=0.5)
TrinaryTrue: Expression = Expression(data=1.0)


def trinary_logic_not(expression: ScalarData) -> Expression:
    """
            |   Not
    ------------------
    True    |  False
    Unknown | Unknown
    False   |  True
    """
    return Expression(data=1 - expression)


def trinary_logic_and(*args: ScalarData) -> ScalarData:
    """
      AND   |  True   | Unknown | False
    ------------------+---------+-------
    True    |  True   | Unknown | False
    Unknown | Unknown | Unknown | False
    False   |  False  |  False  | False
    """
    assert len(args) >= 2, "and must be called with at least 2 arguments"
    # if there is any False, return False
    if any(x for x in args if is_const_binary_false(x)):
        return TrinaryFalse
    # filter all True
    args = [x for x in args if not is_const_binary_true(x)]
    if len(args) == 0:
        return TrinaryTrue
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        cas_a = to_sx(args[0])
        cas_b = to_sx(args[1])
        return min(cas_a, cas_b)
    else:
        return trinary_logic_and(args[0], trinary_logic_and(*args[1:]))


def trinary_logic_or(*args: ScalarData) -> ScalarData:
    """
       OR   |  True   | Unknown | False
    ------------------+---------+-------
    True    |  True   |  True   | True
    Unknown |  True   | Unknown | Unknown
    False   |  True   | Unknown | False
    """
    assert len(args) >= 2, "and must be called with at least 2 arguments"
    # if there is any False, return False
    if any(x for x in args if is_const_binary_true(x)):
        return TrinaryTrue
    # filter all True
    args = [x for x in args if not is_const_binary_true(x)]
    if len(args) == 0:
        return TrinaryFalse
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        cas_a = to_sx(args[0])
        cas_b = to_sx(args[1])
        return max(cas_a, cas_b)
    else:
        return trinary_logic_or(args[0], trinary_logic_or(*args[1:]))


def is_const_trinary_true(expression: Expression) -> bool:
    """
    Checks if the expression has not free variables and is equal to TrinaryTrue.
    If you need this check as an expression use expression == TrinaryTrue.
    """
    try:
        return bool((expression == TrinaryTrue).to_np())
    except Exception as e:
        return False


def is_const_trinary_false(expression: Expression) -> bool:
    """
    Checks if the expression has not free variables and is equal to TrinaryFalse.
    If you need this check as an expression use expression == TrinaryFalse.
    """
    try:
        return bool((expression == TrinaryFalse).to_np())
    except Exception as e:
        return False


def is_const_trinary_unknown(expression: Expression) -> bool:
    """
    Checks if the expression has not free variables and is equal to TrinaryUnknown.
    If you need this check as an expression use expression == TrinaryUnknown.
    """
    try:
        return bool((expression == TrinaryUnknown).to_np())
    except Exception as e:
        return False


def trinary_logic_to_str(expression: Expression) -> str:
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
        if is_const_trinary_true(cas_expr):
            return "True"
        if is_const_trinary_false(cas_expr):
            return "False"
        if is_const_trinary_unknown(cas_expr):
            return "Unknown"
        return f'"{expression}"'

    match cas_expr.op():
        case _ca.OP_SUB:  # trinary "not" is 1-x
            return f"not {trinary_logic_to_str(cas_expr.dep(1))}"
        case _ca.OP_FMIN:  # trinary "and" is min(left, right)
            left = trinary_logic_to_str(cas_expr.dep(0))
            right = trinary_logic_to_str(cas_expr.dep(1))
            return f"({left} and {right})"
        case _ca.OP_FMAX:  # trinary "or" is max(left, right)
            left = trinary_logic_to_str(cas_expr.dep(0))
            right = trinary_logic_to_str(cas_expr.dep(1))
            return f"({left} or {right})"
        case _:
            raise SymbolicMathError(f"cannot convert {expression} to a string")


# %% ifs


def _create_return_type_for_if(if_result) -> _te.Type[SymbolicType]:
    if isinstance(if_result, (int, float, FloatVariable)):
        return Expression
    return type(if_result)


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
    if isinstance(if_result, NumericalScalar):
        if_result = Expression(data=if_result)
    if isinstance(else_result, NumericalScalar):
        else_result = Expression(data=else_result)
    if_result_sx = to_sx(if_result)
    else_result_sx = to_sx(else_result)
    return_type = _create_return_type_for_if(if_result)
    return return_type.from_casadi_sx(
        _ca.if_else(condition, if_result_sx, else_result_sx)
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
    return if_else(_ca.gt(a, b), if_result, else_result)


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
    return if_else(_ca.lt(a, b), if_result, else_result)


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
    return if_else(_ca.gt(condition, 0), if_result, else_result)


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
    return if_else(_ca.ge(a, b), if_result, else_result)


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
    return if_else(_ca.eq(a, b), if_result, else_result)


def if_eq_cases(
    a: ScalarData,
    b_result_cases: _te.Iterable[_te.Tuple[ScalarData, GenericSymbolicType]],
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
        result_sx = _ca.if_else(_ca.eq(a_sx, b_sx), b_result_sx, result_sx)
    return _create_return_type_for_if(a).from_casadi_sx(result_sx)


def if_cases(
    cases: _te.Sequence[_te.Tuple[ScalarData, GenericSymbolicType]],
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
        result_sx = _ca.if_else(case, case_result, result_sx)
    return _create_return_type_for_if(else_result).from_casadi_sx(result_sx)


def if_less_eq_cases(
    a: ScalarData,
    b_result_cases: _te.Sequence[_te.Tuple[ScalarData, GenericSymbolicType]],
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
        result_sx = _ca.if_else(_ca.le(a_sx, b_sx), b_result_sx, result_sx)
    return _create_return_type_for_if(a).from_casadi_sx(result_sx)


# %% type hints

NumericalScalar = _te.Union[int, float, _IntEnum]
NumericalArray = _te.Union[_np.ndarray, _te.Iterable[NumericalScalar]]
Numerical2dMatrix = _te.Union[_np.ndarray, _te.Iterable[NumericalArray]]
NumericalData = _te.Union[NumericalScalar, NumericalArray, Numerical2dMatrix]

SymbolicScalar = _te.Union[FloatVariable, Expression]

ScalarData = _te.Union[NumericalScalar, SymbolicScalar]

GenericSymbolicType = _te.TypeVar(
    "GenericSymbolicType",
    bound=SymbolicType,
)
