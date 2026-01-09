krrood.symbolic_math.symbolic_math
==================================

.. py:module:: krrood.symbolic_math.symbolic_math

.. autoapi-nested-parse::

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



Attributes
----------

.. autoapisummary::

   krrood.symbolic_math.symbolic_math.EPS
   krrood.symbolic_math.symbolic_math.abs
   krrood.symbolic_math.symbolic_math.floor
   krrood.symbolic_math.symbolic_math.ceil
   krrood.symbolic_math.symbolic_math.sign
   krrood.symbolic_math.symbolic_math.exp
   krrood.symbolic_math.symbolic_math.log
   krrood.symbolic_math.symbolic_math.sqrt
   krrood.symbolic_math.symbolic_math.fmod
   krrood.symbolic_math.symbolic_math.cos
   krrood.symbolic_math.symbolic_math.sin
   krrood.symbolic_math.symbolic_math.tan
   krrood.symbolic_math.symbolic_math.cosh
   krrood.symbolic_math.symbolic_math.sinh
   krrood.symbolic_math.symbolic_math.acos
   krrood.symbolic_math.symbolic_math.atan2
   krrood.symbolic_math.symbolic_math.NumericalScalar
   krrood.symbolic_math.symbolic_math.NumericalVector
   krrood.symbolic_math.symbolic_math.NumericalMatrix
   krrood.symbolic_math.symbolic_math.SymbolicScalar
   krrood.symbolic_math.symbolic_math.ScalarData
   krrood.symbolic_math.symbolic_math.VectorData
   krrood.symbolic_math.symbolic_math.MatrixData
   krrood.symbolic_math.symbolic_math.GenericSymbolicType
   krrood.symbolic_math.symbolic_math.GenericVectorOrMatrixType


Classes
-------

.. autoapisummary::

   krrood.symbolic_math.symbolic_math.VariableGroup
   krrood.symbolic_math.symbolic_math.VariableParameters
   krrood.symbolic_math.symbolic_math.CompiledFunction
   krrood.symbolic_math.symbolic_math.CompiledFunctionWithViews
   krrood.symbolic_math.symbolic_math.SymbolicMathType
   krrood.symbolic_math.symbolic_math.Scalar
   krrood.symbolic_math.symbolic_math.FloatVariable
   krrood.symbolic_math.symbolic_math.Vector
   krrood.symbolic_math.symbolic_math.Matrix


Functions
---------

.. autoapisummary::

   krrood.symbolic_math.symbolic_math.to_sx
   krrood.symbolic_math.symbolic_math.array_like_to_casadi_sx
   krrood.symbolic_math.symbolic_math.create_float_variables
   krrood.symbolic_math.symbolic_math.diag
   krrood.symbolic_math.symbolic_math.vstack
   krrood.symbolic_math.symbolic_math.hstack
   krrood.symbolic_math.symbolic_math.diag_stack
   krrood.symbolic_math.symbolic_math.concatenate
   krrood.symbolic_math.symbolic_math.max
   krrood.symbolic_math.symbolic_math.min
   krrood.symbolic_math.symbolic_math.limit
   krrood.symbolic_math.symbolic_math.dot
   krrood.symbolic_math.symbolic_math.sum
   krrood.symbolic_math.symbolic_math.normalize_angle_positive
   krrood.symbolic_math.symbolic_math.normalize_angle
   krrood.symbolic_math.symbolic_math.shortest_angular_distance
   krrood.symbolic_math.symbolic_math.safe_acos
   krrood.symbolic_math.symbolic_math.solve_for
   krrood.symbolic_math.symbolic_math.gauss
   krrood.symbolic_math.symbolic_math.is_const_true
   krrood.symbolic_math.symbolic_math.is_const_false
   krrood.symbolic_math.symbolic_math.logic_and
   krrood.symbolic_math.symbolic_math.logic_or
   krrood.symbolic_math.symbolic_math.logic_not
   krrood.symbolic_math.symbolic_math.logic_any
   krrood.symbolic_math.symbolic_math.logic_all
   krrood.symbolic_math.symbolic_math.trinary_logic_not
   krrood.symbolic_math.symbolic_math.trinary_logic_and
   krrood.symbolic_math.symbolic_math.trinary_logic_or
   krrood.symbolic_math.symbolic_math.trinary_logic_to_str
   krrood.symbolic_math.symbolic_math.if_else
   krrood.symbolic_math.symbolic_math.if_greater
   krrood.symbolic_math.symbolic_math.if_less
   krrood.symbolic_math.symbolic_math.if_greater_zero
   krrood.symbolic_math.symbolic_math.if_greater_eq_zero
   krrood.symbolic_math.symbolic_math.if_greater_eq
   krrood.symbolic_math.symbolic_math.if_less_eq
   krrood.symbolic_math.symbolic_math.if_eq_zero
   krrood.symbolic_math.symbolic_math.if_eq
   krrood.symbolic_math.symbolic_math.if_eq_cases
   krrood.symbolic_math.symbolic_math.if_cases
   krrood.symbolic_math.symbolic_math.if_less_eq_cases
   krrood.symbolic_math.symbolic_math.substitution_cache


Module Contents
---------------

.. py:data:: EPS
   :type:  float

.. py:class:: VariableGroup

   A homogeneous, ordered group of variables that forms one input block.


   .. py:attribute:: variables
      :type:  typing_extensions.Tuple[FloatVariable, Ellipsis]


.. py:class:: VariableParameters

   A collection of variable groups that define the input blocks of a compiled function.


   .. py:attribute:: groups
      :type:  typing_extensions.Tuple[VariableGroup, Ellipsis]


   .. py:method:: flatten() -> typing_extensions.Tuple[FloatVariable, Ellipsis]


   .. py:method:: from_lists(*args: typing_extensions.List[FloatVariable]) -> VariableParameters
      :classmethod:


      Creates a new instance of VariableParameters from multiple lists.

      :param args: A variable number of lists, where each list contains
          FloatVariable instances.
      :return: A new instance of VariableParameters created from the provided lists.



   .. py:method:: to_casadi_parameters() -> typing_extensions.List[casadi.SX]


.. py:class:: CompiledFunction

   A compiled symbolic function that can be efficiently evaluated with CasADi.

   This class compiles symbolic expressions into optimized CasADi functions that can be
   evaluated efficiently. It supports both sparse and dense matrices and handles
   parameter substitution automatically.


   .. py:attribute:: expression
      :type:  SymbolicMathType

      The symbolic expression to compile.



   .. py:attribute:: variable_parameters
      :type:  typing_extensions.Optional[VariableParameters]
      :value: None


      The input parameters for the compiled symbolic expression.



   .. py:attribute:: sparse
      :type:  bool
      :value: False


      Whether to return a sparse matrix or a dense numpy matrix



   .. py:method:: bind_args_to_memory_view(arg_idx: int, numpy_array: numpy.ndarray) -> None

      Binds the arg at index arg_idx to the memoryview of a numpy_array.
      If your args keep the same memory across calls, you only need to bind them once.



   .. py:method:: evaluate() -> numpy.ndarray | scipy.sparse.csc_matrix

      Evaluate the compiled function with the current args.



   .. py:method:: call_with_kwargs(**kwargs: float) -> numpy.ndarray

      Call the object instance with the provided keyword arguments. This method retrieves
      the required arguments from the keyword arguments based on the defined
      `variable_parameters`, compiles them into an array, and then calls the instance
      with the constructed array.

      :param kwargs: A dictionary of keyword arguments containing the parameters
          that match the variables defined in `variable_parameters`.
      :return: A NumPy array resulting from invoking the callable object instance
          with the filtered arguments.



.. py:class:: CompiledFunctionWithViews

   A wrapper for CompiledFunction which automatically splits the result array into multiple views, with minimal
   overhead.
   Useful, when many arrays must be evaluated at the same time, especially when they depend on the same variables.
   __call__ returns first a list of expressions, followed by additional_views.
   e.g. CompiledFunctionWithViews(expressions=[expr1, expr2], additional_views=[(start, end)])
       returns [expr1_result, expr2_result, np.concatenate([expr1_result, expr2_result])[start:end]]


   .. py:attribute:: expressions
      :type:  typing_extensions.List[SymbolicMathType]

      The list of expressions to be compiled.



   .. py:attribute:: parameters
      :type:  VariableParameters

      The input parameters for the compiled symbolic expression.



   .. py:attribute:: additional_views
      :type:  typing_extensions.Optional[typing_extensions.List[slice]]
      :value: []


      If additional views are required that don't correspond to the expressions directly.



   .. py:attribute:: compiled_function
      :type:  CompiledFunction

      Reference to the compiled function.



   .. py:attribute:: split_out_view
      :type:  typing_extensions.List[numpy.ndarray]

      Views to the out buffer of the compiled function.



.. py:class:: SymbolicMathType

   Bases: :py:obj:`abc.ABC`


   A wrapper around CasADi's ca.SX, with better usability


   .. py:method:: from_casadi_sx(casadi_sx: casadi.SX) -> typing_extensions.Self
      :classmethod:



   .. py:property:: casadi_sx
      :type: casadi.SX



   .. py:method:: pretty_str() -> typing_extensions.List[typing_extensions.List[str]]

      Turns a symbolic type into a more or less readable string.



   .. py:method:: is_scalar() -> bool


   .. py:property:: shape
      :type: typing_extensions.Tuple[int, int]



   .. py:method:: free_variables() -> typing_extensions.List[FloatVariable]


   .. py:method:: is_constant() -> bool


   .. py:method:: to_np() -> numpy.ndarray

      Transforms the data into a numpy array.
      Only works if the expression has no free variables.



   .. py:method:: to_list() -> list

      Converts the symbolic expression into a nested Python list, like numpy.tolist.

      The expression must be constant; otherwise a HasFreeVariablesError is raised.



   .. py:method:: safe_division(other: GenericSymbolicType, if_nan: typing_extensions.Optional[ScalarData] = None) -> GenericSymbolicType

      A version of division where no sub-expression is ever NaN. The expression would evaluate to 'if_nan', but
      you should probably never work with the 'if_nan' result. However, if one sub-expressions is NaN, the whole expression
      evaluates to NaN, even if it is only in a branch of an if-else, that is not returned.
      This method is a workaround for such cases.



   .. py:method:: compile(parameters: typing_extensions.Optional[VariableParameters] = None, sparse: bool = False) -> CompiledFunction

      Compiles the function into a representation that can be executed efficiently. This method
      allows for optional parameterization and the ability to specify whether the compilation
      should consider a sparse representation.

      :param parameters: A list of parameter sets, where each set contains variables that define
          the configuration for the compiled function. If set to None, no parameters are applied.
      :param sparse: A boolean that determines whether the compiled function should use a
          sparse representation. Defaults to False.
      :return: The compiled function as an instance of CompiledFunction.



   .. py:method:: evaluate() -> numpy.ndarray

      Substitutes the free variables in this expression using their `resolve` method and compute the result.
      :return: The evaluated value of this expression.



   .. py:method:: substitute(old_variables: typing_extensions.List[FloatVariable], new_variables: typing_extensions.List[ScalarData] | Vector) -> typing_extensions.Self

      Replace variables in an expression with new variables or expressions.

      This function substitutes variables in the given expression with the provided
      new variables or expressions. It ensures that the original expression remains
      unaltered and creates a new instance with the substitutions applied.

      :param old_variables: A list of variables in the expression which need to be replaced.
      :param new_variables: A list of new variables or expressions which will replace the old variables.
          The length of this list must correspond to the `old_variables` list.
      :return: A new expression with the specified variables replaced.



   .. py:method:: equivalent(other: ScalarData) -> bool

      Determines whether two scalar expressions are mathematically equivalent by simplifying
      and comparing them.

      :param other: Second scalar expression to compare
      :return: True if the two expressions are equivalent, otherwise False



   .. py:method:: jacobian(variables: typing_extensions.Iterable[FloatVariable]) -> Matrix

      Compute the Jacobian matrix of a vector of expressions with respect to a vector of variables.

      This function calculates the Jacobian matrix, which is a matrix of all first-order
      partial derivatives of a vector of functions with respect to a vector of variables.

      :param variables: The variables with respect to which the partial derivatives are taken.
      :return: The Jacobian matrix as an SymbolicMathType.



   .. py:method:: jacobian_dot(variables: typing_extensions.Iterable[FloatVariable], variables_dot: typing_extensions.Iterable[FloatVariable]) -> Matrix

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



   .. py:method:: jacobian_ddot(variables: typing_extensions.Iterable[FloatVariable], variables_dot: typing_extensions.Iterable[FloatVariable], variables_ddot: typing_extensions.Iterable[FloatVariable]) -> Matrix

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



   .. py:method:: total_derivative(variables: typing_extensions.Iterable[FloatVariable], variables_dot: typing_extensions.Iterable[FloatVariable]) -> Vector

      Compute the total derivative of an expression with respect to given variables and their derivatives
      (dot variables).

      The total derivative accounts for a dependent relationship where the specified variables represent
      the variables of interest, and the dot variables represent the time derivatives of those variables.

      :param variables: Iterable of variables with respect to which the derivative is computed.
      :param variables_dot: Iterable of dot variables representing the derivatives of the variables.
      :return: The expression resulting from the total derivative computation.



   .. py:method:: second_order_total_derivative(variables: typing_extensions.Iterable[FloatVariable], variables_dot: typing_extensions.Iterable[FloatVariable], variables_ddot: typing_extensions.Iterable[FloatVariable]) -> Vector

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



.. py:class:: Scalar(data: ScalarData = 0)

   Bases: :py:obj:`SymbolicMathType`


   A symbolic type representing a scalar value.


   .. py:attribute:: casadi_sx
      :value: 0



   .. py:method:: const_false() -> typing_extensions.Self
      :classmethod:



   .. py:method:: const_trinary_unknown() -> typing_extensions.Self
      :classmethod:



   .. py:method:: const_true() -> typing_extensions.Self
      :classmethod:



   .. py:method:: is_const_true()


   .. py:method:: is_const_unknown()


   .. py:method:: is_const_false()


   .. py:method:: hessian(variables: typing_extensions.Iterable[FloatVariable]) -> Matrix

      Calculate the Hessian matrix of a given expression with respect to specified variables.

      The function computes the second-order partial derivatives (Hessian matrix) for a
      provided mathematical expression using the specified variables. It utilizes a symbolic
      library for the internal operations to generate the Hessian.

      :param variables: An iterable containing the variables with respect to which the derivatives
          are calculated.
      :return: The resulting Hessian matrix as an expression.



.. py:class:: FloatVariable(name: str)

   Bases: :py:obj:`Scalar`


   A symbolic expression representing a single float variable.
   Applying any operation on a FloatVariable results in a Scalar.


   .. py:attribute:: name
      :type:  str


   .. py:method:: resolve() -> float

      This method is called by SymbolicType.evaluate().
      Subclasses should override this method to return the current float value for this variable.
      :return: This variables' current value.



.. py:class:: Vector(data: typing_extensions.Optional[VectorData] = None)

   Bases: :py:obj:`SymbolicMathType`


   A vector of symbolic expressions.
   Should behave like a numpy array with one dimension.


   .. py:attribute:: casadi_sx
      :value: None



   .. py:method:: zeros(size: int) -> typing_extensions.Self
      :classmethod:



   .. py:method:: ones(size: int) -> typing_extensions.Self
      :classmethod:



   .. py:method:: dot(other: GenericSymbolicType) -> Scalar | Vector

      Same as numpy dot.



   .. py:method:: euclidean_distance(other: typing_extensions.Self) -> Scalar


   .. py:method:: norm() -> Scalar

      Computes the 2-norm (Euclidean norm) of the current object.

      :return: The 2-norm of the object, represented as a `Scalar` type.



   .. py:method:: scale(a: ScalarData) -> Vector

      Scales the current vector proportionally based on the provided scalar value.

      :param a: A scalar value used to scale the vector
      :return: A new vector resulting from the scaling operation



   .. py:method:: concatenate(other: Vector) -> Vector

      Concatenates the calling vector object with another vector, resulting in
      a single unified vector.

      :param other: The vector to concatenate with the current vector.
      :return: A new vector object representing the combined result of the two vectors.



.. py:class:: Matrix(data: typing_extensions.Optional[VectorData | MatrixData] = None)

   Bases: :py:obj:`SymbolicMathType`


   A matrix of symbolic expressions.
   Should behave like a 2d numpy array.


   .. py:attribute:: casadi_sx
      :value: None



   .. py:method:: create_filled_with_variables(shape: typing_extensions.Tuple[int, int], name: str) -> typing_extensions.Self
      :classmethod:



   .. py:method:: dot(other: GenericSymbolicType) -> GenericSymbolicType

      Same as numpy dot.



   .. py:method:: zeros(rows: int, columns: int) -> typing_extensions.Self
      :classmethod:


      See numpy.zeros.



   .. py:method:: ones(x: int, y: int) -> typing_extensions.Self
      :classmethod:


      See numpy.ones.



   .. py:method:: tri(dimension: int) -> typing_extensions.Self
      :classmethod:


      See numpy.tri.



   .. py:method:: eye(size: int) -> typing_extensions.Self
      :classmethod:


      See numpy.eye.



   .. py:method:: diag(args: VectorData) -> typing_extensions.Self
      :classmethod:


      See numpy.diag.



   .. py:method:: vstack(list_of_matrices: VectorData | MatrixData) -> typing_extensions.Self
      :classmethod:


      See numpy.vstack.



   .. py:method:: hstack(list_of_matrices: VectorData | MatrixData) -> typing_extensions.Self
      :classmethod:


      See numpy.hstack.



   .. py:method:: diag_stack(list_of_matrices: VectorData | MatrixData) -> typing_extensions.Self
      :classmethod:


      See numpy.diag_stack.



   .. py:method:: remove(rows: typing_extensions.List[int], columns: typing_extensions.List[int])

      Removes the specified rows and columns from the matrix.
      :param rows: Row ids to be removed
      :param columns: Column ids to be removed



   .. py:method:: sum() -> Scalar

      the equivalent to _np.sum(matrix)



   .. py:method:: sum_row() -> typing_extensions.Self

      the equivalent to _np.sum(matrix, axis=0)



   .. py:method:: sum_column() -> typing_extensions.Self

      the equivalent to _np.sum(matrix, axis=1)



   .. py:method:: trace() -> Scalar

      See numpy.trace.



   .. py:method:: det() -> Scalar

      See numpy.linalg.det.



   .. py:method:: is_square() -> bool


   .. py:property:: T
      :type: typing_extensions.Self


      :return: the Transpose of the matrix.



   .. py:method:: reshape(new_shape: typing_extensions.Tuple[int, int]) -> typing_extensions.Self

      See numpy.reshape.



   .. py:method:: inverse() -> Matrix

      Computes the matrix inverse. Only works if the expression is square.



   .. py:method:: flatten() -> Vector

      Returns a row-major flattened Vector, matching numpy.ndarray.flatten(order='C').



   .. py:method:: kron(other: Matrix) -> typing_extensions.Self

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



.. py:function:: to_sx(data: NumericalScalar | NumericalVector | NumericalMatrix | typing_extensions.Iterable[FloatVariable] | SymbolicMathType) -> casadi.SX

   Tries to turn anything into a casadi SX object.
   :param data: input data to be converted to SX
   :return: casadi SX object


.. py:function:: array_like_to_casadi_sx(data: VectorData) -> casadi.SX

   Converts a given array-like data structure into a CasADi SX matrix. The input
   data can be a list, tuple, or numpy array. Based on the structure of the input
   data, the function determines the dimensions of the resulting CasADi SX object
   and populates it with values using the `to_sx` function.

   :param data: Input array-like data. It can be a 1D or 2D array-like structure,
       such as a list, tuple, or numpy array.
   :return: A CasADi SX object representation of the input data.


.. py:function:: create_float_variables(names: typing_extensions.List[str] | int) -> typing_extensions.List[FloatVariable]

   Generates a list of symbolic objects based on the input names or an integer value.

   This function takes either a list of names or an integer. If an integer is
   provided, it generates symbolic objects with default names in the format
   `s_<index>` for numbers up to the given integer. If a list of names is
   provided, it generates symbolic objects for each name in the list.

   :param names: A list of strings representing names of variables or an integer
       specifying the number of variables to generate.
   :return: A list of symbolic objects created based on the input.


.. py:function:: diag(args: VectorData | MatrixData) -> Matrix

   Places the input along the diagonal of a matrix.

   :param args: A vector, list of scalars, or nested lists representing a matrix.
   :return: A square matrix with the input values on its main diagonal.


.. py:function:: vstack(args: VectorData | MatrixData) -> Matrix

   Stacks vectors or matrices vertically into a single matrix.

   :param args: A sequence of vectors or matrices with matching column sizes.
   :return: A new matrix containing the inputs stacked by rows.


.. py:function:: hstack(args: VectorData | MatrixData) -> Matrix

   Stacks vectors or matrices horizontally into a single matrix.

   :param args: A sequence of vectors or matrices with matching row sizes.
   :return: A new matrix containing the inputs stacked by columns.


.. py:function:: diag_stack(args: VectorData | MatrixData) -> Matrix

   Builds a block diagonal matrix from the provided inputs.

   :param args: A sequence of vectors or matrices to place on the block diagonal.
   :return: A block diagonal matrix.


.. py:function:: concatenate(*vectors: Vector) -> Vector

   Concatenates multiple vectors into a single vector.

   :param vectors: The vectors to concatenate in order.
   :return: A new vector with all inputs concatenated.


.. py:data:: abs

.. py:function:: max(arg1: GenericSymbolicType, arg2: typing_extensions.Optional[GenericSymbolicType] = None) -> GenericSymbolicType

   Returns the maximum element-wise value.

   - With one argument, returns the maximum value across all elements.
   - With two arguments, returns the element-wise maximum.

   :param arg1: The first expression.
   :param arg2: Optional second expression.
   :return: The resulting expression with maximum values.


.. py:function:: min(arg1: GenericSymbolicType, arg2: typing_extensions.Optional[GenericSymbolicType] = None) -> GenericSymbolicType

   Returns the minimum element-wise value.

   - With one argument, returns the minimum value across all elements.
   - With two arguments, returns the element-wise minimum.

   :param arg1: The first expression.
   :param arg2: Optional second expression.
   :return: The resulting expression with minimum values.


.. py:function:: limit(x: GenericSymbolicType, lower_limit: ScalarData, upper_limit: ScalarData) -> GenericSymbolicType

   Clamps values to the closed interval [lower_limit, upper_limit].

   :param x: The expression to clamp.
   :param lower_limit: The lower bound.
   :param upper_limit: The upper bound.
   :return: The clamped expression.


.. py:function:: dot(e1: GenericVectorOrMatrixType, e2: GenericVectorOrMatrixType) -> GenericVectorOrMatrixType

   Computes the dot product following NumPy semantics.

   :param e1: The left vector or matrix.
   :param e2: The right vector or matrix.
   :return: The dot product result.


.. py:function:: sum(*expressions: ScalarData) -> Scalar

   Sums the provided scalar expressions.

   :param expressions: The values to add.
   :return: The total as a scalar expression.


.. py:data:: floor

.. py:data:: ceil

.. py:data:: sign

.. py:data:: exp

.. py:data:: log

.. py:data:: sqrt

.. py:data:: fmod

.. py:function:: normalize_angle_positive(angle: ScalarData) -> Scalar

   Normalizes the angle to be 0 to 2*pi
   It takes and returns radians.


.. py:function:: normalize_angle(angle: ScalarData) -> Scalar

   Normalizes the angle to be -pi to +pi
   It takes and returns radians.


.. py:function:: shortest_angular_distance(from_angle: ScalarData, to_angle: ScalarData) -> Scalar

   Given 2 angles, this returns the shortest angular
   difference.  The inputs and outputs are radians.

   The result would always be -pi <= result <= pi. Adding the result
   to "from" will always get you an equivalent angle to "to".


.. py:function:: safe_acos(angle: GenericSymbolicType) -> GenericSymbolicType

   Limits the angle between -1 and 1 to avoid acos becoming NaN.


.. py:data:: cos

.. py:data:: sin

.. py:data:: tan

.. py:data:: cosh

.. py:data:: sinh

.. py:data:: acos

.. py:data:: atan2

.. py:function:: solve_for(expression: SymbolicMathType, target_value: float, start_value: float = 0.0001, max_tries: int = 10000, eps: float = 1e-10, max_step: float = 1) -> float

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


.. py:function:: gauss(n: ScalarData) -> Scalar

   Calculate the sum of the first `n` natural numbers using the Gauss formula.

   This function computes the sum of an arithmetic series where the first term
   is 1, the last term is `n`, and the total count of the terms is `n`. The
   result is derived from the formula `(n * (n + 1)) / 2`, which simplifies
   to `(n ** 2 + n) / 2`.

   :param n: The upper limit of the sum, representing the last natural number
             of the series to include.
   :return: The sum of the first `n` natural numbers.


.. py:function:: is_const_true(expression: Scalar) -> bool

   Checks whether a scalar expression is the constant truth value.

   :param expression: The scalar expression to test.
   :return: True if the expression is exactly the constant 1, otherwise False.


.. py:function:: is_const_false(expression: Scalar) -> bool

   Checks whether a scalar expression is the constant false value.

   :param expression: The scalar expression to test.
   :return: True if the expression is exactly the constant 0, otherwise False.


.. py:function:: logic_and(left: ScalarData, right: ScalarData) -> Scalar

   Logical conjunction on symbolic scalars.

   :param left: The left operand.
   :param right: The right operand.
   :return: The symbolic result of left AND right.


.. py:function:: logic_or(left: ScalarData, right: ScalarData) -> Scalar

   Logical disjunction on symbolic scalars.

   :param left: The left operand.
   :param right: The right operand.
   :return: The symbolic result of left OR right.


.. py:function:: logic_not(expression: ScalarData) -> Scalar

   Logical negation on a symbolic scalar.

   :param expression: The operand to negate.
   :return: The symbolic result of NOT expression.


.. py:function:: logic_any(args: VectorData | MatrixData) -> Scalar

   Returns True if any element evaluates to True.

   :param args: A vector or matrix of logical scalars.
   :return: A scalar truth value.


.. py:function:: logic_all(args: GenericVectorOrMatrixType) -> Scalar

   Returns True if all elements evaluate to True.

   :param args: A vector or matrix of logical scalars.
   :return: A scalar truth value.


.. py:function:: trinary_logic_not(expression: FloatVariable | Scalar) -> Scalar

           |   Not
   ------------------
   True    |  False
   Unknown | Unknown
   False   |  True


.. py:function:: trinary_logic_and(*args: FloatVariable | Scalar) -> Scalar

     AND   |  True   | Unknown | False
   ------------------+---------+-------
   True    |  True   | Unknown | False
   Unknown | Unknown | Unknown | False
   False   |  False  |  False  | False


.. py:function:: trinary_logic_or(*args: FloatVariable | Scalar) -> Scalar

      OR   |  True   | Unknown | False
   ------------------+---------+-------
   True    |  True   |  True   | True
   Unknown |  True   | Unknown | Unknown
   False   |  True   | Unknown | False


.. py:function:: trinary_logic_to_str(expression: Scalar) -> str

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


.. py:function:: if_else(condition: ScalarData, if_result: GenericSymbolicType, else_result: GenericSymbolicType) -> GenericSymbolicType

   Creates an expression that represents:
   if condition:
       return if_result
   else:
       return else_result


.. py:function:: if_greater(a: ScalarData, b: ScalarData, if_result: GenericSymbolicType, else_result: GenericSymbolicType) -> GenericSymbolicType

   Creates an expression that represents:
   if a > b:
       return if_result
   else:
       return else_result


.. py:function:: if_less(a: ScalarData, b: ScalarData, if_result: GenericSymbolicType, else_result: GenericSymbolicType) -> GenericSymbolicType

   Creates an expression that represents:
   if a < b:
       return if_result
   else:
       return else_result


.. py:function:: if_greater_zero(condition: ScalarData, if_result: GenericSymbolicType, else_result: GenericSymbolicType) -> GenericSymbolicType

   Creates an expression that represents:
   if condition > 0:
       return if_result
   else:
       return else_result


.. py:function:: if_greater_eq_zero(condition: ScalarData, if_result: GenericSymbolicType, else_result: GenericSymbolicType) -> GenericSymbolicType

   Creates an expression that represents:
   if condition >= 0:
       return if_result
   else:
       return else_result


.. py:function:: if_greater_eq(a: ScalarData, b: ScalarData, if_result: GenericSymbolicType, else_result: GenericSymbolicType) -> GenericSymbolicType

   Creates an expression that represents:
   if a >= b:
       return if_result
   else:
       return else_result


.. py:function:: if_less_eq(a: ScalarData, b: ScalarData, if_result: GenericSymbolicType, else_result: GenericSymbolicType) -> GenericSymbolicType

   Creates an expression that represents:
   if a <= b:
       return if_result
   else:
       return else_result


.. py:function:: if_eq_zero(condition: ScalarData, if_result: GenericSymbolicType, else_result: GenericSymbolicType) -> GenericSymbolicType

   Creates an expression that represents:
   if condition == 0:
       return if_result
   else:
       return else_result


.. py:function:: if_eq(a: ScalarData, b: ScalarData, if_result: GenericSymbolicType, else_result: GenericSymbolicType) -> GenericSymbolicType

   Creates an expression that represents:
   if a == b:
       return if_result
   else:
       return else_result


.. py:function:: if_eq_cases(a: ScalarData, b_result_cases: typing_extensions.Iterable[typing_extensions.Tuple[ScalarData, GenericSymbolicType]], else_result: GenericSymbolicType) -> GenericSymbolicType

   if a == b_result_cases[0][0]:
       return b_result_cases[0][1]
   elif a == b_result_cases[1][0]:
       return b_result_cases[1][1]
   ...
   else:
       return else_result


.. py:function:: if_cases(cases: typing_extensions.Sequence[typing_extensions.Tuple[ScalarData, GenericSymbolicType]], else_result: GenericSymbolicType) -> GenericSymbolicType

   if cases[0][0]:
       return cases[0][1]
   elif cases[1][0]:
       return cases[1][1]
   ...
   else:
       return else_result


.. py:function:: if_less_eq_cases(a: ScalarData, b_result_cases: typing_extensions.Sequence[typing_extensions.Tuple[ScalarData, GenericSymbolicType]], else_result: GenericSymbolicType) -> GenericSymbolicType

   This only works if b_result_cases is sorted in ascending order.
   if a <= b_result_cases[0][0]:
       return b_result_cases[0][1]
   elif a <= b_result_cases[1][0]:
       return b_result_cases[1][1]
   ...
   else:
       return else_result


.. py:function:: substitution_cache(method)

   This decorator allows you to speed up complex symbolic math operations.
   The operator computes the expression once with variables and stores it in a cache.
   On subsequent calls, the cached expression is used and the args are substituted into the variables,
   avoiding rebuilding of the computation graph.


.. py:data:: NumericalScalar

.. py:data:: NumericalVector

.. py:data:: NumericalMatrix

.. py:data:: SymbolicScalar

.. py:data:: ScalarData

.. py:data:: VectorData

.. py:data:: MatrixData

.. py:data:: GenericSymbolicType

.. py:data:: GenericVectorOrMatrixType

