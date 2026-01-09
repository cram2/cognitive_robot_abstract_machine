krrood.symbolic_math.exceptions
===============================

.. py:module:: krrood.symbolic_math.exceptions


Exceptions
----------

.. autoapisummary::

   krrood.symbolic_math.exceptions.SymbolicMathError
   krrood.symbolic_math.exceptions.UnsupportedOperationError
   krrood.symbolic_math.exceptions.WrongDimensionsError
   krrood.symbolic_math.exceptions.NotScalerError
   krrood.symbolic_math.exceptions.NotSquareMatrixError
   krrood.symbolic_math.exceptions.HasFreeVariablesError
   krrood.symbolic_math.exceptions.ExpressionEvaluationError
   krrood.symbolic_math.exceptions.WrongNumberOfArgsError
   krrood.symbolic_math.exceptions.DuplicateVariablesError


Module Contents
---------------

.. py:exception:: SymbolicMathError

   Bases: :py:obj:`krrood.utils.DataclassException`


   Represents an error specifically related to symbolic mathematics operations.


.. py:exception:: UnsupportedOperationError

   Bases: :py:obj:`SymbolicMathError`, :py:obj:`TypeError`


   Represents an error for unsupported operations between incompatible types.


   .. py:attribute:: operation
      :type:  str

      The name of the operation that was attempted (e.g., '+', '-', etc.).



   .. py:attribute:: left
      :type:  typing_extensions.Any

      The first argument involved in the operation.



   .. py:attribute:: right
      :type:  typing_extensions.Any

      The second argument involved in the operation.



   .. py:attribute:: message
      :type:  str


.. py:exception:: WrongDimensionsError

   Bases: :py:obj:`SymbolicMathError`


   Represents an error for mismatched dimensions.


   .. py:attribute:: expected_dimensions
      :type:  typing_extensions.Tuple[int, int] | str


   .. py:attribute:: actual_dimensions
      :type:  typing_extensions.Tuple[int, int]


   .. py:attribute:: message
      :type:  str


.. py:exception:: NotScalerError

   Bases: :py:obj:`WrongDimensionsError`


   Exception raised for errors when a non-scalar input is provided.


   .. py:attribute:: expected_dimensions
      :type:  typing_extensions.Tuple[int, int]
      :value: (1, 1)



.. py:exception:: NotSquareMatrixError

   Bases: :py:obj:`WrongDimensionsError`


   Represents an error raised when an operation requires a square matrix but the input is not.


   .. py:attribute:: expected_dimensions
      :type:  typing_extensions.Tuple[int, int]
      :value: 'square'



   .. py:attribute:: actual_dimensions
      :type:  typing_extensions.Tuple[int, int]


.. py:exception:: HasFreeVariablesError

   Bases: :py:obj:`SymbolicMathError`


   Raised when an operation can't be performed on an expression with free variables.


   .. py:attribute:: variables
      :type:  typing_extensions.List[krrood.symbolic_math.symbolic_math.FloatVariable]


   .. py:attribute:: message
      :type:  str


.. py:exception:: ExpressionEvaluationError

   Bases: :py:obj:`SymbolicMathError`


   Represents an exception raised during the evaluation of a symbolic mathematical expression.


.. py:exception:: WrongNumberOfArgsError

   Bases: :py:obj:`ExpressionEvaluationError`


   This error is specifically used in expression evaluation scenarios where a certain number of arguments
   are required and the actual number provided is incorrect.


   .. py:attribute:: expected_number_of_args
      :type:  int


   .. py:attribute:: actual_number_of_args
      :type:  int


   .. py:attribute:: message
      :type:  str


.. py:exception:: DuplicateVariablesError

   Bases: :py:obj:`SymbolicMathError`


   Raised when duplicate variables are found in an operation that requires unique variables.


   .. py:attribute:: variables
      :type:  typing_extensions.List[krrood.symbolic_math.symbolic_math.FloatVariable]


   .. py:attribute:: message
      :type:  str


