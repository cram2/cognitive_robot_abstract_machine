krrood.entity_query_language.failures
=====================================

.. py:module:: krrood.entity_query_language.failures

.. autoapi-nested-parse::

   This module defines some custom exception types used by the entity_query_language package.



Exceptions
----------

.. autoapisummary::

   krrood.entity_query_language.failures.QuantificationNotSatisfiedError
   krrood.entity_query_language.failures.GreaterThanExpectedNumberOfSolutions
   krrood.entity_query_language.failures.LessThanExpectedNumberOfSolutions
   krrood.entity_query_language.failures.MultipleSolutionFound
   krrood.entity_query_language.failures.NoSolutionFound
   krrood.entity_query_language.failures.LogicalError
   krrood.entity_query_language.failures.VariableCannotBeEvaluated
   krrood.entity_query_language.failures.UsageError
   krrood.entity_query_language.failures.NoKwargsInMatchVar
   krrood.entity_query_language.failures.WrongSelectableType
   krrood.entity_query_language.failures.LiteralConditionError
   krrood.entity_query_language.failures.CannotProcessResultOfGivenChildType
   krrood.entity_query_language.failures.NonPositiveLimitValue
   krrood.entity_query_language.failures.UnsupportedOperation
   krrood.entity_query_language.failures.UnSupportedOperand
   krrood.entity_query_language.failures.UnsupportedNegation
   krrood.entity_query_language.failures.QuantificationSpecificationError
   krrood.entity_query_language.failures.QuantificationConsistencyError
   krrood.entity_query_language.failures.NegativeQuantificationError
   krrood.entity_query_language.failures.InvalidChildType
   krrood.entity_query_language.failures.InvalidEntityType
   krrood.entity_query_language.failures.ClassDiagramError
   krrood.entity_query_language.failures.NoneWrappedFieldError


Module Contents
---------------

.. py:exception:: QuantificationNotSatisfiedError

   Bases: :py:obj:`krrood.utils.DataclassException`, :py:obj:`abc.ABC`


   Represents a custom exception where the quantification constraints are not satisfied.

   This exception is used to indicate errors related to the quantification
   of the query results.


   .. py:attribute:: expression
      :type:  krrood.entity_query_language.symbolic.ResultQuantifier

      The result quantifier expression where the error occurred.



   .. py:attribute:: expected_number
      :type:  int

      Expected number of solutions (i.e, quantification constraint value).



.. py:exception:: GreaterThanExpectedNumberOfSolutions

   Bases: :py:obj:`QuantificationNotSatisfiedError`


   Represents an error when the number of solutions exceeds the
   expected threshold.


.. py:exception:: LessThanExpectedNumberOfSolutions

   Bases: :py:obj:`QuantificationNotSatisfiedError`


   Represents an error that occurs when the number of solutions found
   is lower than the expected number.


   .. py:attribute:: found_number
      :type:  int

      The number of solutions found.



.. py:exception:: MultipleSolutionFound

   Bases: :py:obj:`GreaterThanExpectedNumberOfSolutions`


   Raised when a query unexpectedly yields more than one solution where a single
   result was expected.


   .. py:attribute:: expected_number
      :type:  int
      :value: 1


      Expected number of solutions (i.e, quantification constraint value).



.. py:exception:: NoSolutionFound

   Bases: :py:obj:`LessThanExpectedNumberOfSolutions`


   Raised when a query does not yield any solution.


   .. py:attribute:: expected_number
      :type:  int
      :value: 1


      Expected number of solutions (i.e, quantification constraint value).



   .. py:attribute:: found_number
      :type:  int
      :value: 0


      The number of solutions found.



.. py:exception:: LogicalError

   Bases: :py:obj:`krrood.utils.DataclassException`


   Raised when there is an error in the logical structure/evaluation of the query.


.. py:exception:: VariableCannotBeEvaluated

   Bases: :py:obj:`krrood.utils.DataclassException`


   Raised when a variable cannot be evaluated due to missing or invalid information in the variable.


   .. py:attribute:: variable
      :type:  Variable


.. py:exception:: UsageError

   Bases: :py:obj:`krrood.utils.DataclassException`


   Raised when there is an incorrect usage of the entity query language API.


.. py:exception:: NoKwargsInMatchVar

   Bases: :py:obj:`UsageError`


   Raised when a match_variable is used without any keyword arguments.


   .. py:attribute:: match_variable
      :type:  krrood.entity_query_language.match.Match


.. py:exception:: WrongSelectableType

   Bases: :py:obj:`UsageError`


   Raised when a wrong variable type is given to the select() statement.


   .. py:attribute:: wrong_variable_type
      :type:  typing_extensions.Type


   .. py:attribute:: expected_types
      :type:  typing_extensions.List[typing_extensions.Type]


.. py:exception:: LiteralConditionError

   Bases: :py:obj:`UsageError`


   Raised when a literal (i.e. a non-variable) condition is given to the query.
   Example:
       >>> a = True
       >>> body = let(Body, None)
       >>> query = an(entity(body, a))
   This could also happen when you are using a predicate or a symbolic_function and all the given arguments are literals.
   Example:
       >>> predicate = HasType(Body("Body1"), Body)
       >>> query = an(entity(let(Body, None), predicate))
   So make sure that at least one of the arguments to the predicate or symbolic function are variables.


   .. py:attribute:: literal_conditions
      :type:  typing_extensions.List[typing_extensions.Any]


.. py:exception:: CannotProcessResultOfGivenChildType

   Bases: :py:obj:`UsageError`


   Raised when the entity query language API cannot process the results of a given child type during evaluation.


   .. py:attribute:: unsupported_child_type
      :type:  typing_extensions.Type

      The unsupported child type.



.. py:exception:: NonPositiveLimitValue

   Bases: :py:obj:`UsageError`


   Raised when a limit value for the query results is not positive.


   .. py:attribute:: wrong_limit_value
      :type:  int


.. py:exception:: UnsupportedOperation

   Bases: :py:obj:`UsageError`


   Raised when an operation is not supported by the entity query language API.


.. py:exception:: UnSupportedOperand

   Bases: :py:obj:`UnsupportedOperation`


   Raised when an operand is not supported by the operation.


   .. py:attribute:: operation
      :type:  typing_extensions.Type[krrood.entity_query_language.symbolic.SymbolicExpression]

      The operation used.



   .. py:attribute:: unsupported_operand
      :type:  typing_extensions.Any

      The operand that is not supported by the operation.



.. py:exception:: UnsupportedNegation

   Bases: :py:obj:`UnsupportedOperation`


   Raised when negating quantifiers.


   .. py:attribute:: operation_type
      :type:  typing_extensions.Type[krrood.entity_query_language.symbolic.SymbolicExpression]

      The type of the operation that is being negated.



.. py:exception:: QuantificationSpecificationError

   Bases: :py:obj:`UsageError`


   Raised when the quantification constraints specified on the query results are invalid or inconsistent.


.. py:exception:: QuantificationConsistencyError

   Bases: :py:obj:`QuantificationSpecificationError`


   Raised when the quantification constraints specified on the query results are inconsistent.


.. py:exception:: NegativeQuantificationError

   Bases: :py:obj:`QuantificationConsistencyError`


   Raised when the quantification constraints specified on the query results have a negative value.


   .. py:attribute:: message
      :type:  str
      :value: 'ResultQuantificationConstraint must be a non-negative integer.'



.. py:exception:: InvalidChildType

   Bases: :py:obj:`UsageError`


   Raised when an invalid entity type is given to the quantification operation.


   .. py:attribute:: invalid_child_type
      :type:  typing_extensions.Type

      The invalid child type.



   .. py:attribute:: correct_child_types
      :type:  typing_extensions.List[typing_extensions.Type]

      The list of valid child types.



.. py:exception:: InvalidEntityType

   Bases: :py:obj:`InvalidChildType`


   Raised when an invalid entity type is given to the quantification operation.


.. py:exception:: ClassDiagramError

   Bases: :py:obj:`krrood.utils.DataclassException`


   An error related to the class diagram.


.. py:exception:: NoneWrappedFieldError

   Bases: :py:obj:`ClassDiagramError`


   Raised when a field of a class is not wrapped by a WrappedField.


   .. py:attribute:: clazz
      :type:  typing_extensions.Type


   .. py:attribute:: attr_name
      :type:  str


