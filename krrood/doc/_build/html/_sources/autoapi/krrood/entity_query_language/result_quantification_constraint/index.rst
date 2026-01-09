krrood.entity_query_language.result_quantification_constraint
=============================================================

.. py:module:: krrood.entity_query_language.result_quantification_constraint


Classes
-------

.. autoapisummary::

   krrood.entity_query_language.result_quantification_constraint.ResultQuantificationConstraint
   krrood.entity_query_language.result_quantification_constraint.SingleValueQuantificationConstraint
   krrood.entity_query_language.result_quantification_constraint.Exactly
   krrood.entity_query_language.result_quantification_constraint.AtLeast
   krrood.entity_query_language.result_quantification_constraint.AtMost
   krrood.entity_query_language.result_quantification_constraint.Range


Module Contents
---------------

.. py:class:: ResultQuantificationConstraint

   Bases: :py:obj:`abc.ABC`


   A base class that represents a constraint for quantification.


   .. py:method:: assert_satisfaction(number_of_solutions: int, quantifier: krrood.entity_query_language.symbolic.ResultQuantifier, done: bool) -> None
      :abstractmethod:


      Check if the constraint is satisfied, if not, raise a QuantificationNotSatisfiedError exception.

      :param number_of_solutions: The current number of solutions.
      :param quantifier: The quantifier expression of the query.
      :param done: Whether all results have been found.
      :raises: QuantificationNotSatisfiedError: If the constraint is not satisfied.



.. py:class:: SingleValueQuantificationConstraint

   Bases: :py:obj:`ResultQuantificationConstraint`, :py:obj:`abc.ABC`


   A class that represents a single value constraint on the result quantification.


   .. py:attribute:: value
      :type:  int

      The exact value of the constraint.



.. py:class:: Exactly

   Bases: :py:obj:`SingleValueQuantificationConstraint`


   A class that represents an exact constraint on the result quantification.


   .. py:method:: assert_satisfaction(number_of_solutions: int, quantifier: krrood.entity_query_language.symbolic.ResultQuantifier, done: bool) -> None

      Check if the constraint is satisfied, if not, raise a QuantificationNotSatisfiedError exception.

      :param number_of_solutions: The current number of solutions.
      :param quantifier: The quantifier expression of the query.
      :param done: Whether all results have been found.
      :raises: QuantificationNotSatisfiedError: If the constraint is not satisfied.



.. py:class:: AtLeast

   Bases: :py:obj:`SingleValueQuantificationConstraint`


   A class that specifies a minimum number of results as a quantification constraint.


   .. py:method:: assert_satisfaction(number_of_solutions: int, quantifier: krrood.entity_query_language.symbolic.ResultQuantifier, done: bool) -> None

      Check if the constraint is satisfied, if not, raise a QuantificationNotSatisfiedError exception.

      :param number_of_solutions: The current number of solutions.
      :param quantifier: The quantifier expression of the query.
      :param done: Whether all results have been found.
      :raises: QuantificationNotSatisfiedError: If the constraint is not satisfied.



.. py:class:: AtMost

   Bases: :py:obj:`SingleValueQuantificationConstraint`


   A class that specifies a maximum number of results as a quantification constraint.


   .. py:method:: assert_satisfaction(number_of_solutions: int, quantifier: krrood.entity_query_language.symbolic.ResultQuantifier, done: bool) -> None

      Check if the constraint is satisfied, if not, raise a QuantificationNotSatisfiedError exception.

      :param number_of_solutions: The current number of solutions.
      :param quantifier: The quantifier expression of the query.
      :param done: Whether all results have been found.
      :raises: QuantificationNotSatisfiedError: If the constraint is not satisfied.



.. py:class:: Range

   Bases: :py:obj:`ResultQuantificationConstraint`


   A class that represents a range constraint on the result quantification.


   .. py:attribute:: at_least
      :type:  AtLeast

      The minimum value of the range.



   .. py:attribute:: at_most
      :type:  AtMost

      The maximum value of the range.



   .. py:method:: assert_satisfaction(number_of_solutions: int, quantifier: krrood.entity_query_language.symbolic.ResultQuantifier, done: bool) -> None

      Check if the constraint is satisfied, if not, raise a QuantificationNotSatisfiedError exception.

      :param number_of_solutions: The current number of solutions.
      :param quantifier: The quantifier expression of the query.
      :param done: Whether all results have been found.
      :raises: QuantificationNotSatisfiedError: If the constraint is not satisfied.



