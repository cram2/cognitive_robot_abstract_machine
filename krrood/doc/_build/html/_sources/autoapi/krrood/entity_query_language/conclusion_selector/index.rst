krrood.entity_query_language.conclusion_selector
================================================

.. py:module:: krrood.entity_query_language.conclusion_selector


Classes
-------

.. autoapisummary::

   krrood.entity_query_language.conclusion_selector.ConclusionSelector
   krrood.entity_query_language.conclusion_selector.ExceptIf
   krrood.entity_query_language.conclusion_selector.Alternative
   krrood.entity_query_language.conclusion_selector.Next


Module Contents
---------------

.. py:class:: ConclusionSelector

   Bases: :py:obj:`krrood.entity_query_language.symbolic.LogicalBinaryOperator`, :py:obj:`abc.ABC`


   Base class for logical operators that may carry and select conclusions.

   Tracks whether certain conclusion-combinations were already produced so
   they are not duplicated across truth branches.


   .. py:attribute:: concluded_before
      :type:  typing_extensions.Dict[bool, krrood.entity_query_language.cache_data.SeenSet]


   .. py:method:: update_conclusion(output: krrood.entity_query_language.symbolic.OperationResult, conclusions: Set[krrood.entity_query_language.conclusion.Conclusion]) -> None

      Update conclusions if this combination hasn't been seen before.

      Uses canonical tuple keys for stable deduplication.



.. py:class:: ExceptIf

   Bases: :py:obj:`ConclusionSelector`


   Conditional branch that yields left unless the right side produces values.

   This encodes an "except if" behavior: when the right condition matches,
   the left branch's conclusions/outputs are excluded; otherwise, left flows through.


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[krrood.entity_query_language.symbolic.SymbolicExpression] = None) -> typing_extensions.Iterable[krrood.entity_query_language.symbolic.OperationResult]

      Evaluate the ExceptIf condition and yield the results.



   .. py:method:: yield_and_update_conclusion(result: krrood.entity_query_language.symbolic.OperationResult, conclusion: Set[krrood.entity_query_language.conclusion.Conclusion]) -> typing_extensions.Iterable[krrood.entity_query_language.symbolic.OperationResult]


.. py:class:: Alternative

   Bases: :py:obj:`krrood.entity_query_language.symbolic.ElseIf`, :py:obj:`ConclusionSelector`


   A conditional branch that behaves like an "else if" clause where the left branch
   is selected if it is true, otherwise the right branch is selected if it is true else
   none of the branches are selected.

   Uses both variable-based deduplication (from base class via projection) and
   conclusion-based deduplication (via update_conclusion).


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[krrood.entity_query_language.symbolic.SymbolicExpression] = None) -> typing_extensions.Iterable[krrood.entity_query_language.symbolic.OperationResult]

      Constrain the symbolic expression based on the indices of the operands.
      This method overrides the base class method to handle ElseIf logic.



.. py:class:: Next

   Bases: :py:obj:`krrood.entity_query_language.symbolic.Union`, :py:obj:`ConclusionSelector`


   A Union conclusion selector that always evaluates the left and right branches and combines their results.


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[krrood.entity_query_language.symbolic.SymbolicExpression] = None) -> typing_extensions.Iterable[krrood.entity_query_language.symbolic.OperationResult]

      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



