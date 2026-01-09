krrood.entity_query_language.conclusion
=======================================

.. py:module:: krrood.entity_query_language.conclusion


Classes
-------

.. autoapisummary::

   krrood.entity_query_language.conclusion.Conclusion
   krrood.entity_query_language.conclusion.Set
   krrood.entity_query_language.conclusion.Add


Module Contents
---------------

.. py:class:: Conclusion

   Bases: :py:obj:`krrood.entity_query_language.symbolic.SymbolicExpression`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   Base for side-effecting/action clauses that adjust outputs (e.g., Set, Add).

   :ivar var: The variable being affected by the conclusion.
   :ivar value: The value or expression used by the conclusion.


   .. py:attribute:: var
      :type:  krrood.entity_query_language.symbolic.Selectable[krrood.entity_query_language.utils.T]


   .. py:attribute:: value
      :type:  typing_extensions.Any


.. py:class:: Set

   Bases: :py:obj:`Conclusion`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   Set the value of a variable in the current solution binding.


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[krrood.entity_query_language.symbolic.SymbolicExpression] = None) -> typing_extensions.Iterable[krrood.entity_query_language.symbolic.OperationResult]

      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



.. py:class:: Add

   Bases: :py:obj:`Conclusion`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   Add a new value to the domain of a variable.


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[krrood.entity_query_language.symbolic.SymbolicExpression] = None) -> typing_extensions.Iterable[krrood.entity_query_language.symbolic.OperationResult]

      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



