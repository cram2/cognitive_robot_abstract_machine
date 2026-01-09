krrood.entity_query_language.rule
=================================

.. py:module:: krrood.entity_query_language.rule


Functions
---------

.. autoapisummary::

   krrood.entity_query_language.rule.refinement
   krrood.entity_query_language.rule.alternative
   krrood.entity_query_language.rule.next_rule
   krrood.entity_query_language.rule.alternative_or_next


Module Contents
---------------

.. py:function:: refinement(*conditions: krrood.entity_query_language.entity.ConditionType) -> krrood.entity_query_language.symbolic.SymbolicExpression[krrood.entity_query_language.utils.T]

   Add a refinement branch (ExceptIf node with its right the new conditions and its left the base/parent rule/query)
    to the current condition tree.

   Each provided condition is chained with AND, and the resulting branch is
   connected via ExceptIf to the current node, representing a refinement/specialization path.

   :param conditions: The refinement conditions. They are chained with AND.
   :returns: The newly created branch node for further chaining.


.. py:function:: alternative(*conditions: krrood.entity_query_language.entity.ConditionType) -> krrood.entity_query_language.symbolic.SymbolicExpression[krrood.entity_query_language.utils.T]

   Add an alternative branch (logical ElseIf) to the current condition tree.

   Each provided condition is chained with AND, and the resulting branch is
   connected via ElseIf to the current node, representing an alternative path.

   :param conditions: Conditions to chain with AND and attach as an alternative.
   :returns: The newly created branch node for further chaining.


.. py:function:: next_rule(*conditions: krrood.entity_query_language.entity.ConditionType) -> krrood.entity_query_language.symbolic.SymbolicExpression[krrood.entity_query_language.utils.T]

   Add a consequent rule that gets always executed after the current rule.

   Each provided condition is chained with AND, and the resulting branch is
   connected via Next to the current node, representing the next path.

   :param conditions: Conditions to chain with AND and attach as an alternative.
   :returns: The newly created branch node for further chaining.


.. py:function:: alternative_or_next(type_: typing_extensions.Union[krrood.entity_query_language.enums.RDREdge.Alternative, krrood.entity_query_language.enums.RDREdge.Next], *conditions: krrood.entity_query_language.entity.ConditionType) -> krrood.entity_query_language.symbolic.SymbolicExpression[krrood.entity_query_language.utils.T]

   Add an alternative/next branch to the current condition tree.

   Each provided condition is chained with AND, and the resulting branch is
   connected via ElseIf/Next to the current node, representing an alternative/next path.

   :param type_: The type of the branch, either alternative or next.
   :param conditions: Conditions to chain with AND and attach as an alternative.
   :returns: The newly created branch node for further chaining.


