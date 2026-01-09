krrood.entity_query_language.entity
===================================

.. py:module:: krrood.entity_query_language.entity


Attributes
----------

.. autoapisummary::

   krrood.entity_query_language.entity.ConditionType


Functions
---------

.. autoapisummary::

   krrood.entity_query_language.entity.entity
   krrood.entity_query_language.entity.set_of
   krrood.entity_query_language.entity.variable
   krrood.entity_query_language.entity.variable_from
   krrood.entity_query_language.entity.and_
   krrood.entity_query_language.entity.or_
   krrood.entity_query_language.entity.not_
   krrood.entity_query_language.entity.contains
   krrood.entity_query_language.entity.in_
   krrood.entity_query_language.entity.flatten
   krrood.entity_query_language.entity.for_all
   krrood.entity_query_language.entity.exists
   krrood.entity_query_language.entity.inference


Module Contents
---------------

.. py:data:: ConditionType

   The possible types for conditions.


.. py:function:: entity(selected_variable: krrood.entity_query_language.utils.T) -> krrood.entity_query_language.symbolic.Entity[krrood.entity_query_language.utils.T]

   Create an entity descriptor for a selected variable.

   :param selected_variable: The variable to select in the result.
   :return: Entity descriptor.


.. py:function:: set_of(*selected_variables: typing_extensions.Union[krrood.entity_query_language.symbolic.Selectable[krrood.entity_query_language.utils.T], typing_extensions.Any]) -> krrood.entity_query_language.symbolic.SetOf

   Create a set descriptor for the selected variables.

   :param selected_variables: The variables to select in the result set.
   :return: Set descriptor.


.. py:function:: variable(type_: typing_extensions.Type[krrood.entity_query_language.utils.T], domain: krrood.entity_query_language.symbolic.DomainType, name: typing_extensions.Optional[str] = None, inferred: bool = False) -> typing_extensions.Union[krrood.entity_query_language.utils.T, krrood.entity_query_language.symbolic.Selectable[krrood.entity_query_language.utils.T]]

   Declare a symbolic variable that can be used inside queries.

   Filters the domain to elements that are instances of T.

   .. warning::

       If no domain is provided, and the type_ is a Symbol type, then the domain will be inferred from the SymbolGraph,
        which may contain unnecessarily many elements.

   :param type_: The type of variable.
   :param domain: Iterable of potential values for the variable or None.
    If None, the domain will be inferred from the SymbolGraph for Symbol types, else should not be evaluated by EQL
     but by another evaluator (e.g., EQL To SQL converter in Ormatic).
   :param name: The variable name, only required for pretty printing.
   :param inferred: Whether the variable is inferred or not.
   :return: A Variable that can be queried for.


.. py:function:: variable_from(domain: krrood.entity_query_language.symbolic.DomainType, name: typing_extensions.Optional[str] = None) -> typing_extensions.Union[krrood.entity_query_language.utils.T, krrood.entity_query_language.symbolic.Selectable[krrood.entity_query_language.utils.T]]

   Similar to `variable` but constructed from a domain directly wihout specifying its type.


.. py:function:: and_(*conditions: ConditionType)

   Logical conjunction of conditions.

   :param conditions: One or more conditions to combine.
   :type conditions: SymbolicExpression | bool
   :return: An AND operator joining the conditions.
   :rtype: SymbolicExpression


.. py:function:: or_(*conditions)

   Logical disjunction of conditions.

   :param conditions: One or more conditions to combine.
   :type conditions: SymbolicExpression | bool
   :return: An OR operator joining the conditions.
   :rtype: SymbolicExpression


.. py:function:: not_(operand: ConditionType) -> krrood.entity_query_language.symbolic.SymbolicExpression

   A symbolic NOT operation that can be used to negate symbolic expressions.


.. py:function:: contains(container: typing_extensions.Union[typing_extensions.Iterable, krrood.entity_query_language.symbolic.CanBehaveLikeAVariable[krrood.entity_query_language.utils.T]], item: typing_extensions.Any) -> krrood.entity_query_language.symbolic.Comparator

   Check whether a container contains an item.

   :param container: The container expression.
   :param item: The item to look for.
   :return: A comparator expression equivalent to ``item in container``.
   :rtype: SymbolicExpression


.. py:function:: in_(item: typing_extensions.Any, container: typing_extensions.Union[typing_extensions.Iterable, krrood.entity_query_language.symbolic.CanBehaveLikeAVariable[krrood.entity_query_language.utils.T]])

   Build a comparator for membership: ``item in container``.

   :param item: The candidate item.
   :param container: The container expression.
   :return: Comparator expression for membership.
   :rtype: Comparator


.. py:function:: flatten(var: typing_extensions.Union[krrood.entity_query_language.symbolic.CanBehaveLikeAVariable[krrood.entity_query_language.utils.T], typing_extensions.Iterable[krrood.entity_query_language.utils.T]]) -> typing_extensions.Union[krrood.entity_query_language.symbolic.CanBehaveLikeAVariable[krrood.entity_query_language.utils.T], krrood.entity_query_language.utils.T]

   Flatten a nested iterable domain into individual items while preserving the parent bindings.
   This returns a DomainMapping that, when evaluated, yields one solution per inner element
   (similar to SQL UNNEST), keeping existing variable bindings intact.


.. py:function:: for_all(universal_variable: typing_extensions.Union[krrood.entity_query_language.symbolic.CanBehaveLikeAVariable[krrood.entity_query_language.utils.T], krrood.entity_query_language.utils.T], condition: ConditionType)

   A universal on variable that finds all sets of variable bindings (values) that satisfy the condition for **every**
    value of the universal_variable.

   :param universal_variable: The universal on variable that the condition must satisfy for all its values.
   :param condition: A SymbolicExpression or bool representing a condition that must be satisfied.
   :return: A SymbolicExpression that can be evaluated producing every set that satisfies the condition.


.. py:function:: exists(universal_variable: typing_extensions.Union[krrood.entity_query_language.symbolic.CanBehaveLikeAVariable[krrood.entity_query_language.utils.T], krrood.entity_query_language.utils.T], condition: ConditionType)

   A universal on variable that finds all sets of variable bindings (values) that satisfy the condition for **any**
    value of the universal_variable.

   :param universal_variable: The universal on variable that the condition must satisfy for any of its values.
   :param condition: A SymbolicExpression or bool representing a condition that must be satisfied.
   :return: A SymbolicExpression that can be evaluated producing every set that satisfies the condition.


.. py:function:: inference(type_: typing_extensions.Type[krrood.entity_query_language.utils.T]) -> typing_extensions.Union[typing_extensions.Type[krrood.entity_query_language.utils.T], typing_extensions.Callable[[typing_extensions.Any], krrood.entity_query_language.symbolic.Variable[krrood.entity_query_language.utils.T]]]

   This returns a factory function that creates a new variable of the given type and takes keyword arguments for the
   type constructor.

   :param type_: The type of the variable (i.e., The class you want to instantiate).
   :return: The factory function for creating a new variable.


