krrood.entity_query_language.symbolic
=====================================

.. py:module:: krrood.entity_query_language.symbolic

.. autoapi-nested-parse::

   Core symbolic expression system used to build and evaluate entity queries.

   This module defines the symbolic types (variables, sources, logical and
   comparison operators) and the evaluation mechanics.



Attributes
----------

.. autoapisummary::

   krrood.entity_query_language.symbolic.id_generator
   krrood.entity_query_language.symbolic.ResultMapping
   krrood.entity_query_language.symbolic.OperatorOptimizer
   krrood.entity_query_language.symbolic.DomainType


Classes
-------

.. autoapisummary::

   krrood.entity_query_language.symbolic.OperationResult
   krrood.entity_query_language.symbolic.SymbolicExpression
   krrood.entity_query_language.symbolic.Selectable
   krrood.entity_query_language.symbolic.CanBehaveLikeAVariable
   krrood.entity_query_language.symbolic.ResultProcessor
   krrood.entity_query_language.symbolic.Aggregator
   krrood.entity_query_language.symbolic.Count
   krrood.entity_query_language.symbolic.EntityAggregator
   krrood.entity_query_language.symbolic.Sum
   krrood.entity_query_language.symbolic.Average
   krrood.entity_query_language.symbolic.Extreme
   krrood.entity_query_language.symbolic.Max
   krrood.entity_query_language.symbolic.Min
   krrood.entity_query_language.symbolic.ResultQuantifier
   krrood.entity_query_language.symbolic.UnificationDict
   krrood.entity_query_language.symbolic.An
   krrood.entity_query_language.symbolic.The
   krrood.entity_query_language.symbolic.OrderByParams
   krrood.entity_query_language.symbolic.QueryObjectDescriptor
   krrood.entity_query_language.symbolic.SetOf
   krrood.entity_query_language.symbolic.Entity
   krrood.entity_query_language.symbolic.Variable
   krrood.entity_query_language.symbolic.Literal
   krrood.entity_query_language.symbolic.DomainMapping
   krrood.entity_query_language.symbolic.Attribute
   krrood.entity_query_language.symbolic.Index
   krrood.entity_query_language.symbolic.Call
   krrood.entity_query_language.symbolic.Flatten
   krrood.entity_query_language.symbolic.BinaryOperator
   krrood.entity_query_language.symbolic.Comparator
   krrood.entity_query_language.symbolic.LogicalOperator
   krrood.entity_query_language.symbolic.Not
   krrood.entity_query_language.symbolic.LogicalBinaryOperator
   krrood.entity_query_language.symbolic.AND
   krrood.entity_query_language.symbolic.OR
   krrood.entity_query_language.symbolic.Union
   krrood.entity_query_language.symbolic.ElseIf
   krrood.entity_query_language.symbolic.QuantifiedConditional
   krrood.entity_query_language.symbolic.ForAll
   krrood.entity_query_language.symbolic.Exists


Functions
---------

.. autoapisummary::

   krrood.entity_query_language.symbolic.not_contains
   krrood.entity_query_language.symbolic.chained_logic
   krrood.entity_query_language.symbolic.optimize_or


Module Contents
---------------

.. py:data:: id_generator

.. py:class:: OperationResult

   A data structure that carries information about the result of an operation in EQL.


   .. py:attribute:: bindings
      :type:  typing_extensions.Dict[int, typing_extensions.Any]

      The bindings resulting from the operation, mapping variable IDs to their values.



   .. py:attribute:: is_false
      :type:  bool

      Whether the operation resulted in a false value (i.e., The operation condition was not satisfied)



   .. py:attribute:: operand
      :type:  SymbolicExpression

      The operand that produced the result.



   .. py:property:: is_true


   .. py:property:: value
      :type: typing_extensions.Optional[typing_extensions.Any]



.. py:class:: SymbolicExpression

   Bases: :py:obj:`typing_extensions.Generic`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   Base class for all symbolic expressions.

   Symbolic expressions form a tree and are evaluated lazily to produce
   bindings for variables, subject to logical constraints.

   :ivar _child_: Optional child expression.
   :ivar _id_: Unique identifier of this node.
   :ivar _node_: Backing anytree.Node for visualization and traversal.
   :ivar _conclusion_: Set of conclusion actions attached to this node.
   :ivar _is_false_: Internal flag indicating evaluation result for this node.


   .. py:attribute:: _plot_color__
      :type:  typing_extensions.Optional[krrood.entity_query_language.rxnode.ColorLegend]
      :value: None



   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[OperationResult]
      :abstractmethod:


      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



.. py:data:: ResultMapping

   A function that maps the results of a query object descriptor to a new set of results.


.. py:class:: Selectable

   Bases: :py:obj:`SymbolicExpression`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   Base class for all symbolic expressions.

   Symbolic expressions form a tree and are evaluated lazily to produce
   bindings for variables, subject to logical constraints.

   :ivar _child_: Optional child expression.
   :ivar _id_: Unique identifier of this node.
   :ivar _node_: Backing anytree.Node for visualization and traversal.
   :ivar _conclusion_: Set of conclusion actions attached to this node.
   :ivar _is_false_: Internal flag indicating evaluation result for this node.


   .. py:property:: _type__


.. py:class:: CanBehaveLikeAVariable

   Bases: :py:obj:`Selectable`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   This class adds the monitoring/tracking behavior on variables that tracks attribute access, calling,
   and comparison operations.


.. py:class:: ResultProcessor

   Bases: :py:obj:`CanBehaveLikeAVariable`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   Base class for processors that return concrete results from queries, including quantifiers
   (e.g., An, The) and aggregators (e.g., Count, Sum, Max, Min).


   .. py:method:: evaluate() -> typing_extensions.Iterable[typing_extensions.Union[krrood.entity_query_language.utils.T, typing_extensions.Dict[typing_extensions.Union[krrood.entity_query_language.utils.T, SymbolicExpression[krrood.entity_query_language.utils.T]], krrood.entity_query_language.utils.T]]]

      Evaluate the query and map the results to the correct output data structure.
      This is the exposed evaluation method for users.



   .. py:method:: visualize(figsize=(35, 30), node_size=7000, font_size=25, spacing_x: float = 4, spacing_y: float = 4, layout: str = 'tidy', edge_style: str = 'orthogonal', label_max_chars_per_line: typing_extensions.Optional[int] = 13)

      Visualize the query graph, for arguments' documentation see `rustworkx_utils.RWXNode.visualize`.



.. py:class:: Aggregator

   Bases: :py:obj:`ResultProcessor`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   Base class for processors that return concrete results from queries, including quantifiers
   (e.g., An, The) and aggregators (e.g., Count, Sum, Max, Min).


   .. py:method:: evaluate() -> typing_extensions.Iterable[typing_extensions.Union[krrood.entity_query_language.utils.T, typing_extensions.Dict[typing_extensions.Union[krrood.entity_query_language.utils.T, SymbolicExpression[krrood.entity_query_language.utils.T]], krrood.entity_query_language.utils.T]]]

      Evaluate the query and map the results to the correct output data structure.
      This is the exposed evaluation method for users.



   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[OperationResult]

      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



.. py:class:: Count

   Bases: :py:obj:`Aggregator`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   Count the number of child results.


.. py:class:: EntityAggregator

   Bases: :py:obj:`Aggregator`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   Base class for processors that return concrete results from queries, including quantifiers
   (e.g., An, The) and aggregators (e.g., Count, Sum, Max, Min).


.. py:class:: Sum

   Bases: :py:obj:`EntityAggregator`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   Calculate the sum of the child results. If given, make use of the key function to extract the value to be summed.


.. py:class:: Average

   Bases: :py:obj:`EntityAggregator`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   Calculate the average of the child results. If given, make use of the key function to extract the value to be
    averaged.


.. py:class:: Extreme

   Bases: :py:obj:`EntityAggregator`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   Find and return the extreme value among the child results. If given, make use of the key function to extract
    the value to be compared.


.. py:class:: Max

   Bases: :py:obj:`Extreme`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   Find and return the maximum value among the child results. If given, make use of the key function to extract
    the value to be compared.


.. py:class:: Min

   Bases: :py:obj:`Extreme`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   Find and return the minimum value among the child results. If given, make use of the key function to extract
    the value to be compared.


.. py:class:: ResultQuantifier

   Bases: :py:obj:`ResultProcessor`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   Base for quantifiers that return concrete results from entity/set queries
   (e.g., An, The).


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[krrood.entity_query_language.utils.T]

      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



.. py:class:: UnificationDict(dict=None, /, **kwargs)

   Bases: :py:obj:`collections.UserDict`


   A dictionary which maps all expressions that are on a single variable to the original variable id.


.. py:class:: An

   Bases: :py:obj:`ResultQuantifier`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   Quantifier that yields all matching results one by one.


   .. py:method:: evaluate(limit: typing_extensions.Optional[int] = None) -> typing_extensions.Iterable[typing_extensions.Union[krrood.entity_query_language.utils.T, typing_extensions.Dict[typing_extensions.Union[krrood.entity_query_language.utils.T, SymbolicExpression[krrood.entity_query_language.utils.T]], krrood.entity_query_language.utils.T]]]

      Evaluate the query and map the results to the correct output data structure.
      This is the exposed evaluation method for users.



.. py:class:: The

   Bases: :py:obj:`ResultQuantifier`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   Quantifier that expects exactly one result; raises MultipleSolutionFound if more.


   .. py:method:: evaluate() -> typing_extensions.Union[krrood.entity_query_language.utils.T, typing_extensions.Dict[typing_extensions.Union[krrood.entity_query_language.utils.T, SymbolicExpression[krrood.entity_query_language.utils.T]], krrood.entity_query_language.utils.T]]

      Evaluate the query and map the results to the correct output data structure.
      This is the exposed evaluation method for users.



   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[typing_extensions.Union[krrood.entity_query_language.utils.T, typing_extensions.Dict[typing_extensions.Union[krrood.entity_query_language.utils.T, SymbolicExpression[krrood.entity_query_language.utils.T]], krrood.entity_query_language.utils.T]]]

      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



.. py:class:: OrderByParams

   Parameters for ordering the results of a query object descriptor.


   .. py:attribute:: variable
      :type:  Selectable

      The variable to order by.



   .. py:attribute:: descending
      :type:  bool
      :value: False


      Whether to order the results in descending order.



   .. py:attribute:: key
      :type:  typing_extensions.Optional[typing_extensions.Callable]
      :value: None


      A function to extract the key from the variable value.



.. py:class:: QueryObjectDescriptor

   Bases: :py:obj:`SymbolicExpression`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   Describes the queried object(s), could be a query over a single variable or a set of variables,
   also describes the condition(s)/properties of the queried object(s).


   .. py:method:: where(*conditions: krrood.entity_query_language.entity.ConditionType) -> typing_extensions.Self

      Set the conditions that describe the query object. The conditions are chained using AND.

      :param conditions: The conditions that describe the query object.
      :return: This query object descriptor.



   .. py:method:: order_by(variable: Selectable, descending: bool = False, key: typing_extensions.Optional[typing_extensions.Callable] = None) -> typing_extensions.Self

      Order the results by the given variable, using the given key function in descending or ascending order.

      :param variable: The variable to order by.
      :param descending: Whether to order the results in descending order.
      :param key: A function to extract the key from the variable value.



   .. py:method:: distinct(*on: Selectable[krrood.entity_query_language.utils.T]) -> typing_extensions.Self

      Apply distinctness constraint to the query object descriptor results.

      :param on: The variables to be used for distinctness.
      :return: This query object descriptor.



   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[OperationResult]

      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



   .. py:method:: variable_is_inferred(var: CanBehaveLikeAVariable[krrood.entity_query_language.utils.T]) -> bool
      :staticmethod:


      Whether the variable is inferred or not.

      :param var: The variable.
      :return: True if the variable is inferred, otherwise False.



   .. py:method:: any_selected_variable_is_inferred_and_unbound(values: OperationResult) -> bool

      Check if any of the selected variables is inferred and is not bound.

      :param values: The current result with the current bindings.
      :return: True if any of the selected variables is inferred and is not bound, otherwise False.



   .. py:method:: variable_is_bound_or_its_children_are_bound(var: CanBehaveLikeAVariable[krrood.entity_query_language.utils.T], result: OperationResult) -> bool

      Whether the variable is directly bound or all its children are bound.

      :param var: The variable.
      :param result: The current result containing the current bindings.
      :return: True if the variable is bound, otherwise False.



   .. py:method:: evaluate_conclusions_and_update_bindings(child_result: OperationResult)

      Update the bindings of the results by evaluating the conclusions using the received bindings from the child as
      sources.

      :param child_result: The result of the child operation.



   .. py:method:: get_constrained_values(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]]) -> typing_extensions.Iterable[OperationResult]

      Evaluate the child (i.e., the conditions that constrain the domain of the selected variables).

      :param sources: The current bindings.
      :return: The bindings after applying the constraints of the child.



.. py:class:: SetOf

   Bases: :py:obj:`QueryObjectDescriptor`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   A query over a set of variables.


.. py:class:: Entity

   Bases: :py:obj:`QueryObjectDescriptor`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`Selectable`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   A query over a single variable.


   .. py:property:: selected_variable


.. py:class:: Variable

   Bases: :py:obj:`CanBehaveLikeAVariable`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   A Variable that queries will assign. The Variable produces results of type `T`.


   .. py:attribute:: _name__
      :type:  str

      The name of the variable.



   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[OperationResult]

      A variable either is already bound in sources by other constraints (Symbolic Expressions).,
      or will yield from current domain if exists,
      or has no domain and will instantiate new values by constructing the type if the type is given.



.. py:class:: Literal(data: typing_extensions.Any, name: typing_extensions.Optional[str] = None, type_: typing_extensions.Optional[typing_extensions.Type] = None, wrap_in_iterator: bool = True)

   Bases: :py:obj:`Variable`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   Literals are variables that are not constructed by their type but by their given data.


.. py:class:: DomainMapping

   Bases: :py:obj:`CanBehaveLikeAVariable`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   A symbolic expression the maps the domain of symbolic variables.


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[OperationResult]

      Apply the domain mapping to the child's values.



.. py:class:: Attribute

   Bases: :py:obj:`DomainMapping`


   A symbolic attribute that can be used to access attributes of symbolic variables.

   For instance, if Body.name is called, then the attribute name is "name" and `_owner_class_` is `Body`


.. py:class:: Index

   Bases: :py:obj:`DomainMapping`


   A symbolic indexing operation that can be used to access items of symbolic variables via [] operator.


.. py:class:: Call

   Bases: :py:obj:`DomainMapping`


   A symbolic call that can be used to call methods on symbolic variables.


.. py:class:: Flatten

   Bases: :py:obj:`DomainMapping`


   Domain mapping that flattens an iterable-of-iterables into a single iterable of items.

   Given a child expression that evaluates to an iterable (e.g., Views.bodies), this mapping yields
   one solution per inner element while preserving the original bindings (e.g., the View instance),
   similar to UNNEST in SQL.


.. py:class:: BinaryOperator

   Bases: :py:obj:`SymbolicExpression`, :py:obj:`abc.ABC`


   A base class for binary operators that can be used to combine symbolic expressions.


   .. py:attribute:: left
      :type:  SymbolicExpression


   .. py:attribute:: right
      :type:  SymbolicExpression


.. py:function:: not_contains(container, item) -> bool

   The inverted contains operation.

   :param container: The container.
   :param item: The item to test if contained in the container.
   :return:


.. py:class:: Comparator

   Bases: :py:obj:`BinaryOperator`


   A symbolic equality check that can be used to compare symbolic variables using a provided comparison operation.


   .. py:attribute:: left
      :type:  CanBehaveLikeAVariable


   .. py:attribute:: right
      :type:  CanBehaveLikeAVariable


   .. py:attribute:: operation
      :type:  typing_extensions.Callable[[typing_extensions.Any, typing_extensions.Any], bool]


   .. py:attribute:: operation_name_map
      :type:  typing_extensions.ClassVar[typing_extensions.Dict[typing_extensions.Any, str]]


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[OperationResult]

      Compares the left and right symbolic variables using the "operation".



   .. py:method:: apply_operation(operand_values: OperationResult) -> bool


   .. py:method:: get_first_second_operands(sources: typing_extensions.Dict[int, typing_extensions.Any]) -> typing_extensions.Tuple[SymbolicExpression, SymbolicExpression]


.. py:class:: LogicalOperator

   Bases: :py:obj:`SymbolicExpression`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   A symbolic operation that can be used to combine multiple symbolic expressions using logical constraints on their
   truth values. Examples are conjunction (AND), disjunction (OR), negation (NOT), and conditional quantification
   (ForALL, Exists).


.. py:class:: Not

   Bases: :py:obj:`LogicalOperator`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   The logical negation of a symbolic expression. Its truth value is the opposite of its child's truth value. This is
   used when you want bindings that satisfy the negated condition (i.e., that doesn't satisfy the original condition).


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[OperationResult]

      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



.. py:class:: LogicalBinaryOperator

   Bases: :py:obj:`LogicalOperator`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`BinaryOperator`, :py:obj:`abc.ABC`


   A symbolic operation that can be used to combine multiple symbolic expressions using logical constraints on their
   truth values. Examples are conjunction (AND), disjunction (OR), negation (NOT), and conditional quantification
   (ForALL, Exists).


.. py:class:: AND

   Bases: :py:obj:`LogicalBinaryOperator`


   A symbolic AND operation that can be used to combine multiple symbolic expressions.


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[OperationResult]

      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



   .. py:method:: evaluate_right(left_value: OperationResult) -> typing_extensions.Iterable[OperationResult]


.. py:class:: OR

   Bases: :py:obj:`LogicalBinaryOperator`, :py:obj:`abc.ABC`


   A symbolic single choice operation that can be used to choose between multiple symbolic expressions.


   .. py:attribute:: left_evaluated
      :type:  bool
      :value: False



   .. py:attribute:: right_evaluated
      :type:  bool
      :value: False



   .. py:method:: evaluate_left(sources: typing_extensions.Dict[int, typing_extensions.Any]) -> typing_extensions.Iterable[OperationResult]

      Evaluate the left operand, taking into consideration if it should yield when it is False.

      :param sources: The current bindings to use for evaluation.
      :return: The new bindings after evaluating the left operand (and possibly right operand).



   .. py:method:: evaluate_right(sources: typing_extensions.Dict[int, typing_extensions.Any]) -> typing_extensions.Iterable[OperationResult]

      Evaluate the right operand.

      :param sources: The current bindings to use during evaluation.
      :return: The new bindings after evaluating the right operand.



.. py:class:: Union

   Bases: :py:obj:`OR`


   This operator is a version of the OR operator that always evaluates both the left and the right operand.


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[OperationResult]

      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



.. py:class:: ElseIf

   Bases: :py:obj:`OR`


   A version of the OR operator that evaluates the right operand only when the left operand is False.


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[OperationResult]

      Constrain the symbolic expression based on the indices of the operands.
      This method overrides the base class method to handle ElseIf logic.



.. py:class:: QuantifiedConditional

   Bases: :py:obj:`LogicalBinaryOperator`, :py:obj:`abc.ABC`


   This is the super class of the universal, and existential conditional operators. It is a binary logical operator
   that has a quantified variable and a condition on the values of that variable.


   .. py:property:: variable


   .. py:property:: condition


.. py:class:: ForAll

   Bases: :py:obj:`QuantifiedConditional`


   This operator is the universal conditional operator. It returns bindings that satisfy the condition for all the
   values of the quantified variable. It short circuits by ignoring the bindings that doesn't satisfy the condition.


   .. py:property:: condition_unique_variable_ids
      :type: typing_extensions.List[int]



   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[OperationResult]

      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



   .. py:method:: get_all_candidate_solutions(sources: typing_extensions.Dict[int, typing_extensions.Any])


   .. py:method:: evaluate_condition(sources: typing_extensions.Dict[int, typing_extensions.Any]) -> bool


.. py:class:: Exists

   Bases: :py:obj:`QuantifiedConditional`


   An existential checker that checks if a condition holds for any value of the variable given, the benefit
   of this is that this short circuits the condition and returns True if the condition holds for any value without
   getting all the condition values that hold for one specific value of the variable.


   .. py:method:: _evaluate__(sources: typing_extensions.Optional[typing_extensions.Dict[int, typing_extensions.Any]] = None, parent: typing_extensions.Optional[SymbolicExpression] = None) -> typing_extensions.Iterable[OperationResult]

      Evaluate the symbolic expression and set the operands indices.
      This method should be implemented by subclasses.



.. py:data:: OperatorOptimizer

.. py:function:: chained_logic(operator: typing_extensions.Union[typing_extensions.Type[LogicalOperator], OperatorOptimizer], *conditions)

   A chian of logic operation over multiple conditions, e.g. cond1 | cond2 | cond3.

   :param operator: The symbolic operator to apply between the conditions.
   :param conditions: The conditions to be chained.


.. py:function:: optimize_or(left: SymbolicExpression, right: SymbolicExpression) -> OR

.. py:data:: DomainType

