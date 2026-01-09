krrood.entity_query_language.match
==================================

.. py:module:: krrood.entity_query_language.match


Classes
-------

.. autoapisummary::

   krrood.entity_query_language.match.AbstractMatchExpression
   krrood.entity_query_language.match.Match
   krrood.entity_query_language.match.MatchVariable
   krrood.entity_query_language.match.AttributeMatch


Functions
---------

.. autoapisummary::

   krrood.entity_query_language.match.match
   krrood.entity_query_language.match.match_variable


Module Contents
---------------

.. py:class:: AbstractMatchExpression

   Bases: :py:obj:`typing_extensions.Generic`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ], :py:obj:`abc.ABC`


   Abstract base class for constructing and handling a match expression.

   This class is intended to provide a framework for defining and managing match expressions,
   which are used to structural pattern matching in the form of nested match expressions with keyword arguments.


   .. py:attribute:: type_
      :type:  typing_extensions.Optional[typing_extensions.Type[krrood.entity_query_language.utils.T]]
      :value: None


      The type of the variable.



   .. py:attribute:: variable
      :type:  typing_extensions.Optional[krrood.entity_query_language.symbolic.Selectable[krrood.entity_query_language.utils.T]]
      :value: None


      The created variable from the type and kwargs.



   .. py:attribute:: conditions
      :type:  typing_extensions.List[krrood.entity_query_language.entity.ConditionType]
      :value: []


      The conditions that define the match.



   .. py:attribute:: parent
      :type:  typing_extensions.Optional[Match]
      :value: None


      The parent match if this is a nested match.



   .. py:attribute:: node
      :type:  typing_extensions.Optional[krrood.entity_query_language.rxnode.RWXNode]
      :value: None


      The RWXNode representing the match expression in the match query graph.



   .. py:attribute:: resolved
      :type:  bool
      :value: False


      Whether the match is resolved or not.



   .. py:property:: expression
      :type: typing_extensions.Union[krrood.entity_query_language.symbolic.CanBehaveLikeAVariable[krrood.entity_query_language.utils.T], krrood.entity_query_language.utils.T]

      :abstractmethod:


      :return: the entity expression corresponding to the match query.



   .. py:method:: resolve(*args, **kwargs)

      Resolve the match by creating the variable and conditions expressions.



   .. py:property:: name
      :type: str

      :abstractmethod:



   .. py:property:: id


   .. py:property:: type
      :type: typing_extensions.Optional[typing_extensions.Type[krrood.entity_query_language.utils.T]]


      If type is predefined return it, else if the variable is available return its type, else return None.



   .. py:property:: root
      :type: Match


      :return: The root match expression.



.. py:class:: Match

   Bases: :py:obj:`AbstractMatchExpression`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   Construct a query that looks for the pattern provided by the type and the keyword arguments.
   Example usage where we look for an object of type Drawer with body of type Body that has the name"drawer_1":
       >>> @dataclass
       >>> class Body:
       >>>     name: str
       >>> @dataclass
       >>> class Drawer:
       >>>     body: Body
       >>> drawer = match_variable(Drawer, domain=None)(body=match(Body)(name="drawer_1")))


   .. py:attribute:: kwargs
      :type:  typing_extensions.Dict[str, typing_extensions.Any]

      The keyword arguments to match against.



   .. py:property:: expression
      :type: typing_extensions.Union[krrood.entity_query_language.symbolic.An[krrood.entity_query_language.utils.T], krrood.entity_query_language.utils.T]


      Return the entity expression corresponding to the match query.



   .. py:method:: update_fields(variable: typing_extensions.Optional[krrood.entity_query_language.symbolic.Selectable] = None, parent: typing_extensions.Optional[Match] = None)

      Update the match variable, and parent.

      :param variable: The variable to use for the match.
       If None, a new variable will be created.
      :param parent: The parent match if this is a nested match.



   .. py:method:: create_variable()


   .. py:method:: evaluate()

      Evaluate the match expression and return the result.



   .. py:property:: name
      :type: str



.. py:class:: MatchVariable

   Bases: :py:obj:`Match`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   Represents a match variable that operates within a specified domain.

   A class designed to create and manage a variable constrained by a defined
   domain. It provides functionality to add additional constraints via
   keyword arguments and return an expression representing the resolved
   constraints.


   .. py:attribute:: domain
      :type:  krrood.entity_query_language.symbolic.DomainType
      :value: None


      The domain to use for the variable created by the match.



   .. py:method:: create_variable()


.. py:class:: AttributeMatch

   Bases: :py:obj:`AbstractMatchExpression`\ [\ :py:obj:`krrood.entity_query_language.utils.T`\ ]


   A class representing an attribute assignment in a Match statement.


   .. py:attribute:: parent
      :type:  AbstractMatchExpression

      The parent match expression.



   .. py:attribute:: attribute_name
      :type:  str

      The name of the attribute to assign the value to.



   .. py:attribute:: assigned_value
      :type:  typing_extensions.Optional[typing_extensions.Union[krrood.entity_query_language.symbolic.Literal, Match]]
      :value: None


      The value to assign to the attribute, which can be a Match instance or a Literal.



   .. py:attribute:: variable
      :type:  typing_extensions.Union[krrood.entity_query_language.symbolic.Attribute, krrood.entity_query_language.symbolic.Flatten]
      :value: None


      The symbolic variable representing the attribute.



   .. py:property:: expression
      :type: typing_extensions.Union[krrood.entity_query_language.symbolic.CanBehaveLikeAVariable[krrood.entity_query_language.utils.T], krrood.entity_query_language.utils.T]


      Return the entity expression corresponding to the match query.



   .. py:property:: assigned_variable
      :type: krrood.entity_query_language.symbolic.Selectable


      :return: The symbolic variable representing the assigned value.



   .. py:property:: attribute
      :type: krrood.entity_query_language.symbolic.Attribute


      :return: the attribute of the variable.
      :raises NoneWrappedFieldError: If the attribute does not have a WrappedField.



   .. py:property:: is_type_filter_needed

      :return: True if a type filter condition is needed for the attribute assignment, else False.



   .. py:property:: name
      :type: str



.. py:function:: match(type_: typing_extensions.Optional[typing_extensions.Union[typing_extensions.Type[krrood.entity_query_language.utils.T], krrood.entity_query_language.symbolic.Selectable[krrood.entity_query_language.utils.T]]] = None) -> typing_extensions.Union[typing_extensions.Type[krrood.entity_query_language.utils.T], krrood.entity_query_language.symbolic.CanBehaveLikeAVariable[krrood.entity_query_language.utils.T], Match[krrood.entity_query_language.utils.T]]

   Create a symbolic variable matching the type and the provided keyword arguments. This is used for easy variable
    definitions when there are structural constraints.

   :param type_: The type of the variable (i.e., The class you want to instantiate).
   :return: The Match instance.


.. py:function:: match_variable(type_: typing_extensions.Union[typing_extensions.Type[krrood.entity_query_language.utils.T], krrood.entity_query_language.symbolic.Selectable[krrood.entity_query_language.utils.T]], domain: krrood.entity_query_language.symbolic.DomainType) -> typing_extensions.Union[typing_extensions.Type[krrood.entity_query_language.utils.T], krrood.entity_query_language.symbolic.An[krrood.entity_query_language.utils.T], krrood.entity_query_language.symbolic.CanBehaveLikeAVariable[krrood.entity_query_language.utils.T], MatchVariable[krrood.entity_query_language.utils.T]]

   Same as :py:func:`krrood.entity_query_language.match.match` but with a domain to use for the variable created
    by the match.

   :param type_: The type of the variable (i.e., The class you want to instantiate).
   :param domain: The domain used for the variable created by the match.
   :return: The Match instance.


