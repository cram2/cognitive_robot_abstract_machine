krrood.entity_query_language.predicate
======================================

.. py:module:: krrood.entity_query_language.predicate


Classes
-------

.. autoapisummary::

   krrood.entity_query_language.predicate.Symbol
   krrood.entity_query_language.predicate.Predicate
   krrood.entity_query_language.predicate.HasType
   krrood.entity_query_language.predicate.HasTypes


Functions
---------

.. autoapisummary::

   krrood.entity_query_language.predicate.symbolic_function
   krrood.entity_query_language.predicate.update_cache
   krrood.entity_query_language.predicate.get_function_argument_names
   krrood.entity_query_language.predicate.merge_args_and_kwargs


Module Contents
---------------

.. py:function:: symbolic_function(function: typing_extensions.Callable[Ellipsis, krrood.entity_query_language.utils.T]) -> typing_extensions.Callable[Ellipsis, krrood.entity_query_language.symbolic.Variable[krrood.entity_query_language.utils.T]]

   Function decorator that constructs a symbolic expression representing the function call
    when inside a symbolic_rule context.

   When symbolic mode is active, calling the method returns a Call instance which is a SymbolicExpression bound to
   representing the method call that is not evaluated until the evaluate() method is called on the query/rule.

   :param function: The function to decorate.
   :return: The decorated function.


.. py:class:: Symbol

   Base class for things that can be cached in the symbol graph.


.. py:class:: Predicate

   Bases: :py:obj:`Symbol`, :py:obj:`abc.ABC`


   The super predicate class that represents a filtration operation or asserts a relation.


   .. py:attribute:: is_expensive
      :type:  typing_extensions.ClassVar[bool]
      :value: False



.. py:class:: HasType

   Bases: :py:obj:`Predicate`


   Represents a predicate to check if a given variable is an instance of a specified type.

   This class is used to evaluate whether the domain value belongs to a given type by leveraging
   Python's built-in `isinstance` functionality. It provides methods to retrieve the domain and
   range values and perform direct checks.


   .. py:attribute:: variable
      :type:  typing_extensions.Any

      The variable whose type is being checked.



   .. py:attribute:: types_
      :type:  typing_extensions.Type

      The type or tuple of types against which the `variable` is validated.



.. py:class:: HasTypes

   Bases: :py:obj:`HasType`


   Represents a specialized data structure holding multiple types.

   This class is a data container designed to store and manage a tuple of
   types. It inherits from the `HasType` class and extends its functionality
   to handle multiple types efficiently. The primary goal of this class is to
   allow structured representation and access to a collection of type
   information with equality comparison explicitly disabled.


   .. py:attribute:: types_
      :type:  typing_extensions.Tuple[typing_extensions.Type, Ellipsis]

      A tuple containing Type objects that are associated with this instance.



.. py:function:: update_cache(instance: Symbol)

   Updates the cache with the given instance of a symbolic type.

   :param instance: The symbolic instance to be cached.


.. py:function:: get_function_argument_names(function: typing_extensions.Callable) -> typing_extensions.List[str]

   :param function: A function to inspect
   :return: The argument names of the function


.. py:function:: merge_args_and_kwargs(function: typing_extensions.Callable, args, kwargs, ignore_first: bool = False) -> typing_extensions.Dict[str, typing_extensions.Any]

   Merge the arguments and keyword-arguments of a function into a dict of keyword-arguments.

   :param function: The function to get the argument names from
   :param args: The arguments passed to the function
   :param kwargs: The keyword arguments passed to the function
   :param ignore_first: Rather to ignore the first argument or not.
   Use this when `function` contains something like `self`
   :return: The dict of assigned keyword-arguments.


