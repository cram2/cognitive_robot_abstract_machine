krrood.entity_query_language.entity_result_processors
=====================================================

.. py:module:: krrood.entity_query_language.entity_result_processors


Attributes
----------

.. autoapisummary::

   krrood.entity_query_language.entity_result_processors.a


Functions
---------

.. autoapisummary::

   krrood.entity_query_language.entity_result_processors.an
   krrood.entity_query_language.entity_result_processors.the
   krrood.entity_query_language.entity_result_processors.max
   krrood.entity_query_language.entity_result_processors.min
   krrood.entity_query_language.entity_result_processors.sum
   krrood.entity_query_language.entity_result_processors.average
   krrood.entity_query_language.entity_result_processors.count


Module Contents
---------------

.. py:function:: an(entity_: typing_extensions.Union[krrood.entity_query_language.symbolic.SetOf[krrood.entity_query_language.utils.T], krrood.entity_query_language.symbolic.Entity[krrood.entity_query_language.utils.T], krrood.entity_query_language.utils.T, typing_extensions.Iterable[krrood.entity_query_language.utils.T], typing_extensions.Type[krrood.entity_query_language.utils.T]], quantification: typing_extensions.Optional[krrood.entity_query_language.result_quantification_constraint.ResultQuantificationConstraint] = None) -> typing_extensions.Union[krrood.entity_query_language.symbolic.An[krrood.entity_query_language.utils.T], krrood.entity_query_language.utils.T]

   Select all values satisfying the given entity description.

   :param entity_: An entity or a set expression to quantify over.
   :param quantification: Optional quantification constraint.
   :return: A quantifier representing "an" element.
   :rtype: An[T]


.. py:data:: a

   This is an alias to accommodate for words not starting with vowels.


.. py:function:: the(entity_: typing_extensions.Union[krrood.entity_query_language.symbolic.SetOf[krrood.entity_query_language.utils.T], krrood.entity_query_language.symbolic.Entity[krrood.entity_query_language.utils.T], krrood.entity_query_language.utils.T, typing_extensions.Iterable[krrood.entity_query_language.utils.T], typing_extensions.Type[krrood.entity_query_language.utils.T]]) -> typing_extensions.Union[krrood.entity_query_language.symbolic.The[krrood.entity_query_language.utils.T], krrood.entity_query_language.utils.T]

   Select the unique value satisfying the given entity description.

   :param entity_: An entity or a set expression to quantify over.
   :return: A quantifier representing "an" element.
   :rtype: The[T]


.. py:function:: max(variable: krrood.entity_query_language.symbolic.Selectable[krrood.entity_query_language.utils.T], key: typing_extensions.Optional[typing_extensions.Callable] = None, default: typing_extensions.Optional[krrood.entity_query_language.utils.T] = None) -> typing_extensions.Union[krrood.entity_query_language.utils.T, krrood.entity_query_language.symbolic.Max[krrood.entity_query_language.utils.T]]

   Maps the variable values to their maximum value.

   :param variable: The variable for which the maximum value is to be found.
   :param key: A function that extracts a comparison key from each variable value.
   :param default: The value returned when the iterable is empty.
   :return: A Max object that can be evaluated to find the maximum value.


.. py:function:: min(variable: krrood.entity_query_language.symbolic.Selectable[krrood.entity_query_language.utils.T], key: typing_extensions.Optional[typing_extensions.Callable] = None, default: typing_extensions.Optional[krrood.entity_query_language.utils.T] = None) -> typing_extensions.Union[krrood.entity_query_language.utils.T, krrood.entity_query_language.symbolic.Min[krrood.entity_query_language.utils.T]]

   Maps the variable values to their minimum value.

   :param variable: The variable for which the minimum value is to be found.
   :param key: A function that extracts a comparison key from each variable value.
   :param default: The value returned when the iterable is empty.
   :return: A Min object that can be evaluated to find the minimum value.


.. py:function:: sum(variable: krrood.entity_query_language.symbolic.Selectable[krrood.entity_query_language.utils.T], key: typing_extensions.Optional[typing_extensions.Callable] = None, default: typing_extensions.Optional[krrood.entity_query_language.utils.T] = None) -> typing_extensions.Union[krrood.entity_query_language.utils.T, krrood.entity_query_language.symbolic.Sum[krrood.entity_query_language.utils.T]]

   Computes the sum of values produced by the given variable.

   :param variable: The variable for which the sum is calculated.
   :param key: A function that extracts a comparison key from each variable value.
   :param default: The value returned when the iterable is empty.
   :return: A Sum object that can be evaluated to find the sum of values.


.. py:function:: average(variable: krrood.entity_query_language.symbolic.Selectable[krrood.entity_query_language.utils.T], key: typing_extensions.Optional[typing_extensions.Callable] = None, default: typing_extensions.Optional[krrood.entity_query_language.utils.T] = None) -> typing_extensions.Union[krrood.entity_query_language.utils.T, krrood.entity_query_language.symbolic.Average[krrood.entity_query_language.utils.T]]

   Computes the sum of values produced by the given variable.

   :param variable: The variable for which the sum is calculated.
   :param key: A function that extracts a comparison key from each variable value.
   :param default: The value returned when the iterable is empty.
   :return: A Sum object that can be evaluated to find the sum of values.


.. py:function:: count(variable: krrood.entity_query_language.symbolic.Selectable[krrood.entity_query_language.utils.T]) -> typing_extensions.Union[krrood.entity_query_language.utils.T, krrood.entity_query_language.symbolic.Count[krrood.entity_query_language.utils.T]]

   Count the number of values produced by the given variable.

   :param variable: The variable for which the count is calculated.
   :return: A Count object that can be evaluated to count the number of values.


