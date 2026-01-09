krrood.ormatic.alternative_mappings
===================================

.. py:module:: krrood.ormatic.alternative_mappings


Exceptions
----------

.. autoapisummary::

   krrood.ormatic.alternative_mappings.UncallableFunction


Classes
-------

.. autoapisummary::

   krrood.ormatic.alternative_mappings.FunctionMapping


Functions
---------

.. autoapisummary::

   krrood.ormatic.alternative_mappings.raise_uncallable_function


Module Contents
---------------

.. py:exception:: UncallableFunction

   Bases: :py:obj:`NotImplementedError`


   Exception raised when anonymous functions are reconstructed and then called.


   .. py:attribute:: function_mapping
      :type:  FunctionMapping


.. py:function:: raise_uncallable_function(function_mapping: FunctionMapping)

.. py:class:: FunctionMapping

   Bases: :py:obj:`krrood.ormatic.dao.AlternativeMapping`\ [\ :py:obj:`types.FunctionType`\ ]


   Alternative mapping for functions.


   .. py:attribute:: module_name
      :type:  str

      The module name where the function is defined.



   .. py:attribute:: function_name
      :type:  str

      The name of the function.



   .. py:attribute:: class_name
      :type:  typing_extensions.Optional[str]
      :value: None


      The name of the class if the function is defined by a class.



   .. py:method:: from_domain_object(obj: collections.abc.Callable) -> typing_extensions.Self
      :classmethod:


      Create this from a domain object.
      Do not create any DAOs here but the target DAO of `T`.
      The rest of the `to_dao` algorithm will process the fields of the created instance.

      :param obj: The source object.
      :return: A new instance of this mapping class.



   .. py:method:: to_domain_object() -> krrood.ormatic.dao.T

      Create a domain object from this instance.

      :return: The constructed domain object.



