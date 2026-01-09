krrood.class_diagrams.wrapped_field
===================================

.. py:module:: krrood.class_diagrams.wrapped_field


Exceptions
----------

.. autoapisummary::

   krrood.class_diagrams.wrapped_field.TypeResolutionError


Classes
-------

.. autoapisummary::

   krrood.class_diagrams.wrapped_field.WrappedField


Functions
---------

.. autoapisummary::

   krrood.class_diagrams.wrapped_field.manually_search_for_class_name
   krrood.class_diagrams.wrapped_field.search_class_in_globals
   krrood.class_diagrams.wrapped_field.search_class_in_sys_modules


Module Contents
---------------

.. py:exception:: TypeResolutionError

   Bases: :py:obj:`TypeError`


   Error raised when a type cannot be resolved, even if searched for manually.


   .. py:attribute:: name
      :type:  str


.. py:class:: WrappedField

   A class that wraps a field of dataclass and provides some utility functions.


   .. py:attribute:: clazz
      :type:  krrood.class_diagrams.class_diagram.WrappedClass

      The wrapped class that the field was created from.



   .. py:attribute:: field
      :type:  dataclasses.Field

      The dataclass field object that is wrapped.



   .. py:attribute:: public_name
      :type:  typing_extensions.Optional[str]
      :value: None


      If the field is a relationship managed field, this is public name of the relationship that manages the field.



   .. py:attribute:: property_descriptor
      :type:  typing_extensions.Optional[krrood.ontomatic.property_descriptor.PropertyDescriptor]
      :value: None


      The property descriptor instance that manages the field.



   .. py:attribute:: container_types
      :type:  typing_extensions.ClassVar[typing_extensions.List[typing_extensions.Type]]

      A list of container types that are supported by the parser.



   .. py:property:: name


   .. py:property:: resolved_type

      Resolve the type hint for this field.

      Handles forward references by iteratively building a namespace with
      classes from the class diagram and sys.modules until all references
      are resolved.



   .. py:property:: is_builtin_type
      :type: bool



   .. py:property:: is_container
      :type: bool



   .. py:property:: container_type
      :type: typing_extensions.Optional[typing_extensions.Type]



   .. py:property:: is_collection_of_builtins


   .. py:property:: is_optional


   .. py:property:: contained_type


   .. py:property:: is_type_type
      :type: bool



   .. py:property:: is_enum
      :type: bool



   .. py:property:: is_one_to_one_relationship
      :type: bool



   .. py:property:: is_one_to_many_relationship
      :type: bool



   .. py:property:: is_iterable


   .. py:property:: type_endpoint
      :type: typing_extensions.Type



   .. py:property:: is_role_taker
      :type: bool



.. py:function:: manually_search_for_class_name(target_class_name: str) -> typing_extensions.Type

   Searches for a class with the specified name in the current module's `globals()` dictionary
   and all loaded modules present in `sys.modules`. This function attempts to find and resolve
   the first class that matches the given name. If multiple classes are found with the same
   name, a warning is logged, and the first one is returned. If no matching class is found,
   an exception is raised.

   :param target_class_name: Name of the class to search for.
   :return: The resolved class with the matching name.

   :raises ValueError: Raised when no class with the specified name can be found.


.. py:function:: search_class_in_globals(target_class_name: str) -> typing_extensions.List[typing_extensions.Type]

   Searches for a class with the given name in the current module's globals.

   :param target_class_name: The name of the class to search for.
   :return: The resolved classes with the matching name.


.. py:function:: search_class_in_sys_modules(target_class_name: str) -> typing_extensions.List[typing_extensions.Type]

   Searches for a class with the given name in all loaded modules (via sys.modules).


