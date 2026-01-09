krrood.class_diagrams.utils
===========================

.. py:module:: krrood.class_diagrams.utils


Attributes
----------

.. autoapisummary::

   krrood.class_diagrams.utils.T


Classes
-------

.. autoapisummary::

   krrood.class_diagrams.utils.Role


Functions
---------

.. autoapisummary::

   krrood.class_diagrams.utils.classes_of_module
   krrood.class_diagrams.utils.behaves_like_a_built_in_class
   krrood.class_diagrams.utils.is_builtin_class
   krrood.class_diagrams.utils.get_generic_type_param


Module Contents
---------------

.. py:function:: classes_of_module(module) -> typing_extensions.List[typing_extensions.Type]

   Get all classes of a given module.

   :param module: The module to inspect.
   :return: All classes of the given module.


.. py:function:: behaves_like_a_built_in_class(clazz: typing_extensions.Type) -> bool

.. py:function:: is_builtin_class(clazz: typing_extensions.Type) -> bool

.. py:data:: T

.. py:class:: Role

   Bases: :py:obj:`typing_extensions.Generic`\ [\ :py:obj:`T`\ ]


   Represents a role with generic typing. This is used in Role Design Pattern in OOP.

   This class serves as a container for defining roles with associated generic
   types, enabling flexibility and type safety when modeling role-specific
   behavior and data.


.. py:function:: get_generic_type_param(cls, generic_base)

   Given a subclass and its generic base, return the concrete type parameter(s).

   Example:
       get_generic_type_param(Employee, Role) -> (<class '__main__.Person'>,)


