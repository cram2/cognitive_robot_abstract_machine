krrood.utils
============

.. py:module:: krrood.utils


Attributes
----------

.. autoapisummary::

   krrood.utils.T


Exceptions
----------

.. autoapisummary::

   krrood.utils.DataclassException


Functions
---------

.. autoapisummary::

   krrood.utils.recursive_subclasses
   krrood.utils.get_full_class_name
   krrood.utils.inheritance_path_length


Module Contents
---------------

.. py:data:: T

.. py:function:: recursive_subclasses(cls: typing_extensions.Type[T]) -> typing_extensions.List[typing_extensions.Type[T]]

   :param cls: The class.
   :return: A list of the classes subclasses without the class itself.


.. py:exception:: DataclassException

   Bases: :py:obj:`Exception`


   A base exception class for dataclass-based exceptions.
   The way this is used is by inheriting from it and setting the `message` field in the __post_init__ method,
   then calling the super().__post_init__() method.


   .. py:attribute:: message
      :type:  str
      :value: None



.. py:function:: get_full_class_name(cls)

   Returns the full name of a class, including the module name.

   :param cls: The class.
   :return: The full name of the class


.. py:function:: inheritance_path_length(child_class: typing_extensions.Type, parent_class: typing_extensions.Type) -> typing_extensions.Optional[int]

   Calculate the inheritance path length between two classes.
   Every inheritance level that lies between `child_class` and `parent_class` increases the length by one.
   In case of multiple inheritance, the path length is calculated for each branch and the minimum is returned.

   :param child_class: The child class.
   :param parent_class: The parent class.
   :return: The minimum path length between `child_class` and `parent_class` or None if no path exists.


