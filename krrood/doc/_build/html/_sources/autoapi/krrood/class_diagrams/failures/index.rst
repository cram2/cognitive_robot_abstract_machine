krrood.class_diagrams.failures
==============================

.. py:module:: krrood.class_diagrams.failures


Exceptions
----------

.. autoapisummary::

   krrood.class_diagrams.failures.ClassIsUnMappedInClassDiagram
   krrood.class_diagrams.failures.MissingContainedTypeOfContainer


Module Contents
---------------

.. py:exception:: ClassIsUnMappedInClassDiagram(class_: typing_extensions.Type)

   Bases: :py:obj:`Exception`


   Raised when a class is not mapped in the class diagram.


.. py:exception:: MissingContainedTypeOfContainer

   Bases: :py:obj:`Exception`


   Raised when a container type is missing its contained type.
   For example, List without a specified type.


   .. py:attribute:: class_
      :type:  typing_extensions.Type


   .. py:attribute:: field_name
      :type:  str


   .. py:attribute:: container_type
      :type:  typing_extensions.Type


