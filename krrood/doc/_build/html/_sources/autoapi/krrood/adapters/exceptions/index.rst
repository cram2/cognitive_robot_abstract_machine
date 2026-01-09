krrood.adapters.exceptions
==========================

.. py:module:: krrood.adapters.exceptions


Attributes
----------

.. autoapisummary::

   krrood.adapters.exceptions.JSON_TYPE_NAME


Exceptions
----------

.. autoapisummary::

   krrood.adapters.exceptions.JSONSerializationError
   krrood.adapters.exceptions.MissingTypeError
   krrood.adapters.exceptions.InvalidTypeFormatError
   krrood.adapters.exceptions.UnknownModuleError
   krrood.adapters.exceptions.ClassNotFoundError
   krrood.adapters.exceptions.ClassNotSerializableError
   krrood.adapters.exceptions.ClassNotDeserializableError


Module Contents
---------------

.. py:data:: JSON_TYPE_NAME
   :value: '__json_type__'


.. py:exception:: JSONSerializationError

   Bases: :py:obj:`krrood.utils.DataclassException`


   Base exception for JSON (de)serialization errors.


.. py:exception:: MissingTypeError

   Bases: :py:obj:`JSONSerializationError`


   Raised when the 'type' field is missing in the JSON data.


.. py:exception:: InvalidTypeFormatError

   Bases: :py:obj:`JSONSerializationError`


   Raised when the 'type' field value is not a fully qualified class name.


   .. py:attribute:: invalid_type_value
      :type:  str


.. py:exception:: UnknownModuleError

   Bases: :py:obj:`JSONSerializationError`


   Raised when the module specified in the 'type' field cannot be imported.


   .. py:attribute:: module_name
      :type:  str


.. py:exception:: ClassNotFoundError

   Bases: :py:obj:`JSONSerializationError`


   Raised when the class specified in the 'type' field cannot be found in the module.


   .. py:attribute:: class_name
      :type:  str


   .. py:attribute:: module_name
      :type:  str


.. py:exception:: ClassNotSerializableError

   Bases: :py:obj:`JSONSerializationError`


   Raised when the class specified cannot be JSON-serialized.


   .. py:attribute:: clazz
      :type:  typing_extensions.Type


.. py:exception:: ClassNotDeserializableError

   Bases: :py:obj:`JSONSerializationError`


   Raised when the class specified cannot be JSON-deserialized.


   .. py:attribute:: clazz
      :type:  typing_extensions.Type


