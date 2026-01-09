krrood.adapters.json_serializer
===============================

.. py:module:: krrood.adapters.json_serializer


Attributes
----------

.. autoapisummary::

   krrood.adapters.json_serializer.list_like_classes
   krrood.adapters.json_serializer.leaf_types
   krrood.adapters.json_serializer.JSON_DICT_TYPE
   krrood.adapters.json_serializer.JSON_RETURN_TYPE
   krrood.adapters.json_serializer.T


Classes
-------

.. autoapisummary::

   krrood.adapters.json_serializer.JSONSerializableTypeRegistry
   krrood.adapters.json_serializer.SubclassJSONSerializer
   krrood.adapters.json_serializer.ExternalClassJSONSerializer
   krrood.adapters.json_serializer.UUIDJSONSerializer
   krrood.adapters.json_serializer.EnumJSONSerializer
   krrood.adapters.json_serializer.ExceptionJSONSerializer


Functions
---------

.. autoapisummary::

   krrood.adapters.json_serializer.from_json
   krrood.adapters.json_serializer.to_json


Module Contents
---------------

.. py:data:: list_like_classes

.. py:data:: leaf_types

.. py:data:: JSON_DICT_TYPE

.. py:data:: JSON_RETURN_TYPE

.. py:class:: JSONSerializableTypeRegistry

   Singleton registry for custom serializers and deserializers.

   Use this registry when you need to add custom JSON serialization/deserialization logic for a type where you cannot
   control its inheritance.


   .. py:method:: get_external_serializer(clazz: typing_extensions.Type) -> typing_extensions.Type[ExternalClassJSONSerializer]

      Get the external serializer for the given class.

      This returns the serializer of the closest superclass if no direct match is found.

      :param clazz: The class to get the serializer for.
      :return: The serializer class.



.. py:class:: SubclassJSONSerializer

   Class for automatic (de)serialization of subclasses using importlib.

   Stores the fully qualified class name in `type` during serialization and
   imports that class during deserialization.


   .. py:method:: to_json() -> typing_extensions.Dict[str, typing_extensions.Any]


   .. py:method:: from_json(data: typing_extensions.Dict[str, typing_extensions.Any], **kwargs) -> typing_extensions.Self
      :classmethod:


      Create the correct instanceof the subclass from a json dict.

      :param data: The json dict
      :param kwargs: Additional keyword arguments to pass to the constructor of the subclass.
      :return: The correct instance of the subclass



.. py:function:: from_json(data: typing_extensions.Dict[str, typing_extensions.Any], **kwargs) -> typing_extensions.Union[SubclassJSONSerializer, typing_extensions.Any]

   Deserialize a JSON dict to an object.

   :param data: The JSON string
   :return: The deserialized object


.. py:function:: to_json(obj: typing_extensions.Union[SubclassJSONSerializer, typing_extensions.Any]) -> JSON_RETURN_TYPE

   Serialize an object to a JSON dict.

   :param obj: The object to convert to json
   :return: The JSON string


.. py:data:: T

.. py:class:: ExternalClassJSONSerializer

   Bases: :py:obj:`krrood.ormatic.dao.HasGeneric`\ [\ :py:obj:`T`\ ], :py:obj:`abc.ABC`


   ABC for all added JSON de/serializers that are outside the control of your classes.

   Create a new subclass of this class pointing to your original class whenever you can't change its inheritance path
   to `SubclassJSONSerializer`.


   .. py:method:: to_json(obj: typing_extensions.Any) -> typing_extensions.Dict[str, typing_extensions.Any]
      :classmethod:


      Convert an object to a JSON serializable dictionary.

      :param obj: The object to convert.
      :return: The JSON serializable dictionary.



   .. py:method:: from_json(data: typing_extensions.Dict[str, typing_extensions.Any], clazz: typing_extensions.Type[T], **kwargs) -> typing_extensions.Any
      :classmethod:


      Create a class instance from a JSON serializable dictionary.

      :param data: The JSON serializable dictionary.
      :param clazz: The class type to instantiate.
      :param kwargs: Additional keyword arguments for instantiation.
      :return: The instantiated class object.



.. py:class:: UUIDJSONSerializer

   Bases: :py:obj:`ExternalClassJSONSerializer`\ [\ :py:obj:`uuid.UUID`\ ]


   ABC for all added JSON de/serializers that are outside the control of your classes.

   Create a new subclass of this class pointing to your original class whenever you can't change its inheritance path
   to `SubclassJSONSerializer`.


   .. py:method:: to_json(obj: uuid.UUID) -> typing_extensions.Dict[str, typing_extensions.Any]
      :classmethod:


      Convert an object to a JSON serializable dictionary.

      :param obj: The object to convert.
      :return: The JSON serializable dictionary.



   .. py:method:: from_json(data: typing_extensions.Dict[str, typing_extensions.Any], clazz: typing_extensions.Type[uuid.UUID], **kwargs) -> uuid.UUID
      :classmethod:


      Create a class instance from a JSON serializable dictionary.

      :param data: The JSON serializable dictionary.
      :param clazz: The class type to instantiate.
      :param kwargs: Additional keyword arguments for instantiation.
      :return: The instantiated class object.



.. py:class:: EnumJSONSerializer

   Bases: :py:obj:`ExternalClassJSONSerializer`\ [\ :py:obj:`enum.Enum`\ ]


   ABC for all added JSON de/serializers that are outside the control of your classes.

   Create a new subclass of this class pointing to your original class whenever you can't change its inheritance path
   to `SubclassJSONSerializer`.


   .. py:method:: to_json(obj: enum.Enum) -> typing_extensions.Dict[str, typing_extensions.Any]
      :classmethod:


      Convert an object to a JSON serializable dictionary.

      :param obj: The object to convert.
      :return: The JSON serializable dictionary.



   .. py:method:: from_json(data: typing_extensions.Dict[str, typing_extensions.Any], clazz: typing_extensions.Type[enum.Enum], **kwargs) -> enum.Enum
      :classmethod:


      Create a class instance from a JSON serializable dictionary.

      :param data: The JSON serializable dictionary.
      :param clazz: The class type to instantiate.
      :param kwargs: Additional keyword arguments for instantiation.
      :return: The instantiated class object.



.. py:class:: ExceptionJSONSerializer

   Bases: :py:obj:`ExternalClassJSONSerializer`\ [\ :py:obj:`Exception`\ ]


   ABC for all added JSON de/serializers that are outside the control of your classes.

   Create a new subclass of this class pointing to your original class whenever you can't change its inheritance path
   to `SubclassJSONSerializer`.


   .. py:method:: to_json(obj: Exception) -> typing_extensions.Dict[str, typing_extensions.Any]
      :classmethod:


      Convert an object to a JSON serializable dictionary.

      :param obj: The object to convert.
      :return: The JSON serializable dictionary.



   .. py:method:: from_json(data: typing_extensions.Dict[str, typing_extensions.Any], clazz: typing_extensions.Type[Exception], **kwargs) -> Exception
      :classmethod:


      Create a class instance from a JSON serializable dictionary.

      :param data: The JSON serializable dictionary.
      :param clazz: The class type to instantiate.
      :param kwargs: Additional keyword arguments for instantiation.
      :return: The instantiated class object.



