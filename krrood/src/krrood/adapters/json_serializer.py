from __future__ import annotations

import enum
import importlib
import uuid
from dataclasses import dataclass, field
from types import NoneType

from typing_extensions import Dict, Any, Self, Union, Callable, Type

from ..singleton import SingletonMeta
from ..utils import get_full_class_name

list_like_classes = (
    list,
    tuple,
    set,
)  # classes that can be serialized by the built-in JSON module
leaf_types = (
    int,
    float,
    str,
    bool,
    NoneType,
)  # types that can be serialized by the built-in JSON module


JSON_TYPE_NAME = "__json_type__"  # the key used in JSON dicts to identify the class

JSON_DICT_TYPE = Dict[str, Any]  # Commonly referred JSON dict
JSON_RETURN_TYPE = Union[
    JSON_DICT_TYPE, list[JSON_DICT_TYPE], *leaf_types
]  # Commonly referred JSON types


class JSONSerializationError(Exception):
    """Base exception for JSON (de)serialization errors."""


class MissingTypeError(JSONSerializationError):
    """Raised when the 'type' field is missing in the JSON data."""

    def __init__(self):
        super().__init__("Missing 'type' field in JSON data")


@dataclass
class InvalidTypeFormatError(JSONSerializationError):
    """Raised when the 'type' field value is not a fully qualified class name."""

    invalid_type_value: str

    def __post_init__(self):
        super().__init__(f"Invalid type format: {self.invalid_type_value}")


@dataclass
class UnknownModuleError(JSONSerializationError):
    """Raised when the module specified in the 'type' field cannot be imported."""

    module_name: str

    def __post_init__(self):
        super().__init__(f"Unknown module in type: {self.module_name}")


@dataclass
class ClassNotFoundError(JSONSerializationError):
    """Raised when the class specified in the 'type' field cannot be found in the module."""

    class_name: str
    module_name: str

    def __post_init__(self):
        super().__init__(
            f"Class '{self.class_name}' not found in module '{self.module_name}'"
        )


@dataclass
class ClassNotSerializableError(JSONSerializationError):
    """Raised when the class specified cannot be JSON-serialized."""

    clazz: Type

    def __post_init__(self):
        super().__init__(f"Class '{self.clazz.__name__}' cannot be serialized")


@dataclass
class ClassNotDeserializableError(JSONSerializationError):
    """Raised when the class specified cannot be JSON-deserialized."""

    clazz: Type

    def __post_init__(self):
        super().__init__(f"Class '{self.clazz.__name__}' cannot be deserialized")


@dataclass
class JSONSerializableTypeRegistry(metaclass=SingletonMeta):
    """
    Singleton registry for custom serializers and deserializers.

    Use this registry when you need to add custom JSON serialization/deserialization logic for a type where you cannot
    control its inheritance.
    """

    _serializers: Dict[Type, Callable[[Any], Dict[str, Any]]] = field(
        default_factory=dict
    )
    """
    Dictionary mapping types to their respective serializer functions.
    Signature of functions must be like `SubclassJSONSerializer.to_json`
    """

    _deserializers: Dict[Type, Callable[[Dict[str, Any]], Any]] = field(
        default_factory=dict
    )
    """
    Dictionary mapping types to their respective deserializer functions.
    Signature of functions must be like `SubclassJSONSerializer._from_json`
    """

    def register(
        self,
        type_class: Type,
        serializer: Callable[[Any], Dict[str, Any]],
        deserializer: Callable[[Dict[str, Any]], Any],
    ):
        """
        Register a custom serializer and deserializer for a type.

        :param type_class: The type to register
        :param serializer: Function to serialize instances of the type
        :param deserializer: Function to deserialize instances of the type
        """
        self._serializers[type_class] = serializer
        self._deserializers[type_class] = deserializer

    def get_serializer(
        self, type_class: Type
    ) -> Callable[[Any], Dict[str, Any]] | None:
        """
        Get the serializer for an object's type.

        :param type_class: The object to get the serializer for
        :return: The serializer function or None if not registered
        """
        return self._serializers.get(type_class)

    def get_deserializer(
        self, type_class: Type
    ) -> Callable[[Dict[str, Any]], Any] | None:
        """
        Get the deserializer for a type name.

        :param type_class: The class to get the deserializer for
        :return: The deserializer function or None if not registered
        """
        return self._deserializers.get(type_class)


class SubclassJSONSerializer:
    """
    Class for automatic (de)serialization of subclasses using importlib.

    Stores the fully qualified class name in `type` during serialization and
    imports that class during deserialization.
    """

    def to_json(self) -> Dict[str, Any]:
        return {JSON_TYPE_NAME: get_full_class_name(self.__class__)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Create an instance from a json dict.
        This method is called from the from_json method after the correct subclass is determined and should be
        overwritten by the subclass.

        :param data: The JSON dict
        :param kwargs: Additional keyword arguments to pass to the constructor of the subclass.
        :return: The deserialized object
        """
        raise NotImplementedError()

    @classmethod
    def from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Create the correct instanceof the subclass from a json dict.

        :param data: The json dict
        :param kwargs: Additional keyword arguments to pass to the constructor of the subclass.
        :return: The correct instance of the subclass
        """

        if isinstance(data, leaf_types):
            return data

        if isinstance(data, list_like_classes):
            return [from_json(d) for d in data]

        fully_qualified_class_name = data.get(JSON_TYPE_NAME)
        if not fully_qualified_class_name:
            raise MissingTypeError()

        try:
            module_name, class_name = fully_qualified_class_name.rsplit(".", 1)
        except ValueError as exc:
            raise InvalidTypeFormatError(fully_qualified_class_name) from exc

        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise UnknownModuleError(module_name) from exc

        try:
            target_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ClassNotFoundError(class_name, module_name) from exc

        if issubclass(target_cls, enum.Enum):
            return target_cls._from_json(data, **kwargs)

        if issubclass(target_cls, SubclassJSONSerializer):
            return target_cls._from_json(data, **kwargs)

        registered_json_deserializer = JSONSerializableTypeRegistry().get_deserializer(
            target_cls
        )
        if not registered_json_deserializer:
            raise ClassNotDeserializableError(target_cls)

        return registered_json_deserializer(data, **kwargs)


def from_json(data: Dict[str, Any], **kwargs) -> Union[SubclassJSONSerializer, Any]:
    """
    Deserialize a JSON dict to an object.

    :param data: The JSON string
    :return: The deserialized object
    """
    return SubclassJSONSerializer.from_json(data, **kwargs)


def to_json(obj: Union[SubclassJSONSerializer, Any]) -> JSON_RETURN_TYPE:
    """
    Serialize an object to a JSON dict.

    :param obj: The object to serialize
    :return: The JSON string
    """

    if isinstance(obj, leaf_types):
        return obj

    if isinstance(obj, list_like_classes):
        return [to_json(item) for item in obj]

    if isinstance(obj, dict):
        return {to_json(key): to_json(value) for key, value in obj.items()}

    if isinstance(obj, SubclassJSONSerializer):
        return obj.to_json()

    registered_json_serializer = JSONSerializableTypeRegistry().get_serializer(
        type(obj)
    )
    if not registered_json_serializer:
        raise ClassNotSerializableError(type(obj))

    return registered_json_serializer(obj)


# %% UUID serialization functions
def serialize_uuid(obj: uuid.UUID) -> Dict[str, Any]:
    """
    Serialize a UUID to a JSON-compatible dictionary.

    :param obj: The UUID to serialize
    :return: Dictionary with type information and UUID value
    """
    return {
        JSON_TYPE_NAME: get_full_class_name(type(obj)),
        "value": str(obj),
    }


def deserialize_uuid(data: Dict[str, Any]) -> uuid.UUID:
    """
    Deserialize a UUID from a JSON dictionary.

    :param data: Dictionary containing the UUID value
    :return: The deserialized UUID
    """
    return uuid.UUID(data["value"])


# Register UUID with the type registry
JSONSerializableTypeRegistry().register(uuid.UUID, serialize_uuid, deserialize_uuid)
