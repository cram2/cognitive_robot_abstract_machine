from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from json import JSONDecodeError

from typing_extensions import Dict, Any, Self, Union, Callable, Type
import json
import uuid

from ..utils import get_full_class_name
from ..singleton import SingletonMeta

JSON_TYPE_NAME = "__json_type__"  # the key used in JSON dicts to identify the class


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
class TypeRegistry(metaclass=SingletonMeta):
    """Singleton registry for custom serializers and deserializers."""

    _serializers: Dict[Type, Callable[[Any], Dict[str, Any]]] = field(
        default_factory=dict
    )
    _deserializers: Dict[str, Callable[[Dict[str, Any]], Any]] = field(
        default_factory=dict
    )

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
        type_name = f"{type_class.__module__}.{type_class.__name__}"
        self._serializers[type_class] = serializer
        self._deserializers[type_name] = deserializer

    def get_serializer(self, obj: Any) -> Callable[[Any], Dict[str, Any]] | None:
        """
        Get the serializer for an object's type.

        :param obj: The object to get the serializer for
        :return: The serializer function or None if not registered
        """
        return self._serializers.get(type(obj))

    def get_deserializer(
        self, type_name: str
    ) -> Callable[[Dict[str, Any]], Any] | None:
        """
        Get the deserializer for a type name.

        :param type_name: The fully qualified type name
        :return: The deserializer function or None if not registered
        """
        return self._deserializers.get(type_name)


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

        return target_cls._from_json(data, **kwargs)


class SubclassJSONEncoder(json.JSONEncoder):
    """
    Custom encoder to handle classes that inherit from SubClassJSONEncoder and UUIDs.
    """

    def default(self, obj):
        # Check registry first
        serializer = TypeRegistry().get_serializer(obj)
        if serializer:
            return serializer(obj)

        # handle objects that are duck-typed like SubclassJSONSerializer
        if hasattr(obj, "to_json"):
            return obj.to_json()

        # Let the base class handle other objects
        return json.JSONEncoder.default(self, obj)


class SubclassJSONDecoder(json.JSONDecoder):
    """
    Custom decoder to handle classes that inherit from SubClassJSONSerializer and UUIDs.
    """

    def decode(self, s, _w=json.decoder.WHITESPACE.match):
        if not isinstance(s, dict):
            obj = super().decode(s, _w)
        else:
            obj = s
        return self._deserialize_nested(obj)

    def _deserialize_nested(self, obj, **kwargs):
        """
        Recursively deserialize nested objects.
        """
        if isinstance(obj, dict):
            if JSON_TYPE_NAME in obj:
                type_name = obj[JSON_TYPE_NAME]

                # Check registry first
                deserializer = TypeRegistry().get_deserializer(type_name)
                if deserializer:
                    return deserializer(obj)

                # Fall back to SubclassJSONSerializer.from_json
                return SubclassJSONSerializer.from_json(obj, **kwargs)
            else:
                return {
                    k: self._deserialize_nested(v, **kwargs) for k, v in obj.items()
                }
        elif isinstance(obj, list):
            return [self._deserialize_nested(item, **kwargs) for item in obj]
        else:
            return obj


def to_json(obj: Union[SubclassJSONSerializer, Any]) -> str:
    """
    Serialize an object to a JSON string.
    This is a drop-in replacement for json.dumps which handles SubclassJSONSerializer-like objects.

    :param obj: The object to serialize
    :return: The JSON string
    """
    return json.dumps(obj, cls=SubclassJSONEncoder)


def from_json(data: str, **kwargs) -> Union[SubclassJSONSerializer, Any]:
    """
    Deserialize a JSON string to an object.
    This is a drop-in replacement for json.loads which handles SubclassJSONSerializer-like objects.

    :param data: The JSON string
    :return: The deserialized object
    """

    # If we already have a Python container, recursively deserialize nested subclass payloads
    if isinstance(data, (dict, list)):
        decoder = SubclassJSONDecoder()
        return decoder._deserialize_nested(data, **kwargs)

    # If it is not a string (e.g., int, float, bool, None), return as-is
    if not isinstance(data, str):
        return data

    # It is a string: try to parse as JSON; if that fails, treat it as a raw string
    try:
        return json.loads(data, cls=SubclassJSONDecoder)
    except JSONDecodeError:
        return data


# %% UUID serialization functions
def serialize_uuid(obj: uuid.UUID) -> Dict[str, Any]:
    """
    Serialize a UUID to a JSON-compatible dictionary.

    :param obj: The UUID to serialize
    :return: Dictionary with type information and UUID value
    """
    return {
        JSON_TYPE_NAME: f"{uuid.UUID.__module__}.{uuid.UUID.__name__}",
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
TypeRegistry().register(uuid.UUID, serialize_uuid, deserialize_uuid)
