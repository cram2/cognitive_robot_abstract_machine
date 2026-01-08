from __future__ import annotations

import enum
import importlib
import uuid
from abc import ABC
from dataclasses import dataclass
from types import NoneType

from typing_extensions import Dict, Any, Self, Union, Type, TypeVar

from .exceptions import (
    MissingTypeError,
    InvalidTypeFormatError,
    UnknownModuleError,
    ClassNotFoundError,
    ClassNotSerializableError,
    JSON_TYPE_NAME,
)
from ..ormatic.dao import HasGeneric
from ..singleton import SingletonMeta
from ..utils import get_full_class_name, recursive_subclasses, inheritance_path_length

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
)  # containers that can be serialized by the built-in JSON module

JSON_DICT_TYPE = Dict[str, Any]  # Commonly referred JSON dict
JSON_RETURN_TYPE = Union[
    JSON_DICT_TYPE, list[JSON_DICT_TYPE], *leaf_types
]  # Commonly referred JSON types


@dataclass
class JSONSerializableTypeRegistry(metaclass=SingletonMeta):
    """
    Singleton registry for custom serializers and deserializers.

    Use this registry when you need to add custom JSON serialization/deserialization logic for a type where you cannot
    control its inheritance.
    """

    def get_external_serializer(self, clazz: Type) -> Type[ExternalClassJSONSerializer]:
        """
        Get the external serializer for the given class.

        This returns the serializer of the closest superclass if no direct match is found.

        :param clazz: The class to get the serializer for.
        :return: The serializer class.
        """
        if issubclass(clazz, enum.Enum):
            return EnumJSONSerializer

        distances = {}  # mapping of subclasses to the distance to the clazz

        for subclass in recursive_subclasses(ExternalClassJSONSerializer):
            if subclass.original_class() == clazz:
                return subclass
            else:
                distance = inheritance_path_length(clazz, subclass.original_class())
                if distance is not None:
                    distances[subclass] = distance

        if not distances:
            raise ClassNotSerializableError(clazz)
        else:
            return min(distances, key=distances.get)


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

        if issubclass(target_cls, SubclassJSONSerializer):
            return target_cls._from_json(data, **kwargs)

        external_json_deserializer = (
            JSONSerializableTypeRegistry().get_external_serializer(target_cls)
        )

        return external_json_deserializer.from_json(data, clazz=target_cls, **kwargs)


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

    :param obj: The object to convert to json
    :return: The JSON string
    """

    if isinstance(obj, leaf_types):
        return obj

    if isinstance(obj, list_like_classes):
        return [to_json(item) for item in obj]

    if isinstance(obj, SubclassJSONSerializer):
        return obj.to_json()

    registered_json_serializer = JSONSerializableTypeRegistry().get_external_serializer(
        type(obj)
    )

    return registered_json_serializer.to_json(obj)


T = TypeVar("T")


@dataclass
class ExternalClassJSONSerializer(HasGeneric[T], ABC):
    """
    ABC for all added JSON de/serializers that are outside the control of your classes.

    Create a new subclass of this class pointing to your original class whenever you can't change its inheritance path
    to `SubclassJSONSerializer`.
    """

    @classmethod
    def to_json(cls, obj: Any) -> Dict[str, Any]:
        """
        Convert an object to a JSON serializable dictionary.

        :param obj: The object to convert.
        :return: The JSON serializable dictionary.
        """

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type[T], **kwargs) -> Any:
        """
        Create a class instance from a JSON serializable dictionary.

        :param data: The JSON serializable dictionary.
        :param clazz: The class type to instantiate.
        :param kwargs: Additional keyword arguments for instantiation.
        :return: The instantiated class object.
        """


@dataclass
class UUIDJSONSerializer(ExternalClassJSONSerializer[uuid.UUID]):

    @classmethod
    def to_json(cls, obj: uuid.UUID) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(type(obj)),
            "value": str(obj),
        }

    @classmethod
    def from_json(
        cls, data: Dict[str, Any], clazz: Type[uuid.UUID], **kwargs
    ) -> uuid.UUID:
        return clazz(data["value"])


@dataclass
class EnumJSONSerializer(ExternalClassJSONSerializer[enum.Enum]):

    @classmethod
    def to_json(cls, obj: enum.Enum) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(type(obj)),
            "name": obj.name,
        }

    @classmethod
    def from_json(
        cls, data: Dict[str, Any], clazz: Type[enum.Enum], **kwargs
    ) -> enum.Enum:
        return clazz[data["name"]]


@dataclass
class ExceptionJSONSerializer(ExternalClassJSONSerializer[Exception]):
    @classmethod
    def to_json(cls, obj: Exception) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(type(obj)),
            "value": str(obj),
        }

    @classmethod
    def from_json(
        cls, data: Dict[str, Any], clazz: Type[Exception], **kwargs
    ) -> Exception:
        return clazz(data["value"])
