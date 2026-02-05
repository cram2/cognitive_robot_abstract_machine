from __future__ import annotations

import enum
import importlib
import inspect
import uuid
from abc import ABC
from dataclasses import dataclass, fields, is_dataclass
from dataclasses import field
from types import NoneType
from typing import List, Optional

import numpy as np
from typing_extensions import Dict, Any, Self, Union, Type, TypeVar

from .exceptions import (
    MissingTypeError,
    InvalidTypeFormatError,
    UnknownModuleError,
    ClassNotFoundError,
    ClassNotSerializableError,
    JSON_TYPE_NAME,
)
from ..class_diagrams.attribute_introspector import DataclassOnlyIntrospector
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
JSON_IS_CLASS = "__is_class__"
"""
We need to remember if something is a class, because the type of a class is often just type.
"""


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
            if subclass.matches_generic_type(clazz):
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

        if data.get(JSON_IS_CLASS, False):
            return ClassJSONSerializer.from_json(data, clazz=target_cls, **kwargs)

        if issubclass(target_cls, SubclassJSONSerializer):
            return target_cls._from_json(data, **kwargs)

        external_json_deserializer = (
            JSONSerializableTypeRegistry().get_external_serializer(target_cls)
        )

        return external_json_deserializer.from_json(data, clazz=target_cls, **kwargs)

    def update_from_json_diff(self, diffs: List[JSONAttributeDiff], **kwargs) -> None:
        """
        Update the current object from a list of shallow diffs.

        :param diffs: The shallow diffs to apply.
        :param kwargs: Additional keyword arguments to pass to the constructor of the subclass.
        """
        for diff in diffs:
            self._apply_diff(diff, **kwargs)

    def _apply_diff(self, diff: JSONAttributeDiff, **kwargs) -> None:
        """
        Apply a single diff to the current object.
        :param diff: The diff to apply.
        """
        current_value = getattr(self, diff.attribute_name)
        if isinstance(current_value, list):
            for item in diff.removed_values:
                current_value.remove(from_json(item, **kwargs))
            for item in diff.added_values:
                current_value.append(from_json(item, **kwargs))
        else:
            setattr(
                self,
                diff.attribute_name,
                from_json(diff.added_values[0], **kwargs),
            )


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

    if inspect.isclass(obj):
        return ClassJSONSerializer.to_json(obj)

    registered_json_serializer = JSONSerializableTypeRegistry().get_external_serializer(
        type(obj)
    )

    return registered_json_serializer.to_json(obj)


@dataclass
class JSONAttributeDiff(SubclassJSONSerializer):
    """
    A class representing a shallow diff for JSON-serializable keyword arguments.
    """

    attribute_name: str = field(kw_only=True)
    """
    The name of the attribute that has changed.
    """

    added_values: List[Any] = field(default_factory=list)
    """
    The items that have been added to the attribute.
    """

    removed_values: List[Any] = field(default_factory=list)
    """
    The items that have been removed from the attribute.
    """

    def to_json(self) -> Dict[str, Any]:
        super().to_json()
        return {
            JSON_TYPE_NAME: get_full_class_name(self.__class__),
            "attribute_name": self.attribute_name,
            "removed_values": to_json(self.removed_values),
            "added_values": to_json(self.added_values),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            attribute_name=data["attribute_name"],
            removed_values=from_json(data["removed_values"]),
            added_values=from_json(data["added_values"]),
        )


def shallow_diff_json(
    original_json: Dict[str, Any], new_json: Dict[str, Any]
) -> List[JSONAttributeDiff]:
    """
    Create a shallow diff between two JSON dicts. Result describes the changes that need to be applied to first json to get second json.

    :param original_json: The original JSON dict.
    :param new_json: The new JSON dict.

    :return: List of JSONAttributeDiff describing the changes that need to be applied to first json to get second json.
    """
    all_keys = original_json.keys() | new_json.keys()
    diffs: List[JSONAttributeDiff] = [
        diff
        for key in all_keys
        if (diff := _compute_attribute_diff(original_json, new_json, key)) is not None
    ]
    return diffs


def _compute_attribute_diff(
    original_json: Any, new_json: Any, key: str
) -> Optional[JSONAttributeDiff]:
    """
    Compute the attribute diff for a single key between two JSON dicts.

    :param original_json: The original JSON dict.
    :param new_json: The new JSON dict.
    :param key: The key to compute the diff for.

    :return JSONAttributeDiff describing the changes that need to be applied to first json to get second json for a specific key.
    """
    original_value = original_json.get(key)
    new_value = new_json.get(key)

    if not isinstance(original_value, list_like_classes):
        if original_value == new_value:
            return None
        return JSONAttributeDiff(
            attribute_name=key, added_values=[from_json(new_value)]
        )

    add = [from_json(x) for x in new_value if x not in original_value]
    remove = [from_json(x) for x in original_value if x not in new_value]
    if not (add or remove):
        return None
    return JSONAttributeDiff(
        attribute_name=key, added_values=add, removed_values=remove
    )


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

    @classmethod
    def matches_generic_type(cls, clazz: Type) -> bool:
        """
        Determines if the provided class type matches the original class type.

        :param clazz: The class type to compare against the original class type.
        :return: A boolean value indicating whether the provided class type matches
                 the original class type.
        """
        return cls.original_class() == clazz


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
class ClassJSONSerializer(ExternalClassJSONSerializer[None]):
    """
    A class that provides mechanisms for serializing and deserializing Python classes
    to and from JSON representations.
    """

    @classmethod
    def to_json(cls, obj: Type) -> Dict[str, Any]:
        """
        This is a special case because we need to remember that the type of the class is a class, not a type.
        .. note:: We can't do type(obj) because that often returns just `type`.
        """
        return {
            JSON_TYPE_NAME: get_full_class_name(obj),
            JSON_IS_CLASS: inspect.isclass(obj),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type, **kwargs) -> Type:
        return clazz


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


@dataclass
class NumpyNDarrayJSONSerializer(ExternalClassJSONSerializer[np.ndarray]):
    """
    External JSON serializer for numpy ndarrays.
    """

    @classmethod
    def to_json(cls, obj: np.ndarray) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(type(obj)),
            "type": str(obj.dtype),
            "data": obj.tolist(),
        }

    @classmethod
    def from_json(
        cls, data: Dict[str, Any], clazz: Type[np.ndarray], **kwargs
    ) -> np.ndarray:
        return np.array(data["data"], dtype=data["type"])


@dataclass
class DataclassJSONSerializer(ExternalClassJSONSerializer[None]):
    """
    Generic JSON serializer for dataclasses.
    It creates a dict where all fields are serialized using the to_json function.
    If this is not enough, you still need to implement a custom serializer.
    """

    @classmethod
    def to_json(cls, obj) -> Dict[str, Any]:
        result = {JSON_TYPE_NAME: get_full_class_name(type(obj))}
        introspector = DataclassOnlyIntrospector()
        for field_ in introspector.discover(obj.__class__):
            value = getattr(obj, field_.public_name)

            if isinstance(value, (list, set)):
                current_result = [to_json(item) for item in value]
            elif isinstance(value, dict):
                keys = [to_json(k) for k in value.keys()]
                values = [to_json(v) for v in value.values()]
                current_result = {"keys": keys, "values": values}
            else:
                current_result = to_json(value)
            result[field_.public_name] = current_result
        return result

    @classmethod
    def matches_generic_type(cls, clazz: Type) -> bool:
        return is_dataclass(clazz)

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type, **kwargs) -> Self:
        fields_ = {f.name: f for f in fields(clazz)}

        init_args = {}

        for k, v in fields_.items():
            if k not in data.keys():
                continue

            current_data = data[k]

            if isinstance(current_data, list):
                current_result = [from_json(data, **kwargs) for data in current_data]
            elif (
                isinstance(current_data, dict)
                and "keys" in current_data.keys()
                and "values" in current_data.keys()
            ):
                keys = [from_json(data, **kwargs) for data in current_data["keys"]]
                values = [from_json(data, **kwargs) for data in current_data["values"]]
                current_result = dict(zip(keys, values))
            else:
                current_result = from_json(current_data, **kwargs)
            init_args[k] = current_result
        return clazz(**init_args)
