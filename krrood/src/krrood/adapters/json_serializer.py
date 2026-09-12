from __future__ import annotations

import enum
import importlib
import inspect
import uuid
from datetime import timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from dataclasses import field
from types import NoneType
from typing import List, Optional, TypeAlias, TYPE_CHECKING

import numpy as np
from typing_extensions import Dict, Any, Self, Union, Type, TypeVar

from krrood.adapters.exceptions import (
    MissingTypeError,
    InvalidTypeFormatError,
    UnknownModuleError,
    ClassNotFoundError,
    ClassNotSerializableError,
    JSON_TYPE_NAME,
)
from krrood.adapters.keyword_argument import SerializationKeywordArgument
from krrood.class_diagrams.attribute_introspector import DataclassOnlyIntrospector
from krrood.ormatic.data_access_objects.base import HasGeneric
from krrood.singleton import SingletonMeta
from krrood.utils import (
    get_full_class_name,
    recursive_subclasses,
)

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
    JSON_DICT_TYPE, List[Any], *leaf_types
]  # Commonly referred JSON types
JSON_IS_CLASS = "__is_class__"
"""
We need to remember if something is a class, because the type of a class is often just
type.
"""

if TYPE_CHECKING:
    JSONData: TypeAlias = JSON_RETURN_TYPE
else:

    class JSONData:
        """
        Represents raw JSON data.

        Use this type for type hints when you want to tell KRROOD that something is JSON
        data that should not be further processed (e.g. by from_json()).
        """


@dataclass
class JSONSerializableTypeRegistry(metaclass=SingletonMeta):
    """
    Singleton registry for custom serializers and deserializers.

    Use this registry when you need to add custom JSON serialization/deserialization
    logic for a type where you cannot control its inheritance.
    """

    def get_external_serializer(self, clazz: Type) -> Type[ExternalClassJSONSerializer]:
        """
        Get the external serializer for the given class.

        This returns the serializer of the closest superclass if no direct match is
        found.

        :param clazz: The class to get the serializer for.
        :return: The serializer class.
        """
        # Imported lazily to avoid a circular import: inheritance_path_length pulls in the EQL
        # predicate/variable modules, which import back from json_serializer during package load.
        from krrood.inheritance_path_length import inheritance_path_length

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

    Stores the fully qualified class name in `type` during serialization and imports
    that class during deserialization.
    """

    def to_json(self, **kwargs) -> Dict[str, Any]:
        """
        :param kwargs: Keyword arguments to hand on to the ``to_json`` calls for the
            values this object holds.
        :return: The JSON dict
        """
        return {JSON_TYPE_NAME: get_full_class_name(self.__class__)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Create an instance from a json dict.

        This method is called from the from_json method after the correct subclass is
        determined and should be overwritten by the subclass.

        :param data: The JSON dict
        :param kwargs: Additional keyword arguments to pass to the constructor of the
            subclass.
        :return: The deserialized object
        """
        raise NotImplementedError()

    @classmethod
    def from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Create the correct instanceof the subclass from a json dict.

        :param data: The json dict
        :param kwargs: Additional keyword arguments to pass to the constructor of the
            subclass.
        :return: The correct instance of the subclass
        """
        if isinstance(data, leaf_types):
            return data

        if isinstance(data, list_like_classes):
            return [from_json(d, **kwargs) for d in data]

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
        :param kwargs: Additional keyword arguments to pass to the constructor of the
            subclass.
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


def to_json(obj: Union[SubclassJSONSerializer, Any], **kwargs) -> JSON_RETURN_TYPE:
    """
    Serialize an object to a JSON dict.

    :param obj: The object to convert to json
    :param kwargs: Keyword arguments handed on to every nested ``to_json`` call, for
        example a :class:`ReferenceWriter`.
    :return: The JSON string
    """
    if isinstance(obj, dict):
        json_type = obj.get(JSON_TYPE_NAME, None)
        if json_type is not None:
            return obj

    # An enum member that is also a str or an int would pass as a leaf value and lose
    # its enum type on the way back, so enums are checked first.
    if isinstance(obj, enum.Enum):
        return EnumJSONSerializer.to_json(obj)

    if isinstance(obj, (leaf_types)):
        return obj

    if isinstance(obj, list_like_classes):
        return [to_json(item, **kwargs) for item in obj]

    reference_writer = ReferenceWriter.find_for(obj, kwargs)
    if reference_writer is not None:
        return reference_writer.write_reference(obj)

    if isinstance(obj, SubclassJSONSerializer):
        return obj.to_json(**kwargs)

    if inspect.isclass(obj):
        return ClassJSONSerializer.to_json(obj, **kwargs)

    registered_json_serializer = JSONSerializableTypeRegistry().get_external_serializer(
        type(obj)
    )

    return registered_json_serializer.to_json(obj, **kwargs)


@dataclass
class JSONAttributeDiff(SubclassJSONSerializer):
    """
    A class representing a shallow diff for JSON-serializable keyword arguments.
    """

    attribute_name: str = field(kw_only=True)
    """
    The name of the attribute that has changed.
    """

    added_values: List[JSONData] = field(default_factory=list)
    """
    The items that have been added to the attribute.
    """

    removed_values: List[JSONData] = field(default_factory=list)
    """
    The items that have been removed from the attribute.
    """

    def to_json(self, **kwargs) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(self.__class__),
            "attribute_name": self.attribute_name,
            "removed_values": self.removed_values,
            "added_values": self.added_values,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            attribute_name=data["attribute_name"],
            removed_values=data["removed_values"],
            added_values=data["added_values"],
        )


def shallow_diff_json(
    original_json: Dict[str, Any], new_json: Dict[str, Any], **kwargs
) -> List[JSONAttributeDiff]:
    """
    Create a shallow diff between two JSON dicts.

    Result describes the changes that need to be applied to first json to get second
    json.
    :param original_json: The original JSON dict.
    :param new_json: The new JSON dict.
    :return: List of JSONAttributeDiff describing the changes that need to be applied to
        first json to get second json.
    """
    all_keys = original_json.keys() | new_json.keys()
    diffs: List[JSONAttributeDiff] = [
        diff
        for key in all_keys
        if (diff := _compute_attribute_diff(original_json, new_json, key, **kwargs))
        is not None
    ]
    return diffs


def _compute_attribute_diff(
    original_json: Any, new_json: Any, key: str, **kwargs
) -> Optional[JSONAttributeDiff]:
    """
    Compute the attribute diff for a single key between two JSON dicts.

    :param original_json: The original JSON dict.
    :param new_json: The new JSON dict.
    :param key: The key to compute the diff for. :return JSONAttributeDiff describing
        the changes that need to be applied to first json to get second json for a
        specific key.
    """
    original_values = original_json.get(key)
    new_values = new_json.get(key)

    if not isinstance(original_values, list_like_classes):
        if original_values == new_values:
            return None
        return JSONAttributeDiff(
            attribute_name=key,
            added_values=[new_values],
            removed_values=[original_values],
        )

    add = [new_value for new_value in new_values if new_value not in original_values]
    remove = [
        original_value
        for original_value in original_values
        if original_value not in new_values
    ]
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

    Create a new subclass of this class pointing to your original class whenever you
    can't change its inheritance path to `SubclassJSONSerializer`.
    """

    @classmethod
    def to_json(cls, obj: Any, **kwargs) -> Dict[str, Any]:
        """
        Convert an object to a JSON serializable dictionary.

        :param obj: The object to convert.
        :param kwargs: Keyword arguments to hand on to the ``to_json`` calls for the
            values the object holds.
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
        :return: A boolean value indicating whether the provided class type matches the
            original class type.
        """
        return cls.original_class() == clazz


ReferencedType = TypeVar("ReferencedType")


@dataclass
class ReferenceWriter(SerializationKeywordArgument, HasGeneric[ReferencedType], ABC):
    """
    Writes the objects of the type it is bound to as references, because whoever reads
    the document already has them.

    Passed through the keyword arguments of ``to_json``, it replaces such an object with
    its reference wherever the object sits in the document.
    """

    @classmethod
    def find_for(
        cls, obj: Any, to_json_kwargs: Dict[str, Any]
    ) -> Optional[ReferenceWriter]:
        """
        :param obj: The object about to be serialized.
        :param to_json_kwargs: The keyword arguments of the ``to_json`` call.
        :return: The reference writer among the keyword arguments that writes the object
            as a reference, if there is one.
        """
        for value in to_json_kwargs.values():
            if isinstance(value, ReferenceWriter) and isinstance(
                obj, value.original_class()
            ):
                return value
        return None

    @abstractmethod
    def write_reference(self, obj: ReferencedType) -> Dict[str, Any]:
        """
        :param obj: The object to refer to.
        :return: The JSON the document holds in place of the object.
        """


@dataclass
class UUIDJSONSerializer(ExternalClassJSONSerializer[uuid.UUID]):

    @classmethod
    def to_json(cls, obj: uuid.UUID, **kwargs) -> Dict[str, Any]:
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
class TimedeltaJSONSerializer(ExternalClassJSONSerializer[timedelta]):
    """
    External JSON serializer for durations.

    Stored as the three components a duration normalises itself to, so a value survives
    the round trip exactly rather than through a float of seconds.
    """

    @classmethod
    def to_json(cls, obj: timedelta, **kwargs) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(type(obj)),
            "days": obj.days,
            "seconds": obj.seconds,
            "microseconds": obj.microseconds,
        }

    @classmethod
    def from_json(
        cls, data: Dict[str, Any], clazz: Type[timedelta], **kwargs
    ) -> timedelta:
        return clazz(
            days=data["days"],
            seconds=data["seconds"],
            microseconds=data["microseconds"],
        )


@dataclass
class ClassJSONSerializer(ExternalClassJSONSerializer[None]):
    """
    A class that provides mechanisms for serializing and deserializing Python classes to
    and from JSON representations.
    """

    @classmethod
    def to_json(cls, obj: Type, **kwargs) -> Dict[str, Any]:
        """
        This is a special case because we need to remember that the type of the class is
        a class, not a type.

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
    def to_json(cls, obj: enum.Enum, **kwargs) -> Dict[str, Any]:
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
    def to_json(cls, obj: Exception, **kwargs) -> Dict[str, Any]:
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
    def to_json(cls, obj: np.ndarray, **kwargs) -> Dict[str, Any]:
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

    It creates a dict where all fields are serialized using the to_json function. If
    this is not enough, you still need to implement a custom serializer.
    """

    @classmethod
    def to_json(cls, obj, **kwargs) -> Dict[str, Any]:
        result = {JSON_TYPE_NAME: get_full_class_name(type(obj))}
        introspector = DataclassOnlyIntrospector()
        for field_ in introspector.discover(obj.__class__):
            value = getattr(obj, field_.public_name)

            if isinstance(value, (list, set)):
                current_result = [to_json(item, **kwargs) for item in value]
            elif isinstance(value, dict):
                keys = [to_json(k, **kwargs) for k in value.keys()]
                values = [to_json(v, **kwargs) for v in value.values()]
                current_result = {"keys": keys, "values": values}
            else:
                current_result = to_json(value, **kwargs)
            result[field_.public_name] = current_result
        return result

    @classmethod
    def matches_generic_type(cls, clazz: Type) -> bool:
        return is_dataclass(clazz)

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type, **kwargs) -> Self:
        introspector = DataclassOnlyIntrospector()
        discovered_attributes = {
            attr.field.name: attr.field for attr in introspector.discover(clazz)
        }

        init_args = {}
        post_init_args = {}

        for field_name, field_ in discovered_attributes.items():
            if field_name not in data.keys():
                continue

            current_data = data[field_name]

            if isinstance(current_data, list):
                current_result = [from_json(item, **kwargs) for item in current_data]
            elif (
                isinstance(current_data, dict)
                and "keys" in current_data.keys()
                and "values" in current_data.keys()
            ):
                keys = [from_json(item, **kwargs) for item in current_data["keys"]]
                values = [from_json(item, **kwargs) for item in current_data["values"]]
                current_result = dict(zip(keys, values))
            else:
                current_result = from_json(current_data, **kwargs)

            if field_.init:
                init_args[field_name] = current_result
            else:
                post_init_args[field_name] = current_result

        instance = clazz(**init_args)
        for field_name, field_value in post_init_args.items():
            setattr(instance, field_name, field_value)
        return instance


@dataclass
class NumpyFloatJSONSerializer(ExternalClassJSONSerializer[np.float32]):
    """
    External JSON serializer for numpy floats.
    """

    @classmethod
    def to_json(cls, obj: np.float32, **kwargs) -> Dict[str, Any]:
        return {JSON_TYPE_NAME: get_full_class_name(type(obj)), "value": float(obj)}

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type, **kwargs) -> Self:
        return float(data["value"])
