import enum
import importlib
import json
import pathlib

from sqlalchemy import Dialect, TypeDecorator, types
from typing_extensions import Optional, Type

from krrood.adapters.json_serializer import JSONData
from krrood.utils import module_and_class_name


class TypeType(TypeDecorator):
    """
    Type that casts fields that are of type `type` to their class name on serialization and converts the name
    to the class itself through the globals on load.
    """

    impl = types.String(256)
    cache_ok = True

    def process_bind_param(
        self, value: Optional[Type], dialect: Dialect
    ) -> Optional[str]:
        if value is None:
            return None
        return module_and_class_name(value)

    def process_result_value(self, value: impl, dialect: Dialect) -> Optional[Type]:
        if value is None:
            return None

        module_name, class_name = str(value).rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)


class PolymorphicEnumType(TypeDecorator):
    """
    Custom type for storing polymorphic enums by their full path and member name.
    """

    impl = types.String(512)
    cache_ok = True

    def process_bind_param(
        self, value: Optional[enum.Enum], dialect: Dialect
    ) -> Optional[str]:
        if value is None:
            return None
        # Store as 'module.path.ClassName.MEMBER_NAME'
        return f"{value.__class__.__module__}.{value.__class__.__name__}.{value.name}"

    def process_result_value(
        self, value: Optional[str], dialect: Dialect
    ) -> Optional[enum.Enum]:
        if value is None:
            return None

        parts = value.rsplit(".", 2)
        module_name = parts[0]
        class_name = parts[1]
        member_name = parts[2]

        module = importlib.import_module(module_name)
        enum_class = getattr(module, class_name)
        return enum_class[member_name]


class JSONDataType(TypeDecorator):
    """
    Type decorator for JSONData that stores JSON without automatic deserialization.

    Unlike regular JSON columns which use the engine's custom json_deserializer
    (that calls from_json()), this type keeps the data as raw JSON dictionaries/lists.
    This is necessary for fields that should be deserialized later in application code.
    """

    impl = types.String
    cache_ok = True

    def process_bind_param(self, value: Optional[JSONData], dialect: Dialect):
        """Store the value as-is (already JSON-serializable)."""
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value: impl, dialect: Dialect):
        """Return the value as-is (raw JSON, not deserialized)."""
        if value is None:
            return None
        return json.loads(value)


class PathType(TypeDecorator):
    """
    Type decorator for pathlib.Path objects.
    """

    impl = types.Text
    cache_ok = True

    def process_bind_param(
        self, value: Optional[pathlib.Path], dialect: Dialect
    ) -> Optional[str]:
        if value is not None:
            return str(value)
        return value

    def process_result_value(
        self, value: Optional[str], dialect: Dialect
    ) -> Optional[pathlib.Path]:
        if value is not None:
            return pathlib.Path(value)
        return value
