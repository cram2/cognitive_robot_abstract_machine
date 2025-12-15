import enum
import importlib
from typing import List

from typing_extensions import Type, Optional

from sqlalchemy import TypeDecorator
from sqlalchemy import types

from ..adapters.json_serializer import JSON_TYPE_NAME
from ..ormatic.utils import module_and_class_name


class TypeType(TypeDecorator):
    """
    Type that casts fields that are of type `type` to their class name on serialization and converts the name
    to the class itself through the globals on load.
    """

    impl = types.String(256)

    def process_bind_param(self, value: Type, dialect):
        return module_and_class_name(value)

    def process_result_value(self, value: impl, dialect) -> Optional[Type]:
        if value is None:
            return None

        module_name, class_name = str(value).rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)


class EnumListType(TypeDecorator):
    """
    TypeDecorator for storing lists of enum values as JSON.

    Stores the enum class reference and a list of enum values. This provides
    database-backend independence by using native JSON storage.
    """

    impl = types.JSON

    def __init__(self, enum_class: Type[enum.Enum]):
        super().__init__()
        self.enum_class = enum_class

    def process_bind_param(self, values, dialect):
        return {
            JSON_TYPE_NAME: module_and_class_name(self.enum_class),
            "values": [item.value for item in values],
        }

    def process_result_value(self, value: impl, dialect) -> Optional[List[enum.Enum]]:
        if value is None:
            return None

        module_name, class_name = value[JSON_TYPE_NAME].rsplit(".", 1)
        module = importlib.import_module(module_name)
        enum_class = getattr(module, class_name)

        return [enum_class(item) for item in value["values"]]
