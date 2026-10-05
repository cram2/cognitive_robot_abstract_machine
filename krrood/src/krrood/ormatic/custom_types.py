import datetime
import enum
import fractions
import importlib
import io
import ipaddress
import json
import pathlib
import zoneinfo
from functools import cache
from typing import Generic

import numpy as np
from sqlalchemy import Dialect, TypeDecorator, types
from typing_extensions import Any, Optional, Type, TypeVar

from krrood.adapters.json_serializer import JSONData
from krrood.ormatic.exceptions import ZoneInfoWithoutKey
from krrood.utils import (
    get_generic_type_parameters,
    module_and_class_name,
    resolve_class_from_full_name,
)

T = TypeVar("T")

# %% base types


class ValueType(TypeDecorator, Generic[T]):
    """
    Column type for values of the Python type bound to its type parameter.

    ..note:: Concrete subclasses must set ``cache_ok`` themselves, as SQLAlchemy only
        reads it from the class that is used as the column type.

    ..note:: This cannot use
        :class:`~krrood.patterns.subclass_safe_generic.SubClassSafeGeneric`: as a
        dataclass it makes instances unhashable, which SQLAlchemy requires them to be.
    """

    __class_getitem__ = Generic.__dict__["__class_getitem__"]
    """
    SQLAlchemy type classes ignore subscription, so ``Generic``'s is restored to bind
    the type parameter.
    """

    @property
    def python_type(self) -> Type[T]:
        """
        :return: The type of the values this column holds.
        """
        return self.bound_type()

    @classmethod
    @cache
    def bound_type(cls) -> Type[T]:
        """
        :return: The type bound to the type parameter of this class, looked up once per
            class since it is read for every value that is loaded.
        """
        return get_generic_type_parameters(cls, ValueType)[0]


class TextValueType(ValueType[T]):
    """
    Column type for values that are stored as their text form and rebuilt by calling
    their type on that text.
    """

    impl = types.Text

    def process_bind_param(self, value: Optional[T], dialect: Dialect) -> Optional[str]:
        if value is None:
            return None
        return str(value)

    def process_result_value(
        self, value: Optional[str], dialect: Dialect
    ) -> Optional[T]:
        if value is None:
            return None
        return self.python_type(value)


class NumpyScalarType(ValueType[T]):
    """
    Column type for numpy scalars, stored as the matching Python scalar.
    """

    def process_bind_param(self, value: Optional[T], dialect: Dialect) -> Any:
        if value is None:
            return None
        return value.item()

    def process_result_value(self, value: Any, dialect: Dialect) -> Optional[T]:
        if value is None:
            return None
        return self.python_type(value)


# %% types and enums


class TypeType(ValueType[type]):
    """
    Type that casts fields that are of type `type` to their class name on serialization
    and converts the name to the class itself through the globals on load.
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
        return resolve_class_from_full_name(str(value))


class PolymorphicEnumType(ValueType[enum.Enum]):
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


class JSONDataType(ValueType[JSONData]):
    """
    Type decorator for JSONData that stores JSON without automatic deserialization.

    Unlike regular JSON columns which use the engine's custom json_deserializer (that
    calls from_json()), this type keeps the data as raw JSON dictionaries/lists. This is
    necessary for fields that should be deserialized later in application code.
    """

    impl = types.String
    cache_ok = True

    def process_bind_param(self, value: Optional[JSONData], dialect: Dialect):
        """
        Store the value as-is (already JSON-serializable).
        """
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value: impl, dialect: Dialect):
        """
        Return the value as-is (raw JSON, not deserialized).
        """
        if value is None:
            return None
        return json.loads(value)


# %% dates and times


class DateTimeType(ValueType[datetime.datetime]):
    """
    Column type for points in time that stores timezone-aware values as the same instant
    in UTC.

    ..note:: Databases without timezone support, such as SQLite, return those values as
        naive datetimes in UTC.
    """

    impl = types.DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self, value: Optional[datetime.datetime], dialect: Dialect
    ) -> Optional[datetime.datetime]:
        if value is None or value.tzinfo is None:
            return value
        return value.astimezone(datetime.timezone.utc)


class ZoneInfoType(TextValueType[zoneinfo.ZoneInfo]):
    """
    Column type for IANA timezones, stored by their key.
    """

    impl = types.String(256)
    cache_ok = True

    def process_bind_param(
        self, value: Optional[zoneinfo.ZoneInfo], dialect: Dialect
    ) -> Optional[str]:
        if value is not None and value.key is None:
            raise ZoneInfoWithoutKey(value)
        return super().process_bind_param(value, dialect)


# %% numbers


class FractionType(TextValueType[fractions.Fraction]):
    """
    Column type for rational numbers, stored as text such as ``1/3``.
    """

    cache_ok = True


class ComplexType(TextValueType[complex]):
    """
    Column type for complex numbers, stored as text such as ``(1.5-2j)``.
    """

    cache_ok = True


# %% paths and network addresses


class PathType(TextValueType[pathlib.Path]):
    """
    Type decorator for pathlib.Path objects.
    """

    cache_ok = True


class IPv4AddressType(TextValueType[ipaddress.IPv4Address]):
    """
    Column type for IPv4 addresses.
    """

    impl = types.String(15)
    cache_ok = True


class IPv6AddressType(TextValueType[ipaddress.IPv6Address]):
    """
    Column type for IPv6 addresses.
    """

    impl = types.String(45)
    cache_ok = True


class IPv4NetworkType(TextValueType[ipaddress.IPv4Network]):
    """
    Column type for IPv4 networks, stored in CIDR notation.
    """

    impl = types.String(18)
    cache_ok = True


class IPv6NetworkType(TextValueType[ipaddress.IPv6Network]):
    """
    Column type for IPv6 networks, stored in CIDR notation.
    """

    impl = types.String(49)
    cache_ok = True


# %% byte sequences


class ByteArrayType(ValueType[bytearray]):
    """
    Column type for mutable byte sequences.
    """

    impl = types.LargeBinary
    cache_ok = True

    def process_bind_param(
        self, value: Optional[bytearray], dialect: Dialect
    ) -> Optional[bytes]:
        if value is None:
            return None
        return bytes(value)

    def process_result_value(
        self, value: Optional[bytes], dialect: Dialect
    ) -> Optional[bytearray]:
        if value is None:
            return None
        return bytearray(value)


# %% numpy


class NumpyType(ValueType[np.ndarray]):
    """
    Type decorator for numpy arrays, stored as raw float64 bytes.

    ..note:: The shape is not stored, so arrays are read back as one-dimensional.
        :class:`NumpyArrayType` keeps the shape and dtype.
    """

    impl = types.LargeBinary(4 * 1024 * 1024 * 1024 - 1)
    cache_ok = True

    def process_bind_param(
        self, value: Optional[np.ndarray], dialect: Dialect
    ) -> Optional[bytes]:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        return array.tobytes(order="C")

    def process_result_value(
        self, value: Optional[bytes], dialect: Dialect
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        return np.frombuffer(value, dtype=np.float64)


class NumpyArrayType(ValueType[np.ndarray]):
    """
    Column type for numpy arrays, stored in the ``.npy`` format so that their shape and
    dtype are kept.

    ..note:: Arrays of Python objects cannot be stored.
    """

    impl = types.LargeBinary(4 * 1024 * 1024 * 1024 - 1)
    cache_ok = True

    def process_bind_param(
        self, value: Optional[np.ndarray], dialect: Dialect
    ) -> Optional[bytes]:
        if value is None:
            return None
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return buffer.getvalue()

    def process_result_value(
        self, value: Optional[bytes], dialect: Dialect
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        return np.load(io.BytesIO(value), allow_pickle=False)


class NumpyFloat16Type(NumpyScalarType[np.float16]):
    """
    Column type for numpy half precision floats.
    """

    impl = types.Float
    cache_ok = True


class NumpyFloat32Type(NumpyScalarType[np.float32]):
    """
    Column type for numpy single precision floats.
    """

    impl = types.Float
    cache_ok = True


class NumpyFloat64Type(NumpyScalarType[np.float64]):
    """
    Column type for numpy double precision floats.
    """

    impl = types.Float
    cache_ok = True


class NumpyInt8Type(NumpyScalarType[np.int8]):
    """
    Column type for numpy 8 bit integers.
    """

    impl = types.Integer
    cache_ok = True


class NumpyInt16Type(NumpyScalarType[np.int16]):
    """
    Column type for numpy 16 bit integers.
    """

    impl = types.Integer
    cache_ok = True


class NumpyInt32Type(NumpyScalarType[np.int32]):
    """
    Column type for numpy 32 bit integers.
    """

    impl = types.Integer
    cache_ok = True


class NumpyInt64Type(NumpyScalarType[np.int64]):
    """
    Column type for numpy 64 bit integers.
    """

    impl = types.BigInteger
    cache_ok = True


class NumpyUInt8Type(NumpyScalarType[np.uint8]):
    """
    Column type for numpy 8 bit unsigned integers.
    """

    impl = types.Integer
    cache_ok = True


class NumpyUInt16Type(NumpyScalarType[np.uint16]):
    """
    Column type for numpy 16 bit unsigned integers.
    """

    impl = types.Integer
    cache_ok = True


class NumpyUInt32Type(NumpyScalarType[np.uint32]):
    """
    Column type for numpy 32 bit unsigned integers, which need a 64 bit column.
    """

    impl = types.BigInteger
    cache_ok = True


class NumpyBoolType(NumpyScalarType[np.bool_]):
    """
    Column type for numpy booleans.
    """

    impl = types.Boolean
    cache_ok = True


class Datetime64Type(TextValueType[np.datetime64]):
    """
    Column type for numpy points in time, stored as ISO 8601 text whose precision keeps
    the time unit.
    """

    impl = types.String(64)
    cache_ok = True
