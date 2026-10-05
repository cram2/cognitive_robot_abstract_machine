from __future__ import annotations

import datetime
import decimal
import enum
import fractions
import ipaddress
import pathlib
import uuid
import zoneinfo
from dataclasses import dataclass
from types import NoneType

import numpy as np
import sqlalchemy
from typing_extensions import Type

from krrood.adapters.json_serializer import JSONData, SubclassJSONSerializer
from krrood.ormatic.custom_types import (
    ByteArrayType,
    ComplexType,
    DateTimeType,
    Datetime64Type,
    FractionType,
    IPv4AddressType,
    IPv4NetworkType,
    IPv6AddressType,
    IPv6NetworkType,
    JSONDataType,
    NumpyArrayType,
    NumpyBoolType,
    NumpyFloat16Type,
    NumpyFloat32Type,
    NumpyFloat64Type,
    NumpyInt16Type,
    NumpyInt32Type,
    NumpyInt64Type,
    NumpyInt8Type,
    NumpyUInt16Type,
    NumpyUInt32Type,
    NumpyUInt8Type,
    PathType,
    PolymorphicEnumType,
    TypeType,
    ZoneInfoType,
)


@dataclass(frozen=True)
class TypeMapping:
    """
    A Python type and the SQLAlchemy column type that stores its values.
    """

    python_type: Type
    """
    The type of the values.
    """

    column_type: Type[sqlalchemy.types.TypeEngine]
    """
    The column type that stores the values.
    """


class DefaultTypeMapping(TypeMapping, enum.Enum):
    """
    The type mappings ORMatic uses for every type the caller gives no mapping for.

    Each member is a :class:`TypeMapping` of its Python type and column type.
    """

    # %% types, enums, identifiers and JSON

    TYPE = type, TypeType
    ENUM = enum.Enum, PolymorphicEnumType
    SUBCLASS_JSON_SERIALIZER = SubclassJSONSerializer, sqlalchemy.JSON
    UUID = uuid.UUID, sqlalchemy.UUID
    PATH = pathlib.Path, PathType
    JSON_DATA = JSONData, JSONDataType
    NONE = NoneType, TypeType

    # %% dates and times

    DATE = datetime.date, sqlalchemy.Date
    TIME = datetime.time, sqlalchemy.Time
    TIMEDELTA = datetime.timedelta, sqlalchemy.Interval
    DATETIME = datetime.datetime, DateTimeType
    ZONE_INFO = zoneinfo.ZoneInfo, ZoneInfoType

    # %% numbers

    DECIMAL = decimal.Decimal, sqlalchemy.Numeric
    FRACTION = fractions.Fraction, FractionType
    COMPLEX = complex, ComplexType

    # %% network addresses

    IPV4_ADDRESS = ipaddress.IPv4Address, IPv4AddressType
    IPV6_ADDRESS = ipaddress.IPv6Address, IPv6AddressType
    IPV4_NETWORK = ipaddress.IPv4Network, IPv4NetworkType
    IPV6_NETWORK = ipaddress.IPv6Network, IPv6NetworkType

    # %% byte sequences

    BYTE_ARRAY = bytearray, ByteArrayType

    # %% numpy

    NUMPY_ARRAY = np.ndarray, NumpyArrayType
    NUMPY_FLOAT16 = np.float16, NumpyFloat16Type
    NUMPY_FLOAT32 = np.float32, NumpyFloat32Type
    NUMPY_FLOAT64 = np.float64, NumpyFloat64Type
    NUMPY_INT8 = np.int8, NumpyInt8Type
    NUMPY_INT16 = np.int16, NumpyInt16Type
    NUMPY_INT32 = np.int32, NumpyInt32Type
    NUMPY_INT64 = np.int64, NumpyInt64Type
    NUMPY_UINT8 = np.uint8, NumpyUInt8Type
    NUMPY_UINT16 = np.uint16, NumpyUInt16Type
    NUMPY_UINT32 = np.uint32, NumpyUInt32Type
    NUMPY_BOOL = np.bool_, NumpyBoolType
    NUMPY_DATETIME64 = np.datetime64, Datetime64Type
