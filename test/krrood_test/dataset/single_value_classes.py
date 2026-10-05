"""
Dataclasses that each hold one value of a type ORMatic stores by default, so every
default type mapping can be round-tripped through the database on its own.
"""

from __future__ import annotations

import datetime
import decimal
import fractions
import ipaddress
import pathlib
import re
import zoneinfo
from dataclasses import dataclass

import numpy as np

# %% standard library values with a native SQL type


@dataclass
class StoredDate:
    """
    Holds a calendar date.
    """

    value: datetime.date
    """
    The stored date.
    """


@dataclass
class StoredTime:
    """
    Holds a time of day.
    """

    value: datetime.time
    """
    The stored time of day.
    """


@dataclass
class StoredTimedelta:
    """
    Holds a duration.
    """

    value: datetime.timedelta
    """
    The stored duration.
    """


@dataclass
class StoredDecimal:
    """
    Holds an exact decimal number.
    """

    value: decimal.Decimal
    """
    The stored decimal number.
    """


# %% standard library values stored through a custom type


@dataclass
class StoredDateTime:
    """
    Holds a point in time, with or without a timezone.
    """

    value: datetime.datetime
    """
    The stored point in time.
    """


@dataclass
class StoredTimezone:
    """
    Holds a fixed-offset timezone.
    """

    value: datetime.timezone
    """
    The stored timezone.
    """


@dataclass
class StoredZoneInfo:
    """
    Holds an IANA timezone.
    """

    value: zoneinfo.ZoneInfo
    """
    The stored IANA timezone.
    """


@dataclass
class StoredPurePath:
    """
    Holds a path that is not bound to the file system.
    """

    value: pathlib.PurePath
    """
    The stored path.
    """


@dataclass
class StoredIPv4Address:
    """
    Holds an IPv4 address.
    """

    value: ipaddress.IPv4Address
    """
    The stored address.
    """


@dataclass
class StoredIPv6Address:
    """
    Holds an IPv6 address.
    """

    value: ipaddress.IPv6Address
    """
    The stored address.
    """


@dataclass
class StoredIPv4Network:
    """
    Holds an IPv4 network.
    """

    value: ipaddress.IPv4Network
    """
    The stored network.
    """


@dataclass
class StoredIPv6Network:
    """
    Holds an IPv6 network.
    """

    value: ipaddress.IPv6Network
    """
    The stored network.
    """


@dataclass
class StoredFraction:
    """
    Holds a rational number.
    """

    value: fractions.Fraction
    """
    The stored rational number.
    """


@dataclass
class StoredComplex:
    """
    Holds a complex number.
    """

    value: complex
    """
    The stored complex number.
    """


@dataclass
class StoredByteArray:
    """
    Holds a mutable byte sequence.
    """

    value: bytearray
    """
    The stored bytes.
    """


@dataclass
class StoredRange:
    """
    Holds a range of integers.
    """

    value: range
    """
    The stored range.
    """


@dataclass
class StoredSlice:
    """
    Holds a slice.
    """

    value: slice
    """
    The stored slice.
    """


@dataclass
class StoredPattern:
    """
    Holds a compiled regular expression.
    """

    value: re.Pattern
    """
    The stored regular expression.
    """


@dataclass
class StoredRanges:
    """
    Holds a list of ranges.
    """

    values: list[range]
    """
    The stored ranges.
    """


# %% numpy values


@dataclass
class StoredArray:
    """
    Holds a numpy array.

    ..note:: Arrays define ``==`` elementwise, so compare instances field by field.
    """

    value: np.ndarray
    """
    The stored array.
    """


@dataclass
class StoredFloat16:
    """
    Holds a numpy half precision float.
    """

    value: np.float16
    """
    The stored number.
    """


@dataclass
class StoredFloat32:
    """
    Holds a numpy single precision float.
    """

    value: np.float32
    """
    The stored number.
    """


@dataclass
class StoredFloat64:
    """
    Holds a numpy double precision float.
    """

    value: np.float64
    """
    The stored number.
    """


@dataclass
class StoredInt8:
    """
    Holds a numpy 8 bit integer.
    """

    value: np.int8
    """
    The stored number.
    """


@dataclass
class StoredInt16:
    """
    Holds a numpy 16 bit integer.
    """

    value: np.int16
    """
    The stored number.
    """


@dataclass
class StoredInt32:
    """
    Holds a numpy 32 bit integer.
    """

    value: np.int32
    """
    The stored number.
    """


@dataclass
class StoredInt64:
    """
    Holds a numpy 64 bit integer.
    """

    value: np.int64
    """
    The stored number.
    """


@dataclass
class StoredUInt8:
    """
    Holds a numpy 8 bit unsigned integer.
    """

    value: np.uint8
    """
    The stored number.
    """


@dataclass
class StoredUInt16:
    """
    Holds a numpy 16 bit unsigned integer.
    """

    value: np.uint16
    """
    The stored number.
    """


@dataclass
class StoredUInt32:
    """
    Holds a numpy 32 bit unsigned integer.
    """

    value: np.uint32
    """
    The stored number.
    """


@dataclass
class StoredBool:
    """
    Holds a numpy boolean.
    """

    value: np.bool_
    """
    The stored boolean.
    """


@dataclass
class StoredDatetime64:
    """
    Holds a numpy point in time.
    """

    value: np.datetime64
    """
    The stored point in time.
    """
