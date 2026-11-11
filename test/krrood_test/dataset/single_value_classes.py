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


@dataclass(kw_only=True)
class StoredDate:
    """
    Holds a calendar date.
    """

    value: datetime.date
    """
    The stored date.
    """


@dataclass(kw_only=True)
class StoredTime:
    """
    Holds a time of day.
    """

    value: datetime.time
    """
    The stored time of day.
    """


@dataclass(kw_only=True)
class StoredTimedelta:
    """
    Holds a duration.
    """

    value: datetime.timedelta
    """
    The stored duration.
    """


@dataclass(kw_only=True)
class StoredDecimal:
    """
    Holds an exact decimal number.
    """

    value: decimal.Decimal
    """
    The stored decimal number.
    """


# %% standard library values stored through a custom type


@dataclass(kw_only=True)
class StoredDateTime:
    """
    Holds a point in time, with or without a timezone.
    """

    value: datetime.datetime
    """
    The stored point in time.
    """


@dataclass(kw_only=True)
class StoredTimezone:
    """
    Holds a fixed-offset timezone.
    """

    value: datetime.timezone
    """
    The stored timezone.
    """


@dataclass(kw_only=True)
class StoredZoneInfo:
    """
    Holds an IANA timezone.
    """

    value: zoneinfo.ZoneInfo
    """
    The stored IANA timezone.
    """


@dataclass(kw_only=True)
class StoredPurePath:
    """
    Holds a path that is not bound to the file system.
    """

    value: pathlib.PurePath
    """
    The stored path.
    """


@dataclass(kw_only=True)
class StoredIPv4Address:
    """
    Holds an IPv4 address.
    """

    value: ipaddress.IPv4Address
    """
    The stored address.
    """


@dataclass(kw_only=True)
class StoredIPv6Address:
    """
    Holds an IPv6 address.
    """

    value: ipaddress.IPv6Address
    """
    The stored address.
    """


@dataclass(kw_only=True)
class StoredIPv4Network:
    """
    Holds an IPv4 network.
    """

    value: ipaddress.IPv4Network
    """
    The stored network.
    """


@dataclass(kw_only=True)
class StoredIPv6Network:
    """
    Holds an IPv6 network.
    """

    value: ipaddress.IPv6Network
    """
    The stored network.
    """


@dataclass(kw_only=True)
class StoredFraction:
    """
    Holds a rational number.
    """

    value: fractions.Fraction
    """
    The stored rational number.
    """


@dataclass(kw_only=True)
class StoredComplex:
    """
    Holds a complex number.
    """

    value: complex
    """
    The stored complex number.
    """


@dataclass(kw_only=True)
class StoredByteArray:
    """
    Holds a mutable byte sequence.
    """

    value: bytearray
    """
    The stored bytes.
    """


@dataclass(kw_only=True)
class StoredRange:
    """
    Holds a range of integers.
    """

    value: range
    """
    The stored range.
    """


@dataclass(kw_only=True)
class StoredSlice:
    """
    Holds a slice.
    """

    value: slice
    """
    The stored slice.
    """


@dataclass(kw_only=True)
class StoredPattern:
    """
    Holds a compiled regular expression.
    """

    value: re.Pattern
    """
    The stored regular expression.
    """


@dataclass(kw_only=True)
class StoredRanges:
    """
    Holds a list of ranges.
    """

    values: list[range]
    """
    The stored ranges.
    """


# %% numpy values


@dataclass(kw_only=True)
class StoredArray:
    """
    Holds a numpy array.

    ..note:: Arrays define ``==`` elementwise, so compare instances field by field.
    """

    value: np.ndarray
    """
    The stored array.
    """


@dataclass(kw_only=True)
class StoredFloat16:
    """
    Holds a numpy half precision float.
    """

    value: np.float16
    """
    The stored number.
    """


@dataclass(kw_only=True)
class StoredFloat32:
    """
    Holds a numpy single precision float.
    """

    value: np.float32
    """
    The stored number.
    """


@dataclass(kw_only=True)
class StoredFloat64:
    """
    Holds a numpy double precision float.
    """

    value: np.float64
    """
    The stored number.
    """


@dataclass(kw_only=True)
class StoredInt8:
    """
    Holds a numpy 8 bit integer.
    """

    value: np.int8
    """
    The stored number.
    """


@dataclass(kw_only=True)
class StoredInt16:
    """
    Holds a numpy 16 bit integer.
    """

    value: np.int16
    """
    The stored number.
    """


@dataclass(kw_only=True)
class StoredInt32:
    """
    Holds a numpy 32 bit integer.
    """

    value: np.int32
    """
    The stored number.
    """


@dataclass(kw_only=True)
class StoredInt64:
    """
    Holds a numpy 64 bit integer.
    """

    value: np.int64
    """
    The stored number.
    """


@dataclass(kw_only=True)
class StoredUInt8:
    """
    Holds a numpy 8 bit unsigned integer.
    """

    value: np.uint8
    """
    The stored number.
    """


@dataclass(kw_only=True)
class StoredUInt16:
    """
    Holds a numpy 16 bit unsigned integer.
    """

    value: np.uint16
    """
    The stored number.
    """


@dataclass(kw_only=True)
class StoredUInt32:
    """
    Holds a numpy 32 bit unsigned integer.
    """

    value: np.uint32
    """
    The stored number.
    """


@dataclass(kw_only=True)
class StoredBool:
    """
    Holds a numpy boolean.
    """

    value: np.bool_
    """
    The stored boolean.
    """


@dataclass(kw_only=True)
class StoredDatetime64:
    """
    Holds a numpy point in time.
    """

    value: np.datetime64
    """
    The stored point in time.
    """
