import datetime
import decimal
import fractions
import ipaddress
import pathlib
import re
import zoneinfo

import numpy as np
import pytest
from sqlalchemy import select

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.custom_types import NumpyType, ZoneInfoType
from krrood.ormatic.default_type_mappings import DefaultTypeMapping
from krrood.ormatic.data_access_objects.helper import (
    get_data_access_object_class,
    to_dao,
)
from krrood.ormatic.exceptions import ZoneInfoWithoutKey
from krrood.ormatic.helper import OrmaticInterfaceInformation
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.type_dict import TypeDict
from ..dataset.single_value_classes import *

# %% helpers


def store_and_load(original, session):
    """
    Write an object to the database and read it back, bypassing the session cache.

    :param original: The object to store.
    :param session: The session to store it with.
    :return: The object rebuilt from the database row.
    """
    session.add(to_dao(original))
    session.commit()
    session.expire_all()
    return (
        session.scalars(select(get_data_access_object_class(type(original))))
        .one()
        .from_dao()
    )


# %% round trips

ROUND_TRIP_CASES = [
    StoredDate(datetime.date(2026, 9, 29)),
    StoredTime(datetime.time(12, 30, 15, 123)),
    StoredTimedelta(datetime.timedelta(days=1, seconds=5, microseconds=7)),
    StoredDecimal(decimal.Decimal("3.25")),
    StoredDateTime(datetime.datetime(2026, 9, 29, 12, 0, 1, 5)),
    StoredTimezone(datetime.timezone(datetime.timedelta(hours=2))),
    StoredTimezone(datetime.timezone(datetime.timedelta(hours=-3), "BRT")),
    StoredZoneInfo(zoneinfo.ZoneInfo("Europe/Berlin")),
    StoredPurePath(pathlib.PurePosixPath("/a/b")),
    StoredPurePath(pathlib.PureWindowsPath("C:/a/b")),
    StoredIPv4Address(ipaddress.IPv4Address("10.0.0.1")),
    StoredIPv6Address(ipaddress.IPv6Address("::1")),
    StoredIPv4Network(ipaddress.IPv4Network("10.0.0.0/8")),
    StoredIPv6Network(ipaddress.IPv6Network("2001:db8::/32")),
    StoredFraction(fractions.Fraction(1, 3)),
    StoredComplex(1.5 - 2j),
    StoredByteArray(bytearray(b"\x00\x01abc")),
    StoredRange(range(1, 10, 2)),
    StoredSlice(slice(1, None, -1)),
    StoredPattern(re.compile("a+b", re.IGNORECASE)),
    StoredFloat16(np.float16(1.5)),
    StoredFloat32(np.float32(0.1)),
    StoredFloat64(np.float64(0.1)),
    StoredInt8(np.int8(-8)),
    StoredInt16(np.int16(-16)),
    StoredInt32(np.int32(-32)),
    StoredInt64(np.int64(-(2**62))),
    StoredUInt8(np.uint8(255)),
    StoredUInt16(np.uint16(65535)),
    StoredUInt32(np.uint32(2**32 - 1)),
    StoredBool(np.bool_(True)),
    StoredDatetime64(np.datetime64("2026-09-29")),
    StoredDatetime64(np.datetime64("2026-09-29T12:00:00.000000001")),
]


@pytest.mark.parametrize(
    "original", ROUND_TRIP_CASES, ids=[repr(case) for case in ROUND_TRIP_CASES]
)
def test_value_survives_a_database_round_trip(original, session, database):
    reconstructed = store_and_load(original, session)

    assert reconstructed == original
    assert type(reconstructed.value) is type(original.value)


def test_timezone_keeps_its_name(session, database):
    original = StoredTimezone(datetime.timezone(datetime.timedelta(hours=-3), "BRT"))

    reconstructed = store_and_load(original, session)

    assert reconstructed.value.tzname(None) == original.value.tzname(None)


def test_datetime64_keeps_its_unit(session, database):
    original = StoredDatetime64(np.datetime64("2026-09-29T12:00:00.000000001"))

    reconstructed = store_and_load(original, session)

    assert np.datetime_data(reconstructed.value.dtype) == np.datetime_data(
        original.value.dtype
    )


# %% timezone-aware datetimes


def test_timezone_aware_datetime_is_stored_as_the_same_instant_in_utc(
    session, database
):
    """
    SQLite keeps no timezone, so the instant comes back as naive UTC.
    """
    offset = datetime.timezone(datetime.timedelta(hours=2))
    original = StoredDateTime(datetime.datetime(2026, 9, 29, 12, 0, tzinfo=offset))

    reconstructed = store_and_load(original, session)

    assert reconstructed.value == original.value.astimezone(
        datetime.timezone.utc
    ).replace(tzinfo=None)


def test_naive_datetime_stays_naive(session, database):
    original = StoredDateTime(datetime.datetime(2026, 9, 29, 12, 0))

    reconstructed = store_and_load(original, session)

    assert reconstructed.value.tzinfo is None


# %% lists of values stored as JSON


def test_list_of_values_with_a_json_serializer_survives_a_round_trip(session, database):
    original = StoredRanges([range(3), range(1, 10, 2)])

    reconstructed = store_and_load(original, session)

    assert reconstructed == original


# %% numpy arrays


@pytest.mark.parametrize(
    "array",
    [
        np.arange(6, dtype=np.float32).reshape(2, 3),
        np.array([[1, 2], [3, 4]], dtype=np.int64),
        np.array([True, False]),
        np.zeros((0, 4)),
    ],
    ids=["float32 matrix", "int64 matrix", "bool vector", "empty"],
)
def test_array_keeps_its_values_shape_and_dtype(array, session, database):
    reconstructed = store_and_load(StoredArray(array), session)

    np.testing.assert_array_equal(reconstructed.value, array)
    assert reconstructed.value.shape == array.shape
    assert reconstructed.value.dtype == array.dtype


# %% values that cannot be stored


def test_zone_info_without_a_key_is_rejected():
    with open(pathlib.Path(zoneinfo.TZPATH[0]) / "Europe" / "Berlin", "rb") as file:
        zone = zoneinfo.ZoneInfo.from_file(file)

    with pytest.raises(ZoneInfoWithoutKey):
        ZoneInfoType().process_bind_param(zone, dialect=None)


# %% given type mappings


def test_given_type_mapping_is_kept_over_the_default():
    ormatic = ORMatic(
        ClassDiagram([]),
        interface_information=OrmaticInterfaceInformation(
            type_mappings=TypeDict({np.ndarray: NumpyType})
        ),
    )

    assert ormatic.type_mappings[np.ndarray] is NumpyType


@pytest.mark.parametrize(
    "default",
    list(DefaultTypeMapping),
    ids=[default.name for default in DefaultTypeMapping],
)
def test_default_type_mapping_is_used_when_none_is_given(default):
    ormatic = ORMatic(ClassDiagram([]))

    type_mappings = dict(ormatic.type_mappings.items())

    assert type_mappings[default.python_type] is default.column_type
