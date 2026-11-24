"""
Deep copies of dataclasses built through their constructors.
"""

import copy

import pytest

from krrood.patterns.dataclass_deep_copy import DataclassDeepcopyMixin
from krrood.patterns.exceptions import InterfaceInstantiationError
from krrood.patterns.mixin import Mixin

from ..dataset.deep_copied_classes import (
    AnnotatedMeasurements,
    Logbook,
    LoggedReading,
    Measurements,
    PairOfMeasurements,
    SensorReading,
    UnitOfMeasurement,
)

# %% fixtures


@pytest.fixture
def metre() -> UnitOfMeasurement:
    return UnitOfMeasurement(symbol="m")


@pytest.fixture
def measurements(metre: UnitOfMeasurement) -> AnnotatedMeasurements:
    return AnnotatedMeasurements(
        values=[1.0, 2.0], unit=metre, source="laser", notes=["calibrated"]
    )


# %% constructor fields


def test_a_deep_copy_keeps_the_class(measurements: AnnotatedMeasurements):
    assert type(copy.deepcopy(measurements)) is AnnotatedMeasurements


def test_a_deep_copy_has_equal_constructor_fields(
    measurements: AnnotatedMeasurements,
):
    copied = copy.deepcopy(measurements)
    assert copied.values == measurements.values
    assert copied.source == measurements.source
    assert copied.notes == measurements.notes


def test_a_deep_copy_shares_no_mutable_field(measurements: AnnotatedMeasurements):
    copied = copy.deepcopy(measurements)
    assert copied.values is not measurements.values
    assert copied.notes is not measurements.notes


def test_a_deep_copy_shares_a_field_whose_deep_copy_is_itself(
    measurements: AnnotatedMeasurements, metre: UnitOfMeasurement
):
    assert copy.deepcopy(measurements).unit is metre


# %% fields the constructor does not take


def test_a_deep_copy_derives_fields_the_constructor_does_not_take(
    measurements: AnnotatedMeasurements,
):
    measurements.values.append(4.0)
    copied = copy.deepcopy(measurements)
    assert copied.total == sum(measurements.values)
    assert measurements.total != copied.total


# %% objects reached more than once


def test_a_deep_copy_copies_an_object_reached_twice_once(metre: UnitOfMeasurement):
    shared = Measurements(values=[1.0], unit=metre)
    copied = copy.deepcopy(PairOfMeasurements(first=shared, second=shared))
    assert copied.first is copied.second
    assert copied.first is not shared


# %% fields a deep copy keeps


@pytest.fixture
def logged_reading() -> LoggedReading:
    """
    A reading recorded in a logbook that refers back to it.
    """
    logbook = Logbook()
    reading = LoggedReading(value=1.0, notes=["calibrated"], logbook=logbook)
    logbook.readings.append(reading)
    return reading


def test_a_deep_copy_keeps_a_field_marked_shared(logged_reading: LoggedReading):
    """
    A field marked shared refers to the original's object in the copy, even when that
    object refers back to the original.
    """
    assert copy.deepcopy(logged_reading).logbook is logged_reading.logbook


def test_a_deep_copy_still_copies_the_fields_not_marked_shared(
    logged_reading: LoggedReading,
):
    """
    Marking one field shared leaves the other fields copied.
    """
    assert copy.deepcopy(logged_reading).notes is not logged_reading.notes


# %% equality and hashing


def test_the_mixin_keeps_identity_equality_hashable():
    reading = SensorReading(value=1.0)
    assert {reading: "first"}[reading] == "first"


# %% marking


def test_the_dataclass_deep_copy_is_a_mixin():
    assert issubclass(DataclassDeepcopyMixin, Mixin)


def test_the_dataclass_deep_copy_mixin_cannot_be_instantiated():
    with pytest.raises(InterfaceInstantiationError):
        DataclassDeepcopyMixin()
