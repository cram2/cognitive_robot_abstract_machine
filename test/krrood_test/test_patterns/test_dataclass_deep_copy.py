import copy

import pytest

from krrood.patterns.dataclass_deep_copy import DataclassDeepcopyMixin
from krrood.patterns.exceptions import InterfaceInstantiationError
from krrood.patterns.mixin import Mixin

from ..dataset.deep_copied_classes import (
    AnnotatedMeasurements,
    Measurements,
    PairOfMeasurements,
    SensorReading,
    UnitOfMeasurement,
)

# %% fixtures


@pytest.fixture
def metre() -> UnitOfMeasurement:
    return UnitOfMeasurement("m")


@pytest.fixture
def measurements(metre: UnitOfMeasurement) -> AnnotatedMeasurements:
    return AnnotatedMeasurements(
        [1.0, 2.0], metre, source="laser", notes=["calibrated"]
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
    shared = Measurements([1.0], metre)
    copied = copy.deepcopy(PairOfMeasurements(shared, shared))
    assert copied.first is copied.second
    assert copied.first is not shared


# %% equality and hashing


def test_the_mixin_keeps_identity_equality_hashable():
    reading = SensorReading(1.0)
    assert {reading: "first"}[reading] == "first"


# %% marking


def test_the_dataclass_deep_copy_is_a_mixin():
    assert issubclass(DataclassDeepcopyMixin, Mixin)


def test_the_dataclass_deep_copy_mixin_cannot_be_instantiated():
    with pytest.raises(InterfaceInstantiationError):
        DataclassDeepcopyMixin()
