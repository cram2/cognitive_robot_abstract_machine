"""
Mimic measurements and readings whose type parameters a reading derives from the
measurement type it binds, a generic class given type arguments included.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Generic, TypeVar

from krrood.patterns.classproperty import classproperty
from krrood.patterns.subclass_safe_generic import (
    DerivedGenericParameter,
    SubClassSafeGeneric,
)

UnitType = TypeVar("UnitType")
"""
The type naming the unit a measurement is given in.
"""

MagnitudeType = TypeVar("MagnitudeType")
"""
The type of a measurement's magnitude.
"""

MeasurementType = TypeVar("MeasurementType")
"""
The measurement type a reading reads.
"""

# %% measurements generic over their unit and magnitude


@dataclass(kw_only=True)
class Measurement(Generic[UnitType, MagnitudeType], SubClassSafeGeneric):
    """
    A magnitude in some unit.
    """


@dataclass(kw_only=True)
class Length(Measurement[str, MagnitudeType], Generic[MagnitudeType]):
    """
    A length in a unit named by a string, its magnitude type left open.
    """


# %% readings that derive the unit and magnitude from their measurement type


@dataclass(kw_only=True)
class Reading(Generic[MeasurementType], SubClassSafeGeneric):
    """
    A measurement read, whose unit and magnitude the measurement type fixes.
    """

    measurement: MeasurementType
    """
    The measurement read.
    """

    magnitude: MagnitudeType
    """
    The magnitude read.
    """

    @classproperty
    def derived_generic_parameters(cls) -> tuple[DerivedGenericParameter, ...]:
        """
        The unit and the magnitude, read from the measurement type.
        """
        return (
            DerivedGenericParameter(parameter=UnitType, source=MeasurementType),
            DerivedGenericParameter(parameter=MagnitudeType, source=MeasurementType),
        )


@dataclass(kw_only=True)
class LengthReading(Reading[Length[MagnitudeType]], Generic[MagnitudeType]):
    """
    A reading of a length, its magnitude type left open.
    """
