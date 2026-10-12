"""
A module that defers an import into a method because the imported module imports this
one back.

Mirrors a sensor module offering a source that lives in an optional adapter package:
the adapter imports the sensor, so the sensor may only import the adapter once it is
fully initialized.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Generic, Optional, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

if TYPE_CHECKING:
    from decimal import Decimal

TReading = TypeVar("TReading")


@dataclass
class Reading:
    """
    The value a sensor produces.
    """


@dataclass
class SensorWithGenericReading(Generic[TReading], SubClassSafeGeneric):
    """
    A sensor whose reading type is bound by its subclasses.
    """

    reading: Optional[TReading] = None
    """
    The most recent reading.
    """

    resolution: Optional[Decimal] = None
    """
    The smallest change the sensor distinguishes, annotated with a name that only exists
    while type checking.
    """


@dataclass
class SensorBindingItsReading(SensorWithGenericReading[Reading]):
    """
    A sensor that binds its reading type while this module is still being imported.
    """

    def reading_from_other_module(self) -> Reading:
        """
        :return: A reading from the module that imports this one back.
        """
        from .deferred_import_target import ReadingOfBoundSensor

        return ReadingOfBoundSensor(sensor=self)
