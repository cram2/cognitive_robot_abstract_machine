"""
A module importing a name back from :mod:`deferred_import_owner` at module level, which
is why that module only imports this one inside a method.
"""

from __future__ import annotations

from dataclasses import dataclass

from .deferred_import_owner import Reading, SensorBindingItsReading


@dataclass
class ReadingOfBoundSensor(Reading):
    """
    A reading that remembers the sensor it was taken by.
    """

    sensor: SensorBindingItsReading
    """
    The sensor that took this reading.
    """
