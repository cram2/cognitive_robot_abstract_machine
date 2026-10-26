from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Self

from krrood.patterns.dataclass_deep_copy import DataclassDeepcopyMixin

# %% immutable values a deep copy shares


@dataclass
class UnitOfMeasurement:
    """
    An immutable value that every deep copy shares.
    """

    symbol: str
    """
    The symbol the unit is written with.
    """

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self


# %% dataclasses copied through their constructor


@dataclass
class Measurements(DataclassDeepcopyMixin):
    """
    Measured values whose total the constructor derives.
    """

    values: list[float]
    """
    The measured values.
    """

    unit: UnitOfMeasurement
    """
    The unit every value is measured in.
    """

    source: str = field(kw_only=True, default="sensor")
    """
    The name of what took the measurements.
    """

    total: float = field(init=False)
    """
    The sum of the values, derived when constructed.
    """

    def __post_init__(self):
        self.total = sum(self.values)


@dataclass
class AnnotatedMeasurements(Measurements):
    """
    Measurements that carry notes about how they were taken.
    """

    notes: list[str] = field(default_factory=list)
    """
    The notes about the measurements.
    """


@dataclass
class PairOfMeasurements(DataclassDeepcopyMixin):
    """
    Two measurements that may be the same object.
    """

    first: Measurements
    """
    The first measurements.
    """

    second: Measurements
    """
    The second measurements.
    """


# %% identity-compared dataclasses


@dataclass(eq=False)
class SensorReading(DataclassDeepcopyMixin):
    """
    A reading that is equal only to itself.
    """

    value: float
    """
    The value read.
    """
