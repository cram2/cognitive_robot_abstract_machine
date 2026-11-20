# Derived Generic Parameters

A generic class built on `SubClassSafeGeneric` binds each of its type parameters in a
subclass, and the fields typed with a parameter narrow to the bound type. Often one bound
type already fixes another parameter: a measurement type fixes its unit, a vector type
fixes the kind of frame it is given in. Binding both lets them disagree.

A class declares such a parameter as derived instead of taking it. It binds only the most
specific type it depends on, and the rest is read from that type:

```python
from dataclasses import dataclass

from typing_extensions import Generic, TypeVar

from krrood.patterns.classproperty import classproperty
from krrood.patterns.subclass_safe_generic import (
    DerivedGenericParameter,
    SubClassSafeGeneric,
)

UnitType = TypeVar("UnitType")
MeasurementType = TypeVar("MeasurementType")


@dataclass(kw_only=True)
class Measurement(Generic[UnitType], SubClassSafeGeneric):
    unit: UnitType


@dataclass(kw_only=True)
class Metres(Measurement[str]):
    pass


@dataclass(kw_only=True)
class Reading(Generic[MeasurementType], SubClassSafeGeneric):
    measurement: MeasurementType
    unit: UnitType

    @classproperty
    def derived_generic_parameters(cls) -> tuple[DerivedGenericParameter, ...]:
        return (DerivedGenericParameter(parameter=UnitType, source=MeasurementType),)


@dataclass(kw_only=True)
class LengthReading(Reading[Metres]):
    pass


assert LengthReading.get_type_of_generic_parameter(parameter=UnitType) is str
```

`LengthReading` names the unit nowhere, yet reports it as bound and its `unit` field is
typed `str`. This is what other languages call an *associated type*: `Reading[M]`'s unit
is `M`'s unit.

## Which parameter to bind

Derive only where the bound type determines the derived one. A vector type fixes one kind
of frame, but a kind of frame has a vector type for each coordinate type, so a point
binds its vector type and derives its frame, never the other way round.

## Conflicts

A class that binds a derived parameter through another base, to a type that is neither
the derived type nor a base of it, raises `DerivedGenericParameterConflictError` where it
is defined.

## Limits

Derived parameters exist at runtime only. A static type checker sees the parameter
unbound, since Python's type system cannot express a parameter read from another one.
