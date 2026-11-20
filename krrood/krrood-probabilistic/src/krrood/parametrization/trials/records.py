"""
What one trial of a behaviour varies by, what was measured while it ran, and whether it
succeeded.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from typing_extensions import Any, Generic, Self, TypeVar, get_type_hints

from krrood.entity_query_language.factories import a
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.trials.exceptions import (
    FieldOfUnsupportedType,
    InformedFieldMissing,
)
from krrood.patterns.classproperty import classproperty
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

# %% records of numbers


@dataclass(kw_only=True)
class RecordOfNumbers:
    """
    Numbers, directly or in nested dataclasses of numbers.
    """

    def values_by_attribute_path(self) -> dict[str, float]:
        """
        :return: Every number, keyed by the path of attributes that leads to it from
            this object, such as ``stand_offset.forward``.
        :raises FieldOfUnsupportedType: If a field holds neither a number nor a
            dataclass.
        """
        return self.numbers_below(values=dataclasses.asdict(self), prefix="")

    @classmethod
    def numbers_below(cls, *, values: dict[str, Any], prefix: str) -> dict[str, float]:
        """
        :param values: The fields of the record or of a dataclass nested in it, by
            name, with nested dataclasses as dictionaries of their fields.
        :param prefix: The path of attributes that leads to the fields, ending in a dot,
            or empty for the record itself.
        :return: Every number among the fields, keyed by its path of attributes.
        :raises FieldOfUnsupportedType: If a field holds neither a number nor a
            dataclass.
        """
        numbers = {}
        for name, value in values.items():
            path = f"{prefix}{name}"
            if isinstance(value, dict):
                numbers.update(cls.numbers_below(values=value, prefix=f"{path}."))
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FieldOfUnsupportedType(
                    owner=cls, field_name=path, field_type=type(value)
                )
            numbers[path] = float(value)
        return numbers

    @classmethod
    def pattern(cls, *, marker: Any = ...) -> Match[Self]:
        """
        :param marker: What every number is matched with, such as ``...`` to leave it
            to be drawn or ``cause`` to search it as a cause.
        :return: The pattern of the record, with every number matched with the marker.
        :raises FieldOfUnsupportedType: If a field is neither a number nor a
            dataclass.
        """
        return cls.pattern_of_numbers(record_type=cls, marker=marker)

    @classmethod
    def pattern_of_numbers(cls, *, record_type: type, marker: Any) -> Match:
        """
        :param record_type: The record or a dataclass nested in it.
        :param marker: What every number is matched with.
        :return: The pattern of the dataclass, with every number matched with the
            marker.
        :raises FieldOfUnsupportedType: If a field is neither a number nor a
            dataclass.
        """
        hints = get_type_hints(record_type)
        assigned = {}
        for record_field in dataclasses.fields(record_type):
            hint = hints[record_field.name]
            if dataclasses.is_dataclass(hint):
                assigned[record_field.name] = cls.pattern_of_numbers(
                    record_type=hint, marker=marker
                )
                continue
            if hint not in (float, int):
                raise FieldOfUnsupportedType(
                    owner=record_type, field_name=record_field.name, field_type=hint
                )
            assigned[record_field.name] = marker
        return a(record_type)(**assigned)


@dataclass(kw_only=True)
class InformedField:
    """
    A field of an action whose default a parameter of trials informs, so that the value
    that worked best becomes that default.
    """

    attribute_path: str
    """
    The parameter, as :meth:`RecordOfNumbers.values_by_attribute_path` names it.
    """

    owner: type
    """
    The dataclass of the action that declares the field.
    """

    field_name: str
    """
    The name of the field.
    """

    def __post_init__(self) -> None:
        """
        :raises InformedFieldMissing: If :attr:`owner` declares no field of that name.
        """
        if self.field_name not in {
            declared.name for declared in dataclasses.fields(self.owner)
        }:
            raise InformedFieldMissing(owner=self.owner, field_name=self.field_name)

    @property
    def current_default(self) -> Any:
        """
        The default the field has now, or :data:`dataclasses.MISSING` when it has none.
        """
        [informed] = [
            declared
            for declared in dataclasses.fields(self.owner)
            if declared.name == self.field_name
        ]
        if informed.default_factory is not dataclasses.MISSING:
            return informed.default_factory()
        return informed.default


@dataclass(kw_only=True)
class TrialParameters(RecordOfNumbers):
    """
    Everything one trial of a behaviour varies by.
    """

    @classproperty
    def informed_fields(cls) -> list[InformedField]:
        """
        The fields of actions whose defaults the parameters inform; none by default.
        """
        return []


@dataclass(kw_only=True)
class TrialMeasurements(RecordOfNumbers):
    """
    What was measured while a trial ran, such as the pose actually reached or the
    contacts made, which may mediate between the parameters and the outcome.
    """


TParameters = TypeVar("TParameters", bound=TrialParameters)
"""
The parameters a trial varies by.
"""

TMeasurements = TypeVar("TMeasurements", bound=TrialMeasurements)
"""
What is measured while a trial runs.
"""

# %% one trial


@dataclass(kw_only=True)
class Trial(Generic[TParameters, TMeasurements], SubClassSafeGeneric):
    """
    One try of a behaviour, with what came of it.

    A subclass binds the parameters and the measurements, and may keep more, such as the
    stage that failed.
    """

    parameters: TParameters
    """
    What the trial varied by.
    """

    measurements: TMeasurements
    """
    What was measured while the trial ran.
    """

    succeeded: bool
    """
    Whether the trial did what it was tried for.
    """

    starting_state_id: int | None = None
    """
    The database id of the state the trial started from; ``None`` while it is not
    stored.
    """

    def values_by_attribute_path(self) -> dict[str, float]:
        """
        :return: Every number of the parameters and the measurements, keyed by its path
            of attributes from the trial, such as ``parameters.release.height``.
        """
        return {
            f"{record_name}.{path}": value
            for record_name, record in [
                ("parameters", self.parameters),
                ("measurements", self.measurements),
            ]
            for path, value in record.values_by_attribute_path().items()
        }


TTrial = TypeVar("TTrial", bound=Trial)
"""
A trial, with what came of it.
"""
