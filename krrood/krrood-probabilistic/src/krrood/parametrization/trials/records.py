"""
What one trial of a behaviour varies by, what was measured while it ran, and whether it
succeeded.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Any, Generic, TypeVar

from krrood.parametrization.trials.exceptions import InformedFieldMissing
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

if TYPE_CHECKING:
    from krrood.entity_query_language.core.mapped_variable import Attribute

# %% fields of actions that trials inform


@dataclass(kw_only=True)
class InformedField:
    """
    A field of an action whose default a parameter of trials informs, so that the value
    that worked best becomes that default.
    """

    parameter: Attribute
    """
    The parameter, reached from a variable of the parameters, such as
    ``variable(ThrowParameters, domain=[]).angle``.
    """

    informed: Attribute
    """
    The field of the action, reached from a variable of the action.
    """

    def __post_init__(self) -> None:
        """
        :raises InformedFieldMissing: If the parameter or the field is not declared by
            the dataclass it is reached from, such as when it is misspelt.
        """
        for attribute in [self.parameter, self.informed]:
            if attribute._attribute_name_ not in {
                declared.name
                for declared in dataclasses.fields(attribute._owner_class_)
            }:
                raise InformedFieldMissing(
                    owner=attribute._owner_class_,
                    field_name=attribute._attribute_name_,
                )

    @property
    def current_default(self) -> Any:
        """
        The default the field has now, or :data:`dataclasses.MISSING` when it has none.
        """
        [informed] = [
            declared
            for declared in dataclasses.fields(self.informed._owner_class_)
            if declared.name == self.informed._attribute_name_
        ]
        if informed.default_factory is not dataclasses.MISSING:
            return informed.default_factory()
        return informed.default


@dataclass(kw_only=True)
class AdoptedValue:
    """
    The value of a parameter that worked best, to become the default of the field of an
    action it informs.
    """

    informed_field: InformedField
    """
    The field whose default the parameter informs.
    """

    value: Any
    """
    The value of the parameter.
    """


# %% what a trial varies by and measures


@dataclass(kw_only=True)
class TrialParameters:
    """
    Everything one trial of a behaviour varies by, each a feature of the trial.
    """


@dataclass(kw_only=True)
class TrialMeasurements:
    """
    What was measured while a trial ran, such as the pose actually reached or the
    contacts made, which may mediate between the parameters and the outcome.
    """


ParametersTypeVar = TypeVar("ParametersTypeVar", bound=TrialParameters)
"""
The parameters a trial varies by.
"""

MeasurementsTypeVar = TypeVar("MeasurementsTypeVar", bound=TrialMeasurements)
"""
What is measured while a trial runs.
"""

# %% one trial


@dataclass(kw_only=True)
class Trial(Generic[ParametersTypeVar, MeasurementsTypeVar], SubClassSafeGeneric):
    """
    One try of a behaviour, with what came of it.

    A subclass binds the parameters and the measurements, and may keep more, such as the
    stage that failed.
    """

    parameters: ParametersTypeVar
    """
    What the trial varied by.
    """

    measurements: MeasurementsTypeVar
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


TrialTypeVar = TypeVar("TrialTypeVar", bound=Trial)
"""
A trial, with what came of it.
"""
