"""
What goes wrong when finding out by trials what makes a behaviour succeed.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any

from krrood.exceptions import DataclassException, InputError

# %% records of numbers


@dataclass(kw_only=True)
class FieldOfUnsupportedType(InputError):
    """
    Raised when a field of a record of numbers, such as the parameters of a trial, is
    neither a number nor a dataclass of numbers.
    """

    owner: type
    """
    The dataclass that declares the field.
    """

    field_name: str
    """
    The name of the field.
    """

    field_type: Any
    """
    The type the field is declared with or holds.
    """

    def error_message(self) -> str:
        return (
            f"The field '{self.field_name}' of {self.owner.__name__} is of type "
            f"{self.field_type}, which is neither a number nor a dataclass."
        )

    def suggest_correction(self) -> str:
        return (
            "Declare the field as a float or an int, nest the numbers in a dataclass, "
            "or keep what is no number in a field of the trial itself."
        )


@dataclass(kw_only=True)
class InformedFieldMissing(InputError):
    """
    Raised when parameters are declared to inform a field that their action does not
    declare.
    """

    owner: type
    """
    The dataclass of the action.
    """

    field_name: str
    """
    The name of the field that it does not declare.
    """

    def error_message(self) -> str:
        return f"{self.owner.__name__} declares no field '{self.field_name}'."

    def suggest_correction(self) -> str:
        return "Name a field the action declares, as it is spelled there."


# %% drawing parameters


@dataclass(kw_only=True)
class ParameterWithoutCentre(InputError):
    """
    Raised when a statement has a parameter that the centres drawn near give no value
    for.
    """

    parameter_name: str
    """
    The name of the parameter, as the statement's variables name it.
    """

    def error_message(self) -> str:
        return f"The centres give no value for the parameter '{self.parameter_name}'."

    def suggest_correction(self) -> str:
        return (
            "Draw from a statement whose parameters are exactly those of the centres' "
            "type, or give centres of the type the statement queries."
        )


@dataclass(kw_only=True)
class NoSuccessfulTrials(InputError):
    """
    Raised when the next trials are to be drawn near successful trials, but none
    succeeded.
    """

    number_of_trials: int
    """
    How many trials there were.
    """

    def error_message(self) -> str:
        return f"None of the {self.number_of_trials} trials succeeded."

    def suggest_correction(self) -> str:
        return (
            "Draw the next trials from a uniform prior, or retry the trials whose "
            "measurements came closest to succeeding."
        )


# %% many trials at once


@dataclass(kw_only=True)
class OutcomeDidNotVaryInPilot(DataclassException):
    """
    Raised when every pilot trial of a campaign came out the same, such as when success
    is judged in a way no trial can pass, so the campaign would learn nothing.
    """

    number_of_trials: int
    """
    How many pilot trials were tried.
    """

    succeeded: bool
    """
    Whether every pilot trial succeeded or every one failed.
    """

    def error_message(self) -> str:
        outcome = "succeeded" if self.succeeded else "failed"
        return f"All {self.number_of_trials} pilot trials {outcome}."

    def suggest_correction(self) -> str:
        return (
            "Check that the judgement of success can come out both ways, for example on "
            "a state where it must pass; if successes are only rare, try more pilot "
            "trials."
        )


# %% what made trials succeed


@dataclass(kw_only=True)
class NoTrials(InputError):
    """
    Raised when trials are analysed without any trial to analyse.
    """

    def error_message(self) -> str:
        return "There are no trials to analyse."

    def suggest_correction(self) -> str:
        return "Analyse at least one trial."
