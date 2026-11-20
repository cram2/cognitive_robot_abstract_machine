"""
What goes wrong when finding out by trials what makes a behaviour succeed.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException, InputError

# %% fields of actions that trials inform


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


# %% drawing near successful trials


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
