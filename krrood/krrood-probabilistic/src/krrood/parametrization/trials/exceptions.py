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


# %% many trials at once


@dataclass(kw_only=True)
class NotEnoughMemoryForWorkers(DataclassException):
    """
    Raised when the workers of a campaign of trials would need more working memory than
    is free.
    """

    number_of_workers: int
    """
    How many workers were asked for.
    """

    memory_per_worker: int
    """
    How much working memory each worker needs, in bytes.
    """

    free_memory: int
    """
    How much working memory is free, in bytes.
    """

    def error_message(self) -> str:
        return (
            f"{self.number_of_workers} workers of {self.memory_per_worker / 1e9:.1f} GB "
            f"need more than the {self.free_memory / 1e9:.1f} GB of free memory."
        )

    def suggest_correction(self) -> str:
        return (
            f"Use at most {self.free_memory // self.memory_per_worker} workers, or "
            f"free working memory first."
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
