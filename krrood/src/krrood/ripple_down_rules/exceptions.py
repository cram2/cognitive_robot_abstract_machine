from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import InputError, DataclassException

if TYPE_CHECKING:
    from krrood.ripple_down_rules.experts import Expert


@dataclass
class RDRLoadError(DataclassException):
    """
    Raised when there is an error loading the RDR model.
    """

    model_name: str
    """
    The name of the model that failed to load.
    """
    model_path: str
    """
    The path to the model that failed to load.
    """

    def error_message(self) -> str:
        return (
            f"Could not load the rdr model {self.model_name} from {self.model_path}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NoSavePathFoundForExpertAnswers(InputError):
    """
    Exception raised when no save path is found for expert answers.
    """

    expert: Expert
    """
    The expert for which no save path is found.
    """

    def error_message(self) -> str:
        return (
            f"No save path found for expert {self.expert}, either provide a path or set the "
            f"answers_save_path attribute."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NoLoadPathFoundForExpertAnswers(InputError):
    """
    Exception raised when no load path is found for expert answers.
    """

    expert: Expert
    """
    The expert for which no load path is found.
    """

    def error_message(self) -> str:
        return (
            f"No load path found for expert {self.expert}, either provide a path or set the "
            f"answers_save_path attribute."
        )

    def suggest_correction(self) -> str:
        return ""
