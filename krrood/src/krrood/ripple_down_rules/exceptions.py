from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, TYPE_CHECKING

from krrood.exceptions import InputError, DataclassException

if TYPE_CHECKING:
    from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
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
        return f"Could not load the rdr model {self.model_name} from {self.model_path}"

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


@dataclass
class NonInteractiveTerminalError(InputError):
    """
    Raised when the embedded Ipython expert shell is started without an interactive
    terminal attached to stdin, since it would then have no way to prompt for input.
    """

    def error_message(self) -> str:
        return (
            "Cannot start the embedded Ipython expert shell: stdin is not an interactive "
            "terminal, so the shell cannot prompt for input."
        )

    def suggest_correction(self) -> str:
        return (
            "Provide expert answers programmatically, e.g. via "
            "Human(use_loaded_answers=True), instead of relying on an interactive prompt in "
            "a non-interactive environment such as CI."
        )


@dataclass
class NoDistinguishingAttributeFound(InputError):
    """
    Raised by :class:`krrood.ripple_down_rules.experts.Oracle` when none of a case's
    simple (bool/int/float/str/None) attributes differ from the corner case it conflicts
    with, so no differentiating condition can be constructed for it.
    """

    case: Any
    """
    The case being classified.
    """
    corner_case: Any
    """
    The corner case that the rule under refinement was created from.
    """

    def error_message(self) -> str:
        return (
            f"Could not find any simple attribute that differentiates {self.case} from "
            f"its corner case {self.corner_case}."
        )

    def suggest_correction(self) -> str:
        return (
            "Add a simple (bool/int/float/str) attribute to the case that differs from "
            "the corner case, or provide this answer manually instead of using the "
            "Oracle expert."
        )


@dataclass
class OracleCannotInferConclusionWithoutTarget(InputError):
    """
    Raised by :class:`krrood.ripple_down_rules.experts.Oracle` when asked for a
    conclusion on a case query with no known target value: the oracle only ever
    describes a target it is already given, it never guesses one.
    """

    case_query: CaseQuery
    """
    The case query with no known target value.
    """

    def error_message(self) -> str:
        return (
            f"The Oracle expert cannot infer a conclusion for {self.case_query} since it "
            f"has no known target value."
        )

    def suggest_correction(self) -> str:
        return (
            "Provide a target value on the case query (CaseQuery._target), or use a "
            "different expert that can genuinely infer conclusions, such as Human or AI."
        )
