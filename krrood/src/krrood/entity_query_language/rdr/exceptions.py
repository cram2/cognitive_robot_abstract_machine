"""
Exceptions raised by the EQL-RDR subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any

from krrood.exceptions import DataclassException


# %% branch-semantics dispatch


@dataclass(kw_only=True)
class AmbiguousBranchSemanticsError(DataclassException):
    """
    Two or more branch-semantics classes are equally specific for the same conclusion
    selector, so the winner would otherwise be decided by declaration order.

    Surfaced as an error so an accidental overlap is caught rather than masked.
    """

    selector: object
    """
    The conclusion selector node being dispatched when the collision occurred.
    """

    candidates: list[type]
    """
    The equally-specific branch-semantics classes that collided.
    """

    def error_message(self) -> str:
        names = ", ".join(sorted(candidate.__name__ for candidate in self.candidates))
        return f"{names} are equally specific for {type(self.selector).__name__}."

    def suggest_correction(self) -> str:
        return (
            "Give each class a distinct ``selector``, or have one subclass the other to "
            "declare it the more-specific special case."
        )


# %% underspecified inference targets


@dataclass(kw_only=True)
class NoInferenceTarget(DataclassException):
    """
    Raised when an underspecified ``Match`` has no ``...`` attribute to infer.
    """

    case_type: type
    """
    The type whose instances were queried.
    """

    def error_message(self) -> str:
        return f"{self.case_type.__name__} has no underspecified (`...`) attribute to infer."

    def suggest_correction(self) -> str:
        return "Mark the attribute to infer with `...`, e.g. an(Animal)(species=...)."


@dataclass(kw_only=True)
class MultipleInferenceTargets(DataclassException):
    """
    Raised when a single-class RDR is handed more than one ``...`` attribute.
    """

    attribute_names: list[str]
    """
    The names of the attributes that were all marked with ``...``.
    """

    def error_message(self) -> str:
        return (
            "Single-class RDR infers one attribute, but several were underspecified: "
            f"{self.attribute_names}."
        )

    def suggest_correction(self) -> str:
        return "Use a separate RDR per attribute (or a future MultiClassRDR)."


@dataclass(kw_only=True)
class UnsupportedInferenceTarget(DataclassException):
    """
    Raised when an ``...`` attribute is an unbounded iterable.

    .. note::
        Inferring an unbounded-iterable (multi-valued) attribute is not currently
        supported; it is planned as a future feature of ``MultiClassRDR``.
    """

    case_type: type
    """
    The type whose instances were queried.
    """

    attribute_name: str
    """
    The name of the unbounded-iterable attribute that was marked with ``...``.
    """

    def error_message(self) -> str:
        return (
            f"{self.case_type.__name__}.{self.attribute_name} is an unbounded iterable; "
            "single-class RDR only infers single-valued attributes."
        )

    def suggest_correction(self) -> str:
        return "This will be supported by a future MultiClassRDR."


@dataclass(kw_only=True)
class UndeclaredInferenceTarget(DataclassException):
    """
    Raised when an ``...`` attribute is not a declared field of the case type, so the
    shape of the value to infer is unknown.
    """

    case_type: type
    """
    The type whose instances were queried.
    """

    attribute_name: str
    """
    The name of the attribute that was marked with ``...``.
    """

    def error_message(self) -> str:
        return (
            f"{self.case_type.__name__}.{self.attribute_name} is not a declared field of "
            f"{self.case_type.__name__}, so the value to infer cannot be checked."
        )

    def suggest_correction(self) -> str:
        return (
            f"Make {self.case_type.__name__} a dataclass that declares "
            f"{self.attribute_name} as a field."
        )


# %% rule-tree serialization


@dataclass
class EmptyRuleTreeError(DataclassException):
    """
    Raised when serializing an RDR that has no rules yet.
    """

    def error_message(self) -> str:
        return "Cannot serialize an empty RDR (no rules have been added)."

    def suggest_correction(self) -> str:
        return "Fit at least one rule before saving."
