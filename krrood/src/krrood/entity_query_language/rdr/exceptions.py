"""
Exceptions raised by the EQL-RDR subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List, Type

from krrood.exceptions import DataclassException


@dataclass
class AmbiguousBranchSemanticsError(DataclassException):
    """
    Two or more branch-semantics classes are equally specific for the same conclusion
    selector, so the winner would otherwise be decided by declaration order.

    Surfaced as an error so an accidental overlap is caught rather than masked.
    """

    selector: object
    """The conclusion selector node being dispatched when the collision occurred."""

    candidates: List[Type]
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
