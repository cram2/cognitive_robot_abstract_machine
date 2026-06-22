"""
Custom exceptions raised by the EQL verbalization subsystem.

Each is a :class:`~krrood.exceptions.DataclassException` — a dataclass exception that carries the
structured cause as fields and composes its human-readable message from ``error_message()`` /
``suggest_correction()`` — so the failure data is inspectable (not only a formatted string),
consistent with the rest of krrood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression
    from krrood.entity_query_language.verbalization.fragments.base import Fragment
    from krrood.entity_query_language.verbalization.grammar.conditions.forms import (
        ConditionForm,
    )


@dataclass
class UnverbalizableExpressionError(DataclassException):
    """
    No grammar rule covers an EQL construct — a coverage gap, surfaced as an error rather than
    silently degrading the node to its class name.
    """

    node: "SymbolicExpression"
    """The EQL expression no grammar rule matched."""

    def error_message(self) -> str:
        return (
            f"No verbalization rule for {type(self.node).__name__!r} "
            f"(name={getattr(self.node, '_name_', None)!r})."
        )

    def suggest_correction(self) -> str:
        return (
            "Add a PhraseRule in the construct's grammar/<construct>/rules.py module."
        )


@dataclass
class UnloweredFragmentError(DataclassException):
    """
    A fragment kind with no fold handler reached a renderer — a realisation pass was skipped,
    leaving an un-lowered ``NounPhrase`` or ``PossessiveChain`` in the tree.
    """

    fragment: "Fragment"
    """The un-lowered fragment that reached the renderer."""

    def error_message(self) -> str:
        return (
            f"fold_fragment received a {type(self.fragment).__name__}; NounPhrase / "
            "PossessiveChain nodes must be lowered before a renderer folds the tree."
        )

    def suggest_correction(self) -> str:
        return "Run the realisation passes (realize_tree) before rendering."


@dataclass
class UndeclaredFormSlotError(DataclassException):
    """
    A concrete :class:`~krrood.entity_query_language.verbalization.grammar.conditions.forms.ConditionForm`
    subclass did not declare its ``slot`` class variable — caught at class-definition time rather
    than as a silent ``AttributeError`` deep in ``place``.
    """

    form: "type[ConditionForm]"
    """The condition-form subclass missing its ``slot``."""

    def error_message(self) -> str:
        return (
            f"{self.form.__name__!r} must declare the `slot` class variable — "
            "it sets where the form's output attaches."
        )

    def suggest_correction(self) -> str:
        return "Add e.g. `slot = Slot.WHOSE` to the class body."
