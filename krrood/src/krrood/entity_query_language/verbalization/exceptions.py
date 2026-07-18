"""
Custom exceptions raised by the EQL verbalization subsystem.

Each is a :class:`~krrood.exceptions.DataclassException` — a dataclass exception that
carries the structured cause as fields and composes its human-readable message from
``error_message()`` / ``suggest_correction()`` — so the failure data is inspectable (not
only a formatted string), consistent with the rest of krrood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression
    from krrood.entity_query_language.core.variable import InstantiatedVariable
    from krrood.entity_query_language.operators.aggregators import Aggregator
    from krrood.entity_query_language.verbalization.fragments.base import (
        VerbalizationFragment,
    )
    from krrood.entity_query_language.verbalization.grammar.conditions.placement import (
        ConditionForm,
    )


@dataclass
class PredicateFragmentRequiredError(DataclassException):
    """
    A :class:`~krrood.entity_query_language.predicate.Verbalizable` predicate was verbalized without
    implementing ``_verbalization_fragment_``. A predicate must state its surface as a structured
    fragment clause (there is no name-based string fallback), so the omission is an error rather than
    a degraded *"a HasHighSalary, where …"* rendering.
    """

    node: "InstantiatedVariable"
    """The instantiated predicate whose type supplies no verbalization fragment."""

    def error_message(self) -> str:
        return (
            f"{self.node._type_.__name__!r} is a predicate but does not implement "
            "`_verbalization_fragment_`, so it has no verbalization."
        )

    def suggest_correction(self) -> str:
        return (
            "Implement `_verbalization_fragment_(cls, fields)` returning a clause built with "
            "`vocabulary.parts_of_speech.clause(Noun(...), Verb(...), …)`."
        )


@dataclass
class NonFragmentPredicateError(DataclassException):
    """
    A predicate's ``_verbalization_fragment_`` returned something that is not a
    :class:`VerbalizationFragment` (e.g. a leftover format string).

    Fragments are required so the surface composes with the realisation passes
    (negation, coreference, morphology).
    """

    predicate_type: type
    """The predicate type whose hook returned a non-fragment."""

    returned: object
    """
    The non-fragment value the hook returned.
    """

    def error_message(self) -> str:
        return (
            f"{self.predicate_type.__name__}._verbalization_fragment_ returned a "
            f"{type(self.returned).__name__}, not a VerbalizationFragment."
        )

    def suggest_correction(self) -> str:
        return (
            "Return a VerbalizationFragment — build the clause with "
            "`vocabulary.parts_of_speech.clause(...)`, not a string template."
        )


@dataclass
class UnverbalizableExpressionError(DataclassException):
    """
    No grammar rule covers an EQL construct — a coverage gap, surfaced as an error
    rather than silently degrading the node to its class name.
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
    A fragment kind with no fold handler reached a renderer — a realisation pass was
    skipped, leaving an un-lowered ``NounPhrase`` or ``PossessiveChain`` in the tree.
    """

    fragment: "VerbalizationFragment"
    """The un-lowered fragment that reached the renderer."""

    def error_message(self) -> str:
        return (
            f"fold_fragment received a {type(self.fragment).__name__}; NounPhrase / "
            "PossessiveChain nodes must be lowered before a renderer folds the tree."
        )

    def suggest_correction(self) -> str:
        return "Run the realisation passes (realize_tree) before rendering."


@dataclass
class UndeclaredFormPositionError(DataclassException):
    """
    A concrete :class:`~krrood.entity_query_language.verbalization.grammar.conditions.pl
    acement.ConditionForm` subclass did not declare its ``position`` class variable —
    caught at class-definition time rather than as a silent ``AttributeError`` deep in
    ``place``.
    """

    form: "type[ConditionForm]"
    """The condition-form subclass missing its ``position``."""

    def error_message(self) -> str:
        return (
            f"{self.form.__name__!r} must declare the `position` class variable — "
            "it sets where the form's output attaches."
        )

    def suggest_correction(self) -> str:
        return "Add e.g. `position = SurfacePosition.WHOSE` to the class body."


@dataclass
class UnknownAggregatorError(DataclassException):
    """
    An aggregator type has no verbalization phrase — a coverage gap surfaced when a new
    :class:`~krrood.entity_query_language.operators.aggregators.Aggregator` subtype is
    added without a matching entry in :class:`~krrood.entity_query_language.verbalizatio
    n.vocabulary.english.Aggregations`.
    """

    aggregator_type: "type[Aggregator]"
    """The aggregator type that maps to no aggregation phrase."""

    def error_message(self) -> str:
        return (
            f"No aggregation phrase for aggregator {self.aggregator_type.__name__!r}."
        )

    def suggest_correction(self) -> str:
        return (
            "Add the aggregator to `_AGGREGATOR_PHRASES` in "
            "verbalization/vocabulary/english.py."
        )


@dataclass
class AmbiguousRuleError(DataclassException):
    """
    Two or more grammar rules (or specificity-ranked forms) are equally specific for the
    same dispatch target — a collision that would otherwise resolve silently by
    registration order.

    Surfaced as an error so an accidental overlap is caught rather than masked.
    """

    subject: object
    """The expression or request being dispatched when the collision occurred."""

    candidates: "list[type]"
    """
    The equally-specific rule/form classes that collided.
    """

    def error_message(self) -> str:
        names = ", ".join(sorted(candidate.__name__ for candidate in self.candidates))
        return f"{names} are equally specific for {self.subject!r}."

    def suggest_correction(self) -> str:
        return (
            "Make the colliding guards mutually exclusive, or have one rule subclass the other "
            "to declare it the more-specific special case."
        )
