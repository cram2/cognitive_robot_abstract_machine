"""
The *policy* half of the expert split: what to ask, and how to validate the answer.

An :class:`Expert` decides which answers a new rule needs — only the *conditions* when the
target conclusion is known (ground-truth fit), or *both* a conclusion and its conditions
when no target is given (the expert labels the case). It owns the validators (conditions
must be a live EQL :class:`SymbolicExpression`; a conclusion must lie in the attribute's
resolved :class:`~krrood.entity_query_language.rdr.conclusion_domain.ConclusionDomain`) and
delegates the actual expert interaction to its :class:`ExpertInterface`.

Answers are live EQL expression objects built over the shared ``case_variable`` — never
strings or lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import CanBehaveLikeAVariable
from krrood.entity_query_language.rdr.conclusion_domain import ConclusionDomain
from krrood.entity_query_language.rdr.interface import (
    CASE_INSTANCE_NAME,
    CASE_VARIABLE_NAME,
    AnswerRequest,
    CaseContext,
    ExpertAbort,
    ExpertInterface,
)
from krrood.entity_query_language.rdr.utils import UNSET

if TYPE_CHECKING:
    from krrood.entity_query_language.rdr.aid import ConclusionAid
    from krrood.entity_query_language.rdr.observer import ClassificationTrace

#: The namespace name the expert assigns their condition expression to.
ANSWER_NAME = "conditions"

#: The namespace name the expert assigns their conclusion to (unknown-target fitting).
CONCLUSION_NAME = "conclusion"


class NoConditionsProvided(Exception):
    """Raised when the session ended without a valid ``conditions`` expression."""


class NoConclusionProvided(Exception):
    """Raised when unknown-target fitting ended without a ``conclusion``."""


def _validate_conditions(value: Any) -> Optional[str]:
    if isinstance(value, SymbolicExpression):
        return None
    if value is None:
        return (
            f"Assign an EQL condition to `{ANSWER_NAME}`, built over `{CASE_VARIABLE_NAME}` "
            f"(e.g. `{ANSWER_NAME} = {CASE_VARIABLE_NAME}.some_attr == True`)."
        )
    return (
        f"`{ANSWER_NAME}` must be a live EQL expression over `{CASE_VARIABLE_NAME}` "
        f"(got {type(value).__name__}). Did you build it over `{CASE_INSTANCE_NAME}` "
        f"(the concrete case) instead of `{CASE_VARIABLE_NAME}`?"
    )


def _domain_hint(domain: ConclusionDomain) -> str:
    """:return: A short clause naming the allowable values (enumerable) or expected type."""
    if domain.is_enumerable:
        return f"one of: {domain.display()}"
    return f"a {domain.type_display}"


def make_conclusion_validator(
    domain: ConclusionDomain, allow_unset: bool
) -> Callable[[Any], Optional[str]]:
    """
    Build a validator for a conclusion answer from its resolved domain.

    The checks layer, in order: an *unset* answer (the ``UNSET`` sentinel) is acceptable only
    when ``allow_unset`` (a current conclusion already stands and is not known to be wrong);
    ``None`` only when the declared type admits it; an enumerable domain requires membership;
    otherwise the value must be an instance of the expected type(s). An unresolved domain
    (no expected types) accepts any non-``None`` value.

    :param domain: The resolved allowable-value domain of the conclusion attribute.
    :param allow_unset: Whether leaving the conclusion unset is acceptable.
    :return: A validator returning an error message, or ``None`` when the value is acceptable.
    """

    def validate(value: Any) -> Optional[str]:
        if value is UNSET:
            if allow_unset:
                return None
            return (
                f"No rule fired for this case — assign a conclusion to `{CONCLUSION_NAME}` "
                f"({_domain_hint(domain)})."
            )
        if value is None:
            if domain.allows_none:
                return None
            return f"`{CONCLUSION_NAME}` may not be None — set {_domain_hint(domain)}."
        if domain.is_enumerable:
            if domain.contains(value):
                return None
            return f"`{CONCLUSION_NAME}` must be one of: {domain.display()} (got {value!r})."
        if domain.expected_types:
            if isinstance(value, domain.expected_types):
                return None
            return (
                f"`{CONCLUSION_NAME}` must be a {domain.type_display} "
                f"(got {type(value).__name__})."
            )
        return None

    return validate


@dataclass
class Expert:
    """Supplies a new rule's answers when the RDR mis/under-classifies a case.

    Holds an :class:`ExpertInterface` that performs the actual expert interaction; this class only
    builds the request specs and translates an :class:`ExpertAbort` into the policy-level
    :class:`NoConditionsProvided` / :class:`NoConclusionProvided`.
    """

    interface: ExpertInterface
    """
    The interface to use to interact with the expert.
    """
    aids: List[ConclusionAid] = field(default_factory=list)
    """Optional task-specific aids consulted while labelling a case: each may present an
    information / visual aid and / or suggest a conclusion (see :class:`ConclusionAid`)."""

    def ask_for_conditions(
        self,
        case: Any,
        case_variable: CanBehaveLikeAVariable,
        target_conclusion: Any,
        current_conclusion: Any = UNSET,
        trace: Optional[ClassificationTrace] = None,
        corner_case: Optional[Any] = None,
        suggestion: Optional[SymbolicExpression] = None,
    ) -> SymbolicExpression:
        """
        :param case: The case being fit (e.g. an ``Animal`` instance).
        :param case_variable: The RDR's shared EQL variable; conditions must be built over it.
        :param target_conclusion: The known correct conclusion.
        :param current_conclusion: What the RDR currently concludes (``_UNSET`` if no rule fired).
        :param trace: The classification trace, for visualizing the rule tree to the expert.
        :param corner_case: The corner case of the firing rule, for side-by-side display.
        :param suggestion: Optional auto-resolved condition to pre-seed; displayed as a hint
            and used as the namespace default so the expert can accept it by pressing CTRL+D
            or overwrite it with any other expression.
        :return: A live EQL condition expression that holds for ``case`` and distinguishes it.
        """
        context = CaseContext(
            case_instance=case,
            case_variable=case_variable,
            current_conclusion=current_conclusion,
            target_conclusion=target_conclusion,
            trace=trace,
            corner_case=corner_case,
            suggested_condition=suggestion,
        )
        request = AnswerRequest(
            name=ANSWER_NAME,
            validate=_validate_conditions,
            example=f"{ANSWER_NAME} = {CASE_VARIABLE_NAME}.some_attr == True",
            default=suggestion,
        )
        try:
            return self.interface.interact(context, [request])[ANSWER_NAME]
        except ExpertAbort:
            raise NoConditionsProvided(
                "The expert cancelled without supplying conditions."
            )

    def ask_for_rule(
        self,
        case: Any,
        case_variable: CanBehaveLikeAVariable,
        conclusion_domain: ConclusionDomain,
        current_conclusion: Any = UNSET,
        trace: Optional[ClassificationTrace] = None,
        corner_case: Optional[Any] = None,
    ) -> Tuple[Any, Optional[SymbolicExpression]]:
        """
        Ask the expert to label the case (no ground truth), then justify the label.

        Sequential: first a focused **conclusion-only** question — the allowable values are
        shown and a valid aid suggestion pre-seeds the answer — then, when the chosen
        conclusion differs from the current one, the **conditions** are requested via
        :meth:`ask_for_conditions` (the full conditions-only flow with the chosen conclusion as
        the target). Leaving the conclusion unset (only permitted when a current conclusion
        already stands) keeps the current conclusion and skips the conditions step.

        :param case: The case being fit.
        :param case_variable: The RDR's shared EQL variable; conditions are built over it.
        :param conclusion_domain: The resolved allowable-value domain of the conclusion attribute.
        :param current_conclusion: What the RDR currently concludes (``UNSET`` if no rule fired).
        :param trace: The classification trace, for visualizing the rule tree to the expert.
        :param corner_case: The corner case of the firing rule, for side-by-side display.
        :return: ``(conclusion, conditions)``; ``conditions`` is ``None`` when the expert kept
            the current conclusion (nothing to insert).
        """
        context = CaseContext(
            case_instance=case,
            case_variable=case_variable,
            current_conclusion=current_conclusion,
            conclusion_domain=conclusion_domain,
            aids=self.aids,
            trace=trace,
            corner_case=corner_case,
        )
        conclusion = self._ask_for_conclusion(context, conclusion_domain)
        if conclusion is UNSET or conclusion == current_conclusion:
            return current_conclusion, None
        conditions = self.ask_for_conditions(
            case, case_variable, conclusion, current_conclusion, trace, corner_case
        )
        return conclusion, conditions

    def _ask_for_conclusion(
        self, context: CaseContext, conclusion_domain: ConclusionDomain
    ) -> Any:
        """Run the focused conclusion-only question; ``UNSET`` means "keep the current one"."""
        validator = make_conclusion_validator(
            conclusion_domain, allow_unset=context.has_current_conclusion
        )
        request = AnswerRequest(
            name=CONCLUSION_NAME,
            validate=validator,
            example=conclusion_domain.example_for(CONCLUSION_NAME),
            default=self._suggested_conclusion(context, validator),
        )
        try:
            return self.interface.interact(context, [request])[CONCLUSION_NAME]
        except ExpertAbort:
            raise NoConclusionProvided(
                "The expert cancelled without supplying a conclusion."
            )

    def _suggested_conclusion(
        self, context: CaseContext, validator: Callable[[Any], Optional[str]]
    ) -> Any:
        """:return: The first aid suggestion that validates, else ``UNSET`` (no pre-seed)."""
        for aid in self.aids:
            suggestion = aid.suggest(context)
            if suggestion is not None and validator(suggestion) is None:
                return suggestion
        return UNSET
