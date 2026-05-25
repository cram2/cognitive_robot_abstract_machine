"""
The *policy* half of the expert split: what to ask, and how to validate the answer.

An :class:`Expert` decides which answers a new rule needs — only the *conditions* when the
target conclusion is known (ground-truth fit), or *both* a conclusion and its conditions
when no target is given (the expert labels the case). It owns the validators (conditions
must be a live EQL :class:`SymbolicExpression`; a conclusion must be non-``None``) and
delegates the actual expert interaction to its :class:`ExpertInterface`.

Answers are live EQL expression objects built over the shared ``case_variable`` — never
strings or lists.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Any, Optional, Tuple

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import CanBehaveLikeAVariable
from krrood.entity_query_language.rdr.interface import (
    CASE_INSTANCE_NAME,
    CASE_VARIABLE_NAME,
    AnswerRequest,
    CaseContext,
    ExpertAbort,
    ExpertInterface
)
from krrood.entity_query_language.rdr.utils import UNSET

if TYPE_CHECKING:
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


def _validate_conclusion(value: Any) -> Optional[str]:
    if value is not None:
        return None
    return f"Assign the correct conclusion to `{CONCLUSION_NAME}`."


@dataclass
class Expert:
    """Supplies a new rule's answers when the RDR mis/under-classifies a case.

    Holds an :class:`ExpertInterface` that performs the actual expert interaction; this class only
    builds the request specs and translates an :class:`ExpertAbort` into the policy-level
    :class:`NoConditionsProvided` / :class:`NoConclusionProvided`.
    """

    interface: ExpertInterface

    def ask_for_conditions(self, case: Any, case_variable: CanBehaveLikeAVariable, target_conclusion: Any,
                           current_conclusion: Any = UNSET, trace: Optional[ClassificationTrace] = None) -> SymbolicExpression:
        """
        :param case: The case being fit (e.g. an ``Animal`` instance).
        :param case_variable: The RDR's shared EQL variable; conditions must be built over it.
        :param target_conclusion: The known correct conclusion.
        :param current_conclusion: What the RDR currently concludes (``_UNSET`` if no rule fired).
        :param trace: The classification trace, for visualizing the rule tree to the expert.
        :return: A live EQL condition expression that holds for ``case`` and distinguishes it.
        """
        context = CaseContext(
            case_instance=case,
            case_variable=case_variable,
            current_conclusion=current_conclusion,
            target_conclusion=target_conclusion,
            trace=trace,
        )
        request = AnswerRequest(
            name=ANSWER_NAME,
            validate=_validate_conditions,
            example=f"{ANSWER_NAME} = {CASE_VARIABLE_NAME}.some_attr == True",
        )
        try:
            return self.interface.interact(context, [request])[ANSWER_NAME]
        except ExpertAbort:
            raise NoConditionsProvided(
                "The expert cancelled without supplying conditions."
            )

    def ask_for_rule(self, case: Any, case_variable: CanBehaveLikeAVariable, current_conclusion: Any = UNSET,
                     trace: Optional[ClassificationTrace] = None) -> Tuple[Any, SymbolicExpression]:
        """
        Ask the expert for **both** a conclusion and its conditions, for fitting when no
        ground-truth target is supplied (the expert labels the case).

        :param case: The case being fit.
        :param case_variable: The RDR's shared EQL variable; conditions are built over it.
        :param current_conclusion: What the RDR currently concludes (``_UNSET`` if no rule fired).
        :param trace: The classification trace, for visualizing the rule tree to the expert.
        :return: ``(conclusion, conditions)`` — the value to conclude and a live EQL
            condition expression over ``case_variable`` that justifies it.
        """
        context = CaseContext(
            case_instance=case,
            case_variable=case_variable,
            current_conclusion=current_conclusion,
            trace=trace,
        )
        requests = [
            AnswerRequest(
                name=CONCLUSION_NAME,
                validate=_validate_conclusion,
                example=f"{CONCLUSION_NAME} = SomeValue",
            ),
            AnswerRequest(
                name=ANSWER_NAME,
                validate=_validate_conditions,
                example=f"{ANSWER_NAME} = {CASE_VARIABLE_NAME}.some_attr == True",
            ),
        ]
        try:
            answers = self.interface.interact(context, requests)
        except ExpertAbort as abort:
            if CONCLUSION_NAME in abort.missing:
                raise NoConclusionProvided(
                    "The expert cancelled without supplying a conclusion."
                )
            raise NoConditionsProvided(
                "The expert cancelled without supplying conditions."
            )
        return answers[CONCLUSION_NAME], answers[ANSWER_NAME]
