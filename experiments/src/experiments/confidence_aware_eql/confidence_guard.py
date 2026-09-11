"""Block a rule's conclusions from committing to an out-of-distribution object.

This wires the confidence model into EQL's evaluation pipeline through the
``on_conclusions_processed`` hook of
:class:`~krrood.entity_query_language.evaluation_context.EvaluationObserver`: whenever a
rule's conclusions are processed, the object each conclusion just bound is checked
against a fitted :class:`~experiments.confidence_aware_eql.confidence_model.ConfidenceModel`,
and evaluation is aborted if the object is unfamiliar.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.core.base_expressions import (
    OperationResult,
    SymbolicExpression,
)
from krrood.entity_query_language.evaluation import create_default_evaluation_context
from krrood.entity_query_language.evaluation_context import (
    EvaluationObserver,
    get_evaluation_context,
    set_evaluation_context,
)
from krrood.exceptions import DataclassException
from typing_extensions import Any, Iterator

from experiments.confidence_aware_eql.confidence_model import ConfidenceModel


@dataclass
class UnfamiliarInstanceError(DataclassException):
    """Raised when a rule concludes on an instance the confidence model judges unfamiliar."""

    instance: Any
    """The instance a rule's conclusion bound that was judged unfamiliar."""

    log_likelihood: float
    """The instance's log-likelihood under the confidence model."""

    threshold: float
    """The familiarity threshold the log-likelihood fell below."""

    def error_message(self) -> str:
        return (
            f"{self.instance!r} is unfamiliar under the fitted confidence model: "
            f"log-likelihood {self.log_likelihood} is below threshold {self.threshold}."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ConfidenceGuardObserver(EvaluationObserver):
    """Raises :class:`UnfamiliarInstanceError` for any conclusion bound to an unfamiliar instance."""

    confidence_model: ConfidenceModel
    """The fitted confidence model every concluded instance is checked against."""

    def on_conclusions_processed(
        self, expression: SymbolicExpression, result: OperationResult
    ) -> None:
        """
        Check every instance this rule just concluded and raise on the first unfamiliar one.

        :param expression: The rule whose conclusions were processed.
        :param result: The result carrying the bindings the conclusions just updated.
        :raises UnfamiliarInstanceError: If a conclusion's bound instance is unfamiliar.
        """
        for conclusion in expression._conclusions_:
            instance = result.bindings.get(conclusion.variable._id_)
            if instance is None:
                continue
            log_likelihood = self.confidence_model.log_likelihood_of(instance)
            threshold = self.confidence_model.threshold_for(instance)
            if log_likelihood < threshold:
                raise UnfamiliarInstanceError(instance, log_likelihood, threshold)


def evaluate_with_confidence_guard(
    expression: SymbolicExpression, confidence_model: ConfidenceModel
) -> Iterator[Any]:
    """Evaluate a rule expression with a :class:`ConfidenceGuardObserver` attached.

    Builds the default evaluation context up front (rather than letting ``expression``
    create its own while evaluating) so the extra observer can be appended, and so the
    conditions root can be claimed before evaluation starts; both steps normally happen
    inside :meth:`~krrood.entity_query_language.core.base_expressions.SymbolicExpression._evaluate_`
    itself, which only runs them when it creates its own context.

    :param expression: The rule expression to evaluate under the guard.
    :param confidence_model: The confidence model every concluded object is checked
        against.
    :return: An iterator over the expression's results.
    :raises UnfamiliarInstanceError: If any processed conclusion binds an instance the
        confidence model judges unfamiliar.
    """
    previous_context = get_evaluation_context()
    context = create_default_evaluation_context()
    context.observers.append(ConfidenceGuardObserver(confidence_model))
    context.active_conditions_root.set_active_root_if_not_set(
        expression._conditions_root_, has_condition=expression._has_condition_
    )
    set_evaluation_context(context)
    try:
        yield from expression.evaluate()
    finally:
        set_evaluation_context(previous_context)
