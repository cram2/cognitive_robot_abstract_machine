from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.core.base_expressions import (
    OperationResult,
    SymbolicExpression,
)
from krrood.entity_query_language.evaluation_context import EvaluationObserver
from typing_extensions import Any, Optional, Type

from experiments.confidence_aware_eql.engine.evaluator import ConfidenceAwareEvaluator
from experiments.confidence_aware_eql.exceptions import UnfamiliarSampleException


@dataclass
class ConfidenceAwareEvaluationObserver(EvaluationObserver):
    """Raises when a rule fires for an out-of-distribution instance.

    The observer is registered on an evaluation context so the query engine
    notifies it after a rule's conclusions are processed, which happens only when
    the rule's conditions hold. It recovers the instance bound by the rule and
    scores it with the confidence evaluator. When the instance is unfamiliar it
    raises an :class:`UnfamiliarSampleException` naming the rule node, so the
    caller can intercept the unfamiliar case rather than act on the rule's
    deterministic conclusion.
    """

    evaluator: ConfidenceAwareEvaluator
    """The evaluator scoring the bound instance."""

    instance_class: Type
    """The class of the instances that should be checked."""

    def on_conclusions_processed(
        self, expression: SymbolicExpression, result: OperationResult
    ) -> None:
        """Check the instance a fired rule reasoned about, raising when unfamiliar.

        :param expression: The rule expression whose conclusions were processed.
        :param result: The operation result carrying the rule's bindings.
        :raises UnfamiliarSampleException: When the bound instance is unfamiliar.
        """
        instance = self._bound_instance(result)
        if instance is None:
            return
        node_name = expression._name_
        check = self.evaluator.check(instance, node_name=node_name)
        if not check.is_familiar:
            raise UnfamiliarSampleException(
                node_name=node_name,
                log_likelihood=check.log_likelihood,
                threshold=self.evaluator.threshold.value,
            )

    def _bound_instance(self, result: Optional[OperationResult]) -> Optional[Any]:
        """Return the instance of the observed class bound in the result.

        :param result: The operation result to read the bindings from.
        :return: The bound instance, or ``None`` when the result holds none of the
            observed class.
        """
        if result is None:
            return None
        for value in result.bindings.values():
            if isinstance(value, self.instance_class):
                return value
        return None