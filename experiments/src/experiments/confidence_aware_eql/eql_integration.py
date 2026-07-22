from __future__ import annotations

from dataclasses import dataclass, field

from krrood.entity_query_language.core.base_expressions import (
    OperationResult,
    SymbolicExpression,
)
from krrood.entity_query_language.evaluation_context import EvaluationObserver
from typing_extensions import Any, List, Optional, Set, Tuple, Type

from experiments.confidence_aware_eql.engine.evaluator import ConfidenceAwareEvaluator
from experiments.confidence_aware_eql.exceptions import UnfamiliarSampleWarning


@dataclass
class ConfidenceAwareEvaluationObserver(EvaluationObserver):
    """
    Runs a familiarity check at every node of an entity-query-language tree.

    The observer is registered on an evaluation context so that the query engine
    notifies it as each node of the rule tree is entered. For every node it recovers the
    instance currently bound to the query variable and scores it with the confidence
    evaluator. Instances that are unfamiliar produce an :class:`UnfamiliarSampleWarning`
    naming the node that rejected them, so the deterministic result of the rule tree can
    be accompanied by an explicit statement of doubt. The evaluation itself is not
    altered.
    """

    evaluator: ConfidenceAwareEvaluator
    """The evaluator scoring each bound instance."""

    instance_class: Type
    """
    The class of the instances that should be checked.
    """

    warnings: List[UnfamiliarSampleWarning] = field(default_factory=list)
    """
    Warnings collected during the evaluation, in the order they were raised.
    """

    checked_nodes: Set[Tuple[int, str]] = field(default_factory=set)
    """
    Identity and node-name pairs already checked, preventing duplicate warnings.
    """

    def on_evaluate_enter(
        self, expression: SymbolicExpression, sources: Optional[OperationResult] = None
    ) -> None:
        """
        Check the instance bound at the node the query engine is entering.

        :param expression: The rule-tree node currently being evaluated.
        :param sources: The result carrying the bindings of the enclosing operation, or
            ``None`` when the node is evaluated without bindings.
        """
        instance = self._bound_instance(sources)
        if instance is None:
            return

        node_name = expression._name_
        identity = (id(instance), node_name)
        if identity in self.checked_nodes:
            return
        self.checked_nodes.add(identity)

        result = self.evaluator.check(instance, node_name=node_name)
        if result.warning is not None:
            self.warnings.append(result.warning)

    def _bound_instance(self, sources: Optional[OperationResult]) -> Optional[Any]:
        """
        Return the instance of the observed class bound in the operation result.

        :param sources: The operation result to read the bindings from.
        :return: The bound instance, or ``None`` when the result holds no instance of
            :attr:`instance_class`.
        """
        if sources is None:
            return None
        for value in sources.bindings.values():
            if isinstance(value, self.instance_class):
                return value
        return None
