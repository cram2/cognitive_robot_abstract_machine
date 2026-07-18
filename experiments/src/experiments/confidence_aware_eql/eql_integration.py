from dataclasses import dataclass, field

from typing_extensions import Any, List, Optional, Type

from krrood.entity_query_language.evaluation_context import EvaluationObserver

from experiments.confidence_aware_eql.engine.evaluator import ConfidenceAwareEvaluator
from experiments.confidence_aware_eql.exceptions import UnfamiliarSampleWarning


@dataclass
class ConfidenceAwareEvaluationObserver(EvaluationObserver):
    """Runs a familiarity check at every node of an entity-query-language tree.

    The observer is registered on an evaluation context so that the query engine
    notifies it as each node of the rule tree is entered. For every node it
    recovers the instance currently bound to the query variable and scores it
    with the confidence evaluator. Instances that are unfamiliar produce an
    :class:`UnfamiliarSampleWarning` naming the node that rejected them, so the
    deterministic result of the rule tree can be accompanied by an explicit
    statement of doubt.
    """

    evaluator: ConfidenceAwareEvaluator
    """The evaluator scoring each bound instance."""

    instance_class: Type
    """The class of the instances that should be checked."""

    warnings: List[UnfamiliarSampleWarning] = field(default_factory=list)
    """Warnings collected during the evaluation, in the order they were raised."""

    _checked_nodes: set = field(default_factory=set, repr=False)
    """Instance and node pairs already checked, preventing duplicate warnings."""

    def on_evaluate_enter(self, expression: Any, sources: Optional[Any] = None) -> None:
        """Check the instance bound at the node the query engine is entering."""
        instance = self._bound_instance(sources)
        if instance is None:
            return

        node_name = self._node_name(expression)
        identity = (id(instance), node_name)
        if identity in self._checked_nodes:
            return
        self._checked_nodes.add(identity)

        result = self.evaluator.check(instance, node_name=node_name)
        if result.warning is not None:
            self.warnings.append(result.warning)

    def _bound_instance(self, sources: Optional[Any]) -> Optional[Any]:
        """Return the instance bound in the operation result, if there is one."""
        if sources is None:
            return None
        bindings = getattr(sources, "bindings", None)
        if not bindings:
            return None
        for value in bindings.values():
            if isinstance(value, self.instance_class):
                return value
        return None

    @staticmethod
    def _node_name(expression: Any) -> str:
        """Return a readable name for the rule-tree node being evaluated."""
        name = getattr(expression, "name", None)
        if name:
            return str(name)
        return type(expression).__name__
