"""
eql_integration.py — plug the confidence check into KRROOD's EQL evaluation.

KRROOD's EQL fires events while it walks a rule tree, through the public
``EvaluationObserver`` interface. We subclass it: every time a node is entered,
we run an out-of-distribution check on the object being evaluated and collect a
warning if it is unfamiliar. NOTHING in KRROOD is modified — we register the
observer through ``set_evaluation_context`` (the same public mechanism the
repo's monitoring experiments use).

NOTE (validate after install): the exact way to pull the "object under
evaluation" out of ``sources`` depends on KRROOD's binding structure at runtime.
``object_extractor`` is therefore a small pluggable function you finalise once
you can print a real ``sources`` object. Everything else is the standard,
documented observer wiring.
"""

from dataclasses import dataclass, field
from typing_extensions import Callable, List, Optional, Dict, Any

from krrood.entity_query_language.evaluation_context import (
    EvaluationObserver,
    EvaluationContext,
    set_evaluation_context,
)

from .engine.evaluator import ConfidenceAwareEvaluator
from .engine.warning import UnfamiliarSampleWarning


                                                                                 
                                                                               
ObjectExtractor = Callable[[Any, Any], Optional[Dict]]


@dataclass
class ConfidenceObserver(EvaluationObserver):
    """Runs an OOD check on each evaluated node and records warnings."""

    evaluator: ConfidenceAwareEvaluator
    object_extractor: ObjectExtractor
    warnings: List[UnfamiliarSampleWarning] = field(default_factory=list)
    seen: set = field(default_factory=set)

    def on_evaluate_enter(self, expression, sources=None) -> None:
        obj = self.object_extractor(expression, sources)
        if obj is None:
            return

        node_name = getattr(expression, "name", None) or type(expression).__name__

                                                                         
        key = (id(obj) if not isinstance(obj, dict) else tuple(sorted(obj.items())), node_name)
        if key in self.seen:
            return
        self.seen.add(key)

        _, warning = self.evaluator.check(obj, node_name=node_name)
        if warning is not None:
            self.warnings.append(warning)


def run_with_confidence(query, evaluator: ConfidenceAwareEvaluator,
                        object_extractor: ObjectExtractor):
    """Evaluate an EQL query while checking confidence at each node.

    Returns (results, warnings). The deterministic EQL results are unchanged;
    warnings is the list of UnfamiliarSampleWarning raised during the walk.
    """
    observer = ConfidenceObserver(evaluator=evaluator, object_extractor=object_extractor)
    context = EvaluationContext(observers=[observer])
    token = set_evaluation_context(context)
    try:
        results = list(query.evaluate())
    finally:
        set_evaluation_context(None)                                                    
    return results, observer.warnings
