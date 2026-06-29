"""
Shared behavioural contract for any domain.

assert_domain_behaves(module) runs the full pipeline for a domain plugin and
checks: familiar objects pass, anomalous objects are flagged, incomplete
objects are flagged, and the held-out detection / false-positive rates are
good. Every per-domain test (and the auto-discovery test) calls this, so the
behavioural spec is defined in one place.
"""

from experiments.confidence_aware_eql.engine import build_evaluator, evaluate_detection


def assert_domain_behaves(module, *, min_detection=0.95, max_false_positive=0.05):
    domain, spec = module.DOMAIN, module.SPEC
    evaluator, model, strategy, data = build_evaluator(domain, spec, seed=0)

                                     
    for name, obj in getattr(module, "FAMILIAR", []):
        _, w = evaluator.check(obj, node_name=name)
        assert w is None, f"{domain.name}: familiar '{name}' was wrongly flagged"

                                  
    for name, obj in getattr(module, "ANOMALOUS", []):
        _, w = evaluator.check(obj, node_name=name)
        assert w is not None, f"{domain.name}: anomaly '{name}' was missed"

                                                                 
    for name, obj in getattr(module, "INCOMPLETE", []):
        lp, w = evaluator.check(obj, node_name=name)
        assert w is not None, f"{domain.name}: incomplete '{name}' was not flagged"
        assert lp is None, f"{domain.name}: incomplete '{name}' should not be scored"

                        
    tp, fp = evaluate_detection(evaluator, domain, spec)
    assert tp >= min_detection, f"{domain.name}: detection rate {tp:.2%} < {min_detection:.0%}"
    assert fp <= max_false_positive, f"{domain.name}: false-positive {fp:.2%} > {max_false_positive:.0%}"
