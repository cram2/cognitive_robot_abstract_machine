"""
Example of a PER-DOMAIN test file.

The auto-discovery test already covers kitchen via the standard contract. This
file shows the pattern for when you want EXTRA, domain-specific assertions —
copy it to tests/test_<your_domain>.py and add your own checks. To add a domain
you do NOT have to write this; the auto test covers it. This is only for custom
checks.
"""

import experiments.confidence_aware_eql.domains.kitchen as kitchen
from experiments.confidence_aware_eql.engine import build_evaluator
from experiments.confidence_aware_eql.tests._assertions import assert_domain_behaves


def test_kitchen_standard_contract():
    assert_domain_behaves(kitchen)


def test_kitchen_specific_50kg_cup_is_extreme():
    """A domain-specific check: the 50 kg cup should be MASSIVELY unlikely."""
    evaluator, model, strategy, data = build_evaluator(kitchen.DOMAIN, kitchen.SPEC, seed=0)
    lp, w = evaluator.check({"weight": 50.0, "size": 0.10, "material": "glass"})
    assert w is not None
    assert lp < strategy.threshold - 100, "impossible cup should be far below threshold"
