"""
Engine-level tests — core invariants, independent of any domain.

Checks the things that must always hold: the compiled circuit's likelihood
matches the underlying GMM (i.e. variable-order alignment is correct), the
percentile threshold strategy behaves, and missing features are handled.
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from experiments.confidence_aware_eql.engine import (
    Domain, Feature, CircuitModel, PercentileThreshold, ConfidenceAwareEvaluator,
    generate_dataset,
)


def _toy_domain():
    return Domain("toy", [
        Feature("a", "continuous"),
        Feature("b", "continuous"),
        Feature("c", "continuous"),
    ])


def test_circuit_matches_gmm():
    """The compiled circuit must give the SAME log-likelihood as the GMM.

    This is the regression test for the variable-ordering bug: the circuit
    sorts variables, so inputs must be aligned. CircuitModel does this
    internally — verify it.
    """
    rng = np.random.default_rng(0)
    data = np.hstack([
        rng.normal(0.2, 0.05, (80, 1)),
        rng.normal(0.1, 0.02, (80, 1)),
        rng.normal(3.0, 0.40, (80, 1)),
    ])
    names = ["a", "b", "c"]                                                          
    model = CircuitModel.fit(data, names, n_components=2, seed=0)

    gmm = GaussianMixture(n_components=2, covariance_type="diag",
                          random_state=0, reg_covar=1e-4).fit(data)

    test = np.array([[0.2, 0.1, 3.0], [50.0, 0.1, 3.0]])
    assert np.allclose(model.log_likelihood(test), gmm.score_samples(test), atol=1e-5)


def test_percentile_threshold_is_a_percentile():
    lls = np.linspace(-100, 0, 1001)
    thr = PercentileThreshold(percentile=1.0)
    value = thr.fit(lls)
    assert np.isclose(value, np.percentile(lls, 1.0))


def test_missing_feature_is_flagged_not_scored():
    domain = _toy_domain()
    spec = {"x": {"a": (0.2, 0.05), "b": (0.1, 0.02), "c": (3.0, 0.4)}}
    data = generate_dataset(domain, spec, n_per_class=60, seed=0)
    model = CircuitModel.fit(data, domain.names, n_components=1, seed=0)
    thr = PercentileThreshold(1.0); thr.fit(model.log_likelihood(data))
    ev = ConfidenceAwareEvaluator(domain, model, thr.threshold)

    lp, w = ev.check({"a": 0.2, "b": 0.1, "c": None})             
    assert lp is None and w is not None and "incomplete" in w.reason


def test_marginal_reduces_dimensionality():
    domain = _toy_domain()
    spec = {"x": {"a": (0.2, 0.05), "b": (0.1, 0.02), "c": (3.0, 0.4)}}
    data = generate_dataset(domain, spec, n_per_class=60, seed=0)
    model = CircuitModel.fit(data, domain.names, n_components=1, seed=0)
    sub = model.marginal(["a"])
                                                                   
    val = float(sub.log_likelihood(np.array([[0.2]]))[0])
    assert np.isfinite(val)
