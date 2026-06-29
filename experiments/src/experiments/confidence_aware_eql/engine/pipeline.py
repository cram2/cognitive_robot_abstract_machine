"""
pipeline.py — the generalized pipeline, written ONCE.

build_evaluator(): data -> learned circuit -> data-driven threshold -> evaluator.
evaluate_detection(): held-out detection-rate / false-positive-rate.

Both the CLI runner and the test suite call these, so the pipeline logic lives
in exactly one place. Adding a domain never touches this file.
"""

from typing_extensions import Tuple
import numpy as np

from .domain import Domain
from .circuit_model import CircuitModel
from .threshold import PercentileThreshold
from .evaluator import ConfidenceAwareEvaluator
from .datasets import generate_dataset, sample_objects


def build_evaluator(domain: Domain, spec: dict, *, percentile: float = 1.0,
                    n_per_class: int = 80, seed: int = 0, n_components="auto"):
    """Run steps 2–4 for a domain and return (evaluator, model, strategy, data)."""
    data = generate_dataset(domain, spec, n_per_class=n_per_class, seed=seed)
    model = CircuitModel.fit(data, domain.names, n_components=n_components, seed=seed)
    strategy = PercentileThreshold(percentile=percentile)
    strategy.fit(model.log_likelihood(data))
    evaluator = ConfidenceAwareEvaluator(domain, model, strategy.threshold)
    return evaluator, model, strategy, data


def _first_continuous(domain: Domain) -> str:
    for f in domain.features:
        if f.kind == "continuous":
            return f.name
    return domain.features[0].name


def evaluate_detection(evaluator: ConfidenceAwareEvaluator, domain: Domain, spec: dict,
                       *, n: int = 200, seed: int = 7,
                       anomaly_range: Tuple[float, float] = (20.0, 80.0)) -> Tuple[float, float]:
    """Return (detection_rate, false_positive_rate) on freshly sampled held-out sets.

    Anomalies are familiar objects with their first continuous feature pushed to
    an implausible value (default 20–80, e.g. weight in kg).
    """
    corrupt = _first_continuous(domain)
    familiar = sample_objects(domain, spec, n=n, seed=seed)
    rng = np.random.default_rng(seed + 100)
    anomalies = []
    for o in sample_objects(domain, spec, n=n, seed=seed + 1):
        bad = dict(o)
        bad[corrupt] = float(rng.uniform(*anomaly_range))
        anomalies.append(bad)

    fp = sum(0 if evaluator.is_familiar(o) else 1 for o in familiar) / len(familiar)
    tp = sum(1 if not evaluator.is_familiar(o) else 0 for o in anomalies) / len(anomalies)
    return tp, fp
