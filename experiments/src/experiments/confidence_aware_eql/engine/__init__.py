"""
confidence_engine — a domain-agnostic confidence-aware evaluation engine.

Learns a tractable probabilistic circuit FROM data for any domain, sets a
data-driven threshold, and flags anomalous / incomplete objects with traceable
warnings. The circuit supports log_likelihood / marginal / conditional (the
same kind of object a JPT produces), learned without the `jpt` package.

To add a DOMAIN: drop a file in domains/  (no edits here).
To add TESTS:    drop a file in tests/    (no edits here).
Pick a domain at runtime:  python run.py
"""

from .domain import Domain, Feature
from .circuit_model import CircuitModel
from .threshold import PercentileThreshold, StdDevThreshold
from .warning import UnfamiliarSampleWarning
from .evaluator import ConfidenceAwareEvaluator
from .datasets import generate_dataset, sample_objects
from .pipeline import build_evaluator, evaluate_detection

__all__ = [
    "Domain", "Feature", "CircuitModel",
    "PercentileThreshold", "StdDevThreshold",
    "UnfamiliarSampleWarning", "ConfidenceAwareEvaluator",
    "generate_dataset", "sample_objects",
    "build_evaluator", "evaluate_detection",
]
