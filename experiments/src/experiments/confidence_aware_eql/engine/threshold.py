"""
Threshold strategies — choose the familiarity threshold FROM data.

Instead of a magic number, we derive the threshold from the log-likelihoods the
fitted circuit assigns to the (familiar) training objects. Two interchangeable
strategies are provided; both expose .fit(train_log_likelihoods) -> threshold.

PercentileThreshold(p):
    threshold = p-th percentile of training log-likelihoods.
    Interpretation: about p% of genuinely-familiar objects will fall below it,
    i.e. p is (approximately) the expected false-positive rate. Set p=1.0 for a
    ~1% false alarm budget. This makes the threshold explainable and tunable.

StdDevThreshold(k):
    threshold = mean - k * std of training log-likelihoods.
    A Gaussian-tail style cutoff; flags objects k standard deviations less
    likely than the average familiar object.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PercentileThreshold:
    percentile: float = 1.0
    threshold: float = None

    def fit(self, train_log_likelihoods) -> float:
        lls = np.asarray(train_log_likelihoods, dtype=float)
        self.threshold = float(np.percentile(lls, self.percentile))
        return self.threshold

    def describe(self) -> str:
        return (f"PercentileThreshold(p={self.percentile}) "
                f"-> {self.threshold:.2f} (~{self.percentile:.1f}% expected false positives)")


@dataclass
class StdDevThreshold:
    k: float = 3.0
    threshold: float = None

    def fit(self, train_log_likelihoods) -> float:
        lls = np.asarray(train_log_likelihoods, dtype=float)
        self.threshold = float(lls.mean() - self.k * lls.std())
        return self.threshold

    def describe(self) -> str:
        return f"StdDevThreshold(k={self.k}) -> {self.threshold:.2f} (mean - {self.k}*std)"
