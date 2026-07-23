from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FamiliarityThreshold(ABC):
    """
    A configurable cutoff separating familiar from unfamiliar instances.

    A threshold is fitted from the log-likelihoods of the training instances and then
    compares any future instance's log-likelihood against the learned value.
    """

    value: float = field(init=False, default=0.0)
    """
    The fitted log-likelihood cutoff; instances below this are unfamiliar.
    """

    @abstractmethod
    def fit(self, training_log_likelihoods: np.ndarray) -> float:
        """
        Learn the cutoff from the training log-likelihoods and return it.
        """

    def is_familiar(self, log_likelihood: float) -> bool:
        """
        Whether a log-likelihood is at or above the fitted cutoff.
        """
        return log_likelihood >= self.value


@dataclass
class PercentileThreshold(FamiliarityThreshold):
    """
    Places the cutoff at a low percentile of the training log-likelihoods.

    Choosing the first percentile budgets roughly one percent of familiar instances to
    be flagged, which bounds the false-positive rate directly.
    """

    percentile: float = 1.0
    """
    The percentile of training log-likelihoods used as the cutoff.
    """

    def fit(self, training_log_likelihoods: np.ndarray) -> float:
        """
        Set the cutoff to the configured percentile of the training values.
        """
        self.value = float(np.percentile(training_log_likelihoods, self.percentile))
        return self.value


@dataclass
class StandardDeviationThreshold(FamiliarityThreshold):
    """
    Places the cutoff a number of standard deviations below the mean.
    """

    number_of_standard_deviations: float = 3.0
    """
    How many standard deviations below the mean the cutoff sits.
    """

    def fit(self, training_log_likelihoods: np.ndarray) -> float:
        """
        Set the cutoff below the mean of the training values.
        """
        mean = float(np.mean(training_log_likelihoods))
        standard_deviation = float(np.std(training_log_likelihoods))
        self.value = mean - self.number_of_standard_deviations * standard_deviation
        return self.value
