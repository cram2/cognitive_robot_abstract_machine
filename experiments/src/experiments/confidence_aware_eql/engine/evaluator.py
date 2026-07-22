from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from probabilistic_model.probabilistic_model import ProbabilisticModel
from random_events.variable import Variable
from typing_extensions import Any, List, Optional

from experiments.confidence_aware_eql.engine.schema import FeatureSchema
from experiments.confidence_aware_eql.engine.threshold import FamiliarityThreshold
from experiments.confidence_aware_eql.exceptions import UnfamiliarSampleWarning


@dataclass
class FamiliarityResult:
    """
    The outcome of checking one instance against the learned distribution.
    """

    log_likelihood: float
    """Log-likelihood of the instance under the model."""

    warning: Optional[UnfamiliarSampleWarning]
    """
    The warning raised for an unfamiliar instance, or ``None`` when familiar.
    """

    @property
    def is_familiar(self) -> bool:
        """
        Whether the instance was accepted as familiar.
        """
        return self.warning is None


@dataclass
class ConfidenceAwareEvaluator:
    """
    Scores instances and flags those unlikely under the learned distribution.

    Features that are not observed on an instance are marginalised out of the
    model, so an incomplete instance is still scored on the information it does
    carry instead of being rejected outright.

    .. note::
        A marginal log-likelihood is not on the same scale as the joint
        log-likelihood the threshold was calibrated on, so an instance with
        missing features is compared against a threshold that was fitted for the
        full feature set. Calibrating one threshold per observed subset would
        remove this approximation.
    """

    schema: FeatureSchema
    """The feature schema used to encode instances."""

    model: ProbabilisticModel
    """
    The probabilistic model scoring encoded instances.
    """

    threshold: FamiliarityThreshold
    """The fitted cutoff separating familiar from unfamiliar instances."""

    def check(self, instance: Any, node_name: str) -> FamiliarityResult:
        """
        Check one instance and report whether it is familiar.

        :param instance: The instance whose features are read by the schema.
        :param node_name: Name of the rule node performing the check, recorded on any
            warning for traceability.
        :return: The log-likelihood of the instance and a warning when it is unfamiliar.
        """
        row = self.schema.encode(instance)
        observed_variables = self.schema.observed_variables(row)
        if len(observed_variables) == len(self.schema.variables):
            log_likelihood = float(self.model.log_likelihood(row[np.newaxis, :])[0])
        else:
            log_likelihood = self._marginal_log_likelihood(row, observed_variables)
        if self.threshold.is_familiar(log_likelihood):
            return FamiliarityResult(log_likelihood, None)
        return FamiliarityResult(
            log_likelihood,
            UnfamiliarSampleWarning(node_name, log_likelihood, self.threshold.value),
        )

    def _marginal_log_likelihood(
        self, row: np.ndarray, observed_variables: List[Variable]
    ) -> float:
        """
        Score a partially observed instance by marginalising the missing features.

        :param row: The encoded row, where unobserved features are ``numpy.nan``.
        :param observed_variables: The variables that were actually observed.
        :return: The log-likelihood of the observed features under the marginal model.
        """
        marginal_model = self.model.marginal(observed_variables)
        observed_values = {
            variable.name: value
            for variable, value in zip(self.schema.variables, row)
            if not np.isnan(value)
        }
        aligned_row = np.array(
            [observed_values[variable.name] for variable in marginal_model.variables],
            dtype=float,
        )
        return float(marginal_model.log_likelihood(aligned_row[np.newaxis, :])[0])
