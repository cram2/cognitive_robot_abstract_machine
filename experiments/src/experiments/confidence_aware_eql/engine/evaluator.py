from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Optional

from experiments.confidence_aware_eql.engine.circuit_model import GaussianMixtureCircuit
from experiments.confidence_aware_eql.engine.schema import FeatureSchema
from experiments.confidence_aware_eql.engine.threshold import FamiliarityThreshold
from experiments.confidence_aware_eql.exceptions import UnfamiliarSampleWarning


@dataclass
class FamiliarityResult:
    """The outcome of checking one instance against the learned distribution."""

    log_likelihood: Optional[float]
    """Log-likelihood of the instance, or ``None`` when it could not be scored."""

    warning: Optional[UnfamiliarSampleWarning]
    """The warning raised for an unfamiliar instance, or ``None`` when familiar."""

    @property
    def is_familiar(self) -> bool:
        """Whether the instance was accepted as familiar."""
        return self.warning is None


@dataclass
class ConfidenceAwareEvaluator:
    """Scores instances and flags those unlikely under the learned distribution.

    The evaluator encodes an instance through its schema, computes the
    log-likelihood with the compiled circuit, and returns a
    :class:`FamiliarityResult` carrying a warning when the instance is incomplete
    or falls below the fitted threshold.
    """

    schema: FeatureSchema
    """The feature schema used to encode instances."""

    circuit: GaussianMixtureCircuit
    """The compiled probabilistic circuit scoring encoded instances."""

    threshold: FamiliarityThreshold
    """The fitted cutoff separating familiar from unfamiliar instances."""

    def check(self, instance: object, node_name: str) -> FamiliarityResult:
        """Check one instance and report whether it is familiar.

        :param instance: The instance whose features are read by the schema.
        :param node_name: Name of the rule node performing the check, recorded on
            any warning for traceability.
        """
        encoded = self.schema.encode(instance)
        if not encoded.is_complete:
            reason = f"incomplete features {encoded.missing_features}"
            return FamiliarityResult(None, UnfamiliarSampleWarning(node_name, reason))

        log_likelihood = float(self.circuit.log_likelihood(encoded.row)[0])
        if self.threshold.is_familiar(log_likelihood):
            return FamiliarityResult(log_likelihood, None)

        reason = (
            f"log-likelihood {log_likelihood:.2f} below threshold "
            f"{self.threshold.value:.2f}"
        )
        return FamiliarityResult(
            log_likelihood,
            UnfamiliarSampleWarning(node_name, reason, log_likelihood),
        )
