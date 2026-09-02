"""Learn a confidence model over familiar objects and score new ones.

The confidence model is a joint probability tree fitted on the feature vectors
of familiar objects. It answers one question about a new object: how likely is
it under the distribution of the familiar ones. An object whose likelihood falls
below a calibrated threshold does not resemble anything the model was trained on
and is judged unfamiliar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from typing_extensions import Any, List

from experiments.confidence_aware_eql.feature_pipeline import extract_feature_dataframe


@dataclass
class ConfidenceModel:
    """A fitted joint probability tree scoring the familiarity of an object."""

    circuit: ProbabilisticCircuit
    """The tractable circuit learned from the familiar objects."""

    threshold: float
    """The log-likelihood below which an object is judged unfamiliar."""

    @classmethod
    def fit_from_instances(cls, instances: List[Any]) -> ConfidenceModel:
        """Fit a confidence model on the feature vectors of familiar instances.

        The familiarity threshold is calibrated as the first percentile of the
        log-likelihoods of the familiar instances, so an instance less likely than
        almost every familiar one is judged unfamiliar.

        :param instances: The familiar instances the model is learned from.
        :return: A fitted confidence model ready to score new instances.
        """
        dataframe = extract_feature_dataframe(instances)
        annotated_variables = infer_variables_from_dataframe(dataframe)
        tree = JointProbabilityTree(
            annotated_variables=annotated_variables, min_samples_per_leaf=10
        )
        circuit = tree.fit(dataframe)
        ordered = dataframe[[variable.name for variable in circuit.variables]]
        training_log_likelihoods = circuit.log_likelihood(ordered.to_numpy())
        threshold = float(np.percentile(training_log_likelihoods, 1.0))
        return cls(circuit, threshold)

    @property
    def column_order(self) -> List[str]:
        """
        :return: The feature names in the order the circuit expects them.
        """
        return [variable.name for variable in self.circuit.variables]

    def log_likelihood_of(self, instance: Any) -> float:
        """Return the log-likelihood of a single object under the model.

        The object is turned into a one-row feature dataframe and its columns are
        aligned to the order the circuit expects before scoring, so a categorical
        column is never read into a continuous variable.

        :param instance: The object whose familiarity is scored.
        :return: The log-likelihood of the object under the learned distribution.
        """
        dataframe = extract_feature_dataframe([instance])
        ordered = dataframe[self.column_order]
        return float(self.circuit.log_likelihood(ordered.to_numpy())[0])

    def is_familiar(self, instance: Any) -> bool:
        """Whether an object is familiar under the model.

        :param instance: The object to judge.
        :return: ``True`` when the object's log-likelihood is at or above the
            familiarity threshold, ``False`` otherwise.
        """
        return self.log_likelihood_of(instance) >= self.threshold
