from typing import Dict, Any

from jaxtyping import Array

from typing_extensions import Self

import jax
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableInputLayer,
)
import jax.numpy as jnp


class DifferentiableDiscreteLayer(DifferentiableInputLayer):
    """
    A layer that represents discrete distributions over a single variable.
    """

    log_probabilities: Array
    """
    The logarithm of probability for each state of the variable.
    
    The shape is (#nodes, #states).
    """

    def __init__(self, variable: int, log_probabilities: jnp.array):
        super().__init__(variable)
        self.log_probabilities = log_probabilities

    def validate(self):
        return True

    @property
    def log_normalization_constant(self) -> Array:
        return jax.scipy.special.logsumexp(self.log_probabilities, axis=1)

    @property
    def normalized_log_probabilities(self) -> Array:
        return self.log_probabilities - self.log_normalization_constant[:, None]

    @property
    def number_of_nodes(self) -> int:
        return self.log_probabilities.shape[0]

    def log_likelihood_of_nodes_of_value(self, value: Array) -> Array:
        return self.normalized_log_probabilities[:, value.astype(int)][:, 0]

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "variable": self.variable,
            "log_probabilities": self.log_probabilities.tolist(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(data["variable"], jnp.array(data["log_probabilities"]))
