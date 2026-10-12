from typing import Dict, Any

import jax
from jax import numpy as jnp
from jaxtyping import Array

from probabilistic_model.exceptions import ShapeMismatchError
from typing_extensions import Self

from probabilistic_model.probabilistic_circuit.jax.input_layer import (
    DifferentiableContinuousLayerWithFiniteSupport,
)


class DifferentiableUniformLayer(DifferentiableContinuousLayerWithFiniteSupport):
    """
    A layer that represents uniform distributions over a single variable.
    """

    def validate(self):
        if not self.lower.shape == self.upper.shape:
            raise ShapeMismatchError(self.upper.shape, self.lower.shape)

    @property
    def number_of_nodes(self) -> int:
        return len(self.lower)

    def log_density(self) -> Array:
        """
        Calculate the log-density of the uniform distribution.
        """
        return -jnp.log(self.upper - self.lower)

    def log_likelihood_of_nodes_of_value(self, value: Array) -> Array:
        return jnp.where(self.included_condition(value), self.log_density(), -jnp.inf)

    def log_likelihood_of_nodes(self, x: Array) -> Array:
        return jax.vmap(self.log_likelihood_of_nodes_single)(x)

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(data["variable"], jnp.array(data["interval"]))
