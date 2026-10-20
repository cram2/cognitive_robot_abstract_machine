from typing import Dict, Any

import equinox as eqx
import jax
from jax import numpy as jnp

from probabilistic_model.exceptions import ShapeMismatchError
from typing_extensions import Optional, Self

from probabilistic_model.probabilistic_circuit.jax.input_layer import (
    DifferentiableContinuousLayer,
)


class DifferentiableGaussianLayer(DifferentiableContinuousLayer):
    """
    A layer that represents Gaussian distributions over a single variable.
    """

    location: jax.Array
    """
    The location of the Gaussian distributions.
    """

    log_scale: jax.Array
    """
    The logarithm of the scale of the Gaussian distributions.
    """

    minimum_scale: jax.Array = eqx.field(static=True, default=0.01)
    """
    The minimum scale of the Gaussian distributions.
    """

    def __init__(
        self,
        variable: int,
        location: jax.Array,
        log_scale: jax.Array,
        minimum_scale: jax.Array,
    ):
        super().__init__(variable)
        self.location = location
        self.log_scale = log_scale
        self.minimum_scale = minimum_scale

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        result = DifferentiableGaussianLayer(
            self.variable, self.location, self.log_scale, self.minimum_scale
        )
        memo[id_self] = result
        return result

    def validate(self):
        if not self.location.shape == self.log_scale.shape:
            raise ShapeMismatchError(self.log_scale.shape, self.location.shape)
        if not self.minimum_scale.shape == self.log_scale.shape:
            raise ShapeMismatchError(self.log_scale.shape, self.minimum_scale.shape)
        if not jnp.all(self.minimum_scale >= 0):
            raise ValueError("The minimum scale must be positive.")

    @property
    def number_of_nodes(self) -> int:
        return self.location.shape[0]

    @property
    def scale(self) -> jax.Array:
        return jnp.exp(self.log_scale) + self.minimum_scale

    def log_likelihood_of_nodes_of_value(self, value: jax.Array) -> jax.Array:
        return jax.scipy.stats.norm.logpdf(value, loc=self.location, scale=self.scale)

    def log_likelihood_of_nodes(self, x: jax.Array) -> jax.Array:
        return jax.vmap(self.log_likelihood_of_nodes_single)(x)

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "variable": self.variable,
            "location": self.location.tolist(),
            "scale": self.log_scale.tolist(),
            "minimum_scale": self.minimum_scale.tolist(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            data["variable"],
            jnp.array(data["location"]),
            jnp.array(data["scale"]),
            jnp.array(data["minimum_scale"]),
        )
