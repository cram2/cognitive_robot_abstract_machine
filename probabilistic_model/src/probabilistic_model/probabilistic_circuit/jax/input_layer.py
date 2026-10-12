from __future__ import annotations

from abc import ABC
from typing import Dict, Any

import equinox as eqx
import jax
from jax import numpy as jnp

from probabilistic_model.exceptions import ShapeMismatchError
from typing_extensions import Optional, Self

from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableInputLayer,
)


class DifferentiableContinuousLayer(DifferentiableInputLayer, ABC):
    """
    Abstract base class for continuous univariate input units.
    """


class DifferentiableContinuousLayerWithFiniteSupport(
    DifferentiableContinuousLayer, ABC
):
    """
    Abstract class for continuous univariate input units with finite support.
    """

    interval: jax.Array = eqx.field(static=True)
    """
    The interval of the distribution as a array of shape (#nodes, 2).

    The first column contains the lower bounds and the second column the upper bounds.
    The intervals are treated as open intervals (>/< comparator).
    """

    def __init__(self, variable: int, interval: jax.Array):
        super().__init__(variable)
        self.interval = interval

    @property
    def lower(self) -> jax.Array:
        return self.interval[:, 0]

    @property
    def upper(self) -> jax.Array:
        return self.interval[:, 1]

    def left_included_condition(self, x: jax.Array) -> jax.Array:
        """
        Check if x is included in the left bound of the intervals.

        :param x: The data
        :return: A boolean array of shape (#x, #nodes)
        """
        return self.lower < x

    def right_included_condition(self, x: jax.Array) -> jax.Array:
        """
        Check if x is included in the right bound of the intervals.

        :param x: The data
        :return: A boolean array of shape (#x, #nodes)
        """
        return x < self.upper

    def included_condition(self, x: jax.Array) -> jax.Array:
        """
        Check if x is included in the interval.

        :param x: The data
        :return: A boolean array of shape (#x, #nodes)
        """
        return self.left_included_condition(x) & self.right_included_condition(x)

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["interval"] = self.interval.tolist()
        return result

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        result = self.__class__(self.variables[0].item(), self.interval.copy())
        memo[id_self] = result
        return result


class DifferentiableDiracDeltaLayer(DifferentiableContinuousLayer):
    """
    A layer that represents Dirac delta distributions over a single variable.
    """

    location: jax.Array = eqx.field(static=True)
    """
    The locations of the Dirac delta distributions.
    """

    density_cap: jax.Array = eqx.field(static=True)
    """
    The density caps of the Dirac delta distributions.

    This value will be used to replace infinity in likelihoods.
    """

    def __init__(self, variable: int, location: jax.Array, density_cap: jax.Array):
        super().__init__(variable)
        self.location = location
        self.density_cap = density_cap

    def validate(self):
        if not self.location.shape == self.density_cap.shape:
            raise ShapeMismatchError(self.density_cap.shape, self.location.shape)

    @property
    def number_of_nodes(self) -> int:
        return len(self.location)

    def log_likelihood_of_nodes(self, x: jax.Array) -> jax.Array:
        return jax.vmap(self.log_likelihood_of_nodes_single)(x)

    def log_likelihood_of_nodes_of_value(self, value: jax.Array) -> jax.Array:
        return jnp.where(value == self.location, jnp.log(self.density_cap), -jnp.inf)

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["location"] = self.location.tolist()
        result["density_cap"] = self.density_cap.tolist()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            data["variable"],
            jnp.array(data["location"]),
            jnp.array(data["density_cap"]),
        )
