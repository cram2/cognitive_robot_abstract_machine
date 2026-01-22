from __future__ import annotations

from abc import ABC
from typing import List, Dict, Any

import equinox as eqx
import jax
import tqdm
from jax import numpy as jnp
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Tuple, Type, Self, Optional

from .inner_layer import InputLayer, NXConverterLayer
from ..rx.probabilistic_circuit import Unit, ProbabilisticCircuit as NXProbabilisticCircuit, UnivariateContinuousLeaf
from ...distributions import DiracDeltaDistribution


class ContinuousLayer(InputLayer, ABC):
    """
    Abstract base class for continuous univariate input units.
    """


class ContinuousLayerWithFiniteSupport(ContinuousLayer, ABC):
    """
    Abstract class for continuous univariate input units with finite support.
    """

    interval: jax.Array = eqx.field(static=True)
    """
    The interval of the distribution as a array of shape (num_nodes, 2).
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

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        result = self.__class__(self.variables[0].item(), self.interval.copy())
        memo[id_self] = result
        return result


class DiracDeltaLayer(ContinuousLayer):
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

    def __init__(self, variable_index, location, density_cap):
        super().__init__(variable_index)
        self.location = location
        self.density_cap = density_cap

    def validate(self):
        assert len(self.location) == len(self.density_cap), "The number of locations and density caps must match."

    @property
    def number_of_nodes(self):
        return len(self.location)

    def log_likelihood_of_nodes(self, x: jax.Array) -> jax.Array:
        return jax.vmap(self.log_likelihood_of_nodes_single)(x)

    def log_likelihood_of_nodes_single(self, x: jax.Array) -> jax.Array:
        return jnp.where(x == self.location, jnp.log(self.density_cap), -jnp.inf)

    @classmethod
    def nx_classes(cls) -> Tuple[Type, ...]:
        return DiracDeltaDistribution,

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[UnivariateContinuousLeaf],
                                                         child_layers: List[NXConverterLayer],
                                                         progress_bar: bool = True) -> \
            NXConverterLayer:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        locations = jnp.array([node.distribution.location for node in nodes], dtype=jnp.float32)
        density_caps = jnp.array([node.distribution.density_cap for node in nodes], dtype=jnp.float32)
        result = cls(nodes[0].probabilistic_circuit.variables.index(nodes[0].variable), locations, density_caps)
        return NXConverterLayer(result, nodes, hash_remap)

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["location"] = self.location.tolist()
        result["density_cap"] = self.density_cap.tolist()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(data["variable"], jnp.array(data["location"]), jnp.array(data["density_cap"]))

    def to_nx(self, variables: SortedSet[Variable], result: NXProbabilisticCircuit,
              progress_bar: Optional[tqdm.tqdm] = None) -> List[
        Unit]:
        variable = variables[self.variable]

        if progress_bar:
            progress_bar.set_postfix_str(f"Creating Dirac Delta distributions for variable {variable.name}")

        nodes = [UnivariateContinuousLeaf(DiracDeltaDistribution(variable, location.item(), density_cap.item()), result)
                 for location, density_cap in zip(self.location, self.density_cap)]
        progress_bar.update(self.number_of_nodes)
        return nodes
