from typing import List, Dict, Any, Optional

import equinox as eqx
import jax
import tqdm
from jax import numpy as jnp

from probabilistic_model.exceptions import ShapeMismatchError
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Type, Tuple, Self

from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    RustworkxLayerConverter,
)
from probabilistic_model.probabilistic_circuit.jax.input_layer import ContinuousLayer
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    Unit,
    ProbabilisticCircuit as NXProbabilisticCircuit,
    UnivariateContinuousLeaf,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution


class GaussianLayer(ContinuousLayer):
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

    min_scale: jax.Array = eqx.field(static=True, default=0.01)
    """
    The minimum scale of the Gaussian distributions.
    """

    def __init__(
        self,
        variable: int,
        location: jax.Array,
        log_scale: jax.Array,
        min_scale: jax.Array,
    ):
        super().__init__(variable)
        self.location = location
        self.log_scale = log_scale
        self.min_scale = min_scale

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        result = GaussianLayer(
            self.variable, self.location, self.log_scale, self.min_scale
        )
        memo[id_self] = result
        return result

    @classmethod
    def rustworkx_classes(cls) -> Tuple[Type, ...]:
        return (GaussianDistribution,)

    def validate(self):
        if not self.location.shape == self.log_scale.shape:
            raise ShapeMismatchError(self.log_scale.shape, self.location.shape)
        if not self.min_scale.shape == self.log_scale.shape:
            raise ShapeMismatchError(self.log_scale.shape, self.min_scale.shape)
        if not jnp.all(self.min_scale >= 0):
            raise ValueError("The minimum scale must be positive.")

    @property
    def number_of_nodes(self) -> int:
        return self.location.shape[0]

    @property
    def scale(self) -> jax.Array:
        return jnp.exp(self.log_scale) + self.min_scale

    def log_likelihood_of_nodes_single(self, x: jax.Array) -> jax.Array:
        return jax.scipy.stats.norm.logpdf(x, loc=self.location, scale=self.scale)

    def log_likelihood_of_nodes(self, x: jax.Array) -> jax.Array:
        return jax.vmap(self.log_likelihood_of_nodes_single)(x)

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(
        cls,
        nodes: List[UnivariateContinuousLeaf],
        child_layers: List[RustworkxLayerConverter],
        progress_bar: bool = True,
    ) -> RustworkxLayerConverter:
        """
        Create a GaussianLayer from a list of UnivariateContinuousLeaf nodes that have the same variable and scope.
        """
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}

        variable = nodes[0].variable

        parameters = jnp.vstack(
            [
                jnp.array([node.distribution.location, node.distribution.scale, 0.01])
                for node in (
                    tqdm.tqdm(
                        nodes,
                        desc=f"Creating guassian layer for variable {variable.name}",
                    )
                    if progress_bar
                    else nodes
                )
            ]
        )

        result = cls(
            nodes[0].probabilistic_circuit.variables.index(variable),
            parameters[:, 0],
            jnp.log(parameters[:, 1]),
            parameters[:, 2],
        )
        return RustworkxLayerConverter(result, nodes, hash_remap)

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "variable": self.variable,
            "location": self.location.tolist(),
            "scale": self.log_scale.tolist(),
            "min_scale": self.min_scale.tolist(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            data["variable"],
            jnp.array(data["location"]),
            jnp.array(data["scale"]),
            jnp.array(data["min_scale"]),
        )

    def to_rustworkx(
        self,
        variables: SortedSet[Variable],
        result: NXProbabilisticCircuit,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Unit]:
        variable = variables[self.variable]

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Creating Gaussian distributions for variable {variable.name}"
            )

        nodes = [
            UnivariateContinuousLeaf(
                GaussianDistribution(
                    variable=variable, location=location.item(), scale=scale.item()
                ),
                probabilistic_circuit=result,
            )
            for location, scale in zip(self.location, self.scale)
        ]

        if progress_bar:
            progress_bar.update(self.number_of_nodes)

        return nodes
