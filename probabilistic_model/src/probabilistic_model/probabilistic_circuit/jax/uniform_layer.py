from typing import List, Dict, Any, Optional

import jax
import tqdm
from jax import numpy as jnp
from jaxtyping import Array

from probabilistic_model.exceptions import ShapeMismatchError
from random_events.interval import SimpleInterval, Bound
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Type, Tuple, Self

from probabilistic_model.probabilistic_circuit.jax.inner_layer import RustworkxLayerConverter
from probabilistic_model.probabilistic_circuit.jax.input_layer import (
    ContinuousLayerWithFiniteSupport,
)
from probabilistic_model.probabilistic_circuit.jax.utils import (
    simple_interval_to_open_array,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    Unit,
    ProbabilisticCircuit as NXProbabilisticCircuit,
    UnivariateContinuousLeaf,
)
from probabilistic_model.distributions.uniform import UniformDistribution


class UniformLayer(ContinuousLayerWithFiniteSupport):
    """
    A layer that represents uniform distributions over a single variable.
    """

    @classmethod
    def rustworkx_classes(cls) -> Tuple[Type, ...]:
        return (UniformDistribution,)

    def validate(self):
        if not self.lower.shape == self.upper.shape:
            raise ShapeMismatchError(self.upper.shape, self.lower.shape)

    @property
    def number_of_nodes(self) -> int:
        return len(self.lower)

    def log_pdf_value(self) -> Array:
        """
        Calculate the log-density of the uniform distribution.
        """
        return -jnp.log(self.upper - self.lower)

    def log_likelihood_of_nodes_single(self, x: Array) -> Array:
        return jnp.where(self.included_condition(x), self.log_pdf_value(), -jnp.inf)

    def log_likelihood_of_nodes(self, x: Array) -> Array:
        return jax.vmap(self.log_likelihood_of_nodes_single)(x)

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(
        cls,
        nodes: List[UnivariateContinuousLeaf],
        child_layers: List[RustworkxLayerConverter],
        progress_bar: bool = True,
    ) -> RustworkxLayerConverter:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}

        variable = nodes[0].variable

        intervals = jnp.vstack(
            [
                simple_interval_to_open_array(node.distribution.interval)
                for node in (
                    tqdm.tqdm(
                        nodes,
                        desc=f"Creating uniform layer for variable {variable.name}",
                    )
                    if progress_bar
                    else nodes
                )
            ]
        )

        result = cls(
            nodes[0].probabilistic_circuit.variables.index(variable), intervals
        )
        return RustworkxLayerConverter(result, nodes, hash_remap)

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(data["variable"], jnp.array(data["interval"]))

    def to_rustworkx(
        self,
        variables: SortedSet[Variable],
        result: NXProbabilisticCircuit,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Unit]:
        variable = variables[self.variable]

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Creating Uniform distributions for variable {variable.name}"
            )

        nodes = [
            UnivariateContinuousLeaf(
                UniformDistribution(
                    variable=variable,
                    interval=SimpleInterval.from_data(
                        lower.item(), upper.item(), Bound.OPEN, Bound.OPEN
                    ),
                ),
                probabilistic_circuit=result,
            )
            for lower, upper in self.interval
        ]

        if progress_bar:
            progress_bar.update(self.number_of_nodes)

        return nodes
