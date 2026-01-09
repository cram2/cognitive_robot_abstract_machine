from typing import Tuple, Type, List, Dict, Any

from random_events.set import SetElement
from random_events.variable import Symbolic, Variable
from sortedcontainers import SortedSet
from typing_extensions import Self, Optional

import jax
from . import NXConverterLayer
from .inner_layer import InputLayer
import jax.numpy as jnp

from ..rx.probabilistic_circuit import Unit
from ...distributions import SymbolicDistribution
import tqdm
import numpy as np
from ..rx.probabilistic_circuit import ProbabilisticCircuit as NXProbabilisticCircuit, UnivariateDiscreteLeaf

from ...utils import MissingDict


class DiscreteLayer(InputLayer):

    log_probabilities: jnp.array
    """
    The logarithm of probability for each state of the variable.
    
    The shape is (#nodes, #states).
    """

    def __init__(self, variable: int, log_probabilities: jnp.array):
        super().__init__(variable)
        self.log_probabilities = log_probabilities

    @classmethod
    def nx_classes(cls) -> Tuple[Type, ...]:
        return SymbolicDistribution,

    def validate(self):
        return True

    @property
    def log_normalization_constant(self) -> jnp.array:
        return jax.scipy.special.logsumexp(self.log_probabilities, axis=1)

    @property
    def normalized_log_probabilities(self) -> jnp.array:
        return self.log_probabilities - self.log_normalization_constant[:, None]

    @property
    def number_of_nodes(self) -> int:
        return self.log_probabilities.shape[0]

    def log_likelihood_of_nodes_single(self, x: jnp.array) -> jnp.array:
        return self.normalized_log_probabilities[:, x.astype(int)][:, 0]


    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[UnivariateDiscreteLeaf],
                                                         child_layers: List[NXConverterLayer],
                                                         progress_bar: bool = True) -> \
            NXConverterLayer:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}

        variable: Symbolic = nodes[0].variable

        parameters = np.zeros((len(nodes), len(variable.domain.simple_sets)))

        for node in (tqdm.tqdm(nodes, desc=f"Creating discrete layer for variable {variable.name}")
                     if progress_bar else nodes):
            for state, value in node.distribution.probabilities.items():
                parameters[hash_remap[hash(node)], state] = value


        result = cls(nodes[0].probabilistic_circuit.variables.index(variable), jnp.log(parameters))
        return NXConverterLayer(result, nodes, hash_remap)

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(),
                "variable": self.variable, "log_probabilities": self.log_probabilities.tolist()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(data["variable"], jnp.array(data["log_probabilities"]))

    def to_nx(self, variables: SortedSet[Variable], result: NXProbabilisticCircuit,
                progress_bar: Optional[tqdm.tqdm] = None) -> List[Unit]:

        variable = variables[self.variable]

        if progress_bar:
            progress_bar.set_postfix_str(f"Creating discrete distributions for variable {variable.name}")

        nodes = [UnivariateDiscreteLeaf(SymbolicDistribution(variable, MissingDict(float,
            {state: value.item() for state, value in enumerate(log_probabilities)})), probabilistic_circuit=result)
                 for log_probabilities in jnp.exp(self.normalized_log_probabilities)]

        if progress_bar:
            progress_bar.update(self.number_of_nodes)
        return nodes







