"""
Hidden Chow-Liu tree circuits of :cite:`liu2021tractable`: a Chow-Liu tree
:cite:`chow1968approximating` over the variables whose every variable is replaced by a
latent variable with a fixed number of states that the observed variable hangs below.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from random_events.variable import Continuous, Symbolic, Variable
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from sortedcontainers import SortedSet

from probabilistic_model.learning.structure.exceptions import (
    UnsupportedStructureVariableError,
)
from probabilistic_model.learning.structure.mutual_information import (
    BinnedMutualInformation,
    MutualInformationEstimator,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    DenseEdgeValues,
    VariableIndices,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    Layer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    DenseEdgeMatrix,
    SparseEntries,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.discrete_layer import (
    SymbolicLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)


# %% Chow-Liu tree


@dataclass(kw_only=True)
class ChowLiuTree:
    """
    The spanning tree over the variables that maximizes the summed pairwise mutual
    information, rooted at the first variable.
    """

    parents: VariableIndices
    """
    The parent of every variable, ``-1`` for the root.
    """

    order: VariableIndices
    """
    The variables in breadth-first order from the root, parents before children.
    """

    @classmethod
    def from_mutual_information(
        cls, *, mutual_information: npt.NDArray[np.float64]
    ) -> ChowLiuTree:
        """
        :param mutual_information: The symmetric matrix of pairwise mutual information.
        :return: The maximum spanning tree of the complete graph weighted by it.
        """
        # shift the weights to be positive, since the spanning tree reads zero as no edge
        costs = mutual_information.max() + 1.0 - mutual_information
        np.fill_diagonal(costs, 0.0)
        tree = minimum_spanning_tree(costs)
        order, predecessors = breadth_first_order(
            tree, i_start=0, directed=False, return_predecessors=True
        )
        parents = np.where(predecessors < 0, -1, predecessors)
        return cls(parents=parents.astype(np.int64), order=order.astype(np.int64))


# %% initial parameters


@dataclass(kw_only=True)
class UniformWeightRange:
    """
    The interval unnormalized initial weights are drawn uniformly from.
    """

    lowest: float = 0.1
    """
    The lowest weight, above zero so that no state or transition starts impossible.
    """

    highest: float = 1.0
    """
    The highest weight.
    """

    def sample(
        self, *, random_number_generator: np.random.Generator, shape: tuple[int, ...]
    ) -> npt.NDArray[np.float64]:
        """
        :param random_number_generator: The source of the weights.
        :param shape: The shape of the weights.
        :return: Weights drawn uniformly from the interval.
        """
        return random_number_generator.uniform(self.lowest, self.highest, size=shape)


@dataclass(kw_only=True)
class RandomInitialParameters:
    """
    Random initial parameters of a hidden Chow-Liu tree circuit: categorical
    probabilities and transition weights drawn from :attr:`weight_range` and normalized,
    Gaussians centered at random values of their column with the column's standard
    deviation.
    """

    variables: SortedSet
    """
    The variables of the circuit, one per column of the data.
    """

    data: npt.NDArray[np.float64]
    """
    The rows, encoded as the circuits read them.
    """

    hidden_states: int
    """
    The number of states of every latent variable.
    """

    seed: int
    """
    The seed of the parameters.
    """

    weight_range: UniformWeightRange = field(default_factory=UniformWeightRange)
    """
    The interval the categorical probabilities and transition weights are drawn from.
    """

    random_number_generator: np.random.Generator = field(init=False, repr=False)
    """
    The source of the parameters, drawn from :attr:`seed`.
    """

    def __post_init__(self) -> None:
        """
        Seeds :attr:`random_number_generator`.
        """
        self.random_number_generator = np.random.default_rng(self.seed)

    def input_layer(self, *, variable_index: int) -> Layer:
        """
        :param variable_index: The index of the variable.
        :return: One input distribution of the variable per state of its latent
            variable.
        :raises UnsupportedStructureVariableError: If the variable is neither symbolic
            nor continuous.
        """
        variable: Variable = self.variables[variable_index]
        if isinstance(variable, Symbolic):
            number_of_states = len(variable.domain.simple_sets)
            probabilities = self.weight_range.sample(
                random_number_generator=self.random_number_generator,
                shape=(self.hidden_states, number_of_states),
            )
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            return SymbolicLayer(
                node_variables=np.full(self.hidden_states, variable_index),
                states=np.arange(number_of_states),
                table=DenseProbabilityTable(log_probabilities=np.log(probabilities)),
            )
        if isinstance(variable, Continuous):
            column = self.data[:, variable_index]
            scale = max(float(column.std()), GaussianLayer.minimum_scale)
            return GaussianLayer(
                node_variables=np.full(self.hidden_states, variable_index),
                location=self.random_number_generator.choice(
                    column, size=self.hidden_states
                ),
                log_scale=np.full(self.hidden_states, np.log(scale)),
            )
        raise UnsupportedStructureVariableError(variable=variable)

    def transition_weights(self, *, parent: int | None) -> DenseEdgeValues:
        """
        :param parent: The index of the parent variable, ``None`` for the root.
        :return: The unnormalized weight of every state of the child's latent variable
            given every state of the parent's, shape (#hidden states or 1 for the root,
            #hidden states).
        """
        rows = 1 if parent is None else self.hidden_states
        return self.weight_range.sample(
            random_number_generator=self.random_number_generator,
            shape=(rows, self.hidden_states),
        )


# %% hidden Chow-Liu tree


@dataclass(kw_only=True)
class HiddenChowLiuTree:
    """
    Builds a hidden Chow-Liu tree circuit from data.

    Every latent variable is a layer of :attr:`hidden_states` product units, each an
    input distribution of its variable times one mixture over the units of every child.
    """

    variables: SortedSet
    """
    The variables of the circuit, one per column of the data.
    """

    hidden_states: int = 12
    """
    The number of states of every latent variable.
    """

    mutual_information: MutualInformationEstimator = field(
        default_factory=BinnedMutualInformation
    )
    """
    The estimator of the mutual information the tree is spanned over.
    """

    seed: int = 0
    """
    The seed of the random initial parameters.
    """

    initial_weight_range: UniformWeightRange = field(default_factory=UniformWeightRange)
    """
    The interval the initial categorical probabilities and transition weights are drawn
    from.
    """

    def build_circuit(
        self, *, data: npt.NDArray[np.float64]
    ) -> LayeredProbabilisticCircuit:
        """
        :param data: The rows, encoded as the circuits read them.
        :return: The hidden Chow-Liu tree circuit with random initial numpy parameters.
        """
        tree = ChowLiuTree.from_mutual_information(
            mutual_information=self.mutual_information.estimate(
                variables=self.variables, data=data
            )
        )
        parameters = RandomInitialParameters(
            variables=self.variables,
            data=data,
            hidden_states=self.hidden_states,
            seed=self.seed,
            weight_range=self.initial_weight_range,
        )
        # the mixture of every variable's units under each state of its parent's latent
        # variable, or under the single state above the root
        mixtures: dict[int, SumLayer] = {}
        for variable_index in tree.order[::-1].tolist():
            factors = [parameters.input_layer(variable_index=variable_index)]
            factors.extend(
                mixtures.pop(child)
                for child in np.flatnonzero(tree.parents == variable_index).tolist()
            )
            units = (
                factors[0]
                if len(factors) == 1
                else ProductLayer.node_wise_product_of(factors)
            )
            parent = int(tree.parents[variable_index])
            weights = parameters.transition_weights(
                parent=None if parent < 0 else parent
            )
            rows, columns = np.nonzero(np.ones_like(weights))
            mixtures[variable_index] = SumLayer.from_entries(
                [units],
                SparseEntries(np.log(weights[rows, columns]), rows, columns),
                weights.shape[0],
                DenseEdgeMatrix,
            )
        return LayeredProbabilisticCircuit(
            SortedSet(self.variables), mixtures[int(tree.order[0])]
        )
