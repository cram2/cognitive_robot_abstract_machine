"""
Hidden Chow-Liu tree circuits of :cite:`liu2021tractable`: a Chow-Liu tree
:cite:`chow1968approximating` over the variables whose every variable is replaced by a
latent variable with a fixed number of states that the observed variable hangs below.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

import numpy as np
from random_events.variable import Continuous, Symbolic, Variable
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from sortedcontainers import SortedSet

from probabilistic_model.learning.structure.exceptions import (
    UnsupportedStructureVariableError,
)
from probabilistic_model.learning.structure.mutual_information import (
    DiscretizedData,
    PairwiseMutualInformation,
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

    parents: np.ndarray
    """
    The parent of every variable, ``-1`` for the root.
    """

    order: np.ndarray
    """
    The variables in breadth-first order from the root, parents before children.
    """

    @classmethod
    def from_mutual_information(cls, *, mutual_information: np.ndarray) -> ChowLiuTree:
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
        return cls(parents=parents, order=order)


# %% initial parameters


@dataclass(kw_only=True)
class RandomInitialParameters:
    """
    Random initial parameters of a hidden Chow-Liu tree circuit: categorical
    probabilities and transition weights drawn from [0.1, 1] and normalized, Gaussians
    centered at random values of their column with the column's standard deviation.
    """

    variables: SortedSet
    """
    The variables of the circuit, one per column of the data.
    """

    data: np.ndarray
    """
    The rows, encoded as the circuits read them.
    """

    hidden_states: int
    """
    The number of states of every latent variable.
    """

    random_number_generator: np.random.Generator
    """
    The source of the parameters.
    """

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
            probabilities = self.random_number_generator.uniform(
                0.1, 1.0, size=(self.hidden_states, number_of_states)
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

    def transition_weights(self, *, parent: int | None) -> np.ndarray:
        """
        :param parent: The index of the parent variable, ``None`` for the root.
        :return: The unnormalized weight of every state of the child's latent variable
            given every state of the parent's, shape (#hidden states or 1 for the root,
            #hidden states).
        """
        rows = 1 if parent is None else self.hidden_states
        return self.random_number_generator.uniform(
            0.1, 1.0, size=(rows, self.hidden_states)
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

    number_of_bins: int = 10
    """
    The number of quantile bins a continuous variable is cut into for the mutual
    information.
    """

    seed: int = 0
    """
    The seed of the random initial parameters.
    """

    def build_circuit(self, *, data: np.ndarray) -> LayeredProbabilisticCircuit:
        """
        :param data: The rows, encoded as the circuits read them.
        :return: The hidden Chow-Liu tree circuit with random initial numpy parameters.
        """
        tree = ChowLiuTree.from_mutual_information(
            mutual_information=PairwiseMutualInformation(
                discretized=DiscretizedData(
                    variables=self.variables,
                    data=data,
                    number_of_bins=self.number_of_bins,
                )
            ).compute()
        )
        parameters = RandomInitialParameters(
            variables=self.variables,
            data=data,
            hidden_states=self.hidden_states,
            random_number_generator=np.random.default_rng(self.seed),
        )
        units: dict[int, Layer] = {}
        for variable_index in tree.order[::-1].tolist():
            factors = [parameters.input_layer(variable_index=variable_index)]
            factors.extend(
                self.mixture_layer(
                    child_layer=units.pop(child),
                    weights=parameters.transition_weights(parent=variable_index),
                )
                for child in np.flatnonzero(tree.parents == variable_index).tolist()
            )
            units[variable_index] = (
                factors[0]
                if len(factors) == 1
                else ProductLayer.node_wise_product_of(factors)
            )
        root_variable = int(tree.order[0])
        root = self.mixture_layer(
            child_layer=units[root_variable],
            weights=parameters.transition_weights(parent=None),
        )
        return LayeredProbabilisticCircuit(SortedSet(self.variables), root)

    @staticmethod
    def mixture_layer(*, child_layer: Layer, weights: np.ndarray) -> SumLayer:
        """
        :param child_layer: The layer whose units are mixed.
        :param weights: The unnormalized weight of every unit of the child layer in
            every mixture, shape (#mixtures, #child units).
        :return: The layer of the mixtures.
        """
        rows, columns = np.nonzero(np.ones_like(weights))
        return SumLayer.from_entries(
            [child_layer],
            SparseEntries(np.log(weights[rows, columns]), rows, columns),
            weights.shape[0],
            DenseEdgeMatrix,
        )
