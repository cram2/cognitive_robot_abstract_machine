"""
Hidden Chow-Liu tree circuits of :cite:`liu2021tractable`: a Chow-Liu tree
:cite:`chow1968approximating` over the variables whose every variable is replaced by a
latent variable with a fixed number of states that the observed variable hangs below.
"""

from __future__ import annotations

# %% imports
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from random_events.variable import Continuous, Symbolic, Variable
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from sortedcontainers import SortedSet

from probabilistic_model.learning.structure.exceptions import (
    LatentStatesMismatchError,
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
from probabilistic_model.probabilistic_circuit.tensorized.symbolic_encoding import (
    SymbolicEncoding,
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
class GaussianParameters:
    """
    The location and scale of the Gaussian of a variable under every latent state.
    """

    location: npt.NDArray[np.float64]
    """
    The mean under every latent state.
    """

    scale: npt.NDArray[np.float64]
    """
    The standard deviation under every latent state.
    """


@dataclass(kw_only=True)
class InitialParameters(ABC):
    """
    The parameters a hidden Chow-Liu tree circuit starts from: one input distribution of
    every variable and one transition from every parent's latent state to every child's.
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

    def input_layer(self, *, variable_index: int) -> Layer:
        """
        :param variable_index: The index of the variable.
        :return: One input distribution of the variable per state of its latent
            variable.
        :raises UnsupportedStructureVariableError: If the variable is neither symbolic
            nor continuous.
        """
        variable: Variable = self.variables[variable_index]
        node_variables = np.full(self.hidden_states, variable_index)
        if isinstance(variable, Symbolic):
            return SymbolicLayer(
                node_variables=node_variables,
                states=np.arange(len(variable.domain.simple_sets)),
                table=DenseProbabilityTable(
                    log_probabilities=np.log(
                        self.symbolic_probabilities(variable_index=variable_index)
                    )
                ),
            )
        if isinstance(variable, Continuous):
            gaussians = self.gaussian_parameters(variable_index=variable_index)
            return GaussianLayer(
                node_variables=node_variables,
                location=gaussians.location,
                log_scale=np.log(gaussians.scale),
            )
        raise UnsupportedStructureVariableError(variable=variable)

    @abstractmethod
    def symbolic_probabilities(self, *, variable_index: int) -> npt.NDArray[np.float64]:
        """
        :param variable_index: The index of a symbolic variable.
        :return: The probability of every state of the variable under every latent
            state, shape (#hidden states, #states).
        """

    @abstractmethod
    def gaussian_parameters(self, *, variable_index: int) -> GaussianParameters:
        """
        :param variable_index: The index of a continuous variable.
        :return: The Gaussian of the variable under every latent state.
        """

    @abstractmethod
    def transition_weights(self, *, parent: int | None, child: int) -> DenseEdgeValues:
        """
        :param parent: The index of the parent variable, ``None`` for the root.
        :param child: The index of the child variable.
        :return: The unnormalized weight of every state of the child's latent variable
            given every state of the parent's, shape (#hidden states or 1 for the root,
            #hidden states).
        """


@dataclass(kw_only=True)
class RandomInitialParameters(InitialParameters):
    """
    Random initial parameters: categorical probabilities and transition weights drawn
    from :attr:`weight_range` and normalized, Gaussians centered at random values of
    their column with the column's standard deviation.
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

    def symbolic_probabilities(self, *, variable_index: int) -> npt.NDArray[np.float64]:
        probabilities = self.weight_range.sample(
            random_number_generator=self.random_number_generator,
            shape=(
                self.hidden_states,
                len(self.variables[variable_index].domain.simple_sets),
            ),
        )
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def gaussian_parameters(self, *, variable_index: int) -> GaussianParameters:
        column = self.data[:, variable_index]
        return GaussianParameters(
            location=self.random_number_generator.choice(
                column, size=self.hidden_states
            ),
            scale=np.full(
                self.hidden_states,
                max(float(column.std()), GaussianLayer.minimum_scale),
            ),
        )

    def transition_weights(self, *, parent: int | None, child: int) -> DenseEdgeValues:
        rows = 1 if parent is None else self.hidden_states
        return self.weight_range.sample(
            random_number_generator=self.random_number_generator,
            shape=(rows, self.hidden_states),
        )


@dataclass(kw_only=True)
class CountedParameters(InitialParameters):
    """
    The maximum likelihood parameters of the rows completed with the state of every
    latent variable, smoothed by :attr:`pseudocount`; step 3 of latent variable
    distillation :cite:`liu2023scaling`.
    """

    latent_states: npt.NDArray[np.int64]
    """
    The state of every latent variable in every row, shape (#rows, #variables).
    """

    pseudocount: float = 0.1
    """
    The count added to every value under every latent state and to every transition.
    """

    def __post_init__(self) -> None:
        """
        :raises LatentStatesMismatchError: If the latent states do not hold one state
            below :attr:`hidden_states` per row and variable.
        """
        if self.latent_states.shape != (len(self.data), len(self.variables)) or (
            self.latent_states.size
            and not 0
            <= self.latent_states.min()
            <= self.latent_states.max()
            < self.hidden_states
        ):
            raise LatentStatesMismatchError(
                shape=self.latent_states.shape,
                expected_shape=(len(self.data), len(self.variables)),
                hidden_states=self.hidden_states,
            )

    def symbolic_probabilities(self, *, variable_index: int) -> npt.NDArray[np.float64]:
        variable = self.variables[variable_index]
        states = SymbolicEncoding(variable=variable).indices_of_hashes(
            self.data[:, variable_index]
        )
        counts = np.full(
            (self.hidden_states, len(variable.domain.simple_sets)), self.pseudocount
        )
        np.add.at(counts, (self.latent_states[:, variable_index], states), 1.0)
        return counts / counts.sum(axis=1, keepdims=True)

    def gaussian_parameters(self, *, variable_index: int) -> GaussianParameters:
        column = self.data[:, variable_index]
        latent = self.latent_states[:, variable_index]
        counts = np.bincount(latent, minlength=self.hidden_states)
        rows_per_state = np.maximum(counts, 1)
        location = np.where(
            counts > 0,
            np.bincount(latent, weights=column, minlength=self.hidden_states)
            / rows_per_state,
            column.mean(),
        )
        variance = np.where(
            counts > 1,
            np.bincount(
                latent,
                weights=(column - location[latent]) ** 2,
                minlength=self.hidden_states,
            )
            / rows_per_state,
            column.var(),
        )
        return GaussianParameters(
            location=location,
            scale=np.sqrt(np.maximum(variance, GaussianLayer.minimum_scale**2)),
        )

    def transition_weights(self, *, parent: int | None, child: int) -> DenseEdgeValues:
        child_states = self.latent_states[:, child]
        if parent is None:
            counts = np.full((1, self.hidden_states), self.pseudocount)
            np.add.at(counts, (0, child_states), 1.0)
            return counts
        counts = np.full((self.hidden_states, self.hidden_states), self.pseudocount)
        np.add.at(counts, (self.latent_states[:, parent], child_states), 1.0)
        return counts


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
    The interval the random initial categorical probabilities and transition weights are
    drawn from.
    """

    pseudocount: float = 0.1
    """
    The count added to every value and every transition when the initial parameters are
    counted from latent states.
    """

    def build_circuit(
        self,
        *,
        data: npt.NDArray[np.float64],
        latent_states: npt.NDArray[np.int64] | None = None,
    ) -> LayeredProbabilisticCircuit:
        """
        :param data: The rows, encoded as the circuits read them.
        :param latent_states: The state of every latent variable in every row, shape
            (#rows, #variables); the initial parameters are counted from them, see
            :class:`CountedParameters`, and random if ``None``.
        :return: The hidden Chow-Liu tree circuit with initial numpy parameters.
        """
        tree = ChowLiuTree.from_mutual_information(
            mutual_information=self.mutual_information.estimate(
                variables=self.variables, data=data
            )
        )
        parameters: InitialParameters = (
            RandomInitialParameters(
                variables=self.variables,
                data=data,
                hidden_states=self.hidden_states,
                seed=self.seed,
                weight_range=self.initial_weight_range,
            )
            if latent_states is None
            else CountedParameters(
                variables=self.variables,
                data=data,
                hidden_states=self.hidden_states,
                latent_states=latent_states,
                pseudocount=self.pseudocount,
            )
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
                parent=None if parent < 0 else parent, child=variable_index
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
