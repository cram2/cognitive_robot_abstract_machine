import math
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Union, Optional, List, Iterable, Dict, Any

import numpy as np
import pandas as pd
from jpt.learning.impurity import Impurity

from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from random_events.product_algebra import VariableMap
from random_events.variable import Variable, Continuous, Integer, Symbolic
from typing_extensions import Self

from probabilistic_model.learning.jpt.variables import AnnotatedVariable
from probabilistic_model.learning.nyga_induction import NygaInduction
from probabilistic_model.distributions.distributions import (
    DiracDeltaDistribution,
    SymbolicDistribution,
    IntegerDistribution,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    SumUnit,
    ProductUnit,
    ProbabilisticCircuit,
    UnivariateDiscreteLeaf,
    UnivariateContinuousLeaf,
)
from probabilistic_model.utils import MissingDict


@dataclass
class JointProbabilityTree(SubclassJSONSerializer):
    """
    Class that implements the JPT learning algorithm for probabilistic circuits.
    """

    annotated_variables: Iterable[AnnotatedVariable]
    """
    The variables from initialization. Since variables will be overwritten as soon as the model is learned,
    we need to store the variables from initialization here.
    """

    targets: Optional[Iterable[Variable]] = field(default=None)
    """
    The variables to optimize for.
    """

    features: Optional[Iterable[Variable]] = field(default=None)
    """
    The variables that are used to craft criteria.
    """

    min_samples_per_leaf: Union[int, float] = field(default=1)
    """
    The minimum number of samples to create another sum node. If this is smaller than one, it will be reinterpreted
    as fraction w. r. t. the number of samples total.
    """

    min_impurity_improvement: float = field(default=0.0)
    """
    The minimum impurity improvement to create another sum node.
    """

    max_leaves: Union[int, float] = field(default=float("inf"))
    """
    The maximum number of leaves.
    """

    max_depth: Union[int, float] = field(default=float("inf"))
    """
    The maximum depth of the tree.
    """

    dependencies: Optional[VariableMap] = field(default=None)
    """
    The dependencies between the variables.
    """

    probabilistic_circuit: ProbabilisticCircuit = field(init=False)
    """
    The probabilistic circuit the result will appear in.
    """

    total_samples: int = field(default=1)
    """
    The total amount of samples that were used to fit the model.
    """

    indices: Optional[np.ndarray] = field(default=None)
    """
    The indices of the samples that were used to fit the model.
    """

    impurity: Optional[Impurity] = field(default=None)
    """
    The impurity object that is used to calculate the best split.
    """

    c45queue: deque = field(default_factory=deque)
    """
    The queue used to store the data to be processed by the C4.5 algorithm.
    """

    keep_sample_indices: bool = field(default=False)
    """
    Rather to store the sample indices in the leaves or not.
    """

    root: Optional[SumUnit] = field(default=None)
    """
    The root of the circuit that will be learned.
    """

    def __post_init__(self):
        self.annotated_variables = tuple(sorted(self.annotated_variables))
        self.set_targets_and_features(self.targets, self.features)

        if self.dependencies is None:
            self.dependencies = VariableMap(
                {var: list(self.targets) for var in self.features}
            )

        self.probabilistic_circuit = ProbabilisticCircuit()

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return tuple(
            annotated_variable.variable
            for annotated_variable in self.annotated_variables
        )

    def set_targets_and_features(
        self,
        targets: Optional[Iterable[Variable]],
        features: Optional[Iterable[Variable]],
    ) -> None:
        """
        Set the targets and features of the model.
        If only one of them is provided, the other is set as the complement of the provided one.
        If none are provided, both of them are set as the variables of the model.
        If both are provided, they are taken as given.

        :param targets: The targets of the model.
        :param features: The features of the model.
        :return: None
        """
        # if targets are not specified
        if targets is None:
            # and features are not specified
            if features is None:
                self.targets = self.variables
                self.features = self.variables
            # and features are specified
            else:
                self.targets = tuple(sorted(set(self.variables) - set(features)))
                self.features = tuple(sorted(features))
        # if targets are specified
        else:
            # and features are not specified
            if features is None:
                self.targets = tuple(sorted(set(targets)))
                self.features = tuple(sorted(set(self.variables) - set(targets)))

            # and features are specified
            else:
                self.targets = tuple(sorted(set(targets)))
                self.features = tuple(sorted(set(features)))

    @property
    def min_samples_leaf(self):
        """
        The minimum number of samples to create another sum node.
        """
        if self.min_samples_per_leaf < 1.0:
            return math.ceil(self.min_samples_per_leaf * self.total_samples)
        else:
            return self.min_samples_per_leaf

    @property
    def numeric_variables(self):
        return [
            variable
            for variable in self.variables
            if isinstance(variable, (Continuous, Integer))
        ]

    @property
    def numeric_targets(self):
        return [
            variable
            for variable in self.targets
            if isinstance(variable, (Continuous, Integer))
        ]

    @property
    def numeric_features(self):
        return [
            variable
            for variable in self.features
            if isinstance(variable, (Continuous, Integer))
        ]

    @property
    def symbolic_variables(self):
        return [
            variable for variable in self.variables if isinstance(variable, Symbolic)
        ]

    @property
    def symbolic_targets(self):
        return [variable for variable in self.targets if isinstance(variable, Symbolic)]

    @property
    def symbolic_features(self):
        return [
            variable for variable in self.features if isinstance(variable, Symbolic)
        ]

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the data to be used in the model.

        :param data: The data to preprocess.
        :return: The preprocessed data.
        """

        result = np.zeros(data.shape)

        for variable_index, variable in enumerate(self.variables):
            column = data[variable.name]
            if isinstance(variable, Symbolic):
                all_elements = {
                    element: index
                    for index, element in enumerate(variable.domain.all_elements)
                }
                column = column.apply(lambda x: all_elements[x])
            result[:, variable_index] = column

        return result

    def fit(self, data: pd.DataFrame) -> ProbabilisticCircuit:
        """
        Fit the model to the data.

        :param data: The data to fit the model to.
        :return: The fitted model.
        """
        self.root = SumUnit(probabilistic_circuit=self.probabilistic_circuit)
        preprocessed_data = self.preprocess_data(data)

        self.total_samples = len(preprocessed_data)

        self.indices = np.ascontiguousarray(
            np.arange(preprocessed_data.shape[0], dtype=np.int64)
        )
        self.impurity = self.construct_impurity()
        self.impurity.setup(preprocessed_data, self.indices)

        self.c45queue.append((preprocessed_data, 0, len(preprocessed_data), 0))

        while self.c45queue:
            self.c45(*self.c45queue.popleft())

        return self.probabilistic_circuit

    def c45(self, data: np.ndarray, start: int, end: int, depth: int):
        """
        Construct a DecisionNode or DecomposableProductNode from the data.

        :param data: The data to calculate the impurity from.
        :param start: Starting index in the data.
        :param end: Ending index in the data.
        :param depth: The current depth of the induction
        :return: The constructed decision tree node
        """

        number_of_samples = end - start
        # if the inducing in this step results in inadmissible nodes, skip the impurity calculation
        if depth >= self.max_depth or number_of_samples < 2 * self.min_samples_leaf:
            max_gain = -float("inf")
        else:
            max_gain = self.impurity.compute_best_split(start, end)

        # if the max gain is insufficient
        if max_gain <= self.min_impurity_improvement:

            # create decomposable product node
            leaf_node = self.create_leaf_node(data[self.indices[start:end]])
            weight = number_of_samples / len(data)
            self.root.add_subcircuit(leaf_node, np.log(weight))

            if self.keep_sample_indices:
                leaf_node.sample_indices = self.indices[start:end]

            # terminate the induction
            return

        # if the max gain is sufficient
        split_pos = self.impurity.best_split_pos

        # increase the depth
        new_depth = depth + 1

        # append the new induction steps
        self.c45queue.append((data, start, start + split_pos + 1, new_depth))
        self.c45queue.append((data, start + split_pos + 1, end, new_depth))

    def create_leaf_node(self, data: np.ndarray) -> ProductUnit:
        """
        Create a fully decomposable product node from a 2D data array.

        :param data: The preprocessed data to use for training
        :return: The leaf node.
        """
        result = ProductUnit(probabilistic_circuit=self.probabilistic_circuit)
        result.total_samples = len(data)

        for index, annotated_variable in enumerate(self.annotated_variables):
            if isinstance(annotated_variable.variable, Continuous):
                distribution = NygaInduction(
                    annotated_variable.variable,
                    min_likelihood_improvement=annotated_variable.min_likelihood_improvement,
                    min_samples_per_quantile=annotated_variable.min_samples_per_quantile,
                )
                distribution = distribution.fit(data[:, index])

                if isinstance(
                    distribution.root, UnivariateContinuousLeaf
                ) and isinstance(
                    distribution.root.distribution, DiracDeltaDistribution
                ):
                    distribution.root.distribution.density_cap = (
                        1 / annotated_variable.minimal_distance
                    )
                    distribution.root.distribution.tolerance = 1e-4
                    # print(f"Variable {annotated_variable.variable.name} is constant at {distribution.root.distribution.location}. Density cap: {distribution.root.distribution.density_cap}")
                nyga_root = distribution.root
                new_nodes = self.probabilistic_circuit.mount(nyga_root)
                result.add_subcircuit(new_nodes[nyga_root.index])

            elif isinstance(annotated_variable.variable, Symbolic):
                distribution = SymbolicDistribution(
                    variable=annotated_variable.variable,
                    probabilities=MissingDict(float),
                )
                distribution.fit_from_indices(data[:, index].astype(int))
                distribution = UnivariateDiscreteLeaf(
                    distribution, probabilistic_circuit=self.probabilistic_circuit
                )
                result.add_subcircuit(distribution)

            elif isinstance(annotated_variable.variable, Integer):
                distribution = IntegerDistribution(
                    variable=annotated_variable.variable,
                    probabilities=MissingDict(float),
                )
                distribution.fit(data[:, index])
                distribution = UnivariateDiscreteLeaf(
                    distribution, probabilistic_circuit=self.probabilistic_circuit
                )
                result.add_subcircuit(distribution)

            else:
                raise ValueError(f"Variable {annotated_variable} is not supported.")

        return result

    def construct_impurity(self) -> Impurity:
        """
        Construct the impurity object to be used in the model.
        An impurity object is used to calculate the best split.
        """

        min_samples_leaf = self.min_samples_leaf

        numeric_vars = np.array(
            [
                index
                for index, variable in enumerate(self.variables)
                if variable in self.numeric_targets
            ],
            dtype=int,
        )
        symbolic_vars = np.array(
            [
                index
                for index, variable in enumerate(self.variables)
                if variable in self.symbolic_targets
            ],
            dtype=int,
        )

        invert_impurity = np.array([0] * len(self.symbolic_targets), dtype=int)

        n_sym_vars_total = len(self.symbolic_variables)
        n_num_vars_total = len(self.numeric_variables)

        numeric_features = np.array(
            [
                index
                for index, variable in enumerate(self.variables)
                if variable in self.numeric_features
            ],
            dtype=int,
        )
        symbolic_features = np.array(
            [
                index
                for index, variable in enumerate(self.variables)
                if variable in self.symbolic_features
            ],
            dtype=int,
        )

        symbols = np.array(
            [len(variable.domain.simple_sets) for variable in self.symbolic_variables]
        )
        max_variances = np.array(
            [
                annotated_variable.standard_deviation**2
                for annotated_variable in self.annotated_variables
            ]
        )

        min_impurity_improvement = np.array(
            [
                annotated_variable.min_impurity_improvement
                for annotated_variable in self.annotated_variables
                if annotated_variable.variable in self.features
            ],
            dtype=float,
        )

        dependency_indices = dict()

        for variable, dep_vars in self.dependencies.items():
            # get the index version of the dependent variables and store them
            idx_var = self.variables.index(variable)
            idc_dep = [self.variables.index(var) for var in dep_vars]
            dependency_indices[idx_var] = idc_dep

        return Impurity(
            min_samples_leaf,
            numeric_vars,
            symbolic_vars,
            invert_impurity,
            n_sym_vars_total,
            n_num_vars_total,
            numeric_features,
            symbolic_features,
            symbols,
            max_variances,
            min_impurity_improvement,
            dependency_indices,
        )

    def _variable_dependencies_to_json(self) -> Dict[str, List[str]]:
        """
        Convert the variable dependencies to a json compatible format.
        The result maps variable names to lists of variable names.
        """
        return {
            variable.name: [dependency.name for dependency in dependencies]
            for variable, dependencies in self.dependencies.items()
        }

    def empty_copy(self):
        result = self.__class__(
            annotated_variables=self.annotated_variables,
            targets=self.targets,
            features=self.features,
            min_samples_per_leaf=self.min_samples_leaf,
            min_impurity_improvement=self.min_impurity_improvement,
            max_depth=self.max_depth,
            dependencies=self.dependencies,
        )
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["annotated_variables_from_init"] = [
            to_json(variable) for variable in self.annotated_variables
        ]
        result["targets"] = [variable.name for variable in self.targets]
        result["features"] = [variable.name for variable in self.features]
        result["min_samples_per_leaf"] = self.min_samples_per_leaf
        result["min_impurity_improvement"] = self.min_impurity_improvement
        result["max_leaves"] = self.max_leaves
        result["max_depth"] = self.max_depth
        result["dependencies"] = self._variable_dependencies_to_json()
        result["total_samples"] = self.total_samples
        result["probabilistic_circuit"] = to_json(self.probabilistic_circuit)
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        annotated_variable_from_init: List[AnnotatedVariable] = [
            from_json(annotated_variable)
            for annotated_variable in data["annotated_variables_from_init"]
        ]
        name_to_variable_map: Dict[str, Variable] = {
            annotated_variable.variable.name: annotated_variable.variable
            for annotated_variable in annotated_variable_from_init
        }
        targets: List[Variable] = [
            name_to_variable_map[name] for name in data["targets"]
        ]
        features: List[Variable] = [
            name_to_variable_map[name] for name in data["features"]
        ]
        _min_samples_leaf = data["min_samples_per_leaf"]
        min_impurity_improvement = data["min_impurity_improvement"]
        max_leaves = data["max_leaves"]
        max_depth = data["max_depth"]
        dependencies = VariableMap(
            {
                name_to_variable_map[name]: [
                    name_to_variable_map[dep_name] for dep_name in dep_names
                ]
                for name, dep_names in data["dependencies"].items()
            }
        )
        result = cls(
            annotated_variables=annotated_variable_from_init,
            targets=targets,
            features=features,
            min_samples_per_leaf=_min_samples_leaf,
            min_impurity_improvement=min_impurity_improvement,
            max_leaves=max_leaves,
            max_depth=max_depth,
            dependencies=dependencies,
        )
        result.total_samples = data["total_samples"]
        result.probabilistic_circuit = from_json(data["probabilistic_circuit"])
        return result

    def __eq__(self, other: Self):
        return (
            isinstance(other, self.__class__)
            and self.variables == other.variables
            and self.targets == other.targets
            and self.features == other.features
        )
