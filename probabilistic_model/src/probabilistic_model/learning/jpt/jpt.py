import math
from collections import deque
from typing import Tuple, Union, Optional, List, Iterable, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jpt.learning.impurity import Impurity
from plotly.subplots import make_subplots
from random_events.product_algebra import VariableMap
from random_events.utils import SubclassJSONSerializer
from random_events.variable import Variable
from typing_extensions import Self

from .variables import Continuous, Integer, Symbolic, ScaledContinuous
from ..nyga_distribution import NygaDistribution
from ...distributions import (DiracDeltaDistribution, SymbolicDistribution, IntegerDistribution, UnivariateDistribution)
from ...probabilistic_circuit.rx.probabilistic_circuit import (SumUnit, ProductUnit,
                                                               ProbabilisticCircuit, UnivariateDiscreteLeaf)
from ...utils import MissingDict


class JPT(SubclassJSONSerializer):
    """
    Class that implements the JPT learning algorithm for probabilistic circuits.
    """

    targets: Tuple[Variable, ...]
    """
    The variables to optimize for.
    """

    features: Tuple[Variable, ...]
    """
    The variables that are used to craft criteria.
    """

    _min_samples_leaf: Union[int, float]
    """
    The minimum number of samples to create another sum node. If this is smaller than one, it will be reinterpreted
    as fraction w. r. t. the number of samples total.
    """

    min_impurity_improvement: float
    """
    The minimum impurity improvement to create another sum node.
    """

    max_leaves: Union[int, float]
    """
    The maximum number of leaves.
    """

    max_depth: Union[int, float]
    """
    The maximum depth of the tree.
    """

    dependencies: VariableMap
    """
    The dependencies between the variables.
    """

    total_samples: int = 1
    """
    The total amount of samples that were used to fit the model.
    """

    indices: Optional[np.ndarray] = None
    impurity: Optional[Impurity] = None
    c45queue: deque = deque()
    weights: List[float]

    keep_sample_indices: bool = False
    """
    Rather to store the sample indices in the leaves or not.
    """

    variables: Tuple[Variable, ...]
    """
    The variables from initialization. Since variables will be overwritten as soon as the model is learned,
    we need to store the variables from initialization here.
    """

    probabilistic_circuit: ProbabilisticCircuit
    """
    The probabilistic circuit the result will appear in.
    """

    root: Optional[SumUnit] = None
    """
    The root of the circuit that will be learned.
    """

    def __init__(self, variables: Iterable[Variable], targets: Optional[Iterable[Variable]] = None,
                 features: Optional[Iterable[Variable]] = None, min_samples_leaf: Union[int, float] = 1,
                 min_impurity_improvement: float = 0.0, max_leaves: Union[int, float] = float("inf"),
                 max_depth: Union[int, float] = float("inf"), dependencies: Optional[VariableMap] = None, ):
        self.variables = tuple(sorted(variables))
        self.set_targets_and_features(targets, features)
        self._min_samples_leaf = min_samples_leaf
        self.min_impurity_improvement = min_impurity_improvement
        self.max_leaves = max_leaves
        self.max_depth = max_depth

        if dependencies is None:
            self.dependencies = VariableMap({var: list(self.targets) for var in self.features})
        else:
            self.dependencies = dependencies

        self.probabilistic_circuit = ProbabilisticCircuit()

    def set_targets_and_features(self, targets: Optional[Iterable[Variable]],
                                 features: Optional[Iterable[Variable]]) -> None:
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
        if self._min_samples_leaf < 1.:
            return math.ceil(self._min_samples_leaf * self.total_samples)
        else:
            return self._min_samples_leaf

    @property
    def numeric_variables(self):
        return [variable for variable in self.variables if isinstance(variable, (Continuous, Integer))]

    @property
    def numeric_targets(self):
        return [variable for variable in self.targets if isinstance(variable, (Continuous, Integer))]

    @property
    def numeric_features(self):
        return [variable for variable in self.features if isinstance(variable, (Continuous, Integer))]

    @property
    def symbolic_variables(self):
        return [variable for variable in self.variables if isinstance(variable, Symbolic)]

    @property
    def symbolic_targets(self):
        return [variable for variable in self.targets if isinstance(variable, Symbolic)]

    @property
    def symbolic_features(self):
        return [variable for variable in self.features if isinstance(variable, Symbolic)]

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the data to be used in the model.

        :param data: The data to preprocess.
        :return: The preprocessed data.
        """

        result = np.zeros(data.shape)

        for variable_index, variable in enumerate(self.variables):
            column = data[variable.name]
            if isinstance(variable, ScaledContinuous):
                column = variable.encode(column)
            if isinstance(variable, Symbolic):
                all_elements = {element: index for index, element in enumerate(variable.domain.all_elements)}
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

        self.indices = np.ascontiguousarray(np.arange(preprocessed_data.shape[0], dtype=np.int64))
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

        for index, variable in enumerate(self.variables):
            if isinstance(variable, Continuous):
                distribution = NygaDistribution(variable,
                                                min_likelihood_improvement=variable.min_likelihood_improvement,
                                                min_samples_per_quantile=variable.min_samples_per_quantile)
                distribution = distribution.fit(data[:, index])

                if isinstance(distribution.root, DiracDeltaDistribution):
                    distribution.root.density_cap = 1 / variable.minimal_distance
                nyga_root = distribution.root
                new_nodes = self.probabilistic_circuit.mount(nyga_root)
                result.add_subcircuit(new_nodes[nyga_root.index])

            elif isinstance(variable, Symbolic):
                distribution = SymbolicDistribution(variable, probabilities=MissingDict(float))
                distribution.fit_from_indices(data[:, index].astype(int))
                distribution = UnivariateDiscreteLeaf(distribution,
                                                      probabilistic_circuit=self.probabilistic_circuit)
                result.add_subcircuit(distribution)

            elif isinstance(variable, Integer):
                distribution = IntegerDistribution(variable, probabilities=MissingDict(float))
                distribution.fit(data[:, index])
                distribution = UnivariateDiscreteLeaf(distribution,
                                                      probabilistic_circuit=self.probabilistic_circuit)
                result.add_subcircuit(distribution)

            else:
                raise ValueError(f"Variable {variable} is not supported.")


        return result

    def construct_impurity(self) -> Impurity:
        min_samples_leaf = self.min_samples_leaf

        numeric_vars = (
            np.array([index for index, variable in enumerate(self.variables)
                      if variable in self.numeric_targets], dtype=int))
        symbolic_vars = np.array(
            [index for index, variable in enumerate(self.variables)
             if variable in self.symbolic_targets], dtype=int)

        invert_impurity = np.array([0] * len(self.symbolic_targets), dtype=int)

        n_sym_vars_total = len(self.symbolic_variables)
        n_num_vars_total = len(self.numeric_variables)

        numeric_features = np.array(
            [index for index, variable in enumerate(self.variables)
             if variable in self.numeric_features], dtype=int)
        symbolic_features = np.array(
            [index for index, variable in enumerate(self.variables)
             if variable in self.symbolic_features], dtype=int)

        symbols = np.array([len(variable.domain.simple_sets) for variable in self.symbolic_variables])
        max_variances = np.array([variable.std ** 2 for variable in self.numeric_variables])

        dependency_indices = dict()

        for variable, dep_vars in self.dependencies.items():
            # get the index version of the dependent variables and store them
            idx_var = self.variables.index(variable)
            idc_dep = [self.variables.index(var) for var in dep_vars]
            dependency_indices[idx_var] = idc_dep

        return Impurity(min_samples_leaf, numeric_vars, symbolic_vars, invert_impurity, n_sym_vars_total,
                        n_num_vars_total, numeric_features, symbolic_features, symbols, max_variances,
                        dependency_indices)

    def _variable_dependencies_to_json(self) -> Dict[str, List[str]]:
        """
        Convert the variable dependencies to a json compatible format.
        The result maps variable names to lists of variable names.
        """
        return {variable.name: [dependency.name for dependency in dependencies]
                for variable, dependencies in self.dependencies.items()}

    def empty_copy(self):
        result = self.__class__(variables=self.variables,
                                targets=self.targets,
                                features=self.features,
                                min_samples_leaf=self.min_samples_leaf,
                                min_impurity_improvement=self.min_impurity_improvement,
                                max_depth=self.max_depth,
                                dependencies=self.dependencies)
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["variables_from_init"] = [variable.to_json() for variable in self.variables]
        result["targets"] = [variable.name for variable in self.targets]
        result["features"] = [variable.name for variable in self.features]
        result["_min_samples_leaf"] = self._min_samples_leaf
        result["min_impurity_improvement"] = self.min_impurity_improvement
        result["max_leaves"] = self.max_leaves
        result["max_depth"] = self.max_depth
        result["dependencies"] = self._variable_dependencies_to_json()
        result["total_samples"] = self.total_samples
        result["probabilistic_circuit"] = self.probabilistic_circuit.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variable_from_init = [Variable.from_json(variable) for variable in data["variables_from_init"]]
        name_to_variable_map = {variable.name: variable for variable in variable_from_init}
        targets = [name_to_variable_map[name] for name in data["targets"]]
        features = [name_to_variable_map[name] for name in data["features"]]
        _min_samples_leaf = data["_min_samples_leaf"]
        min_impurity_improvement = data["min_impurity_improvement"]
        max_leaves = data["max_leaves"]
        max_depth = data["max_depth"]
        dependencies = VariableMap({name_to_variable_map[name]: [name_to_variable_map[dep_name] for dep_name
                                                                 in dep_names] for name, dep_names
                                    in data["dependencies"].items()})
        result = cls(variables=variable_from_init, targets=targets, features=features,
                     min_samples_leaf=_min_samples_leaf, min_impurity_improvement=min_impurity_improvement,
                     max_leaves=max_leaves, max_depth=max_depth, dependencies=dependencies)
        result.total_samples = data["total_samples"]
        result.probabilistic_circuit = ProbabilisticCircuit.from_json(data["probabilistic_circuit"])
        return result

    def __eq__(self, other: Self):
        return (isinstance(other, self.__class__) and self.variables == other.variables and
                self.targets == other.targets and self.features == other.features)
