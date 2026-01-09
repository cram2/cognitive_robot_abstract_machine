import itertools

import numpy as np
from random_events.product_algebra import SimpleEvent, Event
from random_events.utils import SubclassJSONSerializer
from random_events.variable import Symbolic, Variable
from typing_extensions import Self, Any, Iterable, List, Optional, Tuple, Dict

from ..probabilistic_circuit.rx.probabilistic_circuit import SymbolicDistribution, UnivariateDiscreteLeaf, ProductUnit, \
    SumUnit, ProbabilisticCircuit, leaf
from ..probabilistic_model import ProbabilisticModel
from ..utils import MissingDict


class MultinomialDistribution(ProbabilisticModel, SubclassJSONSerializer):
    """
    A multinomial distribution over symbolic random variables.
    """

    _variables: Tuple[Symbolic, ...]
    """
    The variables of the distribution.
    """

    probabilities: np.ndarray
    """
    The probability mass function. The dimensions correspond to the variables in the same order.
    The first dimension indexes over the first variable and so on. If no probabilities are provided in the constructor,
    the probabilities are initialized with ones.
    """

    def __init__(self, variables: Iterable[Symbolic], probabilities: Optional[np.ndarray] = None):
        super().__init__()
        self._variables = tuple(variables)

        shape = tuple(len(variable.domain.simple_sets) for variable in self.variables)

        if probabilities is None:
            probabilities = np.ones(shape)
            probabilities /= probabilities.sum()

        if shape != probabilities.shape:
            raise ValueError("The number of variables must match the number of dimensions in the probability array."
                             "Variables: {}".format(self.variables), "Dimensions: {}".format(probabilities.shape))
        self.probabilities = probabilities

    @property
    def variables(self) -> Tuple[Symbolic, ...]:
        return self._variables

    @property
    def support(self) -> Event:
        raise NotImplementedError

    def sample(self, amount: int) -> np.array:
        return None

    def marginal(self, variables: Iterable[Symbolic]) -> Self:

        # calculate which variables to marginalize over as the difference between variables and self.variables
        axis = tuple(self.variables.index(variable) for variable in self.variables if variable not in variables)

        # marginalize the probabilities over the axis
        probabilities = np.sum(self.probabilities, axis=axis)

        result = MultinomialDistribution(variables, probabilities)
        result.normalize()
        return result

    def log_mode(self) -> Tuple[Event, float]:
        likelihood = np.max(self.probabilities)
        indices_of_maximum = np.transpose(np.asarray(self.probabilities == likelihood).nonzero())

        hash_map_variable_values = {variable: list(variable.domain.hash_map.values()) for variable in self.variables}

        mode = None
        for index_of_maximum in indices_of_maximum:

            current_mode = SimpleEvent({variable: hash_map_variable_values[variable][value] for
                                        variable, value in zip(self.variables, index_of_maximum)}).as_composite_set()
            if mode is None:
                mode = current_mode
            else:
                mode |= current_mode

        return mode, np.log(likelihood)

    def log_conditional(self, point: Dict[Variable, Any]) -> Tuple[Optional[Self], float]:
        event = SimpleEvent(point)
        event.fill_missing_variables(self.variables)
        return self.log_truncated(event.as_composite_set())

    def __copy__(self) -> Self:
        """
        :return: a shallow copy of the distribution.
        """
        return MultinomialDistribution(self.variables, self.probabilities)

    def __deepcopy__(self, memo=None) -> Self:
        """
        :param memo: A dictionary that is used to keep track of objects that have already been copied.
        :return: a deep copy of the distribution.
        """
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        import copy
        variables = copy.deepcopy(self.variables, memo)
        probabilities = copy.deepcopy(self.probabilities, memo)
        result = MultinomialDistribution(variables, probabilities)
        memo[id_self] = result
        return result

    def __eq__(self, other: Self) -> bool:
        """Compare self with other and return the boolean result.

        Two discrete random variables are equal only if the probability mass
        functions are equal and the order of dimensions are equal.

        """
        return (isinstance(other,
                           self.__class__) and self.variables == other.variables and self.probabilities.shape == other.probabilities.shape and np.allclose(
            self.probabilities, other.probabilities))

    def __str__(self):
        return "P({}): \n".format(", ".join(var.name for var in self.variables)) + str(self.probabilities)

    def to_tabulate(self) -> List[List[str]]:
        """
        :return: a pretty table of the distribution.
        """
        columns = [[var.name for var in self.variables] + ["P"]]
        events = list(list(event) for event in itertools.product(
            *[[simple_set.element for simple_set in var.domain.simple_sets] for var in self.variables]))
        events = np.concatenate((events, self.probabilities.reshape(-1, 1)), axis=1).tolist()
        table = columns + events
        return table

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        indices = self.indices_from_simple_event(event)
        return self.probabilities[np.ix_(*indices)].sum()

    def log_likelihood(self, events: np.array) -> np.array:
        return np.log(self.probabilities[tuple(events.T)])

    def log_truncated(self, event: Event) -> Tuple[Optional[Self], float]:
        probabilities = np.zeros_like(self.probabilities)

        for simple_event in event.simple_sets:
            probabilities += self.probabilities_from_simple_event(simple_event)

        sum_of_probabilities = probabilities.sum()
        if sum_of_probabilities == 0:
            return None, -np.inf

        result = MultinomialDistribution(self.variables, probabilities)
        result.normalize()
        return result, np.log(sum_of_probabilities)

    def indices_from_simple_event(self, event: SimpleEvent) -> Tuple[List[int], ...]:
        """
        Calculate the indices that can be used to access the underlying probability array from a simple event.

        :param event: The simple event.
        :return: The indices.
        """
        hash_map_variable_keys = {variable: list(variable.domain.hash_map) for variable in self.variables}
        return tuple([hash_map_variable_keys[variable].index(hash(simple_set))
                      for simple_set in event[variable]] for variable in self.variables)

    def probabilities_from_simple_event(self, event: SimpleEvent) -> np.ndarray:
        """
        Calculate the probabilities array for a simple event.

        :param event: The simple event.
        :return: The array of probabilities that apply to this event.
        """
        indices = np.ix_(*self.indices_from_simple_event(event))
        probabilities = np.zeros_like(self.probabilities)
        probabilities[indices] = self.probabilities[indices]
        return probabilities

    def normalize(self):
        """
        Normalize the distribution inplace.
        """
        normalized_probabilities = self.probabilities / np.sum(self.probabilities)
        self.probabilities = normalized_probabilities

    def as_probabilistic_circuit(self) -> SumUnit:
        """
        Convert this distribution to a probabilistic circuit. A deterministic sum unit with decomposable children is
        used to describe every state. The size of the circuit is equal to the size of `self.probabilities`.

        :return: The distribution as a probabilistic circuit.
        """

        pc = ProbabilisticCircuit()

        # initialize the result as a deterministic sum unit
        result = SumUnit(probabilistic_circuit=pc)

        # iterate through all states of this distribution
        for event in itertools.product(*[list(range(len(variable.domain.simple_sets)))
                                         for variable in self.variables]):

            # create a product unit for the current state
            product_unit = ProductUnit(probabilistic_circuit=pc)

            # iterate through all variables
            for variable, value in zip(self.variables, event):
                # create probabilities for the current variables state as one hot encoding
                weights = MissingDict(float)
                weights[hash(value)] = 1.

                # create a distribution for the current variable
                distribution = SymbolicDistribution(variable, weights)

                # mount the distribution to the product unit
                product_unit.add_subcircuit(leaf(distribution, pc))

            # calculate the probability of the current state
            probability = self.likelihood(np.array([event]))[0]

            # mount the product unit to the result
            result.add_subcircuit(product_unit, np.log(probability))

        return result

    def encode_full_evidence_event(self, event: Iterable) -> List[int]:
        """
        Encode a full evidence event into a list of integers.
        :param event: The event to encode.
        :return: The encoded event.
        """
        return [variable.encode(value) for variable, value in zip(self.variables, event)]

    def fit(self, data: np.ndarray) -> Self:
        """
        Fit the distribution to the data.

        :param data: The data to fit the distribution to.
        :return: The fitted distribution.
        """
        self.probabilities = np.zeros_like(self.probabilities)
        uniques, counts = np.unique(data, return_counts=True, axis=0)
        self.probabilities[tuple(uniques.astype(int).T)] = counts
        self.normalize()
        return self

    def to_json(self) -> Dict[str, Any]:
        return {**SubclassJSONSerializer.to_json(self),
                "variables": [variable.to_json() for variable in self.variables],
                "probabilities": self.probabilities.tolist()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variables = [Variable.from_json(variable) for variable in data["variables"]]
        probabilities = np.array(data["probabilities"])
        return cls(variables, probabilities)
