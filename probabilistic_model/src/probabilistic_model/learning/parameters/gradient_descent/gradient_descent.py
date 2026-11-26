"""
Training the parameters of a layered circuit by gradient descent, with a loss that is a
pure function of the parameter arrays, on any array library that a subclass of
:class:`GradientDescent` supports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
import functools
from dataclasses import dataclass, field

import tqdm
import array_api_compat
from array_api_compat import array_namespace
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from random_events.variable import Symbolic
from sortedcontainers import SortedSet
from typing_extensions import Generic, Self

from probabilistic_model.learning.parameters.circuit_parameter_learning import (
    CircuitParameterLearning,
    CircuitParameterTraining,
    LayerType,
)
from probabilistic_model.learning.parameters.training_schedule import (
    LearningRatePlateau,
    MiniBatches,
    PlateauScheduledTraining,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)

# %% losses


@dataclass
class Loss(Generic[LayerType], SubClassSafeGeneric, ABC):
    """
    A scalar loss of the circuit rooted at a layer, computed on any array backend.
    """

    @abstractmethod
    def evaluate(self, layer: LayerType, *arguments: BackendArray) -> BackendArray:
        """
        :param layer: The root of the circuit to evaluate.
        :param arguments: The arrays the loss is computed on, such as data and labels.
        :return: The scalar loss of the circuit on the arguments.
        """
        raise NotImplementedError


@dataclass
class AverageNegativeLogLikelihood(Loss[LayerType]):
    """
    The negative average log-likelihood of the data under the nodes of the root.
    """

    def evaluate(self, layer: LayerType, data: BackendArray) -> BackendArray:
        log_likelihoods = layer.log_likelihood_of_nodes(events=data)
        return -array_namespace(log_likelihoods).mean(log_likelihoods)


@dataclass(kw_only=True)
class CrossEntropy(Loss[LayerType]):
    """
    The average negative log-probability of the category of the predicted variable in
    each row given its other values, the discriminative loss of a RAT-SPN classifier
    (Peharz et al., 2019, equation 4) with the prior the circuit holds.
    """

    predicted_column: int
    """
    The column of the predicted variable in the encoded data.
    """

    number_of_categories: int
    """
    The number of categories of the predicted variable, encoded as their positions in
    its domain.
    """

    @classmethod
    def from_predicted_variable(
        cls, *, predicted_variable: Symbolic, variables: SortedSet
    ) -> Self:
        """
        :param predicted_variable: The symbolic variable the circuit classifies into.
        :param variables: The variables of the circuit, which include the predicted
            variable.
        :return: The cross entropy of the predicted variable of the circuit.
        """
        return cls(
            predicted_column=variables.index(predicted_variable),
            number_of_categories=len(predicted_variable.domain.simple_sets),
        )

    def evaluate(self, layer: LayerType, data: BackendArray) -> BackendArray:
        log_likelihoods = layer.log_likelihood_of_nodes(events=data)[:, 0]
        namespace = array_namespace(log_likelihoods)
        data = namespace.asarray(data, device=array_api_compat.device(log_likelihoods))
        is_predicted_column = (
            namespace.arange(data.shape[1], device=array_api_compat.device(data))
            == self.predicted_column
        )
        log_likelihoods_of_categories = [
            layer.log_likelihood_of_nodes(
                events=namespace.where(
                    is_predicted_column, namespace.full_like(data, category), data
                )
            )[:, 0]
            for category in range(self.number_of_categories)
        ]
        marginal = functools.reduce(namespace.logaddexp, log_likelihoods_of_categories)
        return namespace.mean(marginal - log_likelihoods)


# %% gradient descent


@dataclass(kw_only=True)
class GradientDescentTraining(
    PlateauScheduledTraining, CircuitParameterTraining[LayerType], ABC
):
    """
    Mini-batch training of the parameters of a circuit by a :class:`GradientDescent`.
    """


@dataclass(kw_only=True)
class GradientDescent(CircuitParameterLearning[LayerType], ABC):
    """
    Minimizes a loss of the circuit rooted at a layer by gradient descent, with the
    array library of its subclass, on the flat list of the layer's
    ``circuit_parameters``, projected onto valid parameters after every step.
    """

    loss: Loss[LayerType]
    """
    The loss to minimize.
    """

    learning_rate_plateau: LearningRatePlateau | None = None
    """
    The schedule of the learning rate, copied for every training, a constant learning
    rate if ``None``.
    """

    zero_probability_entries: list[BackendArray] = field(init=False)
    """
    Where every parameter array is ``-inf`` before training, a log-probability or log-
    weight of zero probability, which training keeps.
    """

    def __post_init__(self) -> None:
        """
        Record the :attr:`zero_probability_entries` of the circuit.
        """
        super().__post_init__()
        self.zero_probability_entries = [
            array_namespace(parameter).isinf(parameter) & (parameter < 0)
            for parameter in self.layer.circuit_parameters
        ]

    @abstractmethod
    def start_training(
        self,
        *,
        arguments: tuple[BackendArray, ...],
        mini_batches: MiniBatches,
    ) -> GradientDescentTraining[LayerType]:
        """
        Like :meth:`CircuitParameterLearning.start_training`, with the arrays the
        :attr:`loss` takes after the circuit as ``arguments``.
        """
        raise NotImplementedError

    def copied_learning_rate_plateau(self) -> LearningRatePlateau | None:
        """
        :return: A copy of :attr:`learning_rate_plateau` that no epoch was recorded in
            yet, ``None`` for a constant learning rate.
        """
        if self.learning_rate_plateau is None:
            return None
        return dataclasses.replace(self.learning_rate_plateau)

    def minimize(self, *arguments: BackendArray, epochs: int) -> LayerType:
        """
        :param arguments: The arrays the :attr:`loss` takes after the circuit.
        :param epochs: The number of epochs, one step on all rows each.
        :return: The root of the trained circuit.
        """
        training = self.start_training(arguments=arguments, mini_batches=MiniBatches())
        progress_bar = tqdm.tqdm(range(epochs), desc="Fitting")
        for _ in progress_bar:
            loss = float(training.step_on_batch(batch=arguments))
            progress_bar.set_postfix_str(f"Loss: {loss}")
        return training.trained_root()
