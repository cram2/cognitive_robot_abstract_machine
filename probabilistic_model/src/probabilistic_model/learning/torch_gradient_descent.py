"""
Training the parameters of a tensorized circuit with torch: the parameter arrays are
leaf tensors of a torch optimizer and the loss is differentiated by autograd.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import tqdm
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from typing_extensions import Generic

from probabilistic_model.learning.gradient_descent import (
    ClassConditionalLikelihoodLoss,
    LayerType,
    LikelihoodLoss,
)


# %% gradient descent


@dataclass
class TorchGradientDescent(Generic[LayerType], SubClassSafeGeneric, ABC):
    """
    Minimizes a loss of the circuit rooted at a tensorized layer whose parameters are
    torch tensors, one optimizer step per epoch.

    After every step the parameters are projected onto valid parameters with
    ``projected_circuit_parameters``.
    """

    layer: LayerType
    """
    The root of the circuit to train, which is left unchanged.
    """

    learning_rate: float = 1e-3
    """
    The learning rate of the optimizer.
    """

    optimizer_type: type[torch.optim.Optimizer] = torch.optim.Adam
    """
    The torch optimizer to use.
    """

    parameters: list[torch.Tensor] = field(init=False)
    """
    The leaf tensors the optimizer updates, copies of the parameters of the circuit.
    """

    def __post_init__(self) -> None:
        self.parameters = [
            parameter.detach().clone().requires_grad_(True)
            for parameter in self.layer.circuit_parameters
        ]

    @abstractmethod
    def loss(self, layer: LayerType, *arguments: torch.Tensor) -> torch.Tensor:
        """
        :param layer: The root of the circuit to evaluate.
        :param arguments: The tensors passed to :meth:`minimize`.
        :return: The scalar loss of the circuit on the arguments.
        """

    def minimize(self, epochs: int, *arguments: torch.Tensor) -> LayerType:
        """
        :param epochs: The number of epochs.
        :param arguments: The tensors :meth:`loss` takes after the circuit.
        :return: The root of the trained circuit.
        """
        optimizer = self.optimizer_type(self.parameters, lr=self.learning_rate)
        progress_bar = tqdm.tqdm(range(epochs), desc="Fitting")
        for _ in progress_bar:
            optimizer.zero_grad()
            loss = self.loss(
                self.layer.copy_with_circuit_parameters(self.parameters), *arguments
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                projected = self.layer.projected_circuit_parameters(self.parameters)
                for parameter, valid in zip(self.parameters, projected):
                    parameter.copy_(valid)
            progress_bar.set_postfix_str(f"Loss: {loss.item()}")
        return self.layer.copy_with_circuit_parameters(
            [parameter.detach() for parameter in self.parameters]
        )


# %% losses


@dataclass
class TorchLikelihoodMaximization(
    LikelihoodLoss[LayerType], TorchGradientDescent[LayerType]
):
    """
    Maximizes the average log-likelihood of the data under the nodes of the root with
    torch.
    """


@dataclass
class TorchClassConditionalLikelihoodMaximization(
    ClassConditionalLikelihoodLoss[LayerType], TorchGradientDescent[LayerType]
):
    """
    Maximizes the average log-likelihood of each row under the node of its class with
    torch.
    """
