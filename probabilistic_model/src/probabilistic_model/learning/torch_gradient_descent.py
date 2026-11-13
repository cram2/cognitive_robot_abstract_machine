"""
Training the parameters of a tensorized circuit with torch: the parameter arrays are
leaf tensors of a torch optimizer and the loss is differentiated by autograd.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import tqdm
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from typing_extensions import Generic

from probabilistic_model.learning.gradient_descent import LayerType, Loss


# %% gradient descent


@dataclass
class TorchGradientDescent(Generic[LayerType], SubClassSafeGeneric):
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

    loss: Loss[LayerType]
    """
    The loss to minimize.
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

    def minimize(self, epochs: int, *arguments: torch.Tensor) -> LayerType:
        """
        :param epochs: The number of epochs.
        :param arguments: The tensors the :attr:`loss` takes after the circuit.
        :return: The root of the trained circuit.
        """
        optimizer = self.optimizer_type(self.parameters, lr=self.learning_rate)
        progress_bar = tqdm.tqdm(range(epochs), desc="Fitting")
        for _ in progress_bar:
            optimizer.zero_grad()
            loss = self.loss.evaluate(
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
