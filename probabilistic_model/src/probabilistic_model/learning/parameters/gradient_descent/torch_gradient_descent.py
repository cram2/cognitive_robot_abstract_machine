"""
Training the parameters of a tensorized circuit with torch: the parameter arrays are
leaf tensors of a torch optimizer and the loss is differentiated by autograd.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

import torch

from probabilistic_model.learning.parameters.circuit_parameter_learning import LayerType
from probabilistic_model.learning.parameters.gradient_descent.gradient_descent import (
    GradientDescent,
    GradientDescentTraining,
)
from probabilistic_model.learning.parameters.torch_circuit_parameter_learning import (
    TorchCircuitParameterLearning,
    TorchCircuitParameterTraining,
)
from probabilistic_model.learning.parameters.training_schedule import (
    LearningRatePlateau,
    MiniBatches,
)


# %% gradient descent


@dataclass(kw_only=True)
class TorchGradientDescent(
    GradientDescent[LayerType], TorchCircuitParameterLearning[LayerType]
):
    """
    Gradient descent with torch, one optimizer step per batch, after which the
    parameters are projected onto valid parameters with
    ``projected_circuit_parameters``.

    A recorded step needs an optimizer that accepts ``capturable``.
    """

    learning_rate: float = 1e-3
    """
    The learning rate of the optimizer.
    """

    optimizer_type: type[torch.optim.Optimizer] = torch.optim.Adam
    """
    The torch optimizer to use.
    """

    def start_training(
        self,
        *,
        arguments: tuple[torch.Tensor, ...],
        mini_batches: MiniBatches,
        learning_rate_plateau: LearningRatePlateau | None = None,
    ) -> TorchGradientDescentTraining[LayerType]:
        return TorchGradientDescentTraining(
            arguments=arguments,
            mini_batches=mini_batches,
            learning=self,
            replays_recorded_steps=self.replays_recorded_steps,
            optimizer=(
                self.optimizer_type(
                    self.parameters,
                    lr=torch.tensor(
                        self.learning_rate,
                        dtype=self.parameters[0].dtype,
                        device=self.parameters[0].device,
                    ),
                    capturable=True,
                )
                if self.replays_recorded_steps
                else self.optimizer_type(self.parameters, lr=self.learning_rate)
            ),
            learning_rate_plateau=learning_rate_plateau,
        )

    def compute_step_without_update(self, *, batch: tuple[torch.Tensor, ...]) -> None:
        """
        The loss and its gradient on the batch, see
        :meth:`TorchCircuitParameterLearning.compute_step_without_update`; the gradient
        is discarded afterwards.
        """
        self.loss.evaluate(
            self.layer.copy_with_circuit_parameters(parameters=self.parameters), *batch
        ).backward()
        for parameter in self.parameters:
            parameter.grad = None

    def step(
        self, *, optimizer: torch.optim.Optimizer, arguments: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Update the parameters by one optimizer step and project them onto valid
        parameters.

        :param optimizer: The optimizer of :attr:`parameters`.
        :param arguments: The tensors the :attr:`loss` takes after the circuit.
        :return: The loss of the parameters before the step, a scalar tensor without a
            gradient on the device of the parameters.
        """
        optimizer.zero_grad()
        loss = self.loss.evaluate(
            self.layer.copy_with_circuit_parameters(parameters=self.parameters),
            *arguments,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for parameter, zero_probability in zip(
                self.parameters, self.zero_probability_entries
            ):
                parameter.masked_fill_(zero_probability, -torch.inf)
            projected = self.layer.projected_circuit_parameters(
                parameters=self.parameters
            )
            for parameter, valid in zip(self.parameters, projected):
                parameter.copy_(valid)
        return loss.detach()


# %% training on mini-batches


@dataclass(kw_only=True)
class TorchGradientDescentTraining(
    GradientDescentTraining[LayerType], TorchCircuitParameterTraining[LayerType]
):
    """
    Mini-batch training by :class:`TorchGradientDescent`, one optimizer step per batch.
    """

    learning: TorchGradientDescent[LayerType]
    """
    The gradient descent whose parameters are updated.
    """

    optimizer: torch.optim.Optimizer
    """
    The optimizer of the parameters of :attr:`learning`.
    """

    def take_step(self, *, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        :param batch: The same rows of every training array.
        :return: The loss of the batch before one optimizer step on it.
        """
        return self.learning.step(optimizer=self.optimizer, arguments=batch)

    def step_on_batch(self, *, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.take_step_or_replay(step=self.take_step, batch=batch)

    def scale_learning_rate(self, *, scale: float) -> None:
        learning_rate = self.learning.learning_rate * scale
        for group in self.optimizer.param_groups:
            if isinstance(group["lr"], torch.Tensor):
                # a recorded step reads the learning rate from this tensor
                group["lr"].fill_(learning_rate)
            else:
                group["lr"] = learning_rate
