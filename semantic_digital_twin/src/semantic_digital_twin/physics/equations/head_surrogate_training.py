"""Training for the learned pouring head-above-lip surrogate.

Distills :meth:`~semantic_digital_twin.physics.equations.pouring_equations.ArticulatedPouringEquation.head_above_lip`
into a small smooth MLP with Sobolev (value + gradient) supervision, so the gradients the MPC
linearizes are accurate, not just the values.

Run as a module to train a checkpoint for a cup geometry::

    python -m semantic_digital_twin.physics.equations.head_surrogate_training \\
        --container-height 0.1 --container-width 0.08 --checkpoint head_surrogate.pt

..warning:: This module imports :mod:`torch` at import time. Import it only where the learned
    pouring feature is actually used, so the rest of the package stays importable without torch.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from semantic_digital_twin.physics.equations.head_surrogate_network import HeadSurrogate

TRAINING_TILT_RANGE: tuple[float, float] = (-0.2, math.pi / 2 + 0.2)
"""Tilt-angle interval the surrogate is trained on, in radians; extends slightly past the
physical ``[0, π/2]`` range so the fit stays accurate at the boundaries."""

TRAINING_FILL_RANGE: tuple[float, float] = (0.0, 1.0)
"""Normalized fill-level interval the surrogate is trained on."""


def analytic_head_torch(
    tilt: torch.Tensor,
    fill: torch.Tensor,
    container_height: float,
    container_width: float,
) -> torch.Tensor:
    """
    Torch reimplementation of the analytic head-above-lip, the surrogate's training target.

    :param tilt: Tilt angles about the vertical, in radians.
    :param fill: Normalized fill levels in ``[0, 1]``.
    :param container_height: Inner height of the rectangular container, in metres.
    :param container_width: Inner width of the rectangular container, in metres.
    :return: Head above the pouring lip per row, in metres.
    """
    A = container_height
    r = container_width / 2.0
    empty = A * (1.0 - fill)
    L = torch.sqrt(empty**2 + r**2)
    phi = torch.atan2(empty, torch.full_like(empty, r))
    return torch.relu(L * torch.sin(tilt - phi))


@dataclass
class HeadSurrogateTrainer:
    """
    Trains a :class:`~semantic_digital_twin.physics.equations.head_surrogate_network.HeadSurrogate`
    for one container geometry with Sobolev (value + gradient) supervision against the analytic head.
    """

    container_height: float
    """Inner height of the rectangular container the surrogate is trained for, in metres."""

    container_width: float
    """Inner width of the rectangular container the surrogate is trained for, in metres."""

    seed: int = 0
    """Random seed for the sample draw and the network initialization."""

    sample_count: int = 20000
    """Number of ``(tilt, fill)`` training samples drawn uniformly from the training ranges."""

    epochs: int = 4000
    """Number of full-batch Adam optimization steps."""

    gradient_weight: float = 0.1
    """Weight of the gradient-matching term relative to the value-matching term in the loss."""

    learning_rate: float = 2e-3
    """Adam learning rate."""

    def train(self) -> HeadSurrogate:
        """
        :return: The trained surrogate in evaluation mode.
        """
        torch.manual_seed(self.seed)
        model = HeadSurrogate()
        inputs = self._sample_inputs()
        target_value, target_gradient = self._target_value_and_gradient(inputs)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.MSELoss()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            x = inputs.clone().requires_grad_(True)
            prediction = model(x)
            prediction_gradient = torch.autograd.grad(
                prediction.sum(), x, create_graph=True
            )[0]
            loss = loss_function(prediction, target_value) + self.gradient_weight * (
                loss_function(prediction_gradient, target_gradient)
            )
            loss.backward()
            optimizer.step()
        return model.eval()

    def _sample_inputs(self) -> torch.Tensor:
        """Uniform ``(tilt, fill)`` samples over the training ranges."""
        generator = torch.Generator().manual_seed(self.seed)
        tilt = torch.empty(self.sample_count, 1).uniform_(
            *TRAINING_TILT_RANGE, generator=generator
        )
        fill = torch.empty(self.sample_count, 1).uniform_(
            *TRAINING_FILL_RANGE, generator=generator
        )
        return torch.cat([tilt, fill], dim=1)

    def _target_value_and_gradient(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytic head values and input gradients at ``inputs``, the Sobolev training targets."""
        x = inputs.clone().requires_grad_(True)
        head = analytic_head_torch(
            x[:, 0:1], x[:, 1:2], self.container_height, self.container_width
        )
        gradient = torch.autograd.grad(head.sum(), x, create_graph=False)[0]
        return head.detach(), gradient.detach()


def _main() -> None:
    """Train a head surrogate for the given geometry and save its checkpoint."""
    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("--container-height", type=float, required=True)
    parser.add_argument("--container-width", type=float, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--gradient-weight", type=float, default=0.3)
    arguments = parser.parse_args()
    trainer = HeadSurrogateTrainer(
        container_height=arguments.container_height,
        container_width=arguments.container_width,
        epochs=arguments.epochs,
        gradient_weight=arguments.gradient_weight,
    )
    surrogate = trainer.train()
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(surrogate.state_dict(), str(arguments.checkpoint))
    print(f"saved {arguments.checkpoint}")


if __name__ == "__main__":
    _main()
