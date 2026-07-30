"""
Torch architecture of the learned pouring head-above-lip surrogate.

..warning:: This module imports :mod:`torch` at import time. It is only imported lazily by
    :mod:`semantic_digital_twin.physics.equations.learned_pouring_equations`, so the rest of the
    package stays importable without torch installed.
"""

from __future__ import annotations

import math

import torch


class HeadSurrogate(torch.nn.Module):
    """
    Smooth 2->1 MLP mapping raw ``(tilt[rad], fill)`` to the head above the pouring lip.

    Input normalization is baked into :meth:`forward`, so the network drops into the
    symbolic pipeline unchanged. Trained with Sobolev (value + gradient) supervision
    against the analytic head, so the gradients the MPC linearizes are accurate, not
    just the values.
    """

    def __init__(self, hidden_width: int = 64):
        """
        Build the two-hidden-layer tanh MLP.

        :param hidden_width: Number of units in each of the two hidden layers; a
            checkpoint can only be loaded into a network built with the width it was
            trained with.
        """
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_width),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_width, hidden_width),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_width, 1),
        )
        """
        Layer stack mapping the normalized ``(tilt, fill)`` pair to the predicted head.
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Batch of raw ``(tilt[rad], fill)`` rows.
        :return: Predicted head above the lip per row, in metres.
        """
        tilt = x[:, 0:1] / (math.pi / 2)
        fill = x[:, 1:2]
        return self.net(torch.cat([tilt, fill], dim=1))
