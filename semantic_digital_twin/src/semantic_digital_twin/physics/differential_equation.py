"""
Abstract base for first-order ordinary differential equations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DifferentialEquation(ABC):
    """
    Abstract first-order ODE: dx/dt = f(x, inputs).

    Reads its input signals directly from the world state via the connection
    objects it holds, so callers only need to advance the world and call
    :meth:`step`.
    """

    @abstractmethod
    def step(self, state: float, dt: float) -> float:
        """
        Euler-integrate one step and return the next state x(t + dt).

        :param state: Current value of the governed DOF.
        :param dt: Integration timestep in seconds.
        :return: New state after one Euler step.
        """
