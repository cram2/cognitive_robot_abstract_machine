from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


@dataclass
class DifferentialEquation(ABC):
    """
    Abstract base for first-order ordinary differential equations.
    """
