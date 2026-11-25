"""
The base of the differential equations the physics models are written as.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


@dataclass(kw_only=True)
class DifferentialEquation(ABC):
    """
    Abstract base for first-order ordinary differential equations.
    """
