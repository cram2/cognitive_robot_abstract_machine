"""
Abstract base for first-order ordinary differential equations.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


@dataclass
class DifferentialEquation(ABC):
    """
    Marker base for ODE types whose velocity is expressed as a symbolic constraint
    and registered directly with the QP controller.
    """
