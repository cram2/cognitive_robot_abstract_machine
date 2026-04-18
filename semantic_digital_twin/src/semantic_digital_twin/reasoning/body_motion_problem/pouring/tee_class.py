"""
TEE class definition for the pouring domain (D_pour).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from semantic_digital_twin.reasoning.body_motion_problem.types import TEEClass


@dataclass
class PouringTEEClass(TEEClass):
    """
    D_pour: pouring domain under Torricelli-based fluid dynamics.

    Scopes the BMP to task type ``"pour"`` with the Torricelli ODE as the governing
    physics model. The validity interval ``fill_level ∈ [0, 1]`` bounds the
    normalised fill level for which the physics model produces valid predictions.
    """

    task_types: frozenset = field(default=frozenset({"pour"}))

    validity_intervals: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"fill_level": (0.0, 1.0)}
    )
