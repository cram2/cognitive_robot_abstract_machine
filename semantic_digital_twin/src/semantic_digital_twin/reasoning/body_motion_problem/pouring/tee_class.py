"""
TEE class definition for the pouring domain (D_pour).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.reasoning.body_motion_problem.types import TEEClass


@dataclass
class PouringTEEClass(TEEClass):
    """
    D_pour: pouring domain under Torricelli-based fluid dynamics.

    Scopes the BMP to task type ``"pour"`` with the Torricelli ODE as the governing
    physics model.
    """

    task_types: frozenset = field(default=frozenset({"pour"}))
