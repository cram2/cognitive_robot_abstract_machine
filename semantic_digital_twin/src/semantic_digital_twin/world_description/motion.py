"""
Motion type for the Body Motion Problem framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from semantic_digital_twin.physics.physics_model import PhysicsModel
from semantic_digital_twin.world_description.world_entity import Connection


@dataclass(eq=True)
class Motion:
    """
    Represents a candidate motion as a sequence of actuator positions.

    When a physics model is provided and the trajectory is empty, the model
    is used to generate the trajectory on demand during the causal sufficiency check.
    """

    trajectory: list[float]
    """Trajectory points in actuator space."""

    actuator: Connection
    """The connection (joint) that is manipulated by this motion."""

    motion_model: Optional[PhysicsModel] = field(default=None)
    """Optional physics model used to generate the trajectory when it is empty."""

    secondary_trajectories: list[tuple[Connection, list[float]]] = field(
        default_factory=list
    )
    """Additional coupled actuator trajectories replayed alongside the primary."""

    time_step: Optional[float] = field(default=None)
    """
    Time step between trajectory samples in seconds.

    When set, :meth:`~semantic_digital_twin.reasoning.bmp_predicates.Causes` steps
    all HasUpdateState connections after each position update, allowing coupled
    physics (e.g. fill-level ODEs) to be inferred from the primary trajectory
    without explicit secondary trajectories.
    """
